"""
TRM PyTorch Lightning Module - Corrected to match paper pseudocode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from torch.optim import AdamW
import numpy as np
from lightning import LightningModule

from src.nn.utils import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)

class TRMModule(LightningModule):
    """
    PyTorch Lightning implementation of TRM (Tiny Recursive Model).
    Following the paper pseudocode exactly.
    """
    
    def __init__(self,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 max_grid_size: int = 30,
                 num_colors: int = 10,
                 n_latent_recursions: int = 2,  # n in paper
                 T_deep_recursions: int = 2,    # T in paper
                 N_supervision: int = 16,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 2000,
                 max_steps: int = 100000,
                 output_dir: str = None
                 ):
        """Initialize TRM Lightning Module."""
        super().__init__()
        self.save_hyperparameters()
        
        # CRITICAL: Use manual optimization
        self.automatic_optimization = False
        
        # Model components
        self.input_embedding = nn.Embedding(
            num_colors + 1,  # +1 for padding
            hidden_size
        )
        
        # Build L_net and H_net (hierarchical networks)
        self.L_net = self._build_network(hidden_size, num_layers)
        self.H_net = self._build_network(hidden_size, num_layers)
        
        # Output heads
        self.output_head = nn.Linear(hidden_size, num_colors)
        
        # Q-head returns 2 values: q[0] for halt, q[1] for continue
        self.Q_head = nn.Linear(hidden_size, 2)
        
        # Initialize z (split into zH and zL)
        # Each is half of hidden_size
        self.register_buffer('z_init_H', torch.zeros(1, 1, hidden_size // 2))
        self.register_buffer('z_init_L', torch.zeros(1, 1, hidden_size // 2))
    
    def _build_network(self, hidden_size: int, num_layers: int) -> nn.Module:
        """Build a network (L_net or H_net)."""
        layers = []
        for _ in range(num_layers):
            layers.append(nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_size),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size),
                    nn.Dropout(0.1)
                )
            }))
        return nn.ModuleList(layers)
    
    def _net_forward(self, net: nn.ModuleList, *inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through a network with multiple inputs."""
        # Concatenate or sum inputs
        combined = sum(inputs)  # Could also concatenate
        
        for layer in net:
            normalized = layer['norm'](combined)
            output = layer['mlp'](normalized)
            combined = combined + output  # Residual
        
        return combined
    
    def hrm(self, z: Tuple[torch.Tensor, torch.Tensor], 
            x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Hierarchical Reasoning Module - exactly as in paper pseudocode.
        
        Args:
            z: Tuple of (zH, zL)
            x: Input embeddings [batch, seq_len, hidden]
            
        Returns:
            z: Updated (zH, zL)
            y_pred: Output predictions
            q: Q-values [batch, 2] where q[0]=halt, q[1]=continue
        """
        zH, zL = z
        n = self.hparams.n_latent_recursions
        T = self.hparams.T_deep_recursions
        
        # n*T - 2 iterations without gradients
        with torch.no_grad():
            for i in range(n * T - 2):
                zL = self._net_forward(self.L_net, zL, zH, x)
                if (i + 1) % T == 0:
                    zH = self._net_forward(self.H_net, zH, zL)
        
        # 1-step grad (final 2 steps with gradients)
        zL = self._net_forward(self.L_net, zL, zH, x)
        zH = self._net_forward(self.H_net, zH, zL)
        
        # Output heads
        y_pred = self.output_head(zH)  # [batch, seq_len, num_colors]
        q = self.Q_head(zH.mean(dim=1))  # [batch, 2]
        
        return (zH, zL), y_pred, q
    
    def ACT_halt(self, q: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        ACT halt loss - learns when to stop.
        
        Args:
            q: Q-values [batch, 2]
            y_pred: Predictions [batch, seq_len, num_colors]
            y_true: Ground truth [batch, height, width]
        """
        # Reshape y_pred to match y_true
        batch_size = y_true.shape[0]
        height, width = y_true.shape[1], y_true.shape[2]
        y_pred_classes = y_pred.view(batch_size, height, width, -1).argmax(dim=-1)
        
        # Target: halt if prediction matches ground truth
        target_halt = (y_pred_classes == y_true).float().mean(dim=[1, 2])  # [batch]
        
        # Binary cross entropy with q[0] (halt signal)
        loss = 0.5 * F.binary_cross_entropy_with_logits(q[:, 0], target_halt)
        
        return loss
    
    def ACT_continue(self, q: torch.Tensor, last_step: bool) -> torch.Tensor:
        """
        ACT continue loss - learns to continue if not done.
        
        Args:
            q: Q-values [batch, 2]
            last_step: Whether this is the last supervision step
        """
        if last_step:
            # At last step, target = sigmoid(q[0])
            target_continue = torch.sigmoid(q[:, 0])
        else:
            # Not last step, target = sigmoid(max(q[0], q[1]))
            target_continue = torch.sigmoid(torch.max(q[:, 0], q[:, 1]))
        
        # Binary cross entropy with q[1] (continue signal)
        loss = 0.5 * F.binary_cross_entropy_with_logits(q[:, 1], target_continue)
        
        return loss
    
    def forward(self, x: torch.Tensor, supervision_steps: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass for inference.
        
        Args:
            x: Input tensor [batch, height, width]
            supervision_steps: Number of supervision steps
            
        Returns:
            Output predictions [batch, height, width, num_colors]
        """
        batch_size, height, width = x.shape
        
        # Embed input
        x_emb = self.input_embedding(x)
        x_emb = x_emb.view(batch_size, -1, self.hparams.hidden_size)
        
        # Initialize z
        seq_len = height * width
        zH = self.z_init_H.expand(batch_size, seq_len, -1).clone()
        zL = self.z_init_L.expand(batch_size, seq_len, -1).clone()
        z = (zH, zL)
        
        n_steps = supervision_steps or self.hparams.N_supervision
        
        for step in range(n_steps):
            z, y_pred, q = self.hrm(z, x_emb)
            
            # Early stopping: if q[0] > q[1], halt
            if not self.training:
                with torch.no_grad():
                    if (q[:, 0] > q[:, 1]).all():
                        break
            
            # Detach for next iteration
            zH, zL = z
            z = (zH.detach(), zL.detach())
        
        # Final output
        y_out = y_pred.view(batch_size, height, width, self.hparams.num_colors)
        return y_out
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step following paper pseudocode exactly.
        Uses manual optimization with backward inside the supervision loop.
        """
        # Get optimizer and scheduler
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        x_input = batch['input']
        y_true = batch['output']
        
        batch_size, height, width = x_input.shape
        
        # Initialize z
        seq_len = height * width
        zH = self.z_init_H.expand(batch_size, seq_len, -1).clone()
        zL = self.z_init_L.expand(batch_size, seq_len, -1).clone()
        z = (zH, zL)
        
        total_loss_value = 0
        n_steps = 0
        
        # Deep Supervision loop
        for step in range(self.hparams.N_supervision):
            # Input embedding
            x = self.input_embedding(x_input)
            x = x.view(batch_size, -1, self.hparams.hidden_size)
            
            # Hierarchical reasoning
            z, y_pred, q = self.hrm(z, x)
            
            # Reshape y_pred for cross entropy
            y_pred_reshaped = y_pred.view(batch_size, height, width, self.hparams.num_colors)
            
            # Main classification loss
            loss = F.cross_entropy(
                y_pred_reshaped.permute(0, 3, 1, 2),  # [batch, colors, H, W]
                y_true,
                ignore_index=0
            )
            
            # ACT halt loss
            loss += self.ACT_halt(q, y_pred, y_true)
            
            # ACT continue loss with extra forward pass
            with torch.no_grad():
                z_detached = (z[0].detach(), z[1].detach())
            _, _, q_next = self.hrm(z_detached, x)
            loss += self.ACT_continue(q_next, last_step=(step == self.hparams.N_supervision - 1))
            
            # Detach z for next iteration (before backward)
            zH, zL = z
            z = (zH.detach(), zL.detach())
            
            # CRITICAL: Backward and optimize INSIDE the loop
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            if sch is not None:
                sch.step()
            
            # Track loss for logging
            total_loss_value += loss.item()
            n_steps += 1
            
            # Log per-step metrics
            self.log(f'train/loss_step_{step}', loss.item(), on_step=False, on_epoch=True)
            
            # Early stopping: if q[0] > q[1]
            with torch.no_grad():
                q_halt = q[:, 0].mean().item()
                q_continue = q[:, 1].mean().item()
                if q_halt > q_continue:
                    self.log('train/early_stopped_at', float(step))
                    break
        
        # Log aggregate metrics
        avg_loss = total_loss_value / n_steps if n_steps > 0 else 0
        self.log('train/loss', avg_loss, prog_bar=True)
        self.log('train/n_steps', float(n_steps))
        
        return torch.tensor(avg_loss)
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        x = batch['input']
        y_true = batch['output']
        
        y_pred = self(x, supervision_steps=self.hparams.N_supervision)
        
        # Compute loss
        loss = F.cross_entropy(
            y_pred.permute(0, 3, 1, 2),
            y_true,
            ignore_index=0
        )
        
        # Compute accuracy
        pred_classes = y_pred.argmax(dim=-1)
        correct = (pred_classes == y_true).float()
        mask = (y_true != 0).float()
        
        if mask.sum() > 0:
            accuracy = (correct * mask).sum() / mask.sum()
        else:
            accuracy = correct.mean()
        
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/accuracy', accuracy, prog_bar=True)
        
        return {'loss': loss, 'accuracy': accuracy}
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate schedule with warmup
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / self.hparams.warmup_steps
            else:
                progress = (step - self.hparams.warmup_steps) / (self.hparams.max_steps - self.hparams.warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return [optimizer], [scheduler]