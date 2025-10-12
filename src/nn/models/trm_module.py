"""
TRM PyTorch Lightning Module
Complete training implementation with DataModule integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from lightning import LightningModule

from src.nn.utils import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)

class TRMModule(LightningModule):
    """
    PyTorch Lightning implementation of TRM (Tiny Recursive Model).
    """
    
    def __init__(self,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 max_grid_size: int = 30,
                 num_colors: int = 10,
                 n_latent_recursions: int = 6,
                 T_deep_recursions: int = 3,
                 N_supervision: int = 16,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 2000,
                 max_steps: int = 100000,
                 ema_decay: float = 0.999,
                 use_ema: bool = True,
                 output_dir: str = None
                 ):
        """
        Initialize TRM Lightning Module.
        
        Args:
            hidden_size: Hidden dimension (D=512 in paper)
            num_layers: Number of layers in tiny network (2 is optimal)
            max_grid_size: Maximum grid size
            num_colors: Number of colors
            n_latent_recursions: Number of latent reasoning steps (n)
            T_deep_recursions: Number of deep recursions (T)
            N_supervision: Maximum supervision steps
            learning_rate: Base learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps
            max_steps: Maximum training steps
            ema_decay: Exponential moving average decay
            use_ema: Whether to use EMA
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Model components
        self.input_embedding = nn.Embedding(
            num_colors + 1,  # +1 for padding
            hidden_size
        )
        
        # The tiny network
        self.net = self._build_tiny_network(hidden_size, num_layers)
        
        # Output heads
        self.output_head = nn.Linear(hidden_size, num_colors)
        self.q_head = nn.Linear(hidden_size, 1)  # For early stopping
        
        # Initialize latent buffers
        self.register_buffer('y_init', torch.zeros(1, 1, hidden_size))
        self.register_buffer('z_init', torch.zeros(1, 1, hidden_size))
        
        # EMA model for stability (as mentioned in paper)
        self.ema_model = None
        if use_ema:
            self.setup_ema()
    
    def _build_tiny_network(self, hidden_size: int, num_layers: int) -> nn.Module:
        """Build the tiny network architecture."""
        layers = []
        for _ in range(num_layers):
            layers.append(nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_size),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size),
                    nn.Dropout(0.1)  # Small dropout for regularization
                )
            }))
        return nn.ModuleList(layers)
    
    def net_forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the tiny network."""
        combined = x + y + z
        
        for layer in self.net:
            normalized = layer['norm'](combined)
            output = layer['mlp'](normalized)
            combined = combined + output  # Residual
        
        return combined
    
    def latent_recursion(self, x: torch.Tensor, y: torch.Tensor,
                        z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform latent recursion (Figure 3 from paper)."""
        # Recursively update z for n steps
        for _ in range(self.hparams.n_latent_recursions):
            z = self.net_forward(x, y, z)
        
        # Update y given refined z (no x input when updating y)
        y = self.net_forward(torch.zeros_like(x), y, z)
        
        return y, z
    
    def deep_recursion(self, x: torch.Tensor, y: torch.Tensor,
                  z: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deep recursion with T steps.
        T-1 without gradients, 1 with gradients (during training).
        """
        # T-1 recursions without gradients
        if self.hparams.T_deep_recursions > 1:
            with torch.no_grad():
                for _ in range(self.hparams.T_deep_recursions - 1):
                    y, z = self.latent_recursion(x, y, z)
        
        # Final recursion
        if training:
            y, z = self.latent_recursion(x, y, z)
        else:
            with torch.no_grad():
                y, z = self.latent_recursion(x, y, z)
        
        # Compute halting logits (NOT probabilities)
        q_logits = self.q_head(y.mean(dim=1, keepdim=True))  # Raw logits
        
        return y, z, q_logits
    
    def forward(self, x: torch.Tensor, 
            supervision_steps: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass with deep supervision.
        
        Args:
            x: Input tensor [batch, height, width]
            supervision_steps: Number of supervision steps (None = use default)
            
        Returns:
            Predicted output [batch, height, width, num_colors]
        """
        batch_size, height, width = x.shape
        
        # Embed input
        x_emb = self.input_embedding(x)  # [batch, H, W, hidden]
        x_emb = x_emb.view(batch_size, -1, self.hparams.hidden_size)  # [batch, H*W, hidden]
        
        # Initialize y and z
        seq_len = height * width
        y = self.y_init.expand(batch_size, seq_len, -1).clone()
        z = self.z_init.expand(batch_size, seq_len, -1).clone()
        
        # Deep supervision loop
        n_steps = supervision_steps or self.hparams.N_supervision
        
        for step in range(n_steps):
            y, z, q_logits = self.deep_recursion(x_emb, y, z, training=self.training)
            
            # Early stopping based on confidence (inference only)
            # Apply sigmoid to convert logits to probability
            if not self.training:
                q_prob = torch.sigmoid(q_logits)
                if q_prob.mean() > 0.8:
                    break
            
            # Detach for next iteration (as in paper)
            y = y.detach()
            z = z.detach()
        
        # Final output
        y_out = self.output_head(y)  # [batch, H*W, num_colors]
        y_out = y_out.view(batch_size, height, width, self.hparams.num_colors)
        
        return y_out
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with deep supervision - memory efficient."""
        x = batch['input']
        y_true = batch['output']
        
        batch_size, height, width = x.shape
        
        log.info(f"\tTraining step {self.global_step} with batch size {x.size(0)}")
        
        # Embed input
        x_emb = self.input_embedding(x)
        x_emb = x_emb.view(batch_size, -1, self.hparams.hidden_size)
        
        # Initialize y and z
        seq_len = height * width
        y = self.y_init.expand(batch_size, seq_len, -1).clone()
        z = self.z_init.expand(batch_size, seq_len, -1).clone()
        
        total_loss = 0
        n_steps = 0
        
        # Deep supervision training loop
        for step in range(self.hparams.N_supervision):
            # Deep recursion
            y, z, q_logits = self.deep_recursion(x_emb, y, z, training=True)
            
            # Predict output
            y_hat = self.output_head(y)
            y_hat = y_hat.view(batch_size, height, width, self.hparams.num_colors)
            
            # Classification loss
            loss = F.cross_entropy(
                y_hat.permute(0, 3, 1, 2),
                y_true,
                ignore_index=0
            )
            
            # Halting loss
            with torch.no_grad():  # Don't need gradients for accuracy computation
                correct = (y_hat.argmax(dim=-1) == y_true).float()
                mask = (y_true != 0).float()
                if mask.sum() > 0:
                    correct = (correct * mask).sum() / mask.sum()
                else:
                    correct = correct.mean()
            
            halt_target = (correct > 0.95).float()
            halt_loss = F.binary_cross_entropy_with_logits(
                q_logits.squeeze(),
                halt_target.expand(batch_size)
            )
            
            # Combined loss
            step_loss = loss + 0.5 * halt_loss
            
            # CRITICAL FIX: Accumulate loss values, not tensors
            # This prevents keeping the entire computation graph
            total_loss = total_loss + step_loss.item()  # .item() extracts the scalar
            
            # Keep the last step_loss tensor for backprop
            last_step_loss = step_loss
            
            # Log metrics (use .item() to avoid keeping tensors)
            self.log(f'train/loss_step_{step}', step_loss.item(), on_step=False, on_epoch=True)
            self.log(f'train/accuracy_step_{step}', correct.item(), on_step=False, on_epoch=True)
            
            n_steps += 1
            
            # Early stopping
            with torch.no_grad():
                q_prob = torch.sigmoid(q_logits)
                if q_prob.mean() > 0.8:
                    break
            
            # Detach for next iteration
            y = y.detach()
            z = z.detach()
        
        # Return the last step's loss (which has gradients)
        # This is what gets backpropagated
        final_loss = last_step_loss
        
        # Log average loss for monitoring (scalar only)
        avg_loss_value = total_loss / n_steps
        self.log('train/loss', avg_loss_value, prog_bar=True)
        self.log('train/n_steps', float(n_steps))
        
        # Update EMA
        if self.hparams.use_ema and hasattr(self, 'ema_params') and self.ema_params is not None:
            self.update_ema()
        
        return final_loss

    # def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
    #     """Training step with deep supervision."""
        
    #     x = batch['input']  # [batch, H, W]
    #     y_true = batch['output']  # [batch, H, W]
        
    #     batch_size, height, width = x.shape
        
    #     log.info(f"\tTraining step {self.global_step} with batch size {x.size(0)}")

    #     # Embed input
    #     x_emb = self.input_embedding(x)
    #     x_emb = x_emb.view(batch_size, -1, self.hparams.hidden_size)
        
    #     # Initialize y and z
    #     seq_len = height * width
    #     y = self.y_init.expand(batch_size, seq_len, -1).clone()
    #     z = self.z_init.expand(batch_size, seq_len, -1).clone()
        
    #     total_loss = 0
    #     n_steps = 0
        
    #     # Deep supervision training loop
    #     for step in range(self.hparams.N_supervision):
    #         # Deep recursion
    #         y, z, q_logits = self.deep_recursion(x_emb, y, z, training=True)
            
    #         # Predict output
    #         y_hat = self.output_head(y)
    #         y_hat = y_hat.view(batch_size, height, width, self.hparams.num_colors)
            
    #         # Classification loss
    #         loss = F.cross_entropy(
    #             y_hat.permute(0, 3, 1, 2),  # [batch, colors, H, W]
    #             y_true,
    #             ignore_index=0  # Ignore padding
    #         )
            
    #         # Halting loss (learns when to stop)
    #         correct = (y_hat.argmax(dim=-1) == y_true).float()
    #         # Mask out padding
    #         mask = (y_true != 0).float()
    #         if mask.sum() > 0:
    #             correct = (correct * mask).sum() / mask.sum()
    #         else:
    #             correct = correct.mean()
            
    #         halt_target = (correct > 0.95).float()  # Stop if >95% accurate
            
    #         # USE binary_cross_entropy_with_logits instead of binary_cross_entropy
    #         # q_logits is now raw logits, not probabilities
    #         halt_loss = F.binary_cross_entropy_with_logits(
    #             q_logits.squeeze(),  # Raw logits from q_head
    #             halt_target.expand(batch_size)  # Target should match batch size
    #         )
            
    #         # Combined loss
    #         step_loss = loss + 0.5 * halt_loss
    #         total_loss = total_loss + step_loss
            
    #         # Log metrics
    #         self.log(f'train/loss_step_{step}', step_loss, on_step=True, on_epoch=False)
    #         self.log(f'train/accuracy_step_{step}', correct, on_step=True, on_epoch=False)
            
    #         n_steps += 1
            
    #         # Early stopping based on Q-head (apply sigmoid for threshold check)
    #         q_prob = torch.sigmoid(q_logits)
    #         if q_prob.mean() > 0.8:
    #             break
            
    #         # Detach for next iteration
    #         y = y.detach()
    #         z = z.detach()
        
    #     # Average loss over steps
    #     avg_loss = total_loss / n_steps
        
    #     # Log overall metrics
    #     self.log('train/loss', avg_loss, prog_bar=True)
    #     self.log('train/n_steps', float(n_steps))
        
    #     # Update EMA if enabled
    #     if self.hparams.use_ema and hasattr(self, 'ema_params') and self.ema_params is not None:
    #         self.update_ema()
        
    #     return avg_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        x = batch['input']
        y_true = batch['output']
        
        log.info(f"\tValidation step {self.global_step}")
        
        # Use EMA model for validation if available
        if self.hparams.use_ema and hasattr(self, 'ema_params') and self.ema_params is not None:
            with self.ema_scope():
                y_pred = self(x, supervision_steps=self.hparams.N_supervision)
        else:
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
        # AdamW optimizer as in paper
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95)  # As specified in paper
        )
        
        # Learning rate schedule with warmup
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / self.hparams.warmup_steps
            else:
                progress = (step - self.hparams.warmup_steps) / (self.hparams.max_steps - self.hparams.warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]
    
    def setup_ema(self):
        """Setup exponential moving average model."""
        # Use nn.ParameterDict instead of nn.ModuleDict for storing tensors
        self.ema_model = nn.ParameterDict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                # Store as Parameter (not just tensor)
                self.ema_model[name.replace('.', '_')] = nn.Parameter(
                    param.data.clone(), 
                    requires_grad=False
                )
            
    def update_ema(self):
        """Update EMA parameters."""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    ema_name = name.replace('.', '_')
                    if ema_name in self.ema_model:
                        self.ema_model[ema_name].mul_(self.hparams.ema_decay).add_(
                            param.data, alpha=1 - self.hparams.ema_decay
                        )
    
    def ema_scope(self):
        """Context manager for using EMA parameters."""
        from contextlib import contextmanager
        
        @contextmanager
        def _ema_scope():
            # Save current parameters
            backup = {}
            for name, param in self.named_parameters():
                if param.requires_grad:
                    backup[name] = param.data.clone()
                    ema_name = name.replace('.', '_')
                    if ema_name in self.ema_model:
                        param.data.copy_(self.ema_model[ema_name].data)
            
            try:
                yield
            finally:
                # Restore original parameters
                for name, param in self.named_parameters():
                    if name in backup:
                        param.data.copy_(backup[name])
        
        return _ema_scope()