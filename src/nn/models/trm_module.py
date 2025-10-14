"""
HRM PyTorch Lightning Module - Following Figure 2 pseudocode exactly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from torch.optim import AdamW
from lightning import LightningModule
from src.nn.models.trm_block import TransformerBlock

from src.nn.utils import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)

class TRMModule(LightningModule):
    """
    HRM implementation following Figure 2 pseudocode exactly.
    """
    
    def __init__(self,
                 hidden_size: int = 512,
                 num_layers: int = 4,  # HRM uses 4 layers
                 max_grid_size: int = 30,
                 num_colors: int = 10,
                 n_latent_recursions: int = 2,  # n in HRM
                 T_deep_recursions: int = 2,    # T in HRM
                 N_supervision: int = 16,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 2000,
                 max_steps: int = 100000,
                 output_dir: str = None):
        super().__init__()
        self.save_hyperparameters()
        
        # CRITICAL: Manual optimization
        self.automatic_optimization = False
        
        # Model components
        self.input_embedding = nn.Embedding(num_colors + 1, hidden_size)
        
        # L_net and H_net (two separate networks)
        self.L_net = self._build_transformer(hidden_size, num_layers, num_heads=8, dropout=0.1)
        self.H_net = self._build_transformer(hidden_size, num_layers, num_heads=8, dropout=0.1)

        # Output heads
        self.output_head = nn.Linear(hidden_size, num_colors)
        self.Q_head = nn.Linear(hidden_size, 2)  # Q_head returns 2 values: q[0] and q[1]
        
        self.y_init = None
        self.z_init = None

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small values for stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _net_forward(self, net: nn.ModuleList, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward through Transformer network.
        Sum inputs, then pass through all Transformer blocks.
        """
        x = sum(inputs)
        
        for block in net:
            x = block(x)
        
        return x

    def _build_transformer(self, hidden_size: int, num_layers: int, 
                          num_heads: int, dropout: float) -> nn.ModuleList:
        """Build Transformer as specified in paper."""
        return nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
    
    def _build_network(self, hidden_size: int, num_layers: int) -> nn.Module:
        """Build network (L_net or H_net)."""
        layers = []
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(0.1)
            )
            
            # Small initialization
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            
            layers.append(nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_size),
                'mlp': mlp
            }))
        
        return nn.ModuleList(layers)

    def net_L(self, x, y, z):
        return self._net_forward(self.L_net, x, y, z)

    def net_H(self, y, z):
        return self._net_forward(self.H_net, y, z)

    def latent_recursion(self, x, y, z, n=6):
        for _ in range(n):  # latent reasoning
            z = self.net_L(x, y, z)
            y = self.net_H(y, z)  # refine output answer
        return y, z

    def deep_recursion(self, x, y, z, n=6, T=3):
        # recursing T−1 times to improve y and z (no gradients needed)
        with torch.no_grad():
            for j in range(T-1):
                y, z = self.latent_recursion(x, y, z, n)
        # recursing once to improve y and z
        y, z = self.latent_recursion(x, y, z, n)
        return (y.detach(), z.detach()), self.output_head(y), self.Q_head(y)

    def init_state (self, batch_size: int, seq_len: int):
        self.y_init = self.z_init.expand(batch_size, seq_len, -1).clone()
        self.z_init = self.z_init.expand(batch_size, seq_len, -1).clone()
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        opt = self.optimizers()
        
        x_input = batch['input']
        y_true = batch['output']

        batch_size, height, width = x_input.shape
        seq_len = height * width

        if not self.y_init:
            self.init_state(batch_size, seq_len)
        
        total_loss_value = 0
        n_steps = 0
        y, z = self.y_init, self.z_init
        for step in range(self.hparams.N_supervision):
            x = self.input_embedding(x_input)
            (y, z), y_hat, q_hat = self.deep_recursion(x, y, z)
            loss = F.cross_entropy(y_hat, y_true)
            loss += F.binary_cross_entropy(q_hat, (y_hat == y_true).float())
            loss.backward()
            opt.step()
            opt.zero_grad()
            if q_hat > 0:  # early-stopping
                break

        log.info(f"Training step {batch_idx}, supervision steps: {step+1} loss = {loss.item():.4f}")
        self.log('train/loss', loss.item(), prog_bar=True)
        self.log('train/n_steps', float(n_steps))
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        x = batch['input']
        y_true = batch['output']
        
        y_pred = self(x)
        
        loss = F.cross_entropy(
            y_pred.permute(0, 3, 1, 2),
            y_true,
            ignore_index=0
        )
        
        pred_classes = y_pred.argmax(dim=-1)
        correct = (pred_classes == y_true).float()
        mask = (y_true != 0).float()
        
        accuracy = (correct * mask).sum() / mask.sum() if mask.sum() > 0 else correct.mean()
        
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/accuracy', accuracy, prog_bar=True)
        
        return {'loss': loss, 'accuracy': accuracy}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference."""
        batch_size, height, width = x.shape
        seq_len = height * width
        
        # Initialize z
        z_init_expanded = self.z_init.expand(batch_size, seq_len, -1).clone()
        z = (z_init_expanded, z_init_expanded.clone())
        
        x_emb = self.input_embedding(x).view(batch_size, -1, self.hparams.hidden_size)
        
        # Run supervision steps
        for _ in range(self.hparams.N_supervision):
            z, y_pred, q = self.hrm(z, x_emb, n=self.hparams.n_latent_recursions, T=self.hparams.T_deep_recursions)
            
            if not self.training and q[:, 0].mean() > q[:, 1].mean():
                break
            
            z = (z[0].detach(), z[1].detach())
        
        return y_pred.view(batch_size, height, width, self.hparams.num_colors)
    
    def configure_optimizers(self):
        """Configure optimizer."""
        # Scale learning rate since we do N_supervision updates per batch
        effective_lr = self.hparams.learning_rate / self.hparams.N_supervision
        
        optimizer = AdamW(
            self.parameters(),
            lr=effective_lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95)
        )
        
        return optimizer