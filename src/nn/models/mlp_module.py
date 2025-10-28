"""
Simplified MLP Module for ARC-AGI
Direct input-to-output mapping without fancy features
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


class ConvMLPModule(LightningModule):
    """
    MLP with convolutional feature extraction.
    Better for spatial patterns in ARC tasks.
    """

    def __init__(
        self,
        num_colors: int = 10,
        hidden_channels: int = 128,
        num_conv_layers: int = 4,
        num_mlp_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize Conv+MLP module.

        Args:
            num_colors: Number of colors
            hidden_channels: Hidden channels for conv layers
            num_conv_layers: Number of conv layers
            num_mlp_layers: Number of MLP layers after conv
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        super().__init__()
        self.save_hyperparameters()

        # Input embedding
        self.embedding = nn.Embedding(num_colors + 1, hidden_channels)

        # Convolutional layers
        conv_layers = []
        for i in range(num_conv_layers):
            conv_layers.extend(
                [
                    nn.Conv2d(
                        hidden_channels if i > 0 else hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout2d(dropout),
                ]
            )
        self.conv_net = nn.Sequential(*conv_layers)

        # MLP layers
        mlp_layers = []
        for _ in range(num_mlp_layers):
            mlp_layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, 1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout2d(dropout),
                ]
            )
        self.mlp_net = nn.Sequential(*mlp_layers)

        # Output projection
        self.output_proj = nn.Conv2d(hidden_channels, num_colors, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input grid [batch, height, width]

        Returns:
            Output logits [batch, height, width, num_colors]
        """
        batch_size, height, width = x.shape

        # Embed input
        x = self.embedding(x)  # [batch, H, W, hidden]
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch, hidden, H, W]

        # Convolutional processing
        x = self.conv_net(x)

        # MLP processing
        x = self.mlp_net(x)

        # Output projection
        x = self.output_proj(x)  # [batch, colors, H, W]

        # Back to [batch, H, W, colors]
        x = x.permute(0, 2, 3, 1).contiguous()

        return x

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""

        x = batch["input"]
        y_true = batch["output"]

        y_pred = self(x)

        loss = F.cross_entropy(y_pred.permute(0, 3, 1, 2), y_true, ignore_index=0)

        # Accuracy
        pred_classes = y_pred.argmax(dim=-1)
        mask = (y_true != 0).float()

        if mask.sum() > 0:
            accuracy = ((pred_classes == y_true).float() * mask).sum() / mask.sum()
        else:
            accuracy = (pred_classes == y_true).float().mean()

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/accuracy", accuracy)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        x = batch["input"]
        y_true = batch["output"]

        y_pred = self(x)

        loss = F.cross_entropy(y_pred.permute(0, 3, 1, 2), y_true, ignore_index=0)

        # Accuracy
        pred_classes = y_pred.argmax(dim=-1)
        mask = (y_true != 0).float()
        if mask.sum() > 0:
            accuracy = ((pred_classes == y_true).float() * mask).sum() / mask.sum()
        else:
            accuracy = (pred_classes == y_true).float().mean()

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/accuracy", accuracy, prog_bar=True)

        return {"loss": loss, "accuracy": accuracy}

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
