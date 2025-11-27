"""
Simple MLP Lightning Module for ARC-AGI
A straightforward feedforward approach without recursion
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW


class MLPBlock(nn.Module):
    """Basic MLP block with residual connection."""

    def __init__(self, hidden_size: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * expansion_factor)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_size * expansion_factor, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class MLPEncoder(nn.Module):
    """Encoder that processes input grids."""

    def __init__(
        self, num_colors: int, hidden_size: int, num_layers: int = 6, dropout: float = 0.1
    ):
        super().__init__()

        # Input embedding
        self.embedding = nn.Embedding(num_colors + 1, hidden_size)

        # Positional embedding (learnable)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 900, hidden_size) * 0.02  # 30x30 max grid
        )

        # MLP layers
        self.layers = nn.ModuleList(
            [MLPBlock(hidden_size, dropout=dropout) for _ in range(num_layers)]
        )

        # Final norm
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, height, width]
        Returns:
            Encoded features [batch, height*width, hidden_size]
        """
        batch_size, height, width = x.shape

        # Flatten spatial dimensions
        x = x.view(batch_size, -1)  # [batch, h*w]

        # Embed tokens
        x = self.embedding(x)  # [batch, h*w, hidden]

        # Add positional embedding
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]

        # Apply MLP layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x


class MLPDecoder(nn.Module):
    """Decoder that produces output grids."""

    def __init__(self, hidden_size: int, num_colors: int, num_layers: int = 4):
        super().__init__()

        # MLP layers
        self.layers = nn.ModuleList([MLPBlock(hidden_size) for _ in range(num_layers)])

        # Output projection
        self.output_proj = nn.Linear(hidden_size, num_colors)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Args:
            x: Encoded features [batch, seq_len, hidden_size]
            height: Target height
            width: Target width
        Returns:
            Output logits [batch, height, width, num_colors]
        """
        batch_size = x.size(0)

        # Apply MLP layers
        for layer in self.layers:
            x = layer(x)

        # Project to output space
        x = self.output_proj(x)  # [batch, seq_len, num_colors]

        # Reshape to grid
        x = x[:, : height * width]  # Trim to target size
        x = x.view(batch_size, height, width, -1)

        return x


class LEDModule(LightningModule):
    """
    Simple MLP-based Lightning Module for ARC tasks.
    Processes the entire input at once without recursion.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        encoder_layers: int = 6,
        decoder_layers: int = 4,
        max_grid_size: int = 30,
        num_colors: int = 10,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 50000,
        use_task_embedding: bool = True,
        task_embedding_size: int = 128,
    ):
        """
        Initialize MLP module.

        Args:
            hidden_size: Hidden dimension
            encoder_layers: Number of encoder layers
            decoder_layers: Number of decoder layers
            max_grid_size: Maximum grid size
            num_colors: Number of colors
            dropout: Dropout rate
            learning_rate: Base learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps
            max_steps: Maximum training steps
            use_task_embedding: Whether to use task-specific embeddings
            task_embedding_size: Size of task embedding
        """
        super().__init__()
        self.save_hyperparameters()

        # Encoder
        self.encoder = MLPEncoder(
            num_colors=num_colors,
            hidden_size=hidden_size,
            num_layers=encoder_layers,
            dropout=dropout,
        )

        # Task embedding (optional)
        if use_task_embedding:
            self.task_proj = nn.Linear(task_embedding_size, hidden_size)
            # Random task embeddings for training
            self.register_buffer("task_embeddings", torch.randn(1000, task_embedding_size) * 0.02)

        # Cross-attention for incorporating training examples
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_size)

        # Decoder
        self.decoder = MLPDecoder(
            hidden_size=hidden_size, num_colors=num_colors, num_layers=decoder_layers
        )

        # For storing training examples during forward pass
        self.register_buffer("train_examples", None)
        self.register_buffer("train_outputs", None)

    def encode_examples(
        self, inputs: List[torch.Tensor], outputs: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Encode training examples for context.

        Args:
            inputs: List of input grids
            outputs: Optional list of output grids

        Returns:
            Encoded context [1, context_len, hidden_size]
        """
        encoded = []

        for i, inp in enumerate(inputs):
            # Encode input
            inp_enc = self.encoder(inp.unsqueeze(0))  # [1, seq, hidden]

            if outputs is not None and i < len(outputs):
                # Encode output
                out = outputs[i]
                out_enc = self.encoder(out.unsqueeze(0))

                # Concatenate input-output pair
                pair_enc = torch.cat([inp_enc, out_enc], dim=1)
                encoded.append(pair_enc)
            else:
                encoded.append(inp_enc)

        if encoded:
            # Combine all examples
            context = torch.cat(encoded, dim=1)  # [1, total_seq, hidden]
            return context
        else:
            # Return empty context
            return torch.zeros(1, 1, self.hparams.hidden_size).to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        train_inputs: Optional[List[torch.Tensor]] = None,
        train_outputs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, height, width]
            train_inputs: Optional training example inputs
            train_outputs: Optional training example outputs

        Returns:
            Output logits [batch, height, width, num_colors]
        """
        batch_size, height, width = x.shape

        # Encode input
        x_enc = self.encoder(x)  # [batch, seq_len, hidden]

        # Add context from training examples if available
        if train_inputs is not None:
            context = self.encode_examples(train_inputs, train_outputs)
            context = context.expand(batch_size, -1, -1)  # Match batch size

            # Cross-attention with context
            attended, _ = self.cross_attention(x_enc, context, context)
            x_enc = self.cross_norm(x_enc + attended)

        # Add task embedding if enabled
        if self.hparams.use_task_embedding:
            # Use a random task embedding during training
            task_idx = torch.randint(0, 1000, (1,)).to(self.device)
            task_emb = self.task_embeddings[task_idx]  # [1, task_emb_size]
            task_emb = self.task_proj(task_emb)  # [1, hidden]
            task_emb = task_emb.unsqueeze(1).expand(batch_size, x_enc.size(1), -1)
            x_enc = x_enc + task_emb

        # Decode
        output = self.decoder(x_enc, height, width)

        return output

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch["input"]  # [batch, H, W]
        y_true = batch["output"]  # [batch, H, W]

        # Forward pass
        y_pred = self(x)

        # Compute loss
        loss = F.cross_entropy(
            y_pred.permute(0, 3, 1, 2),  # [batch, colors, H, W]
            y_true,
            ignore_index=0,  # Ignore padding
        )

        # Compute accuracy
        pred_classes = y_pred.argmax(dim=-1)
        correct = (pred_classes == y_true).float()
        mask = (y_true != 0).float()

        if mask.sum() > 0:
            accuracy = (correct * mask).sum() / mask.sum()
        else:
            accuracy = correct.mean()

        # Log metrics
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/accuracy", accuracy)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        x = batch["input"]
        y_true = batch["output"]

        # Forward pass
        y_pred = self(x)

        # Compute loss
        loss = F.cross_entropy(y_pred.permute(0, 3, 1, 2), y_true, ignore_index=0)

        # Compute accuracy
        pred_classes = y_pred.argmax(dim=-1)
        correct = (pred_classes == y_true).float()
        mask = (y_true != 0).float()

        if mask.sum() > 0:
            accuracy = (correct * mask).sum() / mask.sum()
        else:
            accuracy = correct.mean()

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/accuracy", accuracy, prog_bar=True)

        return {"loss": loss, "accuracy": accuracy}

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # AdamW optimizer
        optimizer = AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )

        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / self.hparams.warmup_steps
            else:
                progress = (step - self.hparams.warmup_steps) / (
                    self.hparams.max_steps - self.hparams.warmup_steps
                )
                return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Prediction step for inference."""
        x = batch["input"]

        # Get training examples if available
        train_inputs = batch.get("train_inputs", None)
        train_outputs = batch.get("train_outputs", None)

        # Forward pass with context
        y_pred = self(x, train_inputs, train_outputs)

        # Get predicted classes
        pred_classes = y_pred.argmax(dim=-1)

        return pred_classes
