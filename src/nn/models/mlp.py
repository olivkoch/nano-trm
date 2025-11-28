"""
MLP Baseline Module - Aligned with TRM implementation
Simple feedforward network for comparison with TRM
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from src.nn.modules.sparse_embeddings import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed,
)
from src.nn.modules.trm_block import CastedEmbedding, CastedLinear
from src.nn.modules.utils import compute_lr, stablemax_cross_entropy
from src.nn.utils.constants import IGNORE_LABEL_ID
from src.nn.utils import RankedLogger

try:
    from adam_atan2 import AdamATan2
except ImportError:
    print("Failed to import adam2")

log = RankedLogger(__name__, rank_zero_only=True)


class MLPModule(LightningModule):
    """
    Simple MLP baseline for sequence-to-sequence tasks.
    Aligned with TRM interface for fair comparison.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 4,
        max_grid_size: int = 30,
        ffn_expansion: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        learning_rate_emb: float = 1e-2,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        lr_min_ratio: float = 1.0,
        puzzle_emb_dim: int = 512,
        puzzle_emb_len: int = 16,
        vocab_size: int = 0,  # Should be set from datamodule
        num_puzzles: int = 0,  # Should be set from datamodule
        batch_size: int = 0,  # Should be set from datamodule
        pad_value: int = -1,  # Should be set from datamodule
        seq_len: int = 0,  # Should be set from datamodule
        output_dir: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # CRITICAL: Manual optimization for consistency with TRM
        self.automatic_optimization = False

        self.forward_dtype = torch.float32

        # Token embeddings
        self.embed_scale = math.sqrt(hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.input_embedding = CastedEmbedding(
            vocab_size, hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )

        # Puzzle embeddings (optional, for fair comparison)
        if puzzle_emb_dim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                num_embeddings=num_puzzles,
                embedding_dim=puzzle_emb_dim,
                batch_size=batch_size,
                init_std=0.0,
                cast_to=self.forward_dtype,
            )
            self.puzzle_emb_len = puzzle_emb_len
            log.info(f"Created puzzle_emb with num_puzzles={num_puzzles}, batch_size={batch_size}")
        else:
            log.info("puzzle_emb_dim <= 0, not creating puzzle embeddings")
            self.puzzle_emb = None
            self.puzzle_emb_len = 0

        # MLP layers
        mlp_layers = []
        for i in range(num_layers):
            in_dim = hidden_size if i > 0 else hidden_size
            out_dim = hidden_size
            
            mlp_layers.extend([
                CastedLinear(in_dim, out_dim * ffn_expansion, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                CastedLinear(out_dim * ffn_expansion, out_dim, bias=True),
                nn.Dropout(dropout),
            ])
        
        self.mlp = nn.Sequential(*mlp_layers)

        # Output head
        self.lm_head = CastedLinear(hidden_size, vocab_size, bias=False)

        self.manual_step = 0

    def setup(self, stage: str):
        """Called by Lightning when setting up the model."""
        if stage == "fit":
            # Calculate steps from dataset and epochs
            if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
                train_loader = self.trainer.datamodule.train_dataloader()
                steps_per_epoch = len(train_loader)
            else:
                steps_per_epoch = self.trainer.num_training_batches

            # Compute total steps from epochs
            if self.trainer.max_epochs > 0:
                computed_total_steps = steps_per_epoch * self.trainer.max_epochs
            else:
                computed_total_steps = float("inf")

            # Take minimum of max_steps and computed steps
            if self.trainer.max_steps > 0:
                self.total_steps = min(self.trainer.max_steps, computed_total_steps)
            else:
                self.total_steps = computed_total_steps

            log.info("Training configuration:")
            log.info(f"  Steps per epoch: {steps_per_epoch}")
            log.info(f"  Max epochs: {self.trainer.max_epochs}")
            log.info(f"  Computed total steps: {computed_total_steps}")
            log.info(f"  Max steps limit: {self.trainer.max_steps}")
            log.info(f"  Actual total steps: {self.total_steps}")

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """Create input embeddings with optional puzzle embeddings."""
        # Token embedding
        embedding = self.input_embedding(input.to(torch.int32))

        # Puzzle embeddings
        if self.hparams.puzzle_emb_dim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.hparams.hidden_size - puzzle_embedding.shape[-1]

            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (
                    puzzle_embedding.view(-1, self.puzzle_emb_len, self.hparams.hidden_size),
                    embedding,
                ),
                dim=-2,
            )

        # Scale
        return self.embed_scale * embedding

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            batch: Dictionary with 'input' and 'puzzle_identifiers'

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Get embeddings
        x = self._input_embeddings(batch["input"], batch["puzzle_identifiers"])
        
        # Apply MLP
        x = self.mlp(x)
        
        # Output projection
        logits = self.lm_head(x)
        
        # Remove puzzle embedding positions
        if self.puzzle_emb_len > 0:
            logits = logits[:, self.puzzle_emb_len:]
        
        return logits

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step aligned with TRM."""
        batch_size = batch["input"].shape[0]

        # Handle optimizer
        try:
            opts = self.optimizers()
            if not isinstance(opts, list):
                opts = [opts]
        except RuntimeError:
            if not hasattr(self, "_optimizers"):
                raise RuntimeError("No optimizer available. Set model._optimizers for testing.")
            opts = self._optimizers

        # Forward pass
        logits = self.forward(batch)
        labels = batch["output"]

        # Compute loss with masking
        mask = labels != IGNORE_LABEL_ID
        loss = stablemax_cross_entropy(
            logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask
        )
        
        # Scale loss by sequence length (not batch size) for consistency
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
        loss = (loss / loss_divisor).sum()
        
        # Compute metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics
            valid_metrics = loss_counts > 0
            count = valid_metrics.sum()
            
            if count > 0:
                accuracy = torch.where(
                    valid_metrics, (is_correct.float() / loss_divisor).sum(-1), 0
                ).sum() / count
                exact_accuracy = seq_is_correct.sum().float() / count
            else:
                accuracy = 0.0
                exact_accuracy = 0.0

        # Backward pass
        scaled_loss = loss / batch_size
        scaled_loss.backward()

        # Gradient monitoring
        with torch.no_grad():
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), 
                max_norm=float('inf')
            ).item()
            
            self.log('grad/total_norm', total_grad_norm, on_step=True, prog_bar=True)
            
            if total_grad_norm < 1e-6 or total_grad_norm > 100:
                log.warning(f"Step {self.manual_step}: Gradient norm={total_grad_norm:.2e}")

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Learning rate scheduling
        current_step = self.manual_step
        total_steps = getattr(self, "total_steps", self.hparams.max_steps)

        base_lrs = [self.hparams.learning_rate]
        if len(opts) > 1:
            base_lrs.append(self.hparams.learning_rate_emb)

        lr_this_step = None
        for opt, base_lr in zip(opts, base_lrs):
            if current_step < self.hparams.warmup_steps:
                lr_this_step = compute_lr(
                    base_lr=base_lr,
                    lr_warmup_steps=self.hparams.warmup_steps,
                    lr_min_ratio=self.hparams.lr_min_ratio,
                    current_step=current_step,
                    total_steps=total_steps,
                )
            else:
                lr_this_step = base_lr

            # Update learning rate and step
            if hasattr(opt, "_optimizer"):
                for param_group in opt._optimizer.param_groups:
                    param_group["lr"] = lr_this_step
                opt._optimizer.step()
                opt._optimizer.zero_grad()
            else:
                for param_group in opt.param_groups:
                    param_group["lr"] = lr_this_step
                opt.step()
                opt.zero_grad()

        # Log metrics
        self.log("train/lr", lr_this_step, on_step=True)
        
        if count > 0:
            self.log("train/loss", loss.item() / batch_size, on_step=True, prog_bar=True)
            self.log("train/accuracy", accuracy.item(), on_step=True)
            self.log("train/exact_accuracy", exact_accuracy.item(), prog_bar=True, on_step=True)

        self.manual_step += 1

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step aligned with TRM."""
        batch_size = batch["input"].shape[0]

        with torch.no_grad():
            # Forward pass
            logits = self.forward(batch)
            labels = batch["output"]

            # Compute loss
            mask = labels != IGNORE_LABEL_ID
            loss = stablemax_cross_entropy(
                logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask
            )
            
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            loss = (loss / loss_divisor).sum()

            # Compute metrics
            preds = torch.argmax(logits, dim=-1)
            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            valid_metrics = loss_counts > 0
            count = valid_metrics.sum()

            if count > 0:
                accuracy = torch.where(
                    valid_metrics, (is_correct.float() / loss_divisor).sum(-1), 0
                ).sum() / count
                exact_accuracy = seq_is_correct.sum().float() / count
            else:
                accuracy = 0.0
                exact_accuracy = 0.0

            # Log metrics
            metrics = {
                "val/loss": loss.item() / batch_size,
                "val/accuracy": accuracy.item(),
                "val/exact_accuracy": exact_accuracy.item(),
            }

            for name, value in metrics.items():
                self.log(
                    name,
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(name in ["val/loss", "val/exact_accuracy"]),
                    sync_dist=True,
                )

            return metrics

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step - same as validation."""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """Configure optimizer with different learning rates for different parameter groups."""
        base_lr = self.hparams.learning_rate
        embedding_lr = self.hparams.learning_rate_emb

        optimizers = []

        # Main optimizer
        try:
            main_opt = AdamATan2(
                self.parameters(),
                lr=base_lr,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.95),
            )
        except NameError:
            main_opt = torch.optim.AdamW(
                self.parameters(),
                lr=base_lr,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.95),
            )
        optimizers.append(main_opt)

        # Sparse embedding optimizer (if using puzzle embeddings)
        if hasattr(self, "puzzle_emb") and self.puzzle_emb is not None:
            self.puzzle_emb.local_weights = self.puzzle_emb.local_weights.detach().requires_grad_(
                True
            )

            sparse_opt = CastedSparseEmbeddingSignSGD_Distributed(
                self.puzzle_emb.buffers(),
                lr=embedding_lr,
                weight_decay=self.hparams.weight_decay,
                world_size=1,
            )
            optimizers.append(sparse_opt)

        return optimizers