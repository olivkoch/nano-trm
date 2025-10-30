"""
HRM PyTorch Lightning Module - Following Figure 2 pseudocode exactly
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from adam_atan2 import AdamATan2
from lightning import LightningModule

from src.nn.modules.hybrid_vision_embeddings import HybridDINOEmbedding
from src.nn.modules.sparse_embeddings import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed,
)
from src.nn.modules.trm_block import TransformerBlock
from src.nn.modules.utils import trunc_normal_init
from src.nn.utils import RankedLogger
from src.nn.utils.constants import PAD_VALUE

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class TRMCarry:
    """Carry structure for maintaining state across steps."""

    z_H: torch.Tensor  # High-level state (y in your code)
    z_L: torch.Tensor  # Low-level state (z in your code)
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]  # Stores current batch data


class TRMModule(LightningModule):
    """
    HRM implementation following Figure 2 pseudocode exactly.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 4,  # HRM uses 4 layers
        max_grid_size: int = 30,
        num_colors: int = 10,
        n_latent_recursions: int = 2,  # n in HRM
        T_deep_recursions: int = 2,  # T in HRM
        N_supervision: int = 16,
        ffn_expansion: int = 2,
        learning_rate: float = 1e-4,
        learning_rate_emb: float = 1e-2,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        use_dino_embeddings: bool = True,
        num_puzzles: int = 0,  # Should be set from datamodule
        batch_size: int = 0,  # Should be set from datamodule
        puzzle_emb_dim: int = 512,  # Puzzle embedding dimension
        puzzle_emb_len: int = 16,  # How many tokens for puzzle embedding
        output_dir: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # CRITICAL: Manual optimization
        self.automatic_optimization = False

        # Model components
        if use_dino_embeddings:
            self.input_embedding = HybridDINOEmbedding(
                num_colors=num_colors + 1, hidden_size=hidden_size, freeze_dino=True
            )
        else:
            self.input_embedding = nn.Embedding(
                num_colors + 1, hidden_size, padding_idx=PAD_VALUE
            )  # 0 (padding) + 10 colors

        # TODO: replace with RoPE or Casted Embeddings
        self.pos_embedding = nn.Embedding(
            max_grid_size * max_grid_size + puzzle_emb_len, hidden_size
        )

        # a single network (not two separate networks)
        self.lenet = self._build_transformer(
            hidden_size, num_layers, num_heads=8, ffn_expansion=ffn_expansion, dropout=0.1
        )

        # Output heads
        self.output_head = nn.Linear(hidden_size, num_colors)

        self.embed_scale = math.sqrt(self.hparams.hidden_size)

        # Halting head for adaptive computation
        # Only learn a halting probability through a Binary-Cross-Entropy loss of having
        # reached the correct solution
        self.Q_head = nn.Linear(hidden_size, 1)  # Q_head returns 1 value: q[0]

        # State for carry (persisted across training steps)
        self.carry = None

        self.register_buffer("z_H_init", trunc_normal_init((hidden_size,), std=0.02))
        self.register_buffer("z_L_init", trunc_normal_init((hidden_size,), std=0.02))

        # Initialize weights
        self.apply(self._init_weights)

        # Add puzzle embeddings
        if puzzle_emb_dim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                num_embeddings=num_puzzles,
                embedding_dim=puzzle_emb_dim,
                batch_size=batch_size,
                init_std=0.0,  # Reference uses 0 init
                cast_to=torch.bfloat16,
            )
            self.puzzle_emb_len = puzzle_emb_len
            log.info(f"Created puzzle_emb with batch_size={batch_size}")
            log.info(f"puzzle_emb.local_weights.shape: {self.puzzle_emb.local_weights.shape}")
            log.info(f"puzzle_emb.weights.shape: {self.puzzle_emb.weights.shape}")
        else:
            log.info("puzzle_emb_dim <= 0, not creating puzzle embeddings")
            self.puzzle_emb = None
            self.puzzle_emb_len = 0

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TRMCarry:
        """Create initial carry state for a batch."""
        batch_size = batch["input"].shape[0]
        height, width = batch["input"].shape[1], batch["input"].shape[2]
        seq_len = height * width

        # Account for puzzle embedding positions
        total_seq_len = seq_len + self.puzzle_emb_len

        # Use the actual buffer names
        z_H_init = self.z_H_init.view(1, 1, -1)
        z_L_init = self.z_L_init.view(1, 1, -1)

        current_data = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                current_data[k] = torch.empty_like(v)
            else:
                current_data[k] = v

        return TRMCarry(
            z_H=z_H_init.expand(batch_size, total_seq_len, -1).clone(),
            z_L=z_L_init.expand(batch_size, total_seq_len, -1).clone(),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=self.device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=self.device),
            current_data=current_data,
        )

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

    def _build_transformer(
        self, hidden_size: int, num_layers: int, num_heads: int, ffn_expansion: int, dropout: float
    ) -> nn.ModuleList:
        """Build Transformer as specified in paper."""
        return nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, ffn_expansion, dropout)
                for _ in range(num_layers)
            ]
        )

    def compute_loss_and_metrics(self, carry, batch):
        """Compute loss and metrics without circular reference."""
        # Get model outputs
        new_carry, outputs = self.forward(carry, batch)

        # Extract labels
        y_true = batch["output"]
        batch_size = y_true.shape[0]
        y_true_flat = y_true.flatten(start_dim=1)

        with torch.no_grad():
            # Compute masks and correctness
            mask = y_true_flat != PAD_VALUE
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            # Predictions and correctness
            preds = torch.argmax(outputs["logits"], dim=-1)
            is_correct = mask & (preds == y_true_flat)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (only for halted sequences)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics, (is_correct.float() / loss_divisor).sum(-1), 0
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"].squeeze() >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Compute losses - IMPORTANT: These are per-sequence losses that will be summed
        lm_loss_per_token = F.cross_entropy(
            outputs["logits"].view(-1, self.hparams.num_colors),
            y_true_flat.view(-1),
            ignore_index=PAD_VALUE,
            reduction="none",
        ).view(batch_size, -1)

        # Normalize by number of valid tokens per sequence, then sum
        lm_loss = (lm_loss_per_token / loss_divisor).sum(-1).mean()  # Changed: mean instead of sum

        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"].squeeze(),
            seq_is_correct.float(),
            reduction="mean",  # Changed: mean instead of sum
        )

        metrics.update(
            {
                "lm_loss": lm_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            }
        )

        total_loss = lm_loss + 0.5 * q_halt_loss

        return new_carry, total_loss, metrics, new_carry.halted.all()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step that implements supervision through multiple forward passes.
        Each sequence can run up to N_supervision (halt_max_steps) times.
        """

        # Handle case when not attached to trainer (for testing)
        try:
            opts = self.optimizers()
            if not isinstance(opts, list):
                opts = [opts]
        except RuntimeError:
            # For testing without trainer
            if not hasattr(self, "_optimizers"):
                raise RuntimeError("No optimizer available. Set model._optimizers for testing.")
            opts = self._optimizers

        # Initialize carry if first batch
        if self.carry is None:
            self.carry = self.initial_carry(batch)

        # Accumulate metrics
        accumulated_metrics = {}
        total_loss = 0.0
        n_active_steps = 0

        # Run supervision steps
        for _ in range(self.hparams.N_supervision):
            # Forward with loss computation
            self.carry, loss, metrics, all_halted = self.compute_loss_and_metrics(self.carry, batch)

            # Only backprop for non-halted sequences
            active_mask = ~self.carry.halted
            if active_mask.any():
                masked_loss = loss * active_mask.float().sum() / batch["input"].shape[0]
                masked_loss.backward()
                total_loss += masked_loss.item()
                n_active_steps += 1

                # Accumulate metrics
                for k, v in metrics.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0) + v.item()

            if all_halted:
                break

        # Clip gradients for all parameters (both dense and sparse)
        if n_active_steps > 0:  # Only if we actually computed gradients
            # Get all parameters with gradients
            params_with_grad = []

            # Add regular parameters
            for param in self.parameters():
                if param.grad is not None:
                    params_with_grad.append(param)

            # Add sparse embedding buffers if they have gradients
            if hasattr(self, "puzzle_emb") and self.puzzle_emb is not None:
                for buf in self.puzzle_emb.buffers():
                    if hasattr(buf, "grad") and buf.grad is not None:
                        # For sparse gradients, we need to handle them differently
                        if buf.grad.is_sparse:
                            # Clip sparse gradients by value
                            buf.grad = buf.grad.coalesce()
                            buf.grad._values().clamp_(-1.0, 1.0)
                        else:
                            params_with_grad.append(buf)

            # Clip dense gradients by norm
            if params_with_grad:
                torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=1.0)

        # Optimizer step
        for opt in opts:
            if hasattr(opt, "_optimizer"):
                opt._optimizer.step()
                opt._optimizer.zero_grad()
            else:
                opt.step()
                opt.zero_grad()

        # Log metrics
        if accumulated_metrics.get("count", 0) > 0:
            count = accumulated_metrics["count"]
            self.log("train/loss", total_loss / max(1, n_active_steps), prog_bar=True, on_step=True)
            self.log("train/accuracy", accumulated_metrics.get("accuracy", 0) / count, on_step=True)
            self.log(
                "train/exact_accuracy",
                accumulated_metrics.get("exact_accuracy", 0) / count,
                prog_bar=True,
                on_step=True,
            )
            self.log(
                "train/q_halt_accuracy",
                accumulated_metrics.get("q_halt_accuracy", 0) / count,
                on_step=True,
            )
            self.log(
                "train/steps",
                accumulated_metrics.get("steps", 0) / count,
                prog_bar=True,
                on_step=True,
            )
            self.log("train/lm_loss", accumulated_metrics.get("lm_loss", 0) / count, on_step=True)
            self.log(
                "train/q_halt_loss", accumulated_metrics.get("q_halt_loss", 0) / count, on_step=True
            )

        return torch.tensor(total_loss / max(1, n_active_steps))

    def forward(
        self, carry: TRMCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRMCarry, Dict[str, torch.Tensor]]:
        """Forward pass matching reference implementation with puzzle embeddings."""
        batch_size = batch["input"].shape[0]
        height, width = batch["input"].shape[1], batch["input"].shape[2]
        seq_len = height * width

        # Reset states for halted sequences
        reset_mask = carry.halted.view(-1, 1, 1)

        z_H_init = self.z_H_init.view(1, 1, -1)
        z_L_init = self.z_L_init.view(1, 1, -1)

        # Account for puzzle embedding positions in sequence length
        total_seq_len = seq_len + self.puzzle_emb_len

        # Expand init states to total sequence length
        new_z_H = torch.where(reset_mask, z_H_init.expand(batch_size, total_seq_len, -1), carry.z_H)
        new_z_L = torch.where(reset_mask, z_L_init.expand(batch_size, total_seq_len, -1), carry.z_L)

        # Reset steps for halted sequences
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)

        # Update current_data for halted sequences
        new_current_data = {}
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                mask = carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1))
                new_current_data[k] = torch.where(mask, batch[k], carry.current_data[k])
            else:
                new_current_data[k] = batch[k]

        # Get token embeddings from current data
        x_input = new_current_data["input"]
        token_emb = self.embed_scale * self.input_embedding(x_input)
        token_emb = token_emb.view(batch_size, seq_len, self.hparams.hidden_size)

        # Get and process puzzle embeddings
        if self.puzzle_emb is not None and self.puzzle_emb_len > 0:
            puzzle_ids = new_current_data["puzzle_identifiers"]
            puzzle_embedding = self.puzzle_emb(puzzle_ids)  # [B, puzzle_emb_dim]

            # Reshape puzzle embedding to sequence positions
            # If puzzle_emb_dim doesn't divide evenly into hidden_size, pad
            pad_count = self.puzzle_emb_len * self.hparams.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            # Reshape to [B, puzzle_emb_len, hidden_size]
            puzzle_embedding = puzzle_embedding.view(
                batch_size, self.puzzle_emb_len, self.hparams.hidden_size
            )

            # Concatenate puzzle embeddings with token embeddings
            x_emb = torch.cat([puzzle_embedding, token_emb], dim=1)  # [B, puzzle_len + seq_len, D]
        else:
            x_emb = token_emb
            total_seq_len = seq_len

        # Apply position embeddings
        # Position embeddings should cover total_seq_len now
        if hasattr(self, "pos_embedding"):
            positions = torch.arange(total_seq_len, device=x_emb.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embedding(positions)
            x_emb = x_emb + pos_emb

        # Deep recursion with updated states
        z_H, z_L = new_z_H, new_z_L

        # H_cycles-1 without gradient
        with torch.no_grad():
            for _ in range(self.hparams.T_deep_recursions - 1):
                for _ in range(self.hparams.n_latent_recursions):
                    z_L = self._net_forward(self.lenet, z_L, z_H + x_emb)
                z_H = self._net_forward(self.lenet, z_H, z_L)

        # Last H_cycle WITH gradient
        for _ in range(self.hparams.n_latent_recursions):
            z_L = self._net_forward(self.lenet, z_L, z_H + x_emb)
        z_H = self._net_forward(self.lenet, z_H, z_L)

        # Compute outputs - only from grid positions (skip puzzle positions)
        if self.puzzle_emb_len > 0:
            # Skip puzzle embedding positions when computing output
            grid_z_H = z_H[:, self.puzzle_emb_len :, :]  # [B, seq_len, D]
            logits = self.output_head(grid_z_H)
            # Use first puzzle position for Q-head (like reference)
            q_halt_logits = self.Q_head(z_H[:, 0, :])
        else:
            logits = self.output_head(z_H)
            q_halt_logits = self.Q_head(z_H[:, 0, :])

        outputs = {
            "logits": logits,  # [B, seq_len, num_colors]
            "q_halt_logits": q_halt_logits,  # [B, 1]
        }

        # Update carry for next iteration
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.hparams.N_supervision

            # Halting logic
            if self.training:
                halted = is_last_step | (q_halt_logits.squeeze() > 0)
            else:
                halted = is_last_step

        new_carry = TRMCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            steps=new_steps,
            halted=halted,
            current_data=new_current_data,
        )

        return new_carry, outputs

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Simplified validation using loss head."""

        batch_size = batch["input"].shape[0]

        with torch.no_grad():
            # Create fresh carry for validation
            carry = self.initial_carry(batch)

            # Accumulate metrics across all supervision steps
            accumulated_metrics = {}
            total_loss = 0.0
            n_steps = 0

            # Run up to N_supervision iterations
            for _ in range(self.hparams.N_supervision):
                # Forward with loss computation
                carry, loss, metrics, all_halted = self.compute_loss_and_metrics(carry, batch)

                # Accumulate metrics
                for k, v in metrics.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0) + v.item()

                total_loss += loss.item()
                n_steps += 1

                if all_halted:
                    break

            # Compute averages
            count = accumulated_metrics.get("count", batch_size)
            if count > 0:
                avg_metrics = {
                    "val/loss": total_loss / (n_steps * batch_size),
                    "val/accuracy": accumulated_metrics.get("accuracy", 0) / count,
                    "val/exact_accuracy": accumulated_metrics.get("exact_accuracy", 0) / count,
                    "val/q_halt_accuracy": accumulated_metrics.get("q_halt_accuracy", 0) / count,
                    "val/steps": accumulated_metrics.get("steps", 0) / count,
                    "val/lm_loss": accumulated_metrics.get("lm_loss", 0) / count,
                    "val/q_halt_loss": accumulated_metrics.get("q_halt_loss", 0) / count,
                }
            else:
                avg_metrics = {
                    f"val/{k}": 0.0
                    for k in [
                        "loss",
                        "accuracy",
                        "exact_accuracy",
                        "q_halt_accuracy",
                        "steps",
                        "lm_loss",
                        "q_halt_loss",
                    ]
                }

            # Log metrics
            for name, value in avg_metrics.items():
                self.log(
                    name,
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(name in ["val/loss", "val/exact_accuracy"]),
                    sync_dist=True,
                )

            return avg_metrics

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step - same as validation."""
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_start(self):
        """Don't interfere with training carry during validation."""
        # Note: We DON'T reset self.carry here because that's for training
        pass

    def on_train_epoch_start(self):
        """Reset carry at the beginning of each training epoch."""
        self.carry = None

    def configure_optimizers(self):
        """Configure optimizer with different learning rates for different parameter groups."""

        base_lr = self.hparams.learning_rate / self.hparams.N_supervision
        embedding_lr = self.hparams.learning_rate_emb / self.hparams.N_supervision

        # Collect parameters for main optimizer
        main_params = []
        for name, param in self.named_parameters():
            if "puzzle_emb" not in name:  # Exclude puzzle_emb
                main_params.append(param)

        optimizers = []

        # Main optimizer
        if main_params:
            # Use AdamATan2 if available
            try:
                main_opt = AdamATan2(
                    main_params,
                    lr=base_lr,
                    weight_decay=self.hparams.weight_decay,
                    betas=(0.9, 0.95),
                )
            except ImportError:
                main_opt = torch.optim.AdamW(
                    main_params,
                    lr=base_lr,
                    weight_decay=self.hparams.weight_decay,
                    betas=(0.9, 0.95),
                )
            optimizers.append(main_opt)

        # Force sparse embedding to be leaf tensors
        if hasattr(self, "puzzle_emb") and self.puzzle_emb is not None:
            buffer_list = list(self.puzzle_emb.buffers())
            for i, buf in enumerate(buffer_list):
                if buf.requires_grad and not buf.is_leaf:
                    # This is local_weights - recreate as leaf
                    buffer_list[i] = buf.detach().requires_grad_(True)

            # Add sparse embedding optimizer
            sparse_opt = CastedSparseEmbeddingSignSGD_Distributed(
                buffer_list,  # This should be a list of 3 tensors
                lr=embedding_lr,
                weight_decay=self.hparams.weight_decay,
                world_size=1,
            )
            optimizers.append(sparse_opt)

        return optimizers

        # # Just linear warmup, no decay after
        # def lr_lambda(step):
        #     if step < self.hparams.warmup_steps:
        #         return step / max(1, self.hparams.warmup_steps)
        #     else:
        #         return 1.0  # Constant LR after warmup

        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        # }
