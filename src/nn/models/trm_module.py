"""
HRM PyTorch Lightning Module - Following Figure 2 pseudocode exactly
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn.utils.constants import IGNORE_LABEL_ID
from src.nn.modules.utils import compute_lr

try:
    from adam_atan2 import AdamATan2
except ImportError:
    print("Failed to import adam2")

from lightning import LightningModule

from src.nn.modules.sparse_embeddings import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed,
)
from src.nn.modules.trm_block import (
    CastedEmbedding,
    CastedLinear,
    ReasoningBlock,
    ReasoningBlockConfig,
    ReasoningModule,
    RotaryEmbedding,
)
from src.nn.modules.utils import stablemax_cross_entropy, trunc_normal_init_
from src.nn.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class TRMInnerCarry:
    z_H: torch.Tensor  # High-level state (y in your code)
    z_L: torch.Tensor  # Low-level state (z in your code)


@dataclass
class TRMCarry:
    """Carry structure for maintaining state across steps."""

    inner_carry: TRMInnerCarry
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
        num_layers: int = 2,
        num_heads: int = 8,  # min(2, hidden_size // 64)
        max_grid_size: int = 30,
        H_cycles: int = 3,
        L_cycles: int = 6,
        N_supervision: int = 16,
        ffn_expansion: int = 2,
        learning_rate: float = 1e-4,
        learning_rate_emb: float = 1e-2,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        halt_exploration_prob: float = 0.1,
        puzzle_emb_dim: int = 512,  # Puzzle embedding dimension
        puzzle_emb_len: int = 16,  # How many tokens for puzzle embedding
        rope_theta: int = 10000,
        lr_min_ratio: float = 1.0,
        vocab_size: int = 0,  # Should be set from datamodule
        num_puzzles: int = 0,  # Should be set from datamodule
        batch_size: int = 0,  # Should be set from datamodule
        pad_value: int = -1,  # Should be set from datamodule
        seq_len: int = 0,  # Should be set from datamodule
        output_dir: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # CRITICAL: Manual optimization
        self.automatic_optimization = False

        self.forward_dtype = torch.float32

        # Token embeddings
        self.embed_scale = math.sqrt(hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.input_embedding = CastedEmbedding(
            vocab_size, hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )

        # Positional embeddings with rotary embeddings
        self.pos_embedding = RotaryEmbedding(
            dim=hidden_size // num_heads,
            max_position_embeddings=seq_len + puzzle_emb_len,
            base=rope_theta,
        )

        # a single network (not two separate networks)
        reasoning_config = ReasoningBlockConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=ffn_expansion,
            rms_norm_eps=1e-5,
            seq_len=seq_len,
            mlp_t=False,
            puzzle_emb_ndim=puzzle_emb_dim,
            puzzle_emb_len=puzzle_emb_len,
        )

        self.lenet = ReasoningModule(
            layers=[ReasoningBlock(reasoning_config) for _ in range(num_layers)]
        )

        self.lm_head = CastedLinear(hidden_size, vocab_size, bias=False)
        self.q_head = CastedLinear(hidden_size, 2, bias=True)

        # Halting head for adaptive computation
        # Only learn a halting probability through a Binary-Cross-Entropy loss of having
        # reached the correct solution
        #        self.Q_head = nn.Linear(hidden_size, 1)  # Q_head returns 1 value: q[0]
        with torch.no_grad():
            self.q_head.weight.zero_()
            if self.q_head.bias is not None:
                self.q_head.bias.fill_(-5.0)  # Strong negative bias

        # State for carry (persisted across training steps)
        self.carry = None

        self.z_H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.z_L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        # self.register_buffer("z_H_init", trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        # self.register_buffer("z_L_init", trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Add puzzle embeddings
        if puzzle_emb_dim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                num_embeddings=num_puzzles,
                embedding_dim=puzzle_emb_dim,
                batch_size=batch_size,
                init_std=0.0,  # Reference uses 0 init
                cast_to=self.forward_dtype,
            )
            self.puzzle_emb_len = puzzle_emb_len
            log.info(f"Created puzzle_emb with num_puzzles={num_puzzles}, batch_size={batch_size}")
            log.info(f"puzzle_emb.local_weights.shape: {self.puzzle_emb.local_weights.shape}")
            log.info(f"puzzle_emb.weights.shape: {self.puzzle_emb.weights.shape}")
        else:
            log.info("puzzle_emb_dim <= 0, not creating puzzle embeddings")
            self.puzzle_emb = None
            self.puzzle_emb_len = 0

        self.last_step_time = None

    def setup(self, stage: str):
        """Called by Lightning when setting up the model."""
        if stage == "fit":
            # Calculate steps from dataset and epochs
            if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                train_loader = self.trainer.datamodule.train_dataloader()
                steps_per_epoch = len(train_loader)
            else:
                # Fallback: estimate from limit_train_batches if datamodule not available
                steps_per_epoch = self.trainer.num_training_batches
            
            # Compute total steps from epochs
            if self.trainer.max_epochs > 0:
                computed_total_steps = steps_per_epoch * self.trainer.max_epochs
            else:
                # If max_epochs not set, use a large number
                computed_total_steps = float('inf')
            
            # Take minimum of max_steps and computed steps
            if self.trainer.max_steps > 0:
                self.total_steps = min(self.trainer.max_steps, computed_total_steps)
            else:
                self.total_steps = computed_total_steps
            
            log.info(f"Training configuration:")
            log.info(f"  Steps per epoch: {steps_per_epoch}")
            log.info(f"  Max epochs: {self.trainer.max_epochs}")
            log.info(f"  Computed total steps: {computed_total_steps}")
            log.info(f"  Max steps limit: {self.trainer.max_steps}")
            log.info(f"  Actual total steps: {self.total_steps}")

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.input_embedding(input.to(torch.int32))

        # Puzzle embeddings
        if self.hparams.puzzle_emb_dim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.hparams.hidden_size - puzzle_embedding.shape[-1]

            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            # print(f"Catting {puzzle_embedding.view(-1, self.puzzle_emb_len, self.hparams.hidden_size).shape=} and {embedding.shape=}")
            embedding = torch.cat(
                (
                    puzzle_embedding.view(-1, self.puzzle_emb_len, self.hparams.hidden_size),
                    embedding,
                ),
                dim=-2,
            )

        # Scale
        return self.embed_scale * embedding

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["input"].shape[0]
        device = batch["input"].device

        return TRMCarry(
            inner_carry=self.empty_carry(
                batch_size, device
            ),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),  # Default to halted
            current_data={k: torch.empty_like(v, device=device) for k, v in batch.items()},
        )

    def empty_carry(self, batch_size: int, device: torch.device) -> TRMInnerCarry:
        return TRMInnerCarry(
            z_H=torch.empty(
                batch_size,
                self.hparams.seq_len + self.puzzle_emb_len,
                self.hparams.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                self.hparams.seq_len + self.puzzle_emb_len,
                self.hparams.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMInnerCarry) -> TRMInnerCarry:
        return TRMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.z_H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.z_L_init, carry.z_L),
        )

    def inner_forward(
        self, carry: TRMInnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.pos_embedding() if hasattr(self, "pos_embedding") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["input"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.hparams.H_cycles - 1):
                for _L_step in range(self.hparams.L_cycles):
                    z_L = self.lenet(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
        # 1 with grad
        for _L_step in range(self.hparams.L_cycles):
            z_L = self.lenet(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.lenet(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]
        q_logits = self.q_head(z_H[:, 0]).to(
            torch.float32
        )  # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])

    def forward(
        self, carry: TRMCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRMCarry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner_forward(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.hparams.N_supervision

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.hparams.N_supervision > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes

                halted = halted | (q_halt_logits > 0)

                # Exploration
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.hparams.halt_exploration_prob
                ) * torch.randint_like(new_steps, low=2, high=self.hparams.N_supervision + 1)
                halted = halted & (new_steps >= min_halt_steps)

        return TRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs

    def compute_loss_and_metrics(self, carry, batch):
        """Compute loss and metrics without circular reference."""
        # Get model outputs
        new_carry, outputs = self.forward(carry, batch)
        labels = new_carry.current_data["output"]

        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)

            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
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
        lm_loss = (
            stablemax_cross_entropy(
                outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask
            )
            / loss_divisor
        ).sum()

        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
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
        import time
        t0 = time.time()

        # if self.last_step_time is not None:
        #     print(f"Time since last training step: {time.time() - self.last_step_time:.4f} s")

        batch_size = batch["input"].shape[0]

        # log.info(f"Batch data samples: input: {batch['input'][:25]} \n output: {batch['output'][:25]} \n puzzle_identifiers: {batch['puzzle_identifiers'][:25]}")

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

        # Forward with loss computation
        self.carry, loss, metrics, all_halted = self.compute_loss_and_metrics(self.carry, batch)

        scaled_loss = loss / batch_size
        scaled_loss.backward()

        lr_this_step = None
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Learning rate scheduling with warmup
        current_step = self.global_step
        total_steps = getattr(self, 'total_steps', self.hparams.max_steps)

        # Base learning rates for each optimizer
        base_lrs = [self.hparams.learning_rate]
        if len(opts) > 1:  # If we have puzzle embedding optimizer
            base_lrs.append(self.hparams.learning_rate_emb)
        
        # Compute learning rate for this step
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
                # Constant LR after warmup (you can add decay here if needed)
                lr_this_step = base_lr
            
            # Update learning rate
            if hasattr(opt, "_optimizer"):
                for param_group in opt._optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                opt._optimizer.step()
                opt._optimizer.zero_grad()
            else:
                for param_group in opt.param_groups:
                    param_group['lr'] = lr_this_step
                opt.step()
                opt.zero_grad()
        
        # Log learning rate (will log the last optimizer's LR)
        self.log("train/lr", lr_this_step, on_step=True)

        # Log metrics
        if metrics.get("count", 0) > 0:
            with torch.no_grad():
                count = metrics["count"]
                self.log("train/accuracy", metrics.get("accuracy", 0) / count, on_step=True)
                self.log(
                    "train/exact_accuracy",
                    metrics.get("exact_accuracy", 0) / count,
                    prog_bar=True,
                    on_step=True,
                )
                self.log(
                    "train/q_halt_accuracy",
                    metrics.get("q_halt_accuracy", 0) / count,
                    on_step=True,
                )
                self.log(
                    "train/steps",
                    metrics.get("steps", 0) / count,
                    prog_bar=True,
                    on_step=True,
                )

                self.log("train/lm_loss", metrics.get("lm_loss", 0) / batch_size, on_step=True)
                self.log(
                    "train/q_halt_loss", metrics.get("q_halt_loss", 0) / batch_size, on_step=True
                )

                avg_halt_steps = metrics.get("steps", 0) / metrics["count"]
                early_halt_rate = avg_halt_steps < self.hparams.N_supervision
                self.log("train/early_halt_rate", early_halt_rate, on_step=True)

        t1 = time.time()
        # print(f"Training step time: {t1 - t0:.4f} s")

        self.last_step_time = t1
        
        return loss

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
            while True:
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
                    "val/lm_loss": accumulated_metrics.get("lm_loss", 0) / (n_steps * batch_size),
                    "val/q_halt_loss": accumulated_metrics.get("q_halt_loss", 0)
                    / (n_steps * batch_size),
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
        # self.carry = None
        pass

    def configure_optimizers(self):
        """Configure optimizer with different learning rates for different parameter groups."""

        base_lr = self.hparams.learning_rate
        embedding_lr = self.hparams.learning_rate_emb

        optimizers = []

        # Main optimizer
        # Use AdamATan2 if available
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

        # Force sparse embedding to be leaf tensors
        if hasattr(self, "puzzle_emb") and self.puzzle_emb is not None:
            # Force sparse embedding local weights to be leaf tensors
            self.puzzle_emb.local_weights = self.puzzle_emb.local_weights.detach().requires_grad_(
                True
            )

            # Add sparse embedding optimizer
            sparse_opt = CastedSparseEmbeddingSignSGD_Distributed(
                self.puzzle_emb.buffers(),
                lr=embedding_lr,
                weight_decay=self.hparams.weight_decay,
                world_size=1,
            )
            optimizers.append(sparse_opt)

        return optimizers
