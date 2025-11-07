"""
HRM PyTorch Lightning Module - Following Figure 2 pseudocode exactly
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.nn.utils.constants import IGNORE_LABEL_ID

try:
    from adam_atan2 import AdamATan2
except ImportError:
    print("Failed to import adam2")

from lightning import LightningModule

from src.nn.modules.sparse_embeddings import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed
)
from src.nn.modules.trm_block import (
    CastedEmbedding,
    CastedLinear,
    RotaryEmbedding
)
from src.nn.modules.trm_block import ReasoningBlock, ReasoningBlockConfig, ReasoningModule
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
        num_heads: int = 8, # min(2, hidden_size // 64)
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
        vocab_size: int = 0, # Should be set from datamodule
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
        self.pos_embedding = RotaryEmbedding(dim=hidden_size // num_heads,
                                              max_position_embeddings=seq_len + puzzle_emb_len,
                                              base=rope_theta)
        
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

        self.lenet = ReasoningModule(layers = [ReasoningBlock(reasoning_config) for _ in range(num_layers)])

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

        self.z_H_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.z_L_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        # self.register_buffer("z_H_init", trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        # self.register_buffer("z_L_init", trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1), persistent=True)        
        
        self.last_step_time = None

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
            log.info(f"Created puzzle_emb with batch_size={batch_size}")
            log.info(f"puzzle_emb.local_weights.shape: {self.puzzle_emb.local_weights.shape}")
            log.info(f"puzzle_emb.weights.shape: {self.puzzle_emb.weights.shape}")
        else:
            log.info("puzzle_emb_dim <= 0, not creating puzzle embeddings")
            self.puzzle_emb = None
            self.puzzle_emb_len = 0


    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        input = input.view(input.shape[0], -1) # flatten 2D input
        embedding = self.input_embedding(input.to(torch.int32))

        # Puzzle embeddings
        if self.hparams.puzzle_emb_dim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.hparams.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.hparams.hidden_size), embedding), dim=-2)

        # Scale
        return self.embed_scale * embedding
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["input"].shape[0]
        device = batch["input"].device

        return TRMCarry(
            inner_carry=self.empty_carry(batch_size, device),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted
            current_data={k: torch.empty_like(v, device=device) for k, v in batch.items()}
        )

    def empty_carry(self, batch_size: int, device: torch.device) -> TRMInnerCarry:
        return TRMInnerCarry(
            z_H=torch.empty(batch_size, self.hparams.seq_len + self.puzzle_emb_len, self.hparams.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, self.hparams.seq_len + self.puzzle_emb_len, self.hparams.hidden_size, dtype=self.forward_dtype, device=device),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMInnerCarry) -> TRMInnerCarry:
        return TRMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.z_H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.z_L_init, carry.z_L),
        )

    def inner_forward(self, carry: TRMInnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TRMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
            for _H_step in range(self.hparams.H_cycles-1):
                for _L_step in range(self.hparams.L_cycles):
                    z_L = self.lenet(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
        # 1 with grad
        for _L_step in range(self.hparams.L_cycles):
            z_L = self.lenet(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.lenet(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])
    
    def forward(self, carry: TRMCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TRMCarry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner_forward(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
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
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.hparams.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.hparams.N_supervision + 1)
                halted = halted & (new_steps >= min_halt_steps)

        return TRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
    
    def compute_loss_and_metrics(self, carry, batch):
        """Compute loss and metrics without circular reference."""
        # Get model outputs
        new_carry, outputs = self.forward(carry, batch)
        labels = new_carry.current_data["output"].flatten(start_dim=1)

        # Extract labels
        # y_true = batch["output"]
        # y_true_flat = y_true.flatten(start_dim=1)

        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # print(f"{outputs['preds']=}")
            # print(f"{labels=}")
            # print(f"{outputs['logits']=}")

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
                        
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)

            # print(f"{valid_metrics=}")

            acc = torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum()

            # print(f"{loss_counts=} {loss_divisor=} {is_correct=} {seq_is_correct=} {acc=}")
            # print(f"acc = {acc.item()}")

            # assert acc < 1e-6
            
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
        lm_loss = (stablemax_cross_entropy(
            outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask
        ) / loss_divisor).sum()

        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        total_loss = lm_loss + 0.5 * q_halt_loss

        # print(f"[{self.training=}]\t{lm_loss=} \t {q_halt_loss=} total_loss={total_loss=}")
        
        return new_carry, total_loss, metrics, new_carry.halted.all()
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step that implements supervision through multiple forward passes.
        Each sequence can run up to N_supervision (halt_max_steps) times.
        """
        # Time since last step
        # if self.last_step_time is not None:
        #     gap = time.time() - self.last_step_time
        #     if gap > 0.05:  # Log large gaps
        #         log.info(f"Gap between steps: {gap:.3f}s")

        t_start = time.time()
        
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
        t0_loss = time.time()
        self.carry, loss, metrics, all_halted = self.compute_loss_and_metrics(self.carry, batch)
        t1_loss = time.time()
        # log.info(f"Forward pass time: {t1_loss - t0_loss:.4f}s")

        log.info(f"training_step: loss={loss.item():.4f}, all_halted={all_halted} count = {metrics.get('count', 0)} accuracy = {metrics.get('accuracy', 0)} exact_accuracy = {metrics.get('exact_accuracy', 0)}")

        scaled_loss = loss / batch_size
        scaled_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Optimizer step
        for opt in opts:
            if hasattr(opt, "_optimizer"):
                opt._optimizer.step()
                opt._optimizer.zero_grad()
            else:
                opt.step()
                opt.zero_grad()

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

                log.info(f"Logging lm_loss {metrics.get('lm_loss', 0) / batch_size} \t count = {count.item()} -> train/accuracy = {metrics.get('accuracy', 0) / count} train/exact_accuracy = {metrics.get('exact_accuracy', 0) / count}")

                self.log("train/lm_loss", metrics.get("lm_loss", 0) / batch_size, on_step=True)
                self.log("train/q_halt_loss", metrics.get("q_halt_loss", 0) / batch_size, on_step=True)

                avg_halt_steps = metrics.get("steps", 0) / metrics["count"]
                early_halt_rate = avg_halt_steps < self.hparams.N_supervision
                self.log("train/early_halt_rate", early_halt_rate, on_step=True)

        t_end = time.time()
        # log.info(f"Total training_step time: {t_end - t_start:.4f}s")
        self.last_step_time = time.time()
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Simplified validation using loss head."""

        # log.info(f"*" * 50)
        # log.info("Calling validation!")
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
                    "val/q_halt_loss": accumulated_metrics.get("q_halt_loss", 0) / (n_steps * batch_size),
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
            # log.info(f"\t [val] Logging accuracy = {avg_metrics['val/accuracy']} and exact accuracy {avg_metrics['val/exact_accuracy']}")
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

        base_lr = self.hparams.learning_rate  # / self.hparams.N_supervision
        embedding_lr = self.hparams.learning_rate_emb  # / self.hparams.N_supervision

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
            except NameError:
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
