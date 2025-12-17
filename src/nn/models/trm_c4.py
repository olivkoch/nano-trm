"""
HRM/TRM PyTorch Lightning Module - Refactored to use base class
"""

import os
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn.models.c4_base import C4BaseModule
from src.nn.utils.constants import C4_EMPTY_CELL
from src.nn.modules.utils import compute_lr
from src.nn.utils.trm_debug_viz import debug_viz_training_step

try:
    from adam_atan2 import AdamATan2
except ImportError:
    print("Failed to import adam2")

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
    z_H: torch.Tensor  # High-level state (y in the paper)
    z_L: torch.Tensor  # Low-level state (z in the paper)


@dataclass
class TRMCarry:
    """Carry structure for maintaining state across steps."""
    inner_carry: TRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]  # Stores current batch data


class TRMC4Module(C4BaseModule):
    """
    HRM/TRM implementation - now inherits self-play from base
    """
    TRM_DEFAULTS = dict(
        num_heads=8,
        max_grid_size=30,
        H_cycles=1,
        L_cycles=1,
        N_supervision=1,
        N_supervision_val=1,
        ffn_expansion=2,
        learning_rate_emb=1e-2,
        halt_exploration_prob=0.1,
        puzzle_emb_dim=512,
        puzzle_emb_len=16,
        rope_theta=10000,
        vocab_size=3,
        num_puzzles=1,
        pad_value=-1,
        seq_len=42,
        use_mlp_t=False,
    )

    def __init__(self, **kwargs):
        # Initialize base class with all shared parameters
        merged = {**self.TRM_DEFAULTS, **kwargs}
        super().__init__(**merged)
        
        # self.automatic_optimization = False

        self.forward_dtype = torch.float32
        
        # Token embeddings
        self.embed_scale = math.sqrt(self.hparams.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
                        
        self.input_embedding = CastedEmbedding(
            self.hparams.vocab_size, self.hparams.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        
        # Positional embeddings with rotary embeddings
        self.pos_embedding = RotaryEmbedding(
            dim=self.hparams.hidden_size // self.hparams.num_heads,
            max_position_embeddings=self.hparams.seq_len + self.hparams.puzzle_emb_len,
            base=self.hparams.rope_theta,
        )
        
        # Single network (not two separate networks)
        reasoning_config = ReasoningBlockConfig(
            hidden_size=self.hparams.hidden_size,
            num_heads=self.hparams.num_heads,
            expansion=self.hparams.ffn_expansion,
            rms_norm_eps=1e-5,
            seq_len=self.hparams.seq_len,
            mlp_t=self.hparams.use_mlp_t,
            puzzle_emb_ndim=self.hparams.puzzle_emb_dim,
            puzzle_emb_len=self.hparams.puzzle_emb_len,
        )
        
        self.lenet = ReasoningModule(
            layers=[ReasoningBlock(reasoning_config) for _ in range(self.hparams.num_layers)]
        )
        
        # self.policy_value_cnn = nn.Sequential(
        #     nn.Conv2d(self.hparams.hidden_size, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        # )
        # self.policy_conv = nn.Conv2d(32, 1, kernel_size=1)  # [B, 1, 6, 7] -> pool rows -> [B, 7]
        # self.value_fc = nn.Linear(32 * 6 * 7, 1)

        # self.lm_head = CastedLinear(self.hparams.hidden_size, self.board_cols, bias=False)
        self.lm_head = CastedLinear(self.hparams.hidden_size, self.hparams.vocab_size, bias=False)
        self.value_head = CastedLinear(self.hparams.hidden_size, 1, bias=False)
        self.policy_head = CastedLinear(self.hparams.hidden_size, self.board_cols, bias=False)
        self.q_head = CastedLinear(self.hparams.hidden_size, 1, bias=True) # only learn to stop, not to continue
        
        # Halting head initialization
        with torch.no_grad():
            self.q_head.weight.zero_()
            if self.q_head.bias is not None:
                self.q_head.bias.fill_(-5.0)  # Strong negative bias
        
        # # State for carry (persisted across training steps)
        self.carry = None
        
        self.z_H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.hparams.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.z_L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.hparams.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        
        # Add puzzle embeddings
        if self.hparams.puzzle_emb_dim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                num_embeddings=self.hparams.num_puzzles,
                embedding_dim=self.hparams.puzzle_emb_dim,
                batch_size=self.hparams.batch_size,
                init_std=0.0,  # Reference uses 0 init
                cast_to=self.forward_dtype,
            )
            self.puzzle_emb_len = self.hparams.puzzle_emb_len
            log.info(f"Created puzzle_emb with num_puzzles={self.hparams.num_puzzles}, batch_size={self.hparams.batch_size}")
        else:
            log.info("puzzle_emb_dim <= 0, not creating puzzle embeddings")
            self.puzzle_emb = None
            self.puzzle_emb_len = 0
        
        self.last_step_time = None

        # Debug visualization settings
        self.debug_viz_enabled = False  # Toggle visualization on/off
        self.debug_viz_every_n_steps = 500  # Save every N steps
        self.debug_viz_dir = "debug_viz"  # Output directory
        
    def setup(self, stage: str):

        super().setup(stage)
        
        if stage == "fit":
            # Add torch.compile for faster training
            if "DISABLE_COMPILE" not in os.environ and hasattr(torch, 'compile') and self.device.type == "cuda":
                try:
                    log.info("Compiling inner_forward with torch.compile...")
                    self.inner_forward = torch.compile(
                        self.inner_forward,
                        mode="default",  # TODO: make "reduce-overhead" work, good for repeated calls (H/L cycles)
                        fullgraph=False,         # Allow graph breaks for dynamic control flow
                    )
                    log.info("Compilation successful")
                except Exception as e:
                    log.warning(f"torch.compile failed, running uncompiled: {e}")
            else:
                log.info('*' * 60)
                log.info("torch.compile not available or disabled, running uncompiled")

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor, current_player: torch.Tensor):
        
        batch_size = input.shape[0]

        input = self._canonicalize_board(input, current_player)
        embeddings = self.input_embedding(input.to(torch.int32))
        
        # Puzzle embeddings
        if self.hparams.puzzle_emb_dim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            target_size = self.puzzle_emb_len * self.hparams.hidden_size
            current_size = puzzle_embedding.shape[-1]    

            if current_size < target_size:
                puzzle_embedding = F.pad(puzzle_embedding, (0, target_size - current_size))
            elif current_size > target_size:
                puzzle_embedding = puzzle_embedding[..., :target_size]  # Truncate

            puzzle_embedding = puzzle_embedding.view(batch_size, self.puzzle_emb_len, self.hparams.hidden_size)
        
            embeddings = torch.cat((puzzle_embedding, embeddings), dim=-2)
        
        return self.embed_scale * embeddings
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["boards"].shape[0]
        device = batch["boards"].device
        
        return TRMCarry(
            inner_carry=self.empty_carry(batch_size, device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
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
    ) -> Tuple[TRMInnerCarry, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.pos_embedding() if hasattr(self, "pos_embedding") else None,
        )
        
        # Input encoding
        input_embeddings = self._input_embeddings(batch["boards"], batch["puzzle_identifiers"], batch["current_player"])
        
        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _ in range(self.hparams.H_cycles - 1):
                for _ in range(self.hparams.L_cycles):
                    z_L = self.lenet(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
        # 1 with grad
        for _ in range(self.hparams.L_cycles):
            z_L = self.lenet(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.lenet(z_H, z_L, **seq_info)
        
        # LM Outputs
        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad

        # policy_logits = self.lm_head(z_H[:, 0])
        # value = torch.tanh(self.value_head(z_H[:, 0]))

        # board_repr = z_H[:, self.puzzle_emb_len:]  # [B, 42, hidden] - skip puzzle prefix
        # global_repr = board_repr.mean(dim=1)  # [B, hidden]

        # policy_logits = self.lm_head(global_repr)
        # value = torch.tanh(self.value_head(global_repr))
        # board_repr = z_H[:, self.puzzle_emb_len:]  # [B, 42, hidden]
        # B = board_repr.shape[0]
        
        # # Reshape to 2D spatial format
        # x = board_repr.view(B, 6, 7, -1).permute(0, 3, 1, 2)  # [B, hidden, 6, 7]
        # x = self.policy_value_cnn(x)  # [B, 32, 6, 7]
        
        # # Policy: column-wise (pool over rows)
        # policy_logits = self.policy_conv(x).mean(dim=2).squeeze(1)  # [B, 7]
        
        # # Value: global
        # value = torch.tanh(self.value_fc(x.flatten(1)))  # [B, 1]
        logits = self.lm_head(z_H)[:, self.puzzle_emb_len :]
        policy_logits = self.policy_head(z_H[:, 0])
        value = torch.tanh(self.value_head(z_H[:, 0]))
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
    
        return new_carry, logits, policy_logits, value, q_logits[..., 0]

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
        new_inner_carry, logits, policy_logits, value, q_halt_logits = self.inner_forward(
            new_inner_carry, new_current_data
        )

        boards = new_current_data["boards"]
        boards_reshaped = boards.view(-1, self.board_rows, self.board_cols)
        valid_actions = boards_reshaped[:, 0, :] == C4_EMPTY_CELL
        policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
        
        # Compute policy probabilities
        policy = F.softmax(policy_logits, dim=-1)
        
        outputs = {
            "logits": logits,
            "policy_logits": policy_logits,
            "policy": policy,
            "value": value.squeeze(-1),
            "q_halt_logits": q_halt_logits,
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            n_supervision_steps = (
                self.hparams.N_supervision if self.training else self.hparams.N_supervision_val
            )
            
            is_last_step = new_steps >= n_supervision_steps
            
            halted = is_last_step
            
            # if training, and ACT is enabled
            if self.training and (self.hparams.N_supervision > 1):
                # Halt signal
                halted = halted | (q_halt_logits > 0)
                
                # Exploration
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.hparams.halt_exploration_prob
                ) * torch.randint_like(new_steps, low=2, high=self.hparams.N_supervision + 1)
                halted = halted & (new_steps >= min_halt_steps)
        
        return TRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
    
    def forward_for_mcts(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for MCTS - handles carry and returns policy/value"""

        if self.training:
            log.warning("forward_for_mcts called in training mode!")
    
        batch_size = batch["boards"].shape[0]
        
        with torch.no_grad():

            # Initialize carry for this batch
            carry = self.initial_carry(batch)

            while True:
                carry, outputs = self.forward(carry, batch)

                if carry.halted.all():
                    break

            # Extract only the actual batch results
            return {
                "policy": outputs["policy"][:batch_size],
                "value": outputs["value"][:batch_size]
            }
     
    def compute_loss_and_metrics(self, carry, batch):
        """Compute loss and metrics for TRM"""

        # Get model outputs
        new_carry, outputs = self.forward(carry, batch)
        
        labels = new_carry.current_data["outputs"]
        labels_canonical = self._canonicalize_board(labels, new_carry.current_data["current_player"])

        # Debug visualization
        if getattr(self, 'debug_viz_enabled', True) and hasattr(self, 'manual_step'):
            viz_path = debug_viz_training_step(
                batch=new_carry.current_data,
                outputs=outputs,
                labels=labels,
                step=self.manual_step,
                save_dir=getattr(self, 'debug_viz_dir', 'debug_viz'),
                every_n_steps=getattr(self, 'debug_viz_every_n_steps', 500),
                sample_idx=10,  # Which sample in batch to visualize
                current_player=new_carry.current_data.get("current_player"),
            )
            if viz_path:
                log.info(f"Saved debug viz: {viz_path}")

        # if self.manual_step % 500 == 0:
        #     self._print_debug_samples(batch, n_samples=3)
        labels = labels_canonical

        # Get legal moves for each position
        boards = new_carry.current_data["boards"]
        boards_reshaped = boards.view(-1, 6, 7)
        legal_moves = boards_reshaped[:, 0, :] == C4_EMPTY_CELL  # Top row empty = legal
        
        target_policy = new_carry.current_data["policies"]
        target_policy = target_policy * legal_moves.float()
        target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        
        with torch.no_grad():

            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)
            is_correct = torch.argmax(outputs["logits"], dim=-1) == labels
            label_counts = labels.shape[-1]
            seq_is_correct = is_correct.sum(-1) == label_counts

            pred_actions = outputs["policy"].argmax(dim=-1)
            target_actions = target_policy.argmax(dim=-1)
            policy_accuracy = (pred_actions == target_actions).float()

            # Metrics (halted)
            valid_metrics = new_carry.halted
            
            raw_metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics, (is_correct.float() / label_counts).sum(-1), 0
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "policy_accuracy": torch.where(valid_metrics, policy_accuracy, 0).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"].squeeze() >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }
        
        lm_loss = (
            stablemax_cross_entropy(
                outputs["logits"], labels, ignore_index=-100
            ) / label_counts
        ).sum()

        target_values = new_carry.current_data["values"]  # From dataloader
        value_loss = F.mse_loss(outputs["value"], target_values, reduction="sum")

        policy_loss = -torch.sum(target_policy * torch.log(outputs["policy"] + 1e-8), dim=-1).sum()
        
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        raw_metrics.update(
            {
                "lm_loss": lm_loss.detach(),
                "policy_loss": policy_loss.detach(),
                "value_loss": value_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            }
        )
        
        total_loss = lm_loss + policy_loss + value_loss + 0.5 * q_halt_loss
        
        self.carry = new_carry

        batch_size = batch["boards"].shape[0]
        count = raw_metrics["count"].item()

        metrics = {
            "lm_loss": raw_metrics["lm_loss"].item() / batch_size,
            "policy_loss": raw_metrics["policy_loss"].item() / batch_size,
            "value_loss": raw_metrics["value_loss"].item() / batch_size,
            "q_halt_loss": raw_metrics["q_halt_loss"].item() / batch_size,
            "count": count,
        }

        if count > 0:
            metrics.update({
                "accuracy": raw_metrics["accuracy"].item() / count,
                "exact_accuracy": raw_metrics["exact_accuracy"].item() / count,
                "policy_accuracy": raw_metrics["policy_accuracy"].item() / count,
                "q_halt_accuracy": raw_metrics["q_halt_accuracy"].item() / count,
                "steps": raw_metrics["steps"].item() / count,
                })

        return new_carry, total_loss, metrics, new_carry.halted.all()
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for TRM with carry state"""
        import time
        
        t0 = time.time()
        
        batch_size = batch["boards"].shape[0]
        
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
        self.carry, loss, metrics, _ = self.compute_loss_and_metrics(self.carry, batch)
        
        scaled_loss = loss / batch_size
        scaled_loss.backward()
        
        self.log_grad_norms()
        
        lr_this_step = None
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Learning rate scheduling with warmup
        current_step = self.manual_step
        total_steps = getattr(self, "total_steps", self.max_steps)
        
        # Base learning rates for each optimizer
        base_lrs = [self.hparams.learning_rate]
        if len(opts) > 1:  # If we have puzzle embedding optimizer
            base_lrs.append(self.hparams.learning_rate_emb)
        
        # Compute learning rate for this step
        for i, (opt, base_lr) in enumerate(zip(opts, base_lrs)):
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

            # Update learning rate
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
            
            # Log learning rate
            self.log(f"train/lr_{i}", lr_this_step, on_step=True)

        # Log metrics
        if metrics["count"] > 0:
            self.log("train/accuracy", metrics["accuracy"], prog_bar=True, on_step=True)
            self.log("train/exact_accuracy", metrics["exact_accuracy"], prog_bar=True, on_step=True)
            self.log("train/policy_accuracy", metrics["policy_accuracy"], prog_bar=True, on_step=True)
            self.log("train/q_halt_accuracy", metrics["q_halt_accuracy"], on_step=True)
            self.log("train/steps", metrics["steps"], prog_bar=True, on_step=True)
    
        # These can be logged unconditionally (they're per-batch, not per-halted-sample)
        self.log("train/lm_loss", metrics["lm_loss"], on_step=True)
        self.log("train/value_loss", metrics["value_loss"], on_step=True)
        self.log("train/policy_loss", metrics["policy_loss"], on_step=True)
        self.log("train/q_halt_loss", metrics["q_halt_loss"], on_step=True)
        
        assert not torch.isnan(loss), f"Total loss is NaN at step {self.manual_step}"
        
        t1 = time.time()
        
        self.last_step_time = t1
        self.manual_step += 1
        
        return loss
    
    def log_grad_norms(self):
        # Gradient monitoring
        with torch.no_grad():
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), 
                max_norm=float('inf')  # Don't actually clip, just compute norm
            ).item()
            
            # Key component gradient norms
            grad_metrics = {}
            
            # First attention layer
            # if self.lenet.layers[0].self_attn.qkv_proj.weight.grad is not None:
            #     grad_metrics['first_attn'] = self.lenet.layers[0].self_attn.qkv_proj.weight.grad.norm().item()
            
            # Last MLP layer
            # if self.lenet.layers[-1].mlp.down_proj.weight.grad is not None:
            #     grad_metrics['last_mlp'] = self.lenet.layers[-1].mlp.down_proj.weight.grad.norm().item()
            
            # Output heads
            # if self.lm_head.weight.grad is not None:
            #     grad_metrics['lm_head'] = self.lm_head.weight.grad.norm().item()
            
            if self.q_head.weight.grad is not None:
                grad_metrics['q_head'] = self.q_head.weight.grad.norm().item()
            
            # Log main metric
            self.log('grad/total_norm', total_grad_norm, on_step=True, prog_bar=True)
            
            # Log gradient flow ratio (first vs last layer)
            if 'first_attn' in grad_metrics and 'last_mlp' in grad_metrics:
                ratio = grad_metrics['first_attn'] / (grad_metrics['last_mlp'] + 1e-8)
                self.log('grad/flow_ratio', ratio, on_step=True, prog_bar=True)

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
    
    def on_train_epoch_start(self):
        ans = super().on_train_epoch_start()
        return ans
