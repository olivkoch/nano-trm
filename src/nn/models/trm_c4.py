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
    )

    def __init__(self, **kwargs):
        # Initialize base class with all shared parameters
        merged = {**self.TRM_DEFAULTS, **kwargs}
        super().__init__(**merged)
        
        self.forward_dtype = torch.float32
        
        # Token embeddings
        self.embed_scale = math.sqrt(self.hparams.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        
        self.input_embedding = CastedEmbedding(
            self.hparams.vocab_size, self.hparams.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )
        
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
            mlp_t=False,
            puzzle_emb_ndim=self.hparams.puzzle_emb_dim,
            puzzle_emb_len=self.hparams.puzzle_emb_len,
        )
        
        self.lenet = ReasoningModule(
            layers=[ReasoningBlock(reasoning_config) for _ in range(self.hparams.num_layers)]
        )
        
        self.lm_head = CastedLinear(self.hparams.hidden_size, self.board_cols, bias=False)
        self.value_head = CastedLinear(self.hparams.hidden_size, 1, bias=False)
        self.q_head = CastedLinear(self.hparams.hidden_size, 1, bias=True) # only learn to stop, not to continue
        
        # Halting head initialization
        with torch.no_grad():
            self.q_head.weight.zero_()
            if self.q_head.bias is not None:
                self.q_head.bias.fill_(-5.0)  # Strong negative bias
        
        # State for carry (persisted across training steps)
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
        
        log.info(f"Learning rates: model={self.hparams.learning_rate}, emb={self.hparams.learning_rate_emb} max steps = {self.max_steps}")

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

        # Token embedding
        embedding = self.input_embedding(input.to(torch.int32))
        
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
        
            embedding = torch.cat((puzzle_embedding, embedding), dim=-2)
        
        # Scale
        return self.embed_scale * embedding
    
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
    
    # def reset_carry(self, reset_flag: torch.Tensor, carry: TRMInnerCarry) -> TRMInnerCarry:
    #     """Reset carry with position-aware initialization"""
    #     batch_size = carry.z_H.shape[0]
        
    #     # Expand init to batch size: (seq, hidden) -> (batch, seq, hidden)
    #     z_H_init_expanded = self.z_H_init.unsqueeze(0).expand(batch_size, -1, -1)
    #     z_L_init_expanded = self.z_L_init.unsqueeze(0).expand(batch_size, -1, -1)
        
    #     return TRMInnerCarry(
    #         z_H=torch.where(reset_flag.view(-1, 1, 1), z_H_init_expanded, carry.z_H),
    #         z_L=torch.where(reset_flag.view(-1, 1, 1), z_L_init_expanded, carry.z_L),
    #     )
    
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

        policy_logits = self.lm_head(z_H[:, 0])
        value = torch.tanh(self.value_head(z_H[:, 0]))
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
    
        return new_carry, policy_logits, value, q_logits[..., 0]

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
        new_inner_carry, policy_logits, value, q_halt_logits = self.inner_forward(
            new_inner_carry, new_current_data
        )
        
        boards = new_current_data["boards"]
        boards_reshaped = boards.view(-1, self.board_rows, self.board_cols)
        valid_actions = boards_reshaped[:, 0, :] == C4_EMPTY_CELL
        policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
        
        # Compute policy probabilities
        policy = F.softmax(policy_logits, dim=-1)
        
        outputs = {
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
        # Need to handle padding if batch size doesn't match
        actual_batch_size = batch["boards"].shape[0]
        
        # Pad to expected batch size if necessary
        if actual_batch_size < self.hparams.batch_size:
            pad_size = self.hparams.batch_size - actual_batch_size
            padded_batch = {}
            for k, v in batch.items():
                if k == "boards":
                    padded_batch[k] = F.pad(v, (0, 0, 0, pad_size))
                elif k in ("puzzle_identifiers", "current_player"):
                    padded_batch[k] = F.pad(v, (0, pad_size))
                else:
                    padded_batch[k] = v
        else:
            padded_batch = batch
        
        # Initialize carry for this batch
        carry = self.initial_carry(padded_batch)
        
        # Forward pass
        _, outputs = self.forward(carry, padded_batch)
        
        # Extract only the actual batch results
        return {
            "policy": outputs["policy"][:actual_batch_size],
            "value": outputs["value"][:actual_batch_size]
        }
    def _print_debug_samples(self, batch, n_samples=3):
        """Print debug info for random samples"""
        import random
        
        batch_size = batch["boards"].shape[0]
        indices = random.sample(range(batch_size), min(n_samples, batch_size))
        
        print(f"\n{'='*60}")
        print(f"DEBUG: Step {self.manual_step} - Sample Analysis")
        print(f"{'='*60}")
        
        # Get predictions without affecting training
        with torch.no_grad():
            # Use current carry state
            if self.carry is None:
                temp_carry = self.initial_carry(batch)
            else:
                temp_carry = self.carry
            
            _, outputs = self.forward(temp_carry, batch)
        
        for idx in indices:
            board = batch["boards"][idx].view(6, 7).cpu().numpy().astype(int)
            current_player = batch["current_player"][idx].item()
            target_policy = batch["policies"][idx].cpu().numpy()
            target_value = batch["values"][idx].item()
            
            pred_policy = outputs["policy"][idx].cpu().numpy()
            pred_value = outputs["value"][idx].item()
            
            target_action = target_policy.argmax()
            pred_action = pred_policy.argmax()
            
            print(f"\n--- Sample {idx} ---")
            print(f"Current player: {current_player} ({'X' if current_player == 1 else 'O'})")
            print(f"Board (0=empty, 1=P1, 2=P2):")
            
            # Pretty print board
            symbols = {0: '.', 1: 'X', 2: 'O'}
            for row in board:
                print("  " + " ".join(symbols[c] for c in row))
            print("  " + " ".join(str(i) for i in range(7)))  # Column numbers
            
            # Policies
            print(f"\nTarget policy: {target_policy.round(3)}")
            print(f"Pred policy:   {pred_policy.round(3)}")
            print(f"Target action: {target_action} | Pred action: {pred_action} | {'✓' if target_action == pred_action else '✗'}")
            
            # Values
            print(f"Target value: {target_value:+.3f} | Pred value: {pred_value:+.3f}")
            
            # Legal moves check
            legal_cols = [i for i in range(7) if board[0, i] == 0]
            print(f"Legal columns: {legal_cols}")
            
            # Check if target is putting mass on legal moves only
            illegal_mass = sum(target_policy[i] for i in range(7) if i not in legal_cols)
            if illegal_mass > 0.01:
                print(f"WARNING: Target has {illegal_mass:.2%} mass on illegal moves!")
        
        print(f"\n{'='*60}\n")
        
    def compute_loss_and_metrics(self, batch):
        """Compute loss and metrics for TRM"""

        # Get model outputs
        new_carry, outputs = self.forward(self.carry, batch)
        
        # if self.manual_step % 500 == 0:
        #     self._print_debug_samples(batch, n_samples=3)

        # Get legal moves for each position
        boards = new_carry.current_data["boards"]
        boards_reshaped = boards.view(-1, 6, 7)
        legal_moves = boards_reshaped[:, 0, :] == C4_EMPTY_CELL  # Top row empty = legal
        
        target_policy = new_carry.current_data["policies"]
        target_policy = target_policy * legal_moves.float()
        target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        
        with torch.no_grad():
            pred_actions = outputs["policy"].argmax(dim=-1)
            target_actions = target_policy.argmax(dim=-1)
            policy_accuracy = (pred_actions == target_actions).float()
            
            value_certainty = outputs["value"].abs()  # How sure we are about position
            should_halt = value_certainty > 0.8  # Halt when very certain about outcome
            
            # Metrics (halted)
            valid_metrics = new_carry.halted
            
            raw_metrics = {
                "count": valid_metrics.sum(),
                "policy_accuracy": torch.where(valid_metrics, policy_accuracy, 0).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == should_halt)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }
        
        target_values = new_carry.current_data["values"]  # From dataloader
        value_loss = F.mse_loss(outputs["value"], target_values, reduction="sum")
        policy_loss = -torch.sum(target_policy * torch.log(outputs["policy"] + 1e-8), dim=-1).sum()
        
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            should_halt.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        raw_metrics.update(
            {
                "policy_loss": policy_loss.detach(),
                "value_loss": value_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            }
        )
        
        total_loss = policy_loss + value_loss + q_halt_loss
        
        self.carry = new_carry

        batch_size = batch["boards"].shape[0]
        count = raw_metrics["count"].item()

        metrics = {
            "policy_loss": raw_metrics["policy_loss"].item() / batch_size,
            "q_halt_loss": raw_metrics["q_halt_loss"].item() / batch_size,
            "count": count,
        }

        if count > 0:
            metrics.update({
                "value_loss": raw_metrics["value_loss"].item() / count,
                "policy_accuracy": raw_metrics["policy_accuracy"].item() / count,
                "q_halt_accuracy": raw_metrics["q_halt_accuracy"].item() / count,
                "steps": raw_metrics["steps"].item() / count,
                })

        return total_loss, metrics
    
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
        loss, metrics = self.compute_loss_and_metrics(batch)
        
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
            self.log("train/policy_accuracy", metrics["policy_accuracy"], prog_bar=True, on_step=True)
            self.log("train/value_loss", metrics["value_loss"], on_step=True)
            self.log("train/q_halt_accuracy", metrics["q_halt_accuracy"], on_step=True)
            self.log("train/steps", metrics["steps"], prog_bar=True, on_step=True)
    
        # These can be logged unconditionally (they're per-batch, not per-halted-sample)
        self.log("train/policy_loss", metrics["policy_loss"], on_step=True)
        self.log("train/q_halt_loss", metrics["q_halt_loss"], on_step=True)
        
        # assert not torch.isnan(metrics.get("policy_loss")), f"Policy loss is NaN at step {self.manual_step}"
        
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
            if self.lenet.layers[0].self_attn.qkv_proj.weight.grad is not None:
                grad_metrics['first_attn'] = self.lenet.layers[0].self_attn.qkv_proj.weight.grad.norm().item()
            
            # Last MLP layer
            if self.lenet.layers[-1].mlp.down_proj.weight.grad is not None:
                grad_metrics['last_mlp'] = self.lenet.layers[-1].mlp.down_proj.weight.grad.norm().item()
            
            # Output heads
            if self.lm_head.weight.grad is not None:
                grad_metrics['lm_head'] = self.lm_head.weight.grad.norm().item()
            
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
        # if self.epoch_idx == 0:
        #     results = self.run_all_ablations()
        #     print(results)
            # if not self.test_trm_no_carry():
            #     raise RuntimeError("Model failed to overfit small batch - check architecture!")
        return ans
    
    def test_trm_no_carry(self):
        """Test if TRM can learn without carry complexity"""
        samples = self.replay_buffer.sample(self.hparams.batch_size)
        batch = {
            'boards': torch.stack([s['board'] for s in samples]).to(self.device),
            'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
            'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
            'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
            'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
        }
        
        # Use simple forward without carry
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4)
        
        print("Training TRM without carry...")
        for i in range(500):
            loss, metrics = self.compute_loss_simple(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            if i % 100 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={metrics['policy_accuracy']:.2%}")
        
        print(f"Final: policy_acc={metrics['policy_accuracy']:.2%}")
        return metrics['policy_accuracy'] > 0.9
    
    def forward_no_carry(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward without carry - like CNN baseline but using TRM architecture"""
        
        seq_info = dict(cos_sin=self.pos_embedding())
        
        # Input encoding
        input_emb = self._input_embeddings(
            batch["boards"], 
            batch["puzzle_identifiers"], 
            batch["current_player"]
        )
        
        # Initialize z_H and z_L from input (no carry)
        batch_size = input_emb.shape[0]
        z_H = input_emb  # Start from input embedding
        z_L = input_emb  # Start from input embedding
        
        # Run H/L cycles
        for _ in range(self.hparams.H_cycles):
            for _ in range(self.hparams.L_cycles):
                z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
            z_H = self.lenet(z_H, z_L, **seq_info)
        
        # Output heads
        policy_logits = self.lm_head(z_H[:, 0])
        value = torch.tanh(self.value_head(z_H[:, 0]))
        
        # Mask illegal moves
        boards = batch["boards"].view(-1, 6, 7)
        valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
        policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
        
        policy = F.softmax(policy_logits, dim=-1)
        
        return {
            "policy_logits": policy_logits,
            "policy": policy,
            "value": value.squeeze(-1),
        }


    def compute_loss_simple(self, batch):
        """Simplified loss without carry complexity"""
        outputs = self.forward_no_carry(batch)
        
        # Get legal moves
        boards = batch["boards"].view(-1, 6, 7)
        legal_moves = boards[:, 0, :] == C4_EMPTY_CELL
        
        # Process targets
        target_policy = batch["policies"]
        target_policy = target_policy * legal_moves.float()
        target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        target_values = batch["values"]
        
        # Losses
        policy_loss = -torch.sum(target_policy * torch.log(outputs["policy"] + 1e-8), dim=-1).mean()
        value_loss = F.mse_loss(outputs["value"], target_values)
        
        total_loss = policy_loss + 0.5 * value_loss
        
        with torch.no_grad():
            pred_actions = outputs["policy"].argmax(dim=-1)
            target_actions = target_policy.argmax(dim=-1)
            policy_accuracy = (pred_actions == target_actions).float().mean()
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "policy_accuracy": policy_accuracy.item(),
        }
        
        return total_loss, metrics
    
    def test_ablation_1_no_grad_block(self):
        """Test: Add torch.no_grad() block, keep everything else from working version"""
        samples = self.replay_buffer.sample(self.hparams.batch_size)
        batch = {
            'boards': torch.stack([s['board'] for s in samples]).to(self.device),
            'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
            'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
            'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
            'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
        }
        
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        print("Ablation 1: Adding torch.no_grad() block...")
        for i in range(500):
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            z_H = input_emb
            z_L = input_emb
            
            # H_cycles-1 WITHOUT grad (like real TRM)
            with torch.no_grad():
                for _ in range(self.hparams.H_cycles - 1):
                    for _ in range(self.hparams.L_cycles):
                        z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                    z_H = self.lenet(z_H, z_L, **seq_info)
            
            # Last H-cycle WITH grad
            for _ in range(self.hparams.L_cycles):
                z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
            z_H = self.lenet(z_H, z_L, **seq_info)
            
            # Global pool (working version)
            board_repr = z_H[:, self.puzzle_emb_len:]
            global_repr = board_repr.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            # Mask and loss
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            if i % 100 == 0:
                with torch.no_grad():
                    acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean()
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={acc.item():.2%}")
        
        with torch.no_grad():
            acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean()
        print(f"Ablation 1 Final: {acc.item():.2%}")
        return acc.item() > 0.9


    def test_ablation_2_fixed_init(self):
        """Test: Use z_H_init instead of input_emb for initialization"""
        samples = self.replay_buffer.sample(self.hparams.batch_size)
        batch = {
            'boards': torch.stack([s['board'] for s in samples]).to(self.device),
            'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
            'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
            'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
            'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
        }
        
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        print("Ablation 2: Using fixed z_H_init instead of input_emb...")
        for i in range(500):
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            batch_size = input_emb.shape[0]
            
            # Initialize from fixed init (like real TRM carry reset)
            z_H = self.z_H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, input_emb.shape[1], -1).clone()
            z_L = self.z_L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, input_emb.shape[1], -1).clone()
            
            # All cycles WITH grad for this test
            for _ in range(self.hparams.H_cycles):
                for _ in range(self.hparams.L_cycles):
                    z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
            
            # Global pool
            board_repr = z_H[:, self.puzzle_emb_len:]
            global_repr = board_repr.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            if i % 100 == 0:
                with torch.no_grad():
                    acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean()
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={acc.item():.2%}")
        
        with torch.no_grad():
            acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean()
        print(f"Ablation 2 Final: {acc.item():.2%}")
        return acc.item() > 0.9


    def test_ablation_3_both(self):
        """Test: Both no_grad block AND fixed init"""
        samples = self.replay_buffer.sample(self.hparams.batch_size)
        batch = {
            'boards': torch.stack([s['board'] for s in samples]).to(self.device),
            'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
            'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
            'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
            'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
        }
        
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        print("Ablation 3: Both no_grad block AND fixed init... learning rate =", self.hparams.learning_rate)
        for i in range(500):
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            batch_size = input_emb.shape[0]
            
            # Fixed init
            z_H = self.z_H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, input_emb.shape[1], -1).clone()
            z_L = self.z_L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, input_emb.shape[1], -1).clone()
            
            # no_grad block
            with torch.no_grad():
                for _ in range(self.hparams.H_cycles - 1):
                    for _ in range(self.hparams.L_cycles):
                        z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                    z_H = self.lenet(z_H, z_L, **seq_info)
            
            for _ in range(self.hparams.L_cycles):
                z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
            z_H = self.lenet(z_H, z_L, **seq_info)
            
            # Global pool
            board_repr = z_H[:, self.puzzle_emb_len:]
            global_repr = board_repr.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            if i % 100 == 0:
                with torch.no_grad():
                    acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean()
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={acc.item():.2%}")
        
        with torch.no_grad():
            acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean()
        print(f"Ablation 3 Final: {acc.item():.2%}")
        return acc.item() > 0.9
    
    def test_ablation_4_varying_batches(self):
        """Test: Like ablation_3 but with DIFFERENT batches each step"""
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        print("Ablation 4: Fixed init + no_grad, but VARYING batches...")
        
        final_acc = 0
        for i in range(500):
            # NEW batch each iteration (like real training)
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
            }
            
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            batch_size = input_emb.shape[0]
            
            # Fixed init (like carry reset)
            z_H = self.z_H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, input_emb.shape[1], -1).clone()
            z_L = self.z_L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, input_emb.shape[1], -1).clone()
            
            # no_grad block
            with torch.no_grad():
                for _ in range(self.hparams.H_cycles - 1):
                    for _ in range(self.hparams.L_cycles):
                        z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                    z_H = self.lenet(z_H, z_L, **seq_info)
            
            for _ in range(self.hparams.L_cycles):
                z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
            z_H = self.lenet(z_H, z_L, **seq_info)
            
            # Global pool
            board_repr = z_H[:, self.puzzle_emb_len:]
            global_repr = board_repr.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.9 * final_acc + 0.1 * acc  # EMA
            
            if i % 100 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={acc:.2%}, ema_acc={final_acc:.2%}")
        
        print(f"Ablation 4 Final EMA: {final_acc:.2%}")
        return final_acc > 0.8


    def test_ablation_5_with_carry(self):
        """Test: Add actual carry mechanism"""
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        print("Ablation 5: With CARRY mechanism...")
        
        # Initialize carry (like real training)
        carry_z_H = None
        carry_z_L = None
        
        final_acc = 0
        for i in range(500):
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
            }
            
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            batch_size = input_emb.shape[0]
            
            # Initialize or reuse carry
            if carry_z_H is None:
                z_H = self.z_H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, input_emb.shape[1], -1).clone()
                z_L = self.z_L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, input_emb.shape[1], -1).clone()
            else:
                # Reuse previous carry (this is what real TRM does!)
                z_H = carry_z_H
                z_L = carry_z_L
            
            # no_grad block
            with torch.no_grad():
                for _ in range(self.hparams.H_cycles - 1):
                    for _ in range(self.hparams.L_cycles):
                        z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                    z_H = self.lenet(z_H, z_L, **seq_info)
            
            for _ in range(self.hparams.L_cycles):
                z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
            z_H = self.lenet(z_H, z_L, **seq_info)
            
            # Save carry for next iteration
            carry_z_H = z_H.detach()
            carry_z_L = z_L.detach()
            
            # Global pool
            board_repr = z_H[:, self.puzzle_emb_len:]
            global_repr = board_repr.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.9 * final_acc + 0.1 * acc
            
            if i % 100 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={acc:.2%}, ema_acc={final_acc:.2%}")
        
        print(f"Ablation 5 Final EMA: {final_acc:.2%}")
        return final_acc > 0.8
    
    def test_ablation_6_full_grad_varying(self):
        """Test: Full gradients (no no_grad block), varying batches, fresh init"""
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        print("Ablation 6: FULL gradients, varying batches, fresh init...")
        
        final_acc = 0
        for i in range(500):
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
            }
            
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            # Initialize from input (position-specific)
            z_H = input_emb
            z_L = input_emb
            
            # ALL cycles WITH grad (no no_grad block!)
            for _ in range(self.hparams.H_cycles):
                for _ in range(self.hparams.L_cycles):
                    z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
            
            # Global pool
            board_repr = z_H[:, self.puzzle_emb_len:]
            global_repr = board_repr.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.9 * final_acc + 0.1 * acc
            
            if i % 100 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={acc:.2%}, ema_acc={final_acc:.2%}")
        
        print(f"Ablation 6 Final EMA: {final_acc:.2%}")
        return final_acc > 0.8


    def test_ablation_7_no_grad_input_init(self):
        """Test: no_grad block BUT init from input_emb (not fixed init)"""
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        print("Ablation 7: no_grad block, but init from INPUT_EMB...")
        
        final_acc = 0
        for i in range(500):
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
            }
            
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            # Initialize from input (NOT fixed init)
            z_H = input_emb
            z_L = input_emb
            
            # no_grad for H_cycles - 1
            with torch.no_grad():
                for _ in range(self.hparams.H_cycles - 1):
                    for _ in range(self.hparams.L_cycles):
                        z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                    z_H = self.lenet(z_H, z_L, **seq_info)
            
            # Last cycle with grad
            for _ in range(self.hparams.L_cycles):
                z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
            z_H = self.lenet(z_H, z_L, **seq_info)
            
            # Global pool
            board_repr = z_H[:, self.puzzle_emb_len:]
            global_repr = board_repr.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.9 * final_acc + 0.1 * acc
            
            if i % 100 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={acc:.2%}, ema_acc={final_acc:.2%}")
        
        print(f"Ablation 7 Final EMA: {final_acc:.2%}")
        return final_acc > 0.8
    
    def test_ablation_8_match_cnn_training(self):
        """Test: Match CNN's training setup exactly"""
        
        # Lower LR like CNN baseline tends to use, more steps
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        
        print("Ablation 8: Lower LR (1e-4), more steps (2000)...")
        
        final_acc = 0
        for i in range(2000):
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
            }
            
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            z_H = input_emb
            z_L = input_emb
            
            # Full gradients
            for _ in range(self.hparams.H_cycles):
                for _ in range(self.hparams.L_cycles):
                    z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
            
            board_repr = z_H[:, self.puzzle_emb_len:]
            global_repr = board_repr.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.95 * final_acc + 0.05 * acc
            
            if i % 200 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={acc:.2%}, ema_acc={final_acc:.2%}")
        
        print(f"Ablation 8 Final EMA: {final_acc:.2%}")
        return final_acc > 0.75


    def test_ablation_9_single_layer_no_cycles(self):
        """Test: Simplest possible TRM - single pass, no cycles"""
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        
        print("Ablation 9: Single forward pass (no H/L cycles)...")
        
        final_acc = 0
        for i in range(2000):
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
            }
            
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            # Just ONE forward pass through lenet (like a standard transformer)
            z = self.lenet(input_emb, input_emb, **seq_info)
            
            board_repr = z[:, self.puzzle_emb_len:]
            global_repr = board_repr.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.95 * final_acc + 0.05 * acc
            
            if i % 200 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={acc:.2%}, ema_acc={final_acc:.2%}")
        
        print(f"Ablation 9 Final EMA: {final_acc:.2%}")
        return final_acc > 0.75

    def test_ablation_10_no_puzzle_emb(self):
        """Test: Remove puzzle embeddings entirely"""
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        
        print("Ablation 10: No puzzle embeddings...")
        
        final_acc = 0
        for i in range(2000):
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
            }
            
            # NO RoPE - skip positional embeddings for this test
            seq_info = dict(cos_sin=None)
            
            # SKIP puzzle embeddings - just use board tokens
            boards_canonical = self._canonicalize_board(batch["boards"], batch["current_player"])
            input_emb = self.embed_scale * self.input_embedding(boards_canonical.to(torch.int32))
            # Shape: [batch, 42, hidden] - no puzzle prefix!
            
            z_H = input_emb
            z_L = input_emb
            
            for _ in range(self.hparams.H_cycles):
                for _ in range(self.hparams.L_cycles):
                    z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
            
            # Global pool over ALL positions (no prefix to skip)
            global_repr = z_H.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.95 * final_acc + 0.05 * acc
            
            if i % 200 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={acc:.2%}, ema_acc={final_acc:.2%}")
        
        print(f"Ablation 10 Final EMA: {final_acc:.2%}")
        return final_acc > 0.75


    def test_ablation_11_with_dropout(self):
        """Test: Add dropout like MLP/CNN"""
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        dropout = nn.Dropout(0.1).to(self.device)
        
        print("Ablation 11: With dropout (0.1)...")
        
        final_acc = 0
        for i in range(2000):
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
            }
            
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            z_H = input_emb
            z_L = input_emb
            
            for _ in range(self.hparams.H_cycles):
                for _ in range(self.hparams.L_cycles):
                    z_L = dropout(self.lenet(z_L, z_H + input_emb, **seq_info))
                z_H = dropout(self.lenet(z_H, z_L, **seq_info))
            
            board_repr = z_H[:, self.puzzle_emb_len:]
            global_repr = dropout(board_repr.mean(dim=1))
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.95 * final_acc + 0.05 * acc
            
            if i % 200 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={acc:.2%}, ema_acc={final_acc:.2%}")
        
        print(f"Ablation 11 Final EMA: {final_acc:.2%}")
        return final_acc > 0.75


    def test_ablation_12_raw_linear_embed(self):
        """Test: Use linear projection of raw board instead of token embeddings"""
        
        # Create a simple linear projection - each position gets its own embedding from raw value
        raw_proj = nn.Linear(1, self.hparams.hidden_size).to(self.device)
        
        opt = torch.optim.AdamW(
            list(self.parameters()) + list(raw_proj.parameters()), 
            lr=1e-4, weight_decay=0.01
        )
        
        print("Ablation 12: Linear projection of raw board values...")
        
        final_acc = 0
        for i in range(2000):
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
            }
            
            # No RoPE for simplicity
            seq_info = dict(cos_sin=None)
            
            # Use raw board values with per-position linear projection
            boards_canonical = self._canonicalize_board(
                batch["boards"].float(), batch["current_player"]
            )
            # [batch, 42] -> [batch, 42, 1] -> [batch, 42, hidden]
            input_emb = raw_proj(boards_canonical.unsqueeze(-1))
            
            z_H = input_emb
            z_L = input_emb
            
            for _ in range(self.hparams.H_cycles):
                for _ in range(self.hparams.L_cycles):
                    z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
            
            global_repr = z_H.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.parameters()) + list(raw_proj.parameters()), 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.95 * final_acc + 0.05 * acc
            
            if i % 200 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={acc:.2%}, ema_acc={final_acc:.2%}")
        
        print(f"Ablation 12 Final EMA: {final_acc:.2%}")
        return final_acc > 0.75


    def test_ablation_13_mlp_equivalent(self):
        """Test: Bypass transformer, use MLP on token embeddings (like MLP baseline)"""
        
        # Create MLP that matches the baseline
        mlp = nn.Sequential(
            nn.Linear(42 * self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        ).to(self.device)
        
        policy_head = nn.Linear(self.hparams.hidden_size, 7).to(self.device)
        value_head = nn.Linear(self.hparams.hidden_size, 1).to(self.device)
        
        params = list(self.input_embedding.parameters()) + list(mlp.parameters()) + \
                list(policy_head.parameters()) + list(value_head.parameters())
        opt = torch.optim.AdamW(params, lr=self.hparams.learning_rate, weight_decay=0.01)
        
        print("Ablation 13: MLP on token embeddings (bypass transformer)...")
        
        final_acc = 0
        for i in range(2000):
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
            }
            
            # Get token embeddings (same as TRM)
            boards_canonical = self._canonicalize_board(batch["boards"], batch["current_player"])
            emb = self.embed_scale * self.input_embedding(boards_canonical.to(torch.int32))
            # [batch, 42, hidden]
            
            # Flatten and run through MLP (like MLP baseline)
            x = mlp(emb.flatten(1))  # [batch, 42*hidden] -> [batch, hidden]
            
            policy_logits = policy_head(x)
            value = torch.tanh(value_head(x).squeeze(-1))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value, batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.95 * final_acc + 0.05 * acc
            
            if i % 200 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={acc:.2%}, ema_acc={final_acc:.2%}")
        
        print(f"Ablation 13 Final EMA: {final_acc:.2%}")
        return final_acc > 0.75


    def test_ablation_14_simple_transformer(self):
        """Test: Standard transformer - no H/L cycles, just stack layers once"""
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        
        print("Ablation 14: Simple transformer (single pass through all layers)...")
        
        final_acc = 0
        for i in range(2000):
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
            }
            
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            # Standard transformer: single pass through lenet (which handles all layers)
            z = self.lenet(input_emb, input_emb, **seq_info)
            
            board_repr = z[:, self.puzzle_emb_len:]
            global_repr = board_repr.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.95 * final_acc + 0.05 * acc
            
            if i % 200 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={acc:.2%}, ema_acc={final_acc:.2%}")
        
        print(f"Ablation 14 Final EMA: {final_acc:.2%}")
        return final_acc > 0.75


    def test_ablation_15_exact_mlp_baseline(self):
        """Test: Exact MLP baseline architecture (sanity check that data is correct)"""
        
        # Recreate MLP baseline exactly
        mlp = nn.Sequential(
            nn.Linear(42, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        ).to(self.device)
        
        policy_head = nn.Linear(self.hparams.hidden_size, 7).to(self.device)
        value_head = nn.Linear(self.hparams.hidden_size, 1).to(self.device)
        
        params = list(mlp.parameters()) + list(policy_head.parameters()) + list(value_head.parameters())
        opt = torch.optim.AdamW(params, lr=self.hparams.learning_rate, weight_decay=0.01)
        
        print("Ablation 15: Exact MLP baseline (sanity check)...")
        
        final_acc = 0
        for i in range(4000):
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
            }
            
            # Exact MLP baseline processing
            boards_canonical = self._canonicalize_board(batch["boards"].float(), batch["current_player"])
            x = mlp(boards_canonical)
            
            policy_logits = policy_head(x)
            value = torch.tanh(value_head(x).squeeze(-1))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value, batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.95 * final_acc + 0.05 * acc
            
            if i % 200 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, policy_acc={acc:.2%}, ema_acc={final_acc:.2%}")
        
        print(f"Ablation 15 Final EMA: {final_acc:.2%}")
        return final_acc > 0.75


    def test_ablation_16_attention_only_no_mlp(self):
        """Test: Check if attention itself is learning - remove MLP from transformer"""
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        
        print("Ablation 16: Attention diagnostics - checking attention patterns...")
        
        final_acc = 0
        for i in range(2000):
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
            }
            
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            z_H = input_emb
            z_L = input_emb
            
            # Full H/L cycles with gradient
            for _ in range(self.hparams.H_cycles):
                for _ in range(self.hparams.L_cycles):
                    z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
            
            board_repr = z_H[:, self.puzzle_emb_len:]
            global_repr = board_repr.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.95 * final_acc + 0.05 * acc
            
            # Log embedding stats periodically
            if i % 200 == 0:
                with torch.no_grad():
                    emb_std = input_emb.std().item()
                    z_H_std = z_H.std().item()
                    print(f"Step {i}: loss={loss.item():.4f}, acc={acc:.2%}, ema={final_acc:.2%}, emb_std={emb_std:.3f}, z_H_std={z_H_std:.3f}")
        
        print(f"Ablation 16 Final EMA: {final_acc:.2%}")
        return final_acc > 0.75

    def test_ablation_17_mlp_with_carry(self):
        """Test: MLP model but using the CARRY structure for data flow"""
        
        # Simple MLP (we know this works standalone)
        mlp = nn.Sequential(
            nn.Linear(42, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        ).to(self.device)
        
        policy_head = nn.Linear(self.hparams.hidden_size, 7).to(self.device)
        value_head = nn.Linear(self.hparams.hidden_size, 1).to(self.device)
        
        params = list(mlp.parameters()) + list(policy_head.parameters()) + list(value_head.parameters())
        opt = torch.optim.AdamW(params, lr=self.hparams.learning_rate, weight_decay=0.01)
        
        print("Ablation 17: MLP with CARRY structure (isolate carry bugs)...")
        
        # Initialize carry like TRM does
        carry = None
        
        final_acc = 0
        for i in range(2000):
            # Sample batch (like TRM training_step)
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
            }
            
            # Initialize carry on first batch (like TRM)
            if carry is None:
                carry = {
                    'halted': torch.ones(self.hparams.batch_size, dtype=torch.bool, device=self.device),
                    'steps': torch.zeros(self.hparams.batch_size, dtype=torch.int32, device=self.device),
                    'current_data': {k: torch.empty_like(v) for k, v in batch.items()},
                }
            
            # === CARRY UPDATE LOGIC (copied from TRM forward) ===
            # Reset steps for halted sequences
            new_steps = torch.where(carry['halted'], 0, carry['steps'])
            
            # Update current_data: use new batch data for halted sequences, keep old for non-halted
            new_current_data = {
                k: torch.where(
                    carry['halted'].view((-1,) + (1,) * (batch[k].ndim - 1)), 
                    batch[k], 
                    carry['current_data'][k]
                )
                for k in batch.keys()
            }
            
            # === MLP FORWARD (using new_current_data, not batch!) ===
            boards_canonical = self._canonicalize_board(
                new_current_data["boards"].float(), 
                new_current_data["current_player"]
            )
            x = mlp(boards_canonical)
            
            policy_logits = policy_head(x)
            value = torch.tanh(value_head(x).squeeze(-1))
            
            boards = new_current_data["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            # === HALTING LOGIC (copied from TRM) ===
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.hparams.N_supervision
            halted = is_last_step  # Simplified - no q_halt for MLP
            
            # === LOSS COMPUTATION (using new_current_data) ===
            target_policy = new_current_data["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            target_values = new_current_data["values"]
            
            # Only compute loss for halted sequences (like TRM)
            valid_metrics = halted
            
            if valid_metrics.any():
                policy_loss = -torch.sum(
                    target_policy[valid_metrics] * torch.log(policy[valid_metrics] + 1e-8), 
                    dim=-1
                ).mean()
                value_loss = F.mse_loss(value[valid_metrics], target_values[valid_metrics])
                loss = policy_loss + 0.5 * value_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                
                with torch.no_grad():
                    acc = (policy[valid_metrics].argmax(-1) == target_policy[valid_metrics].argmax(-1)).float().mean().item()
                    final_acc = 0.95 * final_acc + 0.05 * acc
            
            # === UPDATE CARRY FOR NEXT ITERATION ===
            carry = {
                'halted': halted,
                'steps': new_steps,
                'current_data': {k: v.detach() for k, v in new_current_data.items()},
            }
            
            if i % 200 == 0:
                halted_count = halted.sum().item()
                print(f"Step {i}: loss={loss.item() if valid_metrics.any() else 0:.4f}, "
                    f"acc={acc if valid_metrics.any() else 0:.2%}, ema={final_acc:.2%}, "
                    f"halted={halted_count}/{self.hparams.batch_size}")
        
        print(f"Ablation 17 Final EMA: {final_acc:.2%}")
        return final_acc > 0.75


    def test_ablation_18_mlp_carry_always_halt(self):
        """Test: MLP with carry, but ALWAYS halt (N_supervision=1)"""
        
        mlp = nn.Sequential(
            nn.Linear(42, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        ).to(self.device)
        
        policy_head = nn.Linear(self.hparams.hidden_size, 7).to(self.device)
        value_head = nn.Linear(self.hparams.hidden_size, 1).to(self.device)
        
        params = list(mlp.parameters()) + list(policy_head.parameters()) + list(value_head.parameters())
        opt = torch.optim.AdamW(params, lr=self.hparams.learning_rate, weight_decay=0.01)
        
        print("Ablation 18: MLP with carry, N_supervision=1 (always halt)...")
        
        carry = None
        final_acc = 0
        
        for i in range(2000):
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
            }
            
            if carry is None:
                carry = {
                    'halted': torch.ones(self.hparams.batch_size, dtype=torch.bool, device=self.device),
                    'steps': torch.zeros(self.hparams.batch_size, dtype=torch.int32, device=self.device),
                    'current_data': {k: torch.empty_like(v) for k, v in batch.items()},
                }
            
            # Carry update
            new_steps = torch.where(carry['halted'], 0, carry['steps'])
            new_current_data = {
                k: torch.where(
                    carry['halted'].view((-1,) + (1,) * (batch[k].ndim - 1)), 
                    batch[k], 
                    carry['current_data'][k]
                )
                for k in batch.keys()
            }
            
            # MLP forward
            boards_canonical = self._canonicalize_board(
                new_current_data["boards"].float(), 
                new_current_data["current_player"]
            )
            x = mlp(boards_canonical)
            
            policy_logits = policy_head(x)
            value = torch.tanh(value_head(x).squeeze(-1))
            
            boards = new_current_data["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == C4_EMPTY_CELL
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            # ALWAYS halt (N_supervision = 1)
            new_steps = new_steps + 1
            halted = torch.ones_like(new_steps, dtype=torch.bool)  # Always halt!
            
            # Loss (all samples)
            target_policy = new_current_data["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            target_values = new_current_data["values"]
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value, target_values)
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.95 * final_acc + 0.05 * acc
            
            carry = {
                'halted': halted,
                'steps': new_steps,
                'current_data': {k: v.detach() for k, v in new_current_data.items()},
            }
            
            if i % 200 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, acc={acc:.2%}, ema={final_acc:.2%}")
        
        print(f"Ablation 18 Final EMA: {final_acc:.2%}")
        return final_acc > 0.75


    def test_ablation_19_debug_carry_data_flow(self):
        """Test: Print debug info about carry data flow"""
        
        print("Ablation 19: Debug carry data flow...")
        
        carry = None
        
        for i in range(5):  # Just 5 iterations to debug
            samples = self.replay_buffer.sample(self.hparams.batch_size)
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
            }
            
            print(f"\n=== Iteration {i} ===")
            print(f"Batch boards[0,:5]: {batch['boards'][0,:5].tolist()}")
            print(f"Batch policies[0]: {batch['policies'][0].tolist()}")
            
            if carry is None:
                carry = {
                    'halted': torch.ones(self.hparams.batch_size, dtype=torch.bool, device=self.device),
                    'steps': torch.zeros(self.hparams.batch_size, dtype=torch.int32, device=self.device),
                    'current_data': {k: torch.empty_like(v) for k, v in batch.items()},
                }
                print(f"Initialized carry (all halted)")
            
            print(f"Before update - halted[0]: {carry['halted'][0].item()}, steps[0]: {carry['steps'][0].item()}")
            print(f"Before update - current_data boards[0,:5]: {carry['current_data']['boards'][0,:5].tolist()}")
            
            # Carry update
            new_steps = torch.where(carry['halted'], 0, carry['steps'])
            new_current_data = {
                k: torch.where(
                    carry['halted'].view((-1,) + (1,) * (batch[k].ndim - 1)), 
                    batch[k], 
                    carry['current_data'][k]
                )
                for k in batch.keys()
            }
            
            print(f"After update - new_current_data boards[0,:5]: {new_current_data['boards'][0,:5].tolist()}")
            print(f"Boards match batch? {torch.allclose(new_current_data['boards'][0], batch['boards'][0])}")
            
            # Simulate halting (N_supervision=2 for testing)
            new_steps = new_steps + 1
            halted = new_steps >= 2  # Halt after 2 steps
            
            print(f"After step - new_steps[0]: {new_steps[0].item()}, halted[0]: {halted[0].item()}")
            
            carry = {
                'halted': halted,
                'steps': new_steps,
                'current_data': {k: v.detach().clone() for k, v in new_current_data.items()},
            }
        
        print("\n=== Debug complete ===")
        return True  # Always pass, this is just for debugging


    def test_ablation_25_exact_dataloader_path(self):
        """Test: Use exact same data path as real training"""
        
        print("Ablation 25: Using exact dataloader path...")
        
        # Check buffer stats
        print(f"Buffer size: {len(self.replay_buffer)}")
        
        # Sample using EXACT same method as SimpleReplayDataset
        buffer = self.replay_buffer
        n = self.hparams.batch_size
        
        indices = torch.randint(0, len(buffer), (n,))
        samples = [buffer[i] for i in indices]
        
        batch = {
            'boards': torch.stack([s['board'] for s in samples]).to(self.device),
            'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
            'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
            'current_player': torch.tensor([s['current_player'] for s in samples], dtype=torch.long).to(self.device),
            'puzzle_identifiers': torch.zeros(n, dtype=torch.long, device=self.device),
        }
        
        print(f"Sample batch stats:")
        print(f"  boards unique values: {batch['boards'].unique().tolist()}")
        print(f"  policies sum: {batch['policies'].sum(dim=1).mean():.3f}")
        print(f"  values range: [{batch['values'].min():.2f}, {batch['values'].max():.2f}]")
        print(f"  current_player unique: {batch['current_player'].unique().tolist()}")
        
        # Now compare with replay_buffer.sample() method
        print("\nComparing buffer[i] vs buffer.sample():")
        
        # Method 1: buffer[i]
        indices1 = torch.randint(0, len(buffer), (10,))
        samples1 = [buffer[i] for i in indices1]
        
        # Method 2: buffer.sample() - if it exists
        if hasattr(buffer, 'sample'):
            samples2 = buffer.sample(10)
            print(f"  buffer.sample() exists")
            print(f"  samples1[0] keys: {samples1[0].keys()}")
            print(f"  samples2[0] keys: {samples2[0].keys()}")
            
            # Check if they have same structure
            for key in samples1[0].keys():
                v1 = samples1[0][key]
                v2 = samples2[0][key]
                print(f"  {key}: type1={type(v1)}, type2={type(v2)}")
        else:
            print(f"  buffer.sample() does NOT exist - need to use indexing")
        
        return True


    def test_ablation_26_mlp_with_dataloader_sampling(self):
        """Test: MLP using exact same sampling as SimpleReplayDataset"""
        
        mlp = nn.Sequential(
            nn.Linear(42, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        ).to(self.device)
        
        policy_head = nn.Linear(self.hparams.hidden_size, 7).to(self.device)
        value_head = nn.Linear(self.hparams.hidden_size, 1).to(self.device)
        
        params = list(mlp.parameters()) + list(policy_head.parameters()) + list(value_head.parameters())
        opt = torch.optim.AdamW(params, lr=self.hparams.learning_rate, weight_decay=0.01)
        
        buffer = self.replay_buffer
        n = self.hparams.batch_size
        
        print(f"Ablation 26: MLP with dataloader-style sampling (buffer size: {len(buffer)})...")
        
        final_acc = 0
        
        for i in range(4000):
            # EXACT same sampling as SimpleReplayDataset
            indices = torch.randint(0, len(buffer), (n,))
            samples = [buffer[idx] for idx in indices]
            
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples], dtype=torch.long).to(self.device),
            }
            
            boards_canonical = self._canonicalize_board(batch["boards"].float(), batch["current_player"])
            x = mlp(boards_canonical)
            
            policy_logits = policy_head(x)
            value = torch.tanh(value_head(x).squeeze(-1))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == 0  # C4_EMPTY_CELL = 0
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value, batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.95 * final_acc + 0.05 * acc
            
            if i % 200 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, acc={acc:.2%}, ema={final_acc:.2%}")
        
        print(f"Ablation 26 Final EMA: {final_acc:.2%}")
        return final_acc > 0.75


    def test_ablation_27_trm_with_dataloader_sampling(self):
        """Test: TRM using exact same sampling, NO carry, full gradients"""
        
        buffer = self.replay_buffer
        n = self.hparams.batch_size
        
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        
        print(f"Ablation 27: TRM with dataloader-style sampling (buffer size: {len(buffer)})...")
        
        final_acc = 0
        
        for i in range(2000):
            # EXACT same sampling as SimpleReplayDataset
            indices = torch.randint(0, len(buffer), (n,))
            samples = [buffer[idx] for idx in indices]
            
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples], dtype=torch.long).to(self.device),
                'puzzle_identifiers': torch.zeros(n, dtype=torch.long, device=self.device),
            }
            
            # TRM forward - no carry, full gradients, global pooling
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            z_H = input_emb
            z_L = input_emb
            
            for _ in range(self.hparams.H_cycles):
                for _ in range(self.hparams.L_cycles):
                    z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
            
            board_repr = z_H[:, self.puzzle_emb_len:]
            global_repr = board_repr.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == 0
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.95 * final_acc + 0.05 * acc
            
            if i % 200 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, acc={acc:.2%}, ema={final_acc:.2%}")
        
        print(f"Ablation 27 Final EMA: {final_acc:.2%}")
        return final_acc > 0.75

    def test_ablation_29_hidden_size_vs_lr(self):
        """Test: Does hidden_size=512 work with lower LR?"""
        
        print(f"Ablation 29: Testing hidden_size={self.hparams.hidden_size} with various LRs...")
        print(f"Current hidden_size: {self.hparams.hidden_size}")
        
        # Fixed batch for overfitting
        samples = self.replay_buffer.sample(self.hparams.batch_size)
        batch = {
            'boards': torch.stack([s['board'] for s in samples]).to(self.device),
            'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
            'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
            'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
            'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device),
        }
        
        learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
        
        for lr in learning_rates:
            # Reset parameters
            self._reset_parameters()
            opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
            
            print(f"\n  Testing LR={lr}:")
            
            final_acc = 0
            for i in range(500):
                seq_info = dict(cos_sin=self.pos_embedding())
                input_emb = self._input_embeddings(
                    batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
                )
                
                z_H = self.z_H_init.unsqueeze(0).unsqueeze(0).expand(
                    self.hparams.batch_size, input_emb.shape[1], -1
                ).clone()
                z_L = self.z_L_init.unsqueeze(0).unsqueeze(0).expand(
                    self.hparams.batch_size, input_emb.shape[1], -1
                ).clone()
                
                # no_grad block
                with torch.no_grad():
                    for _ in range(self.hparams.H_cycles - 1):
                        for _ in range(self.hparams.L_cycles):
                            z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                        z_H = self.lenet(z_H, z_L, **seq_info)
                
                for _ in range(self.hparams.L_cycles):
                    z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
                
                board_repr = z_H[:, self.puzzle_emb_len:]
                global_repr = board_repr.mean(dim=1)
                
                policy_logits = self.lm_head(global_repr)
                value = torch.tanh(self.value_head(global_repr))
                
                boards = batch["boards"].view(-1, 6, 7)
                valid_actions = boards[:, 0, :] == 0
                policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
                policy = F.softmax(policy_logits, dim=-1)
                
                target_policy = batch["policies"] * valid_actions.float()
                target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                
                policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
                value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
                loss = policy_loss + 0.5 * value_loss
                
                loss.backward()
                
                # Check gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float('inf')).item()
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                
                with torch.no_grad():
                    acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                    final_acc = acc
                
                if i == 0:
                    print(f"    Step 0: loss={loss.item():.4f}, acc={acc:.2%}, grad_norm={grad_norm:.2f}")
                elif i == 499:
                    print(f"    Step 499: loss={loss.item():.4f}, acc={acc:.2%}, grad_norm={grad_norm:.2f}")
            
            print(f"    Final: {final_acc:.2%}")
        
        return True


    def test_ablation_30_mlp_hidden_512_overfit(self):
        """Test: Can MLP with hidden_size=512 overfit a fixed batch?"""
        
        print("Ablation 30: MLP hidden_size=512 overfitting test...")
        
        # Fixed batch
        samples = self.replay_buffer.sample(self.hparams.batch_size)
        batch = {
            'boards': torch.stack([s['board'] for s in samples]).to(self.device),
            'current_player': torch.tensor([s['current_player'] for s in samples]).to(self.device),
            'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
            'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
        }
        
        # MLP with hidden_size=512
        mlp = nn.Sequential(
            nn.Linear(42, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        ).to(self.device)
        
        policy_head = nn.Linear(512, 7).to(self.device)
        value_head = nn.Linear(512, 1).to(self.device)
        
        params = list(mlp.parameters()) + list(policy_head.parameters()) + list(value_head.parameters())
        
        learning_rates = [1e-2, 1e-3, 1e-4]
        
        for lr in learning_rates:
            # Reset
            for layer in mlp:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            policy_head.reset_parameters()
            value_head.reset_parameters()
            
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
            
            print(f"\n  Testing LR={lr}:")
            
            for i in range(500):
                mlp.train()
                boards_canonical = self._canonicalize_board(batch["boards"].float(), batch["current_player"])
                x = mlp(boards_canonical)
                
                policy_logits = policy_head(x)
                value = torch.tanh(value_head(x).squeeze(-1))
                
                boards = batch["boards"].view(-1, 6, 7)
                valid_actions = boards[:, 0, :] == 0
                policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
                policy = F.softmax(policy_logits, dim=-1)
                
                target_policy = batch["policies"] * valid_actions.float()
                target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                
                policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
                value_loss = F.mse_loss(value, batch["values"])
                loss = policy_loss + 0.5 * value_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                
                if i % 100 == 0:
                    with torch.no_grad():
                        acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                        print(f"    Step {i}: loss={loss.item():.4f}, acc={acc:.2%}")
            
            with torch.no_grad():
                mlp.eval()
                x = mlp(boards_canonical)
                policy_logits = policy_head(x)
                policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
                policy = F.softmax(policy_logits, dim=-1)
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                print(f"    Final: {acc:.2%}")
        
        return True


    def test_ablation_31_compare_param_counts(self):
        """Compare parameter counts between models"""
        
        print("Ablation 31: Parameter count comparison...")
        
        # TRM params
        trm_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"TRM (hidden_size={self.hparams.hidden_size}): {trm_params:,} params ({trm_params/1e6:.2f}M)")
        
        # Component breakdown
        print(f"  input_embedding: {sum(p.numel() for p in self.input_embedding.parameters()):,}")
        print(f"  lenet: {sum(p.numel() for p in self.lenet.parameters()):,}")
        print(f"  lm_head: {sum(p.numel() for p in self.lm_head.parameters()):,}")
        print(f"  value_head: {sum(p.numel() for p in self.value_head.parameters()):,}")
        print(f"  q_head: {sum(p.numel() for p in self.q_head.parameters()):,}")
        
        # MLP for comparison
        for hs in [64, 512]:
            mlp = nn.Sequential(
                nn.Linear(42, hs),
                nn.ReLU(),
                nn.Linear(hs, hs),
                nn.ReLU(),
                nn.Linear(hs, hs),
                nn.ReLU(),
                nn.Linear(hs, hs),
                nn.ReLU(),
            )
            policy_head = nn.Linear(hs, 7)
            value_head = nn.Linear(hs, 1)
            
            mlp_params = sum(p.numel() for p in mlp.parameters()) + \
                        sum(p.numel() for p in policy_head.parameters()) + \
                        sum(p.numel() for p in value_head.parameters())
            print(f"MLP (hidden_size={hs}): {mlp_params:,} params ({mlp_params/1e6:.2f}M)")
        
        return True

    def test_ablation_32_trm_with_dropout(self):
        """Test: TRM with dropout to reduce overfitting"""
        
        buffer = self.replay_buffer
        n = self.hparams.batch_size
        dropout = nn.Dropout(0.1).to(self.device)
        
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        
        print(f"Ablation 32: TRM with dropout...")
        
        final_acc = 0
        
        for i in range(4000):
            indices = torch.randint(0, len(buffer), (n,))
            samples = [buffer[idx] for idx in indices]
            
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples], dtype=torch.long).to(self.device),
                'puzzle_identifiers': torch.zeros(n, dtype=torch.long, device=self.device),
            }
            
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            z_H = input_emb
            z_L = input_emb
            
            for _ in range(self.hparams.H_cycles):
                for _ in range(self.hparams.L_cycles):
                    z_L = dropout(self.lenet(z_L, z_H + input_emb, **seq_info))
                z_H = dropout(self.lenet(z_H, z_L, **seq_info))
            
            board_repr = z_H[:, self.puzzle_emb_len:]
            global_repr = dropout(board_repr.mean(dim=1))
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == 0
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.95 * final_acc + 0.05 * acc
            
            if i % 400 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, acc={acc:.2%}, ema={final_acc:.2%}")
        
        print(f"Ablation 32 Final EMA: {final_acc:.2%}")
        return final_acc > 0.75


    def test_ablation_33_trm_fewer_cycles(self):
        """Test: TRM with fewer H/L cycles (simpler model)"""
        
        buffer = self.replay_buffer
        n = self.hparams.batch_size
        
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Use only 1 H-cycle, 2 L-cycles
        h_cycles = 1
        l_cycles = 2
        
        print(f"Ablation 33: TRM with H={h_cycles}, L={l_cycles} (simpler)...")
        
        final_acc = 0
        
        for i in range(4000):
            indices = torch.randint(0, len(buffer), (n,))
            samples = [buffer[idx] for idx in indices]
            
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples], dtype=torch.long).to(self.device),
                'puzzle_identifiers': torch.zeros(n, dtype=torch.long, device=self.device),
            }
            
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            z_H = input_emb
            z_L = input_emb
            
            for _ in range(h_cycles):
                for _ in range(l_cycles):
                    z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
            
            board_repr = z_H[:, self.puzzle_emb_len:]
            global_repr = board_repr.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == 0
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.95 * final_acc + 0.05 * acc
            
            if i % 400 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, acc={acc:.2%}, ema={final_acc:.2%}")
        
        print(f"Ablation 33 Final EMA: {final_acc:.2%}")
        return final_acc > 0.75


    def test_ablation_34_trm_longer_training(self):
        """Test: TRM with more training steps"""
        
        buffer = self.replay_buffer
        n = self.hparams.batch_size
        
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        
        print(f"Ablation 34: TRM with 10000 steps...")
        
        final_acc = 0
        best_acc = 0
        
        for i in range(10000):
            indices = torch.randint(0, len(buffer), (n,))
            samples = [buffer[idx] for idx in indices]
            
            batch = {
                'boards': torch.stack([s['board'] for s in samples]).to(self.device),
                'policies': torch.stack([s['policy'] for s in samples]).to(self.device),
                'values': torch.tensor([s['value'] for s in samples], dtype=torch.float32).to(self.device),
                'current_player': torch.tensor([s['current_player'] for s in samples], dtype=torch.long).to(self.device),
                'puzzle_identifiers': torch.zeros(n, dtype=torch.long, device=self.device),
            }
            
            seq_info = dict(cos_sin=self.pos_embedding())
            input_emb = self._input_embeddings(
                batch["boards"], batch["puzzle_identifiers"], batch["current_player"]
            )
            
            z_H = input_emb
            z_L = input_emb
            
            for _ in range(self.hparams.H_cycles):
                for _ in range(self.hparams.L_cycles):
                    z_L = self.lenet(z_L, z_H + input_emb, **seq_info)
                z_H = self.lenet(z_H, z_L, **seq_info)
            
            board_repr = z_H[:, self.puzzle_emb_len:]
            global_repr = board_repr.mean(dim=1)
            
            policy_logits = self.lm_head(global_repr)
            value = torch.tanh(self.value_head(global_repr))
            
            boards = batch["boards"].view(-1, 6, 7)
            valid_actions = boards[:, 0, :] == 0
            policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
            policy = F.softmax(policy_logits, dim=-1)
            
            target_policy = batch["policies"] * valid_actions.float()
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            
            policy_loss = -torch.sum(target_policy * torch.log(policy + 1e-8), dim=-1).mean()
            value_loss = F.mse_loss(value.squeeze(-1), batch["values"])
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                acc = (policy.argmax(-1) == target_policy.argmax(-1)).float().mean().item()
                final_acc = 0.99 * final_acc + 0.01 * acc
                best_acc = max(best_acc, acc)
            
            if i % 1000 == 0:
                print(f"Step {i}: loss={loss.item():.4f}, acc={acc:.2%}, ema={final_acc:.2%}, best={best_acc:.2%}")
        
        print(f"Ablation 34 Final EMA: {final_acc:.2%}, Best: {best_acc:.2%}")
        return final_acc > 0.75

    def run_all_ablations(self):
        print("=" * 60)
        print("Running ablation study")
        print("=" * 60)
        
        results = {}
        
        # Reset model before each test
        # self._reset_parameters()
        # results['no_carry'] = self.test_trm_no_carry()
        
        # self._reset_parameters()
        # results['ablation_1_no_grad'] = self.test_ablation_1_no_grad_block()
        
        # self._reset_parameters()
        # results['ablation_2_fixed_init'] = self.test_ablation_2_fixed_init()
        
        self._reset_parameters()
        results['ablation_3_both'] = self.test_ablation_3_both()
        
        # self._reset_parameters()
        # results['ablation_4_varying_batches'] = self.test_ablation_4_varying_batches()

        # self._reset_parameters()
        # results['ablation_5_with_carry'] = self.test_ablation_5_with_carry()
        
        # self._reset_parameters()
        # results['ablation_6_full_grad_varying'] = self.test_ablation_6_full_grad_varying()

        # self._reset_parameters()
        # results['ablation_7_no_grad_input_init'] = self.test_ablation_7_no_grad_input_init()

        # self._reset_parameters()
        # results['ablation_8_match_cnn_training'] = self.test_ablation_8_match_cnn_training()

        # self._reset_parameters()
        # results['ablation_9_single_layer_no_cycles'] = self.test_ablation_9_single_layer_no_cycles()

        # self._reset_parameters()
        # results['ablation_10_no_puzzle_emb'] = self.test_ablation_10_no_puzzle_emb()

        # self._reset_parameters()
        # results['ablation_11_with_dropout'] = self.test_ablation_11_with_dropout()

        # self._reset_parameters()
        # results['ablation_12_raw_linear_embed'] = self.test_ablation_12_raw_linear_embed()

        # self._reset_parameters()
        # results['ablation_13_mlp_equivalent'] = self.test_ablation_13_mlp_equivalent()

        # self._reset_parameters()
        # results['ablation_14_simple_transformer'] = self.test_ablation_14_simple_transformer()

        self._reset_parameters()
        results['ablation_15_exact_mlp_baseline'] = self.test_ablation_15_exact_mlp_baseline()

        # self._reset_parameters()
        # results['ablation_16_attention_only_no_mlp'] = self.test_ablation_16_attention_only_no_mlp()

        # self._reset_parameters()
        # results['ablation_17_mlp_with_carry'] = self.test_ablation_17_mlp_with_carry()

        # self._reset_parameters()
        # results['ablation_18_mlp_carry_always_halt'] = self.test_ablation_18_mlp_carry_always_halt()

        # self._reset_parameters()
        # results['ablation_19_debug_carry_data_flow'] = self.test_ablation_19_debug_carry_data_flow()

        # self._reset_parameters()
        # results['ablation_25_exact_dataloader_path'] = self.test_ablation_25_exact_dataloader_path()

        self._reset_parameters()
        results['ablation_26_mlp_with_dataloader_sampling'] = self.test_ablation_26_mlp_with_dataloader_sampling()

        self._reset_parameters()
        results['ablation_27_trm_with_dataloader_sampling'] = self.test_ablation_27_trm_with_dataloader_sampling()

        self._reset_parameters()
        results['ablation_29_hidden_size_vs_lr'] = self.test_ablation_29_hidden_size_vs_lr()

        self._reset_parameters()
        results['ablation_30_mlp_hidden_512_overfit'] = self.test_ablation_30_mlp_hidden_512_overfit()

        self._reset_parameters()
        results['ablation_31_compare_param_counts'] = self.test_ablation_31_compare_param_counts()

        self._reset_parameters()
        results['ablation_32_trm_with_dropout'] = self.test_ablation_32_trm_with_dropout()

        self._reset_parameters()
        results['ablation_33_trm_fewer_cycles'] = self.test_ablation_33_trm_fewer_cycles()

        self._reset_parameters()
        results['ablation_34_trm_longer_training'] = self.test_ablation_34_trm_longer_training()
    
        print("\n" + "=" * 60)
        print("ABLATION RESULTS:")
        print("=" * 60)
        for name, passed in results.items():
            print(f"  {name}: {'✓ PASS' if passed else '✗ FAIL'}")
        
        return results

    def _reset_parameters(self):
        """Reset all parameters to initial values"""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()