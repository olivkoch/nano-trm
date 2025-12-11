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
    z_H: torch.Tensor  # High-level state (y in your code)
    z_L: torch.Tensor  # Low-level state (z in your code)


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
    
    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        max_grid_size: int = 30,
        H_cycles: int = 3,
        L_cycles: int = 6,
        N_supervision: int = 16,
        N_supervision_val: int = 16,
        ffn_expansion: int = 2,
        learning_rate: float = 1e-4,
        learning_rate_emb: float = 1e-2,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        halt_exploration_prob: float = 0.1,
        puzzle_emb_dim: int = 512,
        puzzle_emb_len: int = 16,
        rope_theta: int = 10000,
        lr_min_ratio: float = 1.0,
        steps_per_epoch: int = 1000,
        vocab_size: int = 3,
        num_puzzles: int = 1,
        batch_size: int = 256,
        max_epochs: int = 300,
        pad_value: int = -1,
        seq_len: int = 42,  # Connect Four board size
        
        # Self-play parameters (can enable these)
        enable_selfplay: bool = False,
        selfplay_buffer_size: int = 100000,
        selfplay_games_per_iteration: int = 50,
        selfplay_mcts_simulations: int = 30,
        selfplay_eval_mcts_simulations: int = 100,
        selfplay_parallel_simulations: int = 8, # for debugging, should be much higher on gpu
        selfplay_temperature_moves: int = 15,
        selfplay_update_interval: int = 10,
        selfplay_bootstrap_weight: float = 0.3,  # 0 = pure outcome, 1 = pure MCTS value
        selfplay_temporal_decay: float = 0.95,   # Decay bootstrap for later moves
        
        # Evaluation parameters
        eval_minimax_depth: int = 4,
        eval_minimax_temperature: float = 0.5,
        eval_games_vs_minimax: int = 100,
        eval_games_vs_random: int = 100,
        eval_interval: int = 5,
        eval_use_mcts: bool = True,
        
        output_dir: str = None,
        **kwargs
    ):
        # Initialize base class with all shared parameters
        super().__init__(
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            lr_min_ratio=lr_min_ratio,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            max_epochs=max_epochs,
            enable_selfplay=enable_selfplay,
            selfplay_buffer_size=selfplay_buffer_size,
            selfplay_games_per_iteration=selfplay_games_per_iteration,
            selfplay_mcts_simulations=selfplay_mcts_simulations,
            selfplay_parallel_simulations=selfplay_parallel_simulations,
            selfplay_temperature_moves=selfplay_temperature_moves,
            selfplay_update_interval=selfplay_update_interval,
            selfplay_bootstrap_weight=selfplay_bootstrap_weight,
            selfplay_temporal_decay=selfplay_temporal_decay,
            selfplay_eval_mcts_simulations=selfplay_eval_mcts_simulations,
            eval_minimax_depth=eval_minimax_depth,
            eval_minimax_temperature=eval_minimax_temperature,
            eval_games_vs_minimax=eval_games_vs_minimax,
            eval_games_vs_random=eval_games_vs_random,
            eval_interval=eval_interval,
            eval_use_mcts=eval_use_mcts,
            output_dir=output_dir,
            # TRM-specific parameters
            num_heads=num_heads,
            max_grid_size=max_grid_size,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            N_supervision=N_supervision,
            N_supervision_val=N_supervision_val,
            ffn_expansion=ffn_expansion,
            learning_rate_emb=learning_rate_emb,
            halt_exploration_prob=halt_exploration_prob,
            puzzle_emb_dim=puzzle_emb_dim,
            puzzle_emb_len=puzzle_emb_len,
            rope_theta=rope_theta,
            vocab_size=vocab_size,
            num_puzzles=num_puzzles,
            pad_value=pad_value,
            seq_len=seq_len,
            **kwargs
        )
        
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
        
        # Single network (not two separate networks)
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
        
        self.lm_head = CastedLinear(hidden_size, self.board_cols, bias=False)
        self.value_head = CastedLinear(hidden_size, 1, bias=False)
        self.q_head = CastedLinear(hidden_size, 2, bias=True)
        
        # Halting head initialization
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
        else:
            log.info("puzzle_emb_dim <= 0, not creating puzzle embeddings")
            self.puzzle_emb = None
            self.puzzle_emb_len = 0
        
        self.last_step_time = None
        
        log.info(f"Learning rates: model={learning_rate}, emb={learning_rate_emb} max steps = {self.max_steps}")
    
    def setup(self, stage: str):
        """Called by Lightning when setting up the model."""
        if stage == "fit":
            # Add torch.compile for faster training
            if "DISABLE_COMPILE" not in os.environ and hasattr(torch, 'compile') and self.device.type == "cuda":
                try:
                    log.info("Compiling inner_forward with torch.compile...")
                    self.inner_forward = torch.compile(
                        self.inner_forward,
                        mode="reduce-overhead",  # Good for repeated calls (your H/L cycles)
                        fullgraph=False,         # Allow graph breaks for dynamic control flow
                    )
                    log.info("Compilation successful")
                except Exception as e:
                    log.warning(f"torch.compile failed, running uncompiled: {e}")
            else:
                log.info('*' * 60)
                log.info("torch.compile not available or disabled, running uncompiled")

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor, current_player: torch.Tensor):

        input = self._canonicalize_board(input, current_player)

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
    
    # def reset_carry(self, reset_flag: torch.Tensor, carry: TRMInnerCarry) -> TRMInnerCarry:
    #     return TRMInnerCarry(
    #         z_H=torch.where(reset_flag.view(-1, 1, 1), self.z_H_init, carry.z_H),
    #         z_L=torch.where(reset_flag.view(-1, 1, 1), self.z_L_init, carry.z_L),
    #     )
    
    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMInnerCarry) -> TRMInnerCarry:
        """Reset carry with position-aware initialization"""
        batch_size = carry.z_H.shape[0]
        
        # Expand init to batch size: (seq, hidden) -> (batch, seq, hidden)
        z_H_init_expanded = self.z_H_init.unsqueeze(0).expand(batch_size, -1, -1)
        z_L_init_expanded = self.z_L_init.unsqueeze(0).expand(batch_size, -1, -1)
        
        return TRMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), z_H_init_expanded, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), z_L_init_expanded, carry.z_L),
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
        policy_logits = self.lm_head(z_H[:, 0])
        value = torch.tanh(self.value_head(z_H[:, 0])).to(torch.float32)
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, policy_logits, value, (q_logits[..., 0], q_logits[..., 1])
    
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
        new_inner_carry, policy_logits, value, (q_halt_logits, q_continue_logits) = self.inner_forward(
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
            "q_continue_logits": q_continue_logits,
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
    
    def compute_loss_and_metrics(self, batch):
        """Compute loss and metrics for TRM"""

        # Get model outputs
        new_carry, outputs = self.forward(self.carry, batch)
        
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
        
        target_values = new_carry.current_data["values"]  # From your dataloader
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
        
        total_loss = policy_loss + 0.5 * value_loss  # + 0.5 * q_halt_loss
        
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