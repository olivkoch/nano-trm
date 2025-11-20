"""
Self-Play TRM PyTorch Lightning Module for AlphaZero-style learning
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn.modules.utils import compute_lr

try:
    from adam_atan2 import AdamATan2
except ImportError:
    print("Failed to import adam2")

from lightning import LightningModule

from src.nn.modules.trm_block import (
    CastedLinear,
    ReasoningBlock,
    ReasoningBlockConfig,
    ReasoningModule,
    RotaryEmbedding,
    CastedEmbedding,
)
from src.nn.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class SelfPlayTRMInnerCarry:
    z_H: torch.Tensor  # High-level state
    z_L: torch.Tensor  # Low-level state


@dataclass
class SelfPlayTRMCarry:
    """Carry structure for maintaining state across steps."""
    inner_carry: SelfPlayTRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class SelfPlayTRMModule(LightningModule):
    """
    TRM implementation for self-play with dual outputs: policy and value.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        max_grid_size: int = 42,  # For Connect 4: 6x7 = 42
        action_space_size: int = 7,  # For Connect 4: 7 columns
        H_cycles: int = 3,
        L_cycles: int = 6,
        N_supervision: int = 8,  # Fewer steps needed for game evaluation
        N_supervision_val: int = 8,
        ffn_expansion: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        halt_exploration_prob: float = 0.1,
        rope_theta: int = 10000,
        lr_min_ratio: float = 1.0,
        vocab_size: int = 3,  
        seq_len: int = 42,  # Board size
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        halt_weight: float = 0.5,
        output_dir: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # CRITICAL: Manual optimization (inherited from base TRM)
        self.automatic_optimization = False

        self.forward_dtype = torch.float32

        # Token embeddings for board states
        self.embed_scale = math.sqrt(hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.input_embedding = CastedEmbedding(
            vocab_size, hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype
        )

        # Positional embeddings
        self.pos_embedding = RotaryEmbedding(
            dim=hidden_size // num_heads,
            max_position_embeddings=seq_len,
            base=rope_theta,
        )

        # Reasoning network (unchanged from base TRM)
        reasoning_config = ReasoningBlockConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=ffn_expansion,
            rms_norm_eps=1e-5,
            seq_len=seq_len,
            mlp_t=False,
            puzzle_emb_ndim=0,  # No puzzle embeddings for self-play
            puzzle_emb_len=0,
        )

        self.lenet = ReasoningModule(
            layers=[ReasoningBlock(reasoning_config) for _ in range(num_layers)]
        )

        # DUAL HEADS for self-play
        # Policy head: outputs distribution over actions
        self.policy_head = CastedLinear(hidden_size, action_space_size, bias=False)
        
        # Value head: outputs single scalar value in [-1, 1]
        self.value_head = nn.Sequential(
            CastedLinear(hidden_size, hidden_size // 2, bias=True),
            nn.ReLU(),
            CastedLinear(hidden_size // 2, 1, bias=True),
            nn.Tanh()
        )
        
        # Halting head (adapted for game states)
        # Learns when position is "solved" (clear win/loss/draw)
        self.q_head = CastedLinear(hidden_size, 2, bias=True)
        
        with torch.no_grad():
            self.q_head.weight.zero_()
            if self.q_head.bias is not None:
                self.q_head.bias.fill_(-5.0)

        # Initialize carry states
        self.carry = None
        
        self.z_H_init = nn.Parameter(
            torch.randn(hidden_size, dtype=self.forward_dtype) * 0.02
        )
        self.z_L_init = nn.Parameter(
            torch.randn(hidden_size, dtype=self.forward_dtype) * 0.02
        )

        self.manual_step = 0

    def _input_embeddings(self, board_state: torch.Tensor):
        """Embed board state tokens."""
        embedding = self.input_embedding(board_state.to(torch.int32))
        return self.embed_scale * embedding

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["board_state"].shape[0]
        device = batch["board_state"].device

        return SelfPlayTRMCarry(
            inner_carry=self.empty_carry(batch_size, device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v, device=device) for k, v in batch.items()},
        )

    def empty_carry(self, batch_size: int, device: torch.device) -> SelfPlayTRMInnerCarry:
        return SelfPlayTRMInnerCarry(
            z_H=torch.empty(
                batch_size,
                self.hparams.seq_len,
                self.hparams.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                self.hparams.seq_len,
                self.hparams.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: SelfPlayTRMInnerCarry) -> SelfPlayTRMInnerCarry:
        return SelfPlayTRMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.z_H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.z_L_init, carry.z_L),
        )

    def inner_forward(
        self, 
        carry: SelfPlayTRMInnerCarry, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[SelfPlayTRMInnerCarry, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Inner forward pass producing both policy and value outputs.
        """
        seq_info = dict(
            cos_sin=self.pos_embedding() if hasattr(self, "pos_embedding") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["board_state"])

        # Forward iterations (unchanged from base)
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

        # Use first position as global representation (like Sudoku TRM)
        # This is the "CLS token" position that aggregates board information
        global_repr = z_H[:, 0, :]  # [batch_size, hidden_size]
        
        # Policy output (distribution over actions)
        policy_logits = self.policy_head(global_repr)  # [batch_size, action_space_size]
        
        # Value output (expected game outcome)
        value = self.value_head(global_repr).squeeze(-1)  # [batch_size]
        
        # Halting logits (position complexity)
        q_logits = self.q_head(global_repr).to(torch.float32)
        
        # New carry (detached)
        new_carry = SelfPlayTRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        
        return new_carry, policy_logits, value, (q_logits[..., 0], q_logits[..., 1])

    def forward(
        self, 
        carry: SelfPlayTRMCarry, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[SelfPlayTRMCarry, Dict[str, torch.Tensor]]:
        """
        Forward pass with carry management.
        """
        # Update carry
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

        # Apply valid actions mask if provided
        if "valid_actions" in new_current_data:
            valid_mask = new_current_data["valid_actions"]
            policy_logits = policy_logits.masked_fill(~valid_mask, -1e9)
        
        # Compute policy probabilities
        policy = F.softmax(policy_logits, dim=-1)

        outputs = {
            "policy_logits": policy_logits,
            "policy": policy,
            "value": value,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            n_supervision_steps = (
                self.hparams.N_supervision if self.training else self.hparams.N_supervision_val
            )
            is_last_step = new_steps >= n_supervision_steps
            
            # Halting logic (simpler for games - halt when position is "clear")
            halted = is_last_step
            
            if self.training and (self.hparams.N_supervision > 1):
                # Use q_halt to determine if position is solved
                halted = halted | (q_halt_logits > 0)
                
                # Exploration
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.hparams.halt_exploration_prob
                ) * torch.randint_like(new_steps, low=2, high=self.hparams.N_supervision + 1)
                halted = halted & (new_steps >= min_halt_steps)

        return SelfPlayTRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs

    def compute_loss_and_metrics(self, carry, batch):
        """
        Compute self-play losses: policy loss + value loss + halting loss.
        """
        # Get model outputs
        new_carry, outputs = self.forward(carry, batch)
        
        # Extract targets from batch
        target_policy = new_carry.current_data["mcts_policy"]  # [batch_size, action_space]
        target_value = new_carry.current_data["game_outcome"]  # [batch_size]
        
        with torch.no_grad():
            # Policy metrics
            pred_actions = outputs["policy"].argmax(dim=-1)
            target_actions = target_policy.argmax(dim=-1)
            policy_accuracy = (pred_actions == target_actions).float()
            
            # Value metrics
            value_mae = torch.abs(outputs["value"] - target_value)
            value_mse = (outputs["value"] - target_value) ** 2
            
            # TODO: Alternatives to halting criteria

            # Policy entropy: Low when the model is sure about the best move
            #     policy_entropy = -(outputs["policy"] * torch.log(outputs["policy"] + 1e-8)).sum(dim=-1)
            # normalized_entropy = policy_entropy / torch.log(torch.tensor(self.hparams.action_space_size))
            # policy_confidence = 1.0 - normalized_entropy  # High when entropy is low

            # Agreement with ground truth (if available and trusted)
            # value_agreement = (value_mae < 0.2).float()  # Within 0.2 of MCTS value
            overall_confidence = torch.abs(outputs["value"])
            should_halt = overall_confidence > 0.7
            is_terminal = torch.abs(target_value) > 0.9
            should_halt = should_halt | is_terminal

            # Position complexity metric (is position "solved"?)
            # A position is "solved" if the value is close to -1 or 1
            # position_solved = torch.abs(target_value) > 0.9
            
            # Metrics for halted sequences
            valid_metrics = new_carry.halted
            
            metrics = {
                "count": valid_metrics.sum(),
                "policy_accuracy": torch.where(valid_metrics, policy_accuracy, 0).sum(),
                "value_mae": torch.where(valid_metrics, value_mae, 0).sum(),
                "value_mse": torch.where(valid_metrics, value_mse, 0).sum(),
                "overall_confidence": torch.where(valid_metrics, overall_confidence, 0).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == should_halt)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }
        
        # Compute losses
        # Policy loss: KL divergence (or cross-entropy)
        policy_loss = -torch.sum(target_policy * torch.log(outputs["policy"] + 1e-8), dim=-1).sum()
        
        # Value loss: MSE
        value_loss = F.mse_loss(outputs["value"], target_value, reduction='sum')
        
        # Halting loss: BCE with position complexity
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            should_halt.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        
        metrics.update({
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        
        # Total loss with weights
        total_loss = (
            self.hparams.policy_weight * policy_loss + 
            self.hparams.value_weight * value_loss + 
            self.hparams.halt_weight * q_halt_loss
        )
        
        return new_carry, total_loss, metrics, new_carry.halted.all()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for self-play.
        Expected batch format:
        - board_state: [batch_size, seq_len] - tokenized board
        - mcts_policy: [batch_size, action_space] - MCTS visit counts (normalized)
        - game_outcome: [batch_size] - game outcome from current player's perspective
        - valid_actions: [batch_size, action_space] - boolean mask
        """
        batch_size = batch["board_state"].shape[0]
        
        # Get optimizer
        try:
            opts = self.optimizers()
            if not isinstance(opts, list):
                opts = [opts]
        except RuntimeError:
            if not hasattr(self, "_optimizers"):
                raise RuntimeError("No optimizer available.")
            opts = self._optimizers

        # Initialize carry if needed
        if self.carry is None:
            self.carry = self.initial_carry(batch)

        # Forward with loss computation
        self.carry, loss, metrics, all_halted = self.compute_loss_and_metrics(self.carry, batch)
        
        # Backward
        scaled_loss = loss / batch_size
        scaled_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Learning rate scheduling
        current_step = self.manual_step
        total_steps = getattr(self, "total_steps", self.hparams.max_steps)
        
        for opt in opts:
            if current_step < self.hparams.warmup_steps:
                lr_this_step = compute_lr(
                    base_lr=self.hparams.learning_rate,
                    lr_warmup_steps=self.hparams.warmup_steps,
                    lr_min_ratio=self.hparams.lr_min_ratio,
                    current_step=current_step,
                    total_steps=total_steps,
                )
            else:
                lr_this_step = self.hparams.learning_rate
            
            for param_group in opt.param_groups:
                param_group["lr"] = lr_this_step
            
            opt.step()
            opt.zero_grad()
        
        # Log metrics
        if metrics.get("count", 0) > 0:
            with torch.no_grad():
                count = metrics["count"]
                self.log("train/policy_accuracy", metrics["policy_accuracy"] / count, on_step=True)
                self.log("train/value_mae", metrics["value_mae"] / count, on_step=True, prog_bar=True)
                self.log("train/value_mse", metrics["value_mse"] / count, on_step=True)
                self.log("train/q_halt_accuracy", metrics["q_halt_accuracy"] / count, on_step=True)
                self.log("train/steps", metrics["steps"] / count, on_step=True)
                self.log("train/policy_loss", metrics["policy_loss"] / batch_size, on_step=True)
                self.log("train/value_loss", metrics["value_loss"] / batch_size, on_step=True)
                self.log("train/q_halt_loss", metrics["q_halt_loss"] / batch_size, on_step=True)
                self.log("train/lr", lr_this_step, on_step=True)
        
        self.manual_step += 1
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step for self-play."""
        batch_size = batch["board_state"].shape[0]
        
        with torch.no_grad():
            carry = self.initial_carry(batch)
            accumulated_metrics = {}
            total_loss = 0.0
            n_steps = 0
            
            while True:
                carry, loss, metrics, all_halted = self.compute_loss_and_metrics(carry, batch)
                
                for k, v in metrics.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0) + v.item()
                
                total_loss += loss.item()
                n_steps += 1
                
                if all_halted:
                    break
            
            count = accumulated_metrics.get("count", batch_size)
            if count > 0:
                avg_metrics = {
                    "val/loss": total_loss / (n_steps * batch_size),
                    "val/policy_accuracy": accumulated_metrics["policy_accuracy"] / count,
                    "val/value_mae": accumulated_metrics["value_mae"] / count,
                    "val/value_mse": accumulated_metrics["value_mse"] / count,
                    "val/q_halt_accuracy": accumulated_metrics["q_halt_accuracy"] / count,
                    "val/steps": accumulated_metrics["steps"] / count,
                }
            else:
                avg_metrics = {f"val/{k}": 0.0 for k in ["loss", "policy_accuracy", "value_mae", "value_mse", "q_halt_accuracy", "steps"]}
            
            for name, value in avg_metrics.items():
                self.log(name, value, on_step=False, on_epoch=True, 
                        prog_bar=(name in ["val/loss", "val/value_mae"]), sync_dist=True)
            
            return avg_metrics

    def configure_optimizers(self):
        """Configure optimizer."""
        all_params = list(self.parameters()) + list(self.model.parameters())
        try:
            optimizer = AdamATan2(
                all_params,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.95),
            )
        except NameError:
            optimizer = torch.optim.AdamW(
                all_params,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.95),
            )
        
        return optimizer