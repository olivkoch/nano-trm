"""
HRM PyTorch Lightning Module - Following Figure 2 pseudocode exactly
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.nn.utils.constants import C4_EMPTY_CELL
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


class TRMC4Module(LightningModule):
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
        N_supervision_val: int = 16,
        ffn_expansion: int = 2,
        learning_rate: float = 1e-4,
        learning_rate_emb: float = 1e-2,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        halt_exploration_prob: float = 0.1,
        puzzle_emb_dim: int = 512,  # Puzzle embedding dimension
        puzzle_emb_len: int = 16,  # How many tokens for puzzle embedding
        rope_theta: int = 10000,
        lr_min_ratio: float = 1.0,
        steps_per_epoch: int = 1000,
        vocab_size: int = 3,
        num_puzzles: int = 1,
        batch_size: int = 0,
        max_epochs: int = 300,
        pad_value: int = -1,
        seq_len: int = 0,
        eval_minimax_depth: int = 4,
        eval_minimax_temperature: float = 0.5,
        eval_games_vs_minimax: int = 100,
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

        self.board_rows = 6
        self.board_cols = 7

        self.lm_head = CastedLinear(hidden_size, self.board_cols, bias=False)
        self.value_head = CastedLinear(hidden_size, 1, bias=False)
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
        self.manual_step = 0
        self.games_played = 0
        self.replay_buffer = []
        self.max_steps = self.hparams.steps_per_epoch * self.hparams.max_epochs

        log.info(f"Learning rates: model={learning_rate}, emb={learning_rate_emb} max steps = {self.max_steps}")

        # read game curriculum
        self.load_games_from_file(f"minimax_games_.pkl")

    def setup(self, stage: str):
        """Called by Lightning when setting up the model."""
        pass    

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
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
        input_embeddings = self._input_embeddings(batch["boards"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
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
        policy_logits = self.lm_head(z_H[:,0])
        value = torch.tanh(self.value_head(z_H[:, 0])).to(torch.float32)
        q_logits = self.q_head(z_H[:, 0]).to(
            torch.float32
        )  # Q-head; uses the first puzzle_emb position
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

            metrics = {
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
        metrics.update(
            {
                "policy_loss": policy_loss.detach(),
                "value_loss": value_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            }
        )

        total_loss = policy_loss + 0.5 * value_loss # + 0.5 * q_halt_loss

        return new_carry, total_loss, metrics, new_carry.halted.all()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step that implements supervision through multiple forward passes.
        Each sequence can run up to N_supervision (halt_max_steps) times.
        """
        import time

        t0 = time.time()

        # monitor time since last step, this could be high if a validation occurred
        # if self.last_step_time is not None:
        #     delta = time.time() - self.last_step_time
        #     if delta > 0.2:
        #         print(f"WARNING: Time since last training step is long: {delta:.4f} s")

        batch_size = batch["boards"].shape[0]

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

        # ADD GRADIENT MONITORING HERE
        with torch.no_grad():
            # 1. Total gradient norm
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), 
                max_norm=float('inf')  # Don't actually clip, just compute norm
            ).item()
            
            # 2. Key component gradient norms
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
            
            # Optional: log individual components
            for name, value in grad_metrics.items():
                self.log(f'grad/{name}', value, on_step=True)
            
            # Warning for problematic gradients
            if total_grad_norm < 1e-6 or total_grad_norm > 100:
                log.warning(f"Step {self.manual_step}: Gradient norm={total_grad_norm:.2e}")
        

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
                # Constant LR after warmup (you can add decay here if needed)
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

            # Log learning rate (will log the last optimizer's LR)
            self.log(f"train/lr_{i}", lr_this_step, on_step=True)

        # Log metrics
        if metrics.get("count", 0) > 0:
            with torch.no_grad():
                count = metrics["count"]
                self.log(
                    "train/policy_accuracy",
                    metrics.get("policy_accuracy", 0) / count,
                    prog_bar=True,
                    on_step=True,
                )
                self.log(
                    "train/value_loss",
                    metrics.get("value_loss", 0) / count,
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

                self.log("train/policy_loss", metrics.get("policy_loss", 0) / batch_size, on_step=True)
                self.log(
                    "train/q_halt_loss", metrics.get("q_halt_loss", 0) / batch_size, on_step=True
                )

                avg_halt_steps = metrics.get("steps", 0) / metrics["count"]
                early_halt_rate = avg_halt_steps < self.hparams.N_supervision
                self.log("train/early_halt_rate", early_halt_rate, on_step=True)

        # Assert LM loss is not NaN
        assert not torch.isnan(metrics.get("policy_loss")), f"Policy loss is NaN at step {self.manual_step}"

        t1 = time.time()
        # print(f"Training step time: {t1 - t0:.4f} s")

        self.last_step_time = t1
        self.manual_step += 1

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Simplified validation using loss head."""
        return None

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step - same as validation."""
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_start(self):
        """Don't interfere with training carry during validation."""
        pass

    def on_train_epoch_end(self):
        """Don't interfere with training carry during validation."""
        self.evaluate_vs_minimax_fast()
        pass

    def on_train_epoch_start(self):
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

    def collect_self_play_games_minimax(self, n_games: int, depth: int = 2, temp_player1: float = 0.0, temp_player2: float = 0.5):
        """
        Collect self-play games using Minimax players with different temperatures.
        Lower temperature = stronger/more deterministic play
        Higher temperature = weaker/more exploratory play
        
        Args:
            depth: Minimax search depth for both players
            temp_player1: Temperature for player 1 (X). 0.0 = perfect play, 1.0 = more random
            temp_player2: Temperature for player 2 (O). Different temp creates variety
        """
        from src.nn.modules.minimax import ConnectFourMinimax
        from src.nn.environments.vectorized_c4_env import VectorizedConnectFour

        log.info(f"Collecting self-play {n_games} games using Minimax (depth={depth}, P1_temp={temp_player1}, P2_temp={temp_player2})...")
        
        # Create minimax player
        minimax = ConnectFourMinimax(depth=depth)
        
        # Run multiple games in parallel
        n_parallel = 512
        n_batches = max(1, n_games // n_parallel)
        n_positions_added = 0
        trajectories_lengths = []

        for batch_idx in range(n_batches):
            print(f"  Batch {batch_idx+1}/{n_batches}...")
            vec_env = VectorizedConnectFour(n_envs=n_parallel, device=self.device)
            states = vec_env.reset()
            trajectories = [[] for _ in range(n_parallel)]
            
            # Play all games until completion
            while not states.is_terminal.all():
                active_indices = []
                for i in range(n_parallel):
                    if not states.is_terminal[i]:
                        active_indices.append(i)
                
                if not active_indices:
                    break
                
                # Get minimax move for each active game
                actions = torch.zeros(n_parallel, dtype=torch.long, device=self.device)
                
                for i in active_indices:
                    # Get board and current player
                    board_np = states.boards[i].cpu().numpy()
                    current_player = states.current_players[i].item()
                    
                    # Use different temperatures for different players
                    temperature = temp_player1 if current_player == 1 else temp_player2
                    
                    # Get minimax move with appropriate temperature
                    action = minimax.get_best_move(board_np, current_player, temperature=temperature)
                    actions[i] = action
                    
                    # Create policy target based on minimax evaluation
                    # Instead of one-hot, we can create a distribution based on minimax scores
                    policy = np.zeros(7)
                    policy[action] = 1.0  # One-hot for now; can be improved to a distribution

                    legal_moves = (states.boards[i].view(6, 7)[0, :] == 0).cpu().numpy()
                    assert legal_moves[action], "Target move must be legal!"

                    # Store position
                    trajectories[i].append({
                        'board': states.boards[i].flatten(),
                        'policy': torch.tensor(policy, device=self.device, dtype=torch.float32),
                        'legal_moves': torch.tensor(legal_moves, device=self.device, dtype=torch.bool),
                        'player': current_player
                    })
                
                # Step environment
                states = vec_env.step(actions)
            
            trajectories_lengths.extend([len(t) for t in trajectories])

            # Process completed games and assign values
            for i in range(n_parallel):
                assert(len(trajectories[i]) == states.move_counts[i].item())
                assert(len(trajectories[i]) >= 7)  # Minimum moves to finish a game

                if len(trajectories[i]) < 2:  # Skip very short games
                    continue
                
                winner = states.winners[i].item()
                
                trajectory = trajectories[i]
                for position_index, position in enumerate(trajectory):
                    game_length = len(trajectory)

                    # Distance from end of game
                    distance_from_end = game_length - position_index - 1
    
                    # Decay factor
                    decay = 0.9 ** distance_from_end
                    
                    # Assign value based on outcome and distance
                    if winner == 0:
                        value = 0.0
                    elif winner == position['player']:
                        value = decay  # Good moves closer to win are better
                    else:
                        value = -decay  # Bad moves closer to loss are worse
                    
                    if value > 0.2:
                        self.replay_buffer.append({
                            'board': position['board'],
                            'policy': position['policy'],
                            'value': value
                        })
                        n_positions_added += 1
            
            self.games_played += n_parallel
        
        print(f"Collected {n_positions_added} positions using Minimax (depth={depth}), replay buffer size: {len(self.replay_buffer)}")
        print(f"Average game length: {np.mean(trajectories_lengths):.1f} moves")

    def load_games_from_file(self, input_file: str):
        """
        Load games from a JSON file into the replay buffer.
        
        Args:
            input_file: Path to JSON file containing game data
        """
        
        log.info(f"Loading games from {input_file}...")
        import pickle
        data = pickle.load(open(input_file, 'rb'))
        
        positions = data['positions']
        
        for pos in positions:
            board = torch.tensor(pos['board'], dtype=torch.float32, device=self.device)
            policy = torch.tensor(pos['policy'], dtype=torch.float32, device=self.device)
            value = pos['value']
            
            self.replay_buffer.append({
                'board': board,
                'policy': policy,
                'value': value
            })
        
        log.info(f"Loaded {len(positions)} positions from {data['num_games']} games")
        log.info(f"Replay buffer now contains {len(self.replay_buffer)} positions")

    def train_dataloader(self):
        """Create dataloader for self-play training"""
        from torch.utils.data import DataLoader, Dataset
        import random

        class SelfPlayDataset(Dataset):
            def __init__(self, module, steps_per_epoch):
                self.module = module
                self.steps_per_epoch = steps_per_epoch
                
            def __len__(self):
                return self.steps_per_epoch
            
            def __getitem__(self, idx):
                # Sample batch
                assert(len(self.module.replay_buffer) >= self.module.hparams.batch_size)
                samples = random.sample(
                    self.module.replay_buffer, 
                    self.module.hparams.batch_size
                )
                
                boards = torch.stack([s['board'] for s in samples])
                policies = torch.stack([s['policy'] for s in samples])
                values = torch.tensor([s['value'] for s in samples])
                puzzle_identifiers = torch.zeros(
                    self.module.hparams.batch_size, dtype=torch.long, device=self.module.device
                )  # Dummy puzzle IDs
                
                return {
                    'boards': boards,
                    'policies': policies,
                    'values': values,
                    'puzzle_identifiers': puzzle_identifiers
                }
        
        dataset = SelfPlayDataset(self, self.hparams.steps_per_epoch)
        return DataLoader(dataset, batch_size=None, num_workers=0, shuffle=False)
    
    def evaluate_vs_minimax_fast(self):
        """Fast evaluation using only neural network (no MCTS)"""
        from src.nn.modules.minimax import ConnectFourMinimax
        from src.nn.environments.vectorized_c4_env import VectorizedConnectFour

        log.info(f"Fast eval vs Minimax (depth={self.hparams.eval_minimax_depth}) and temperature={self.hparams.eval_minimax_temperature} over {self.hparams.eval_games_vs_minimax} games...")
        
        minimax = ConnectFourMinimax(depth=self.hparams.eval_minimax_depth)
        n_games = self.hparams.eval_games_vs_minimax
        n_parallel = self.hparams.batch_size  # Can handle more parallel games without MCTS
        n_batches = (n_games + n_parallel - 1) // n_parallel
        
        wins = 0
        draws = 0
        fallback_count = 0
        total_moves = 0

        for batch_idx in range(n_batches):
            games_in_batch = min(n_parallel, n_games - batch_idx * n_parallel)
            vec_env = VectorizedConnectFour(n_envs=games_in_batch, device=self.device)
            states = vec_env.reset()
            
            model_players = [1 if (batch_idx * n_parallel + i) % 2 == 0 else 2 
                            for i in range(games_in_batch)]
            
            while not states.is_terminal.all():
                actions = torch.zeros(games_in_batch, dtype=torch.long, device=self.device)
                
                # Collect all active positions
                active_model_positions = []
                active_model_indices = []
                
                for i in range(games_in_batch):
                    if states.is_terminal[i]:
                        continue
                        
                    current_player = states.current_players[i].item()
                    
                    if current_player == model_players[i]:
                        active_model_positions.append(states.boards[i].flatten())
                        active_model_indices.append(i)
                    else:
                        # Minimax move
                        board_np = states.boards[i].cpu().numpy()
                        actions[i] = minimax.get_best_move(
                            board_np, current_player, 
                            temperature=self.hparams.eval_minimax_temperature
                        )
                
                # Batch evaluate all model positions
                if active_model_positions:
                    with torch.no_grad():
                        boards_tensor = torch.stack(active_model_positions).to(self.device)
                        batch = {'boards': boards_tensor,
                                 'puzzle_identifiers': torch.zeros(len(active_model_positions), dtype=torch.long, device=self.device)}
                        # Pad to batch size
                        actual_batch_size = len(active_model_positions)
                        # print(f"Evaluating {actual_batch_size} positions in batch...")
                        if actual_batch_size < self.hparams.batch_size:
                            pad_size = self.hparams.batch_size - actual_batch_size
                            # print(f"Adding padding of size {pad_size} to match batch size {self.hparams.batch_size}")
                            boards_tensor = F.pad(boards_tensor, (0, 0, 0, pad_size))
                            batch['puzzle_identifiers'] = F.pad(batch['puzzle_identifiers'], (0, pad_size))
                            batch['boards'] = boards_tensor
                        carry = self.initial_carry(batch)
                        _, policies = self.forward(carry, batch)
                        policies = policies['policy']
                        policies = policies[:actual_batch_size]
                        
                        for j, idx in enumerate(active_model_indices):
                            # Apply legal moves mask
                            legal = states.legal_moves[idx]
                            masked_policy = policies[j] * legal.float()
                            if masked_policy.sum() > 0:
                                actions[idx] = masked_policy.argmax().item()
                            else:
                                # Fallback to random legal move
                                legal_actions = legal.nonzero(as_tuple=True)[0]
                                actions[idx] = legal_actions[torch.randint(len(legal_actions), (1,))].item()
                                fallback_count += 1
                            total_moves += 1
                
                states = vec_env.step(actions)
            
            # Count results
            for i in range(games_in_batch):
                winner = states.winners[i].item()
                if winner == model_players[i]:
                    wins += 1
                elif winner == 0:
                    draws += 1
        
        # Log fast eval results
        win_rate = wins / n_games
        draw_rate = draws / n_games
        loss_rate = 1 - win_rate - draw_rate
        fallback_rate = fallback_count / total_moves if total_moves > 0 else 0.0

        self.log('eval/fast_win_rate_vs_minimax', win_rate, prog_bar=True)
        self.log('eval/fast_draw_rate_vs_minimax', draw_rate)
        self.log('eval/fallback_to_random_rate', fallback_rate)
        
        log.info(f"Fast eval vs Minimax: W={win_rate:.1%}, D={draw_rate:.1%}, L={loss_rate:.1%} (fallbacks: {fallback_rate:.1%}) over {n_games} games.")

if __name__ == "__main__":
    # collect some games
    module = TRMC4Module()
    module.replay_buffer = []
    module.collect_self_play_games_minimax(n_games=50000, depth=4, temp_player1=0.1, temp_player2=0.3)

    # Save games to file
    output_file = f"minimax_games_.pkl"
    
    games_data = []
    for item in module.replay_buffer:
        games_data.append({
            'board': item['board'].cpu().numpy().tolist(),
            'policy': item['policy'].cpu().numpy().tolist(),
            'value': float(item['value'])
        })
    data = {}
    data['positions'] = games_data
    data['num_games'] = module.games_played
    data['num_positions'] = len(games_data)
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Saved {len(games_data)} positions from {module.games_played} games to {output_file}")