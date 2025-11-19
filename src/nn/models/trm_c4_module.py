"""
TRM Connect Four Module - Clean Implementation with Proper Optimizer Setup
Pure self-play training following base TRM patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from collections import deque
from typing import Dict, Optional, List, Tuple
import random
import time

from src.nn.models.trm_module import TRMModule
from src.nn.environments.vectorized_c4_env import VectorizedConnectFour, VectorizedC4State
from src.nn.modules.batch_mcts import BatchMCTS
from src.nn.modules.utils import compute_lr
from src.nn.utils.constants import C4_EMPTY_CELL, C4_PLAYER1_CELL, C4_PLAYER2_CELL
from src.nn.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class TRMConnectFourModule(LightningModule):
    """
    TRM for Connect Four with self-play training
    Following base TRM patterns for optimization and scheduling
    """
    
    def __init__(
        self,
        # TRM architecture
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        H_cycles: int = 2,
        L_cycles: int = 4,
        N_supervision: int = 6,
        N_supervision_val: int = 6,
        ffn_expansion: int = 2,
        
        # Optimization
        learning_rate: float = 3e-4,
        learning_rate_emb: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        lr_min_ratio: float = 0.1,
        
        # Self-play
        games_per_epoch: int = 32,
        buffer_size: int = 50000,
        batch_size: int = 128,
        
        # MCTS
        mcts_simulations: int = 50,
        mcts_c_puct: float = 1.5,
        mcts_temperature: float = 1.0,
        
        # Loss weights
        policy_loss_weight: float = 1.0,
        value_loss_weight: float = 1.0,
        
        # Evaluation
        eval_games_vs_minimax: int = 20,
        
        output_dir: str = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # CRITICAL: Manual optimization (following base TRM)
        self.automatic_optimization = False
        
        # Get board dimensions from environment
        temp_env = VectorizedConnectFour(n_envs=1)
        board_rows = temp_env.rows
        board_cols = temp_env.cols
        del temp_env
        
        # Vocabulary: 0=padding, 1=empty, 2=player1, 3=player2
        vocab_size = 4
        seq_len = board_rows * board_cols  # 42
        
        # Create base TRM
        self.trm = TRMModule(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_grid_size=board_cols,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            N_supervision=N_supervision,
            N_supervision_val=N_supervision_val,
            ffn_expansion=ffn_expansion,
            learning_rate=learning_rate,
            learning_rate_emb=learning_rate_emb,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            lr_min_ratio=lr_min_ratio,
            vocab_size=vocab_size,
            num_puzzles=1,  # Single game type
            batch_size=batch_size,
            seq_len=seq_len,
            puzzle_emb_dim=0,  # No puzzle embeddings needed for C4
            puzzle_emb_len=0,
            output_dir=output_dir,
        )
        
        # Game-specific heads
        self.policy_head = nn.Linear(hidden_size, board_cols)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Initialize heads
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head.bias)
        
        # Store board dimensions
        self.board_rows = board_rows
        self.board_cols = board_cols
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Components (initialized in setup)
        self.vec_env = None
        self.mcts = None
        
        # Track steps for learning rate scheduling (like base TRM)
        self.manual_step = 0
        self.total_steps = max_steps  # Will be updated in setup
        
        # Metrics
        self.games_played = 0
        
    def setup(self, stage: str):
        """Initialize components and calculate total steps"""
        if stage == "fit":
            # Setup TRM
            self.trm.setup(stage)
            
            # Calculate total steps (following base TRM pattern)
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
            
            log.info("Connect Four training configuration:")
            log.info(f"  Board size: {self.board_rows}x{self.board_cols}")
            log.info(f"  Steps per epoch: {steps_per_epoch}")
            log.info(f"  Max epochs: {self.trainer.max_epochs}")
            log.info(f"  Total steps: {self.total_steps}")
            
            # Create vectorized environment
            self.vec_env = VectorizedConnectFour(
                n_envs=8,  # Process 8 games in parallel
                device=self.device
            )
            
            # Create MCTS
            self.mcts = BatchMCTS(
                trm_model=self,
                c_puct=self.hparams.mcts_c_puct,
                num_simulations=self.hparams.mcts_simulations,
                batch_size=8,
                device=self.device,
                temperature=self.hparams.mcts_temperature,
                use_compile=False
            )
            
            # Expose base_trm for MCTS compatibility
            self.base_trm = self.trm
    
    @torch.no_grad()
    def get_policy_value(
        self, 
        state_tensor: torch.Tensor, 
        legal_moves: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """Get policy and value for MCTS"""
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        state_tensor = state_tensor.long()
        
        batch = {
            'input': state_tensor,
            'output': torch.zeros_like(state_tensor),
            'puzzle_identifiers': torch.zeros(state_tensor.size(0), dtype=torch.long, device=self.device)
        }
        
        carry = self.trm.initial_carry(batch)
        carry, _ = self.trm(carry, batch)
        
        hidden = carry.inner_carry.z_H[:, 0]
        
        policy_logits = self.policy_head(hidden).squeeze(0)
        value = self.value_head(hidden).squeeze()
        
        if legal_moves is not None:
            illegal_mask = ~legal_moves.to(self.device)
            policy_logits = policy_logits.masked_fill(illegal_mask, -float('inf'))
        
        policy = F.softmax(policy_logits, dim=-1)
        
        return policy, value.item()
    
    def collect_self_play_games(self):
        """Collect self-play games using fully vectorized parallel execution"""
        n_parallel = 8  # Number of parallel games
        n_batches = self.hparams.games_per_epoch // n_parallel
        
        for _ in range(n_batches):
            # Reset all environments at once
            states = self.vec_env.reset()
            
            # Storage for all game trajectories
            trajectories = [[] for _ in range(n_parallel)]
            
            # Play all games in parallel until all are done
            while not states.is_terminal.all():
                # Get MCTS policies for ALL games at once (even terminal ones)
                with torch.no_grad():
                    # Get policies for all games in one batch
                    if self.games_played < 100:
                        # Early training: use raw network for speed
                        state_tensors = states.boards.flatten(1).long()
                        batch = {
                            'input': state_tensors,
                            'output': torch.zeros_like(state_tensors),
                            'puzzle_identifiers': torch.zeros(n_parallel, dtype=torch.long, device=self.device)
                        }
                        
                        carry = self.trm.initial_carry(batch)
                        carry, _ = self.trm(carry, batch)
                        hidden = carry.inner_carry.z_H[:, 0]
                        
                        policy_logits = self.policy_head(hidden)
                        
                        # Apply legal move masks
                        illegal_mask = ~states.legal_moves
                        policy_logits = policy_logits.masked_fill(illegal_mask, -float('inf'))
                        policies = F.softmax(policy_logits, dim=-1)
                    else:
                        # Use MCTS for better policies
                        policies = self.mcts.get_action_probabilities(states, temperature=self.hparams.mcts_temperature)
                    
                    # Ensure policies is 2D
                    if policies.dim() == 1:
                        policies = policies.unsqueeze(0)
                
                # Store trajectories for non-terminal games
                for i in range(n_parallel):
                    if not states.is_terminal[i]:
                        trajectories[i].append({
                            'state': states.boards[i].flatten(),
                            'policy': policies[i].detach(),
                            'player': states.current_players[i].item()
                        })
                
                # Select actions for all games
                actions = torch.zeros(n_parallel, dtype=torch.long, device=self.device)
                
                # Vectorized action selection
                active = ~states.is_terminal
                if active.any():
                    active_policies = policies[active]
                    
                    if self.training and self.hparams.mcts_temperature > 0:
                        # Sample from policy distribution
                        active_actions = torch.multinomial(active_policies, 1).squeeze(-1)
                    else:
                        # Greedy selection
                        active_actions = torch.argmax(active_policies, dim=-1)
                    
                    # Place actions in correct positions
                    actions[active] = active_actions
                
                # Step all environments at once
                states = self.vec_env.step(actions)
            
            # Process all completed games
            for i in range(n_parallel):
                if len(trajectories[i]) < 2:  # Skip trivial games
                    continue
                
                winner = states.winners[i].item()
                
                # Calculate values for all positions in this game
                for step in trajectories[i]:
                    if winner == 0:  # Draw
                        value = 0.0
                    elif winner == step['player']:  # Player won
                        value = 1.0
                    else:  # Player lost
                        value = -1.0
                    
                    self.replay_buffer.append({
                        'state': step['state'],
                        'policy': step['policy'],
                        'value': value
                    })
                
                self.games_played += 1
    
    def on_train_epoch_start(self):
        """Collect new self-play games"""
        if self.mcts is not None:
            self.collect_self_play_games()
            log.info(f"Games played: {self.games_played}, Buffer size: {len(self.replay_buffer)}")
    
    def training_step(self, batch, batch_idx):
        """Training step with manual optimization following base TRM pattern"""
        if len(self.replay_buffer) < self.hparams.batch_size:
            return {'loss': torch.tensor(0.0, device=self.device)}
        
        # Get optimizers
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        
        # Sample batch
        samples = random.sample(self.replay_buffer, self.hparams.batch_size)
        
        states = torch.stack([s['state'] for s in samples])
        policy_targets = torch.stack([s['policy'] for s in samples])
        value_targets = torch.tensor([s['value'] for s in samples], device=self.device)
        
        # Create TRM batch
        trm_batch = {
            'input': states,
            'output': torch.zeros_like(states),
            'puzzle_identifiers': torch.zeros(self.hparams.batch_size, dtype=torch.long, device=self.device)
        }
        
        # Forward pass
        carry = self.trm.initial_carry(trm_batch)
        carry, trm_outputs = self.trm(carry, trm_batch)
        
        hidden = carry.inner_carry.z_H[:, 0]
        policy_logits = self.policy_head(hidden)
        values = self.value_head(hidden).squeeze()
        
        # Losses
        policy_loss = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            policy_targets,
            reduction='batchmean'
        )
        value_loss = F.mse_loss(values, value_targets)
        
        total_loss = (
            self.hparams.policy_loss_weight * policy_loss +
            self.hparams.value_loss_weight * value_loss
        )
        
        # Backward
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Learning rate scheduling (following base TRM exactly)
        current_step = self.manual_step
        total_steps = self.total_steps
        
        base_lrs = [self.hparams.learning_rate]
        if len(opts) > 1:  # If we have embedding optimizer
            base_lrs.append(self.hparams.learning_rate_emb)
        
        # Compute and apply learning rate for each optimizer
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
            
            lr_this_step = self.hparams.learning_rate
            
            # Update learning rate and step
            if hasattr(opt, '_optimizer'):  # Sparse embedding optimizer
                for param_group in opt._optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                opt._optimizer.step()
                opt._optimizer.zero_grad()
            else:  # Regular optimizer
                for param_group in opt.param_groups:
                    param_group['lr'] = lr_this_step
                opt.step()
                opt.zero_grad()
        
        # Increment manual step counter
        self.manual_step += 1
        
        # Logging
        self.log('train/lr', lr_this_step, on_step=True)
        self.log('train/loss', total_loss, prog_bar=True)
        self.log('train/policy_loss', policy_loss)
        self.log('train/value_loss', value_loss)
        self.log('train/buffer_size', float(len(self.replay_buffer)))
        
        return {'loss': total_loss}
    
    def validation_step(self, batch, batch_idx):
        """Validation on replay buffer samples"""
        if len(self.replay_buffer) < 32:
            return
        
        with torch.no_grad():
            samples = random.sample(self.replay_buffer, 32)
            
            states = torch.stack([s['state'] for s in samples])
            policy_targets = torch.stack([s['policy'] for s in samples])
            value_targets = torch.tensor([s['value'] for s in samples], device=self.device)
            
            trm_batch = {
                'input': states,
                'output': torch.zeros_like(states),
                'puzzle_identifiers': torch.zeros(32, dtype=torch.long, device=self.device)
            }
            
            carry = self.trm.initial_carry(trm_batch)
            carry, _ = self.trm(carry, trm_batch)
            
            hidden = carry.inner_carry.z_H[:, 0]
            policy_logits = self.policy_head(hidden)
            values = self.value_head(hidden).squeeze()
            
            policy_acc = (policy_logits.argmax(-1) == policy_targets.argmax(-1)).float().mean()
            value_acc = ((values > 0) == (value_targets > 0)).float().mean()
            
            self.log('val/policy_accuracy', policy_acc, prog_bar=True)
            self.log('val/value_accuracy', value_acc)
    
    def on_validation_epoch_end(self):
        """Periodic evaluation against minimax"""
        self.evaluate_vs_minimax()
    
    def evaluate_vs_minimax(self, depth: int = 2):
        """Evaluate against minimax baseline"""
        from src.nn.modules.minimax import ConnectFourMinimax
        
        minimax = ConnectFourMinimax(depth=depth)
        wins = 0
        
        for game in range(self.hparams.eval_games_vs_minimax):
            self.vec_env.reset(env_ids=torch.tensor([0], device=self.device))
            model_player = 1 if game % 2 == 0 else 2
            
            while not self.vec_env.is_terminal[0]:
                current_player = self.vec_env.current_players[0].item()
                
                if current_player == model_player:
                    state = self.vec_env.get_state()
                    state = VectorizedC4State(
                        boards=state.boards[:1],
                        current_players=state.current_players[:1],
                        move_counts=state.move_counts[:1],
                        winners=state.winners[:1],
                        is_terminal=state.is_terminal[:1],
                        legal_moves=state.legal_moves[:1]
                    )
                    action = self.mcts.select_action(state, deterministic=True)
                else:
                    board = minimax.get_board_from_env(self.vec_env, 0)
                    action = minimax.get_best_move(board, current_player)
                
                actions = torch.zeros(self.vec_env.n_envs, dtype=torch.long, device=self.device)
                actions[0] = action
                self.vec_env.step(actions)
            
            if self.vec_env.winners[0].item() == model_player:
                wins += 1
        
        win_rate = wins / self.hparams.eval_games_vs_minimax
        self.log('eval/win_rate_vs_minimax', win_rate, prog_bar=True)
        log.info(f"Win rate vs minimax: {win_rate:.1%}")
    
    def configure_optimizers(self):
        """Configure optimizers - maintaining base TRM's special sparse embedding optimizer"""
        optimizers = []
        
        # Main optimizer for all regular parameters (including our new heads)
        all_regular_params = []
        
        # Get base TRM parameters EXCEPT sparse embeddings
        for name, param in self.trm.named_parameters():
            if 'puzzle_emb' not in name:  # Exclude sparse embeddings
                all_regular_params.append(param)
        
        # Add our new head parameters
        all_regular_params.extend(self.policy_head.parameters())
        all_regular_params.extend(self.value_head.parameters())
        
        # Main optimizer
        try:
            from adam_atan2 import AdamATan2
            main_opt = AdamATan2(
                all_regular_params,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.95)
            )
        except ImportError:
            main_opt = torch.optim.AdamW(
                all_regular_params,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.95)
            )
        optimizers.append(main_opt)
        
        # Sparse embedding optimizer (if puzzle embeddings exist)
        if hasattr(self.trm, 'puzzle_emb') and self.trm.puzzle_emb is not None:
            from src.nn.modules.sparse_embeddings import CastedSparseEmbeddingSignSGD_Distributed
            
            # Force sparse embedding to be leaf tensor (same as base TRM)
            self.trm.puzzle_emb.local_weights = self.trm.puzzle_emb.local_weights.detach().requires_grad_(True)
            
            sparse_opt = CastedSparseEmbeddingSignSGD_Distributed(
                self.trm.puzzle_emb.buffers(),
                lr=self.hparams.learning_rate_emb,
                weight_decay=self.hparams.weight_decay,
                world_size=1
            )
            optimizers.append(sparse_opt)
        
        return optimizers