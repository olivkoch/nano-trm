"""
TRM Connect Four Lightning Module
Integrates self-play training with your existing Lightning/Hydra setup
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from typing import Dict, List, Optional, Tuple

from src.nn.modules.trm_module import TRMModule
from src.nn.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class GameSample:
    """Single game position with targets"""
    state: torch.Tensor
    policy_target: torch.Tensor
    value_target: float
    move_count: int


class TRMConnectFourModule(LightningModule):
    """
    Lightning Module for TRM Connect Four Self-Play Training
    Wraps your existing TRM and adds game-specific components
    """
    
    def __init__(
        self,
        # TRM base config (same as your existing TRM)
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_grid_size: int = 7,  # For C4, this is cols
        H_cycles: int = 2,
        L_cycles: int = 4,
        N_supervision: int = 6,
        N_supervision_val: int = 6,
        ffn_expansion: int = 2,
        learning_rate: float = 1e-4,
        learning_rate_emb: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        halt_exploration_prob: float = 0.2,
        puzzle_emb_dim: int = 128,
        puzzle_emb_len: int = 4,
        rope_theta: int = 10000,
        lr_min_ratio: float = 0.1,
        vocab_size: int = 4,
        num_puzzles: int = 10000,
        batch_size: int = 32,
        pad_value: int = 0,
        seq_len: int = 42,
        
        # Connect Four specific
        num_actions: int = 7,
        board_rows: int = 6,
        board_cols: int = 7,
        
        # Self-play config
        buffer_size: int = 50000,
        games_per_iteration: int = 25,
        mcts_simulations: int = 50,
        mcts_c_puct: float = 1.5,
        mcts_temperature: float = 1.0,
        trm_iterations_per_eval: int = 3,
        
        # Training config
        policy_loss_weight: float = 1.0,
        value_loss_weight: float = 0.5,
        trm_loss_weight: float = 0.1,
        
        # Evaluation
        eval_games_vs_random: int = 20,
        eval_every_n_val_epochs: int = 1,  # NEW: How often to run game evaluation during validation
        
        output_dir: str = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # CRITICAL: Manual optimization (same as base TRM)
        self.automatic_optimization = False
        
        # Create base TRM
        self.base_trm = TRMModule(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_grid_size=max_grid_size,
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
            halt_exploration_prob=halt_exploration_prob,
            puzzle_emb_dim=puzzle_emb_dim,
            puzzle_emb_len=puzzle_emb_len,
            rope_theta=rope_theta,
            lr_min_ratio=lr_min_ratio,
            vocab_size=vocab_size,
            num_puzzles=num_puzzles,
            batch_size=batch_size,
            pad_value=pad_value,
            seq_len=seq_len,
            output_dir=output_dir
        )
        
        # Add game-specific heads
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()
        )
        
        # Initialize heads
        for module in [self.policy_head, self.value_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # Self-play components (initialized in setup)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.game_env = None
        self.mcts = None
        
        # Metrics
        self.total_games_played = 0
        self.best_win_rate = 0.0
        self.last_eval_step = 0
        self.validation_epoch_count = 0  # Track validation epochs for evaluation frequency
    
    @torch.no_grad()
    def get_policy_value(self, state_tensor: torch.Tensor, legal_moves: torch.Tensor = None) -> Tuple[torch.Tensor, float]:
        """
        Get policy and value for a game position using TRM's recursive reasoning.
        Used by MCTS during self-play.
        """
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        # Create batch for TRM
        batch = {
            'input': state_tensor.to(self.device),
            'output': torch.zeros_like(state_tensor).to(self.device),
            'puzzle_identifiers': torch.zeros(1, dtype=torch.long, device=self.device)
        }
        
        # Initialize carry and run TRM
        carry = self.base_trm.initial_carry(batch)
        carry, trm_outputs = self.base_trm(carry, batch)
        
        # Extract hidden state from first position
        hidden_state = carry.inner_carry.z_H[:, 0]
        
        # Get policy and value
        policy_logits = self.policy_head(hidden_state).squeeze(0)
        value = self.value_head(hidden_state).squeeze()
        
        # Apply legal move mask
        if legal_moves is not None:
            illegal_mask = ~legal_moves.to(self.device)
            policy_logits = policy_logits.masked_fill(illegal_mask, -float('inf'))
        
        # Convert to probabilities
        policy = F.softmax(policy_logits, dim=-1)
        
        return policy, value.item()
    def setup(self, stage: str):
        """Setup called by Lightning"""
        if stage == "fit":
            # Import here to avoid circular dependencies
            from connectfour_env import ConnectFourEnv
            from trm_mcts import TRM_MCTS
            
            # Create game environment
            self.game_env = ConnectFourEnv(device=self.device)
            
            # Setup MCTS to use this module directly
            self.mcts = TRM_MCTS(
                trm_model=self,  # Pass self instead of a wrapper
                c_puct=self.hparams.mcts_c_puct,
                num_simulations=self.hparams.mcts_simulations,
                num_trm_iterations=self.hparams.trm_iterations_per_eval,
                device=self.device,
                temperature=self.hparams.mcts_temperature
            )
            
            # Call base TRM setup
            self.base_trm.setup(stage)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through TRM and game heads"""
        # Get TRM carry and outputs
        carry = self.base_trm.initial_carry(batch)
        carry, trm_outputs = self.base_trm(carry, batch)
        
        # Extract hidden state for game heads
        hidden_state = carry.inner_carry.z_H[:, 0]  # First position
        
        # Compute policy and value
        policy_logits = self.policy_head(hidden_state)
        value = self.value_head(hidden_state).squeeze(-1)
        
        return {
            **trm_outputs,
            'policy_logits': policy_logits,
            'value': value,
            'carry': carry
        }
    
    def self_play_game(self, temperature: float = 1.0) -> List[GameSample]:
        """Play one self-play game"""
        env = self.game_env
        state = env.reset()
        game_history = []
        
        while not state.is_terminal:
            # Get MCTS policy
            policy = self.mcts.get_action_probabilities(state)
            
            # Store position
            game_history.append({
                'state': state.to_trm_input(),
                'policy': policy,
                'player': state.current_player,
                'move_count': state.move_count
            })
            
            # Select and make move
            if temperature > 0:
                action = torch.multinomial(policy, 1).item()
            else:
                action = torch.argmax(policy).item()
            
            state = env.make_move(action)
        
        # Determine outcome
        if state.winner == 1:
            outcomes = {1: 1.0, 2: -1.0}
        elif state.winner == 2:
            outcomes = {1: -1.0, 2: 1.0}
        else:
            outcomes = {1: 0.0, 2: 0.0}
        
        # Convert to training samples
        samples = []
        for position in game_history:
            samples.append(GameSample(
                state=position['state'],
                policy_target=position['policy'],
                value_target=outcomes[position['player']],
                move_count=position['move_count']
            ))
        
        return samples
    
    def on_train_epoch_start(self):
        """Collect self-play games at the start of each epoch"""
        if self.mcts is None:
            return  # Not yet setup
        
        log.info(f"Collecting {self.hparams.games_per_iteration} self-play games...")
        
        for _ in range(self.hparams.games_per_iteration):
            # Vary temperature for diversity
            temp = self.hparams.mcts_temperature if self.total_games_played % 4 != 0 else 0.5
            
            samples = self.self_play_game(temperature=temp)
            self.replay_buffer.extend(samples)
            self.total_games_played += 1
        
        log.info(f"Buffer size: {len(self.replay_buffer)} positions")
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step with manual optimization for Connect Four"""
        
        # Get optimizers (could be multiple due to sparse embeddings)
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        
        # Sample from replay buffer if available
        if len(self.replay_buffer) >= self.hparams.batch_size:
            import random
            samples = random.sample(self.replay_buffer, self.hparams.batch_size)
            
            # Convert to batch
            states = torch.stack([s.state for s in samples]).to(self.device)
            policy_targets = torch.stack([s.policy_target for s in samples]).to(self.device)
            value_targets = torch.tensor([s.value_target for s in samples], 
                                        dtype=torch.float32, device=self.device)
            
            # Create TRM batch
            game_batch = {
                'input': states,
                'output': torch.zeros_like(states),  # Dummy
                'puzzle_identifiers': torch.arange(self.hparams.batch_size, device=self.device)
            }
            
            # Forward pass
            outputs = self.forward(game_batch)
            
            # Compute game losses
            policy_loss = F.kl_div(
                F.log_softmax(outputs['policy_logits'], dim=-1),
                policy_targets,
                reduction='batchmean'
            )
            
            value_loss = F.mse_loss(outputs['value'], value_targets)
            
            # Combine with TRM loss
            carry = outputs['carry']
            _, trm_loss, metrics, _ = self.base_trm.compute_loss_and_metrics(carry, game_batch)
            trm_loss = trm_loss / self.hparams.batch_size
            
            # Total loss
            total_loss = (
                self.hparams.policy_loss_weight * policy_loss +
                self.hparams.value_loss_weight * value_loss +
                self.hparams.trm_loss_weight * trm_loss
            )
            
            # Manual backward and optimization
            total_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Learning rate scheduling (same as base TRM)
            current_step = self.global_step
            total_steps = self.trainer.max_steps if self.trainer.max_steps else 100000
            
            # Base learning rates
            base_lrs = [self.hparams.learning_rate]
            if len(opts) > 1:  # If we have sparse embedding optimizer
                base_lrs.append(self.hparams.learning_rate_emb)
            
            # Apply learning rate schedule
            from src.nn.modules.utils import compute_lr
            for opt, base_lr in zip(opts, base_lrs):
                if current_step < self.hparams.warmup_steps:
                    lr_this_step = compute_lr(
                        base_lr=base_lr,
                        lr_warmup_steps=self.hparams.warmup_steps,
                        lr_min_ratio=self.hparams.lr_min_ratio,
                        current_step=current_step,
                        total_steps=total_steps
                    )
                else:
                    lr_this_step = base_lr
                
                # Update learning rate and step optimizer
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
            
            # Log metrics
            self.log('train/lr', lr_this_step, on_step=True)
            self.log('train/policy_loss', policy_loss, prog_bar=True)
            self.log('train/value_loss', value_loss, prog_bar=True)
            self.log('train/trm_loss', trm_loss)
            self.log('train/total_loss', total_loss)
            self.log('train/buffer_size', float(len(self.replay_buffer)))
            self.log('train/total_games', float(self.total_games_played))
            
        else:
            # Fallback: if no self-play data, just do a forward pass for warmup
            # Don't use base TRM's training_step as it would conflict
            outputs = self.forward(batch)
            total_loss = torch.tensor(0.0, device=self.device)
            self.log('train/warmup', 1.0)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step - evaluate model performance on self-play buffer"""
        
        with torch.no_grad():
            # Only evaluate prediction accuracy on replay buffer data
            if len(self.replay_buffer) >= 32:
                import random
                samples = random.sample(self.replay_buffer, min(32, len(self.replay_buffer)))
                
                # Evaluate value and policy prediction
                states = torch.stack([s.state for s in samples]).to(self.device)
                value_targets = torch.tensor([s.value_target for s in samples], 
                                            dtype=torch.float32, device=self.device)
                policy_targets = torch.stack([s.policy_target for s in samples]).to(self.device)
                
                # Create batch
                game_batch = {
                    'input': states,
                    'output': torch.zeros_like(states),
                    'puzzle_identifiers': torch.arange(len(samples), device=self.device)
                }
                
                # Forward pass
                outputs = self.forward(game_batch)
                
                # Compute validation metrics
                value_loss = F.mse_loss(outputs['value'], value_targets)
                policy_loss = F.kl_div(
                    F.log_softmax(outputs['policy_logits'], dim=-1),
                    policy_targets,
                    reduction='batchmean'
                )
                
                # Value accuracy (correct sign prediction)
                value_correct = ((outputs['value'] > 0) == (value_targets > 0)).float().mean()
                
                # Policy accuracy (top move matches MCTS)
                our_top = torch.argmax(outputs['policy_logits'], dim=-1)
                mcts_top = torch.argmax(policy_targets, dim=-1)
                policy_accuracy = (our_top == mcts_top).float().mean()
                
                # Log metrics
                self.log('val/value_loss', value_loss, sync_dist=True)
                self.log('val/policy_loss', policy_loss, sync_dist=True)
                self.log('val/value_accuracy', value_correct, sync_dist=True)
                self.log('val/policy_accuracy', policy_accuracy, prog_bar=True, sync_dist=True)
                
                return {'val_loss': value_loss + policy_loss}
            else:
                # No data yet
                return {'val_loss': 0.0}
    
    def on_validation_epoch_end(self):
        """Run game evaluation at the end of validation epoch"""
        # Increment validation epoch counter
        self.validation_epoch_count += 1
        
        if self.mcts is None or len(self.replay_buffer) == 0:
            return
        
        # Only run expensive game evaluation every N validation epochs
        if self.validation_epoch_count % self.hparams.eval_every_n_val_epochs == 0:
            self.evaluate_vs_random()
        else:
            # Log that we skipped evaluation this validation epoch
            self.log('val/eval_skipped', 1.0)
    
    def evaluate_vs_random(self):
        """Evaluate against random player"""
        if self.mcts is None:
            return
        
        from connectfour_env import ConnectFourEnv
        
        wins = 0
        for game in range(self.hparams.eval_games_vs_random):
            env = ConnectFourEnv(device=self.device)
            state = env.reset()
            
            model_player = 1 if game % 2 == 0 else 2
            
            while not state.is_terminal:
                if state.current_player == model_player:
                    action = self.mcts.select_action(state, deterministic=True)
                else:
                    legal_actions = state.legal_moves.nonzero(as_tuple=True)[0]
                    action = legal_actions[torch.randint(len(legal_actions), (1,))].item()
                
                state = env.make_move(action)
            
            if state.winner == model_player:
                wins += 1
        
        win_rate = wins / self.hparams.eval_games_vs_random
        self.log('eval/win_rate_vs_random', win_rate, prog_bar=True)
        
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            log.info(f"New best win rate: {win_rate:.1%}")
    
    def configure_optimizers(self):
        """Configure optimizers - maintaining base TRM's special sparse embedding optimizer"""
        optimizers = []
        
        # Main optimizer for all regular parameters (including our new heads)
        all_regular_params = []
        
        # Get base TRM parameters EXCEPT sparse embeddings
        for name, param in self.base_trm.named_parameters():
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
        if hasattr(self.base_trm, 'puzzle_emb') and self.base_trm.puzzle_emb is not None:
            from src.nn.modules.sparse_embeddings import CastedSparseEmbeddingSignSGD_Distributed
            
            # Force sparse embedding to be leaf tensor (same as base TRM)
            self.base_trm.puzzle_emb.local_weights = self.base_trm.puzzle_emb.local_weights.detach().requires_grad_(True)
            
            sparse_opt = CastedSparseEmbeddingSignSGD_Distributed(
                self.base_trm.puzzle_emb.buffers(),
                lr=self.hparams.learning_rate_emb,
                weight_decay=self.hparams.weight_decay,
                world_size=1
            )
            optimizers.append(sparse_opt)
        
        return optimizers