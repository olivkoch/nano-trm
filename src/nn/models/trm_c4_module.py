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

from src.nn.models.trm_module import TRMModule
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
        puzzle_emb_dim: int = 0,  # DISABLE puzzle embeddings for Connect Four
        puzzle_emb_len: int = 0,  # We don't need puzzle-specific embeddings for C4
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
        eval_games_vs_baseline: int = 20,
        eval_every_n_val_epochs: int = 1,  # NEW: How often to run game evaluation during validation
        
        output_dir: str = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # CRITICAL: Manual optimization (same as base TRM)
        self.automatic_optimization = False
        
        # Create base TRM with proper initialization
        # Important: Set batch_size for sparse embeddings initialization
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
            batch_size=batch_size,  # Critical for sparse embeddings
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
        self.vec_env = None  # Vectorized environment for all games
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
        
        batch_size = state_tensor.shape[0]
        
        # Create batch for TRM with proper dimensions
        batch = {
            'input': state_tensor.to(self.device),
            'output': torch.zeros_like(state_tensor).to(self.device),
            'puzzle_identifiers': torch.zeros(batch_size, dtype=torch.long, device=self.device)
        }
        
        # Initialize carry and run TRM
        carry = self.base_trm.initial_carry(batch)
        carry, trm_outputs = self.base_trm(carry, batch)
        
        # Extract hidden state from first token position
        # When puzzle_emb_len = 0, we use the first position directly
        if hasattr(self.base_trm, 'puzzle_emb_len') and self.base_trm.puzzle_emb_len > 0:
            # Use the first position after puzzle embeddings
            hidden_state = carry.inner_carry.z_H[:, self.base_trm.puzzle_emb_len]
        else:
            # No puzzle embeddings, use first position
            hidden_state = carry.inner_carry.z_H[:, 0]
        
        # Ensure hidden_state has correct shape
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
        
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
        # Import here to avoid circular dependencies
        from src.nn.modules.batch_mcts import BatchMCTS
        from src.nn.environments.vectorized_c4_env import VectorizedConnectFour
        
        # Create vectorized environment (even for single games)
        self.vec_env = VectorizedConnectFour(
            n_envs=max(8, self.hparams.games_per_iteration),
            device=self.device
        )
        
        # Setup Batch MCTS (used for vectorized play and evaluation)
        # Note: Individual games in self_play_game create their own MCTS
        # with adaptive parameters
        self.mcts = BatchMCTS(
            trm_model=self,
            c_puct=self.hparams.mcts_c_puct,
            num_simulations=self.hparams.mcts_simulations,
            batch_size=8,  # Evaluate 8 positions at once
            device=self.device,
            temperature=self.hparams.mcts_temperature,
            use_compile=True  # Enable torch.compile
        )

        if stage == "fit":
           
            # Call base TRM setup
            self.base_trm.setup(stage)
            
            # Compile the get_policy_value method for faster inference
            if hasattr(torch, 'compile'):
                try:
                    # Note: We compile the underlying implementation, not the method directly
                    log.info("Compiling model with torch.compile for faster inference...")
                    compile_mode = "reduce-overhead" if self.device.type == "cuda" else "default"
                    
                    # Compile the forward pass of base TRM
                    self.base_trm.forward = torch.compile(
                        self.base_trm.forward,
                        mode=compile_mode,
                        disable=self.device.type == "mps"  # Disable on MPS as it may not be supported
                    )
                    log.info("Model compilation successful")
                except Exception as e:
                    log.info(f"Could not compile model (will run uncompiled): {e}")
    
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
    
    def self_play_game(self, temperature: float = 1.0, verbose: bool = False) -> List[GameSample]:
        """Play one self-play game"""
        print(f"Running self-play game with temperature={temperature}")
        # Reset first slot of vectorized environment
        self.vec_env.reset(env_ids=torch.tensor([0], device=self.device))
        game_history = []
        
        # Adaptive MCTS simulations - fewer at the start of training
        adaptive_sims = min(
            self.hparams.mcts_simulations,
            5 + self.total_games_played // 100  # Start with 5, increase gradually
        )
        
        # Create a game-specific MCTS instance with adaptive parameters
        # Note: We don't reuse self.mcts because:
        # 1. Each game may need different temperature
        # 2. Adaptive simulations change per game
        # 3. Allows parallel game generation in future
        from src.nn.modules.batch_mcts import BatchMCTS
        game_mcts = BatchMCTS(
            self,
            c_puct=self.hparams.mcts_c_puct,
            num_simulations=adaptive_sims,
            batch_size=min(8, adaptive_sims),
            device=self.device,
            temperature=temperature,  # Game-specific temperature
            use_compile=False  # Already compiled in setup
        )
        
        while not self.vec_env.is_terminal[0]:
            # Get current state for first environment only
            from src.nn.environments.vectorized_c4_env import VectorizedC4State

            state = VectorizedC4State(
                boards=self.vec_env.boards[:1],
                current_players=self.vec_env.current_players[:1],
                move_counts=self.vec_env.move_counts[:1],
                winners=self.vec_env.winners[:1],
                is_terminal=self.vec_env.is_terminal[:1],
                legal_moves=(self.vec_env.boards[:1, 0, :] == 0)  # Top row empty = legal
            )
            
            # Get MCTS policy using the adaptive mcts
            policy = game_mcts.get_action_probabilities(state).to(self.device)
            
            # Store position
            state_tensor = state.to_trm_input()[0]  # Keep on device (MPS)
            game_history.append({
                'state': state_tensor,
                'policy': policy,
                'player': state.current_players[0].item(),
                'move_count': state.move_counts[0].item()
            })
            
            # Select and make move
            if temperature > 0:
                action = torch.multinomial(policy, 1).item()
            else:
                action = torch.argmax(policy).item()
            
            # Step only first environment
            actions = torch.zeros(self.vec_env.n_envs, dtype=torch.long, device=self.device)
            actions[0] = action
            self.vec_env.step(actions)

            if verbose:
                print(self.vec_env.render(env_id=0))
        
        # Determine outcome
        winner = self.vec_env.winners[0].item()
        if winner == 1:
            outcomes = {1: 1.0, 2: -1.0}
        elif winner == 2:
            outcomes = {1: -1.0, 2: 1.0}
        else:
            outcomes = {1: 0.0, 2: 0.0}
        
        # Convert to training samples - keep on device
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
        
        # Use vectorized or sequential based on number of games
        if self.hparams.games_per_iteration >= 8:
            # Vectorized self-play for many games
            self.vectorized_self_play()
        else:
            # Sequential for few games
            use_fast_policy = self.trainer.current_epoch < 2
            
            for game_idx in range(self.hparams.games_per_iteration):
                temp = self.hparams.mcts_temperature if self.total_games_played % 4 != 0 else 0.5
                
                if use_fast_policy and game_idx > self.hparams.games_per_iteration // 2:
                    samples = self.fast_self_play_game(temperature=temp)
                else:
                    samples = self.self_play_game(temperature=temp)
                
                self.replay_buffer.extend(samples)
                self.total_games_played += 1
        
        log.info(f"Buffer size: {len(self.replay_buffer)} positions")
    
    def vectorized_self_play(self):
        """Generate multiple games in parallel using vectorized environment"""
        from src.nn.modules.batch_mcts import BatchMCTS
        
        n_parallel = min(8, self.hparams.games_per_iteration)
        n_batches = (self.hparams.games_per_iteration + n_parallel - 1) // n_parallel
        
        for batch_idx in range(n_batches):
            n_games = min(n_parallel, self.hparams.games_per_iteration - batch_idx * n_parallel)
            
            # Reset environments for this batch (only first n_games if needed)
            if n_games < self.vec_env.n_envs:
                # Only use first n_games environments
                states = self.vec_env.get_state()
                # Mark unused environments as terminal so we ignore them
                self.vec_env.is_terminal[n_games:] = True
                states = self.vec_env.get_state()
            else:
                states = self.vec_env.reset()
            
            # Storage for game histories
            game_histories = [[] for _ in range(n_games)]
            
            # Play games until all are terminal (only first n_games)
            while not states.is_terminal[:n_games].all():
                active = ~states.is_terminal[:n_games]
                
                if not active.any():
                    break
                
                # Get policies for all active games in parallel
                active_indices = active.nonzero(as_tuple=True)[0]
                
                if len(active_indices) > 0:
                    # Prepare states for active games
                    active_states = states.boards[active_indices]
                    state_tensors = (active_states.flatten(1) + 1).long()  # Keep on device
                    legal_moves = states.legal_moves[active_indices]
                    
                    # Batch evaluate with MCTS or direct policy
                    if self.trainer.current_epoch < 2 and batch_idx % 2 == 0:
                        # Fast: Use direct policy for early training
                        # Note: get_policy_value_batch is the batch version from BatchMCTS
                        if hasattr(self.mcts, 'get_policy_value_batch'):
                            policies, values = self.mcts.get_policy_value_batch(state_tensors, legal_moves)
                            # Keep policies on device
                        else:
                            # Fallback to individual evaluation
                            policies = []
                            for i in range(len(state_tensors)):
                                policy, _ = self.get_policy_value(state_tensors[i], legal_moves[i])
                                policies.append(policy)  # Keep on device
                            policies = torch.stack(policies)
                        
                        # Select actions
                        if self.hparams.mcts_temperature > 0:
                            actions_active = torch.multinomial(policies, 1).squeeze(-1)
                        else:
                            actions_active = torch.argmax(policies, dim=-1)
                    else:
                        # Full MCTS for better policies
                        policies = []
                        actions_active = []
                        
                        # Process each active game (could be further optimized)
                        for i, env_id in enumerate(active_indices):
                            # Create single state for MCTS using vectorized format
                            from src.nn.environments.vectorized_c4_env import VectorizedC4State
                            single_state = VectorizedC4State(
                                boards=states.boards[env_id:env_id+1],
                                current_players=states.current_players[env_id:env_id+1],
                                move_counts=states.move_counts[env_id:env_id+1],
                                winners=states.winners[env_id:env_id+1],
                                is_terminal=states.is_terminal[env_id:env_id+1],
                                legal_moves=states.legal_moves[env_id:env_id+1]
                            )
                            
                            # Get MCTS policy
                            policy = self.mcts.get_action_probabilities(single_state).to(self.device)
                            policies.append(policy)  # Keep on device
                            
                            # Select action
                            if self.hparams.mcts_temperature > 0:
                                action = torch.multinomial(policy, 1).item()
                            else:
                                action = torch.argmax(policy).item()
                            actions_active.append(action)
                        
                        policies = torch.stack(policies) if policies else torch.zeros((0, 7), device=self.device)
                        actions_active = torch.tensor(actions_active, device=self.device)
                    
                    # Store positions (only for games we're tracking)
                    for i, env_id in enumerate(active_indices):
                        if env_id < n_games:  # Safety check
                            game_histories[env_id].append({
                                'state': state_tensors[i],
                                'policy': policies[i],
                                'player': states.current_players[env_id].item(),
                                'move_count': states.move_counts[env_id].item()
                            })
                
                # Create full action tensor for all environments
                actions = torch.zeros(self.vec_env.n_envs, dtype=torch.long, device=self.device)
                if len(active_indices) > 0:
                    actions[active_indices] = actions_active
                
                # Step all environments
                states = self.vec_env.step(actions)
            
            # Process completed games and add to replay buffer
            for env_id in range(n_games):
                if len(game_histories[env_id]) == 0:
                    continue
                
                # Determine outcome
                winner = states.winners[env_id].item()
                if winner == 1:
                    outcomes = {1: 1.0, 2: -1.0}
                elif winner == 2:
                    outcomes = {1: -1.0, 2: 1.0}
                else:
                    outcomes = {1: 0.0, 2: 0.0}
                
                # Create samples
                samples = []
                for position in game_histories[env_id]:
                    value = outcomes[position['player']]
                    samples.append(GameSample(
                        state=position['state'],
                        policy_target=position['policy'],
                        value_target=value,
                        move_count=position['move_count']
                    ))
                
                self.replay_buffer.extend(samples)
                self.total_games_played += 1
    
    def fast_self_play_game(self, temperature: float = 1.0) -> List[GameSample]:
        """Fast self-play using just the policy network without MCTS"""
        # Reset first slot of vectorized environment
        self.vec_env.reset(env_ids=torch.tensor([0], device=self.device))
        game_history = []
        
        while not self.vec_env.is_terminal[0]:
            # Get current state tensor (keep on device)
            state_tensor = (self.vec_env.boards[0].flatten() + 1).long()
            legal_moves = (self.vec_env.boards[0, 0, :] == 0)  # Keep on device
            
            # Get policy directly from network (no MCTS)
            policy, value = self.get_policy_value(state_tensor, legal_moves)
            # Keep policy on device
            
            # Store position (both state and policy are on device)
            game_history.append({
                'state': state_tensor,  # On device
                'policy': policy,  # On device
                'player': self.vec_env.current_players[0].item(),
                'move_count': self.vec_env.move_counts[0].item()
            })
            
            # Select move
            if temperature > 0:
                action = torch.multinomial(policy, 1).item()
            else:
                action = torch.argmax(policy).item()
            
            # Step only first environment
            actions = torch.zeros(self.vec_env.n_envs, dtype=torch.long, device=self.device)
            actions[0] = action
            self.vec_env.step(actions)
        
        # Convert to samples (same as regular self_play_game)
        winner = self.vec_env.winners[0].item()
        if winner == 1:
            outcomes = {1: 1.0, 2: -1.0}
        elif winner == 2:
            outcomes = {1: -1.0, 2: 1.0}
        else:
            outcomes = {1: 0.0, 2: 0.0}
        
        samples = []
        for position in game_history:
            value = outcomes[position['player']]
            samples.append(GameSample(
                state=position['state'],
                policy_target=position['policy'],
                value_target=value,
                move_count=position['move_count']
            ))
        
        return samples
    
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
        else:
            # Not enough samples yet, skip this training step
            return {'loss': torch.tensor(0.0, device=self.device)}
                
        # Convert to batch - tensors should already be on device
        states = torch.stack([s.state for s in samples])

        policy_targets = torch.stack([s.policy_target for s in samples])
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
        
        # log.info(f"Training Step {self.global_step}: {policy_loss.item()=} {value_loss.item()=} {trm_loss.item()=} {total_loss.item()=}")

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
        
        return {'loss': total_loss}
    
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
            self.evaluate_vs_minimax()
        else:
            # Log that we skipped evaluation this validation epoch
            self.log('val/eval_skipped', 1.0)
    
    def evaluate_vs_minimax(self, minimax_depth: int = 4):
        """
        Evaluate model against minimax player.
        
        Args:
            self: The Lightning module with vec_env and mcts
            minimax_depth: Search depth for minimax (default 4)
        
        This function can be added as a method to your Lightning module,
        just like evaluate_vs_random.
        """
        if self.mcts is None:
            return
        
        # Create minimax player
        from src.nn.modules.minimax import SimpleMinimaxPlayer
        minimax_player = SimpleMinimaxPlayer(depth=minimax_depth)
        
        wins = 0
        total_nodes = 0
        
        for game in range(self.hparams.eval_games_vs_baseline):  # Use same config
            # Reset first environment
            self.vec_env.reset(env_ids=torch.tensor([0], device=self.device))
            
            # Alternate who goes first
            model_player = 1 if game % 2 == 0 else 2
            
            move_count = 0
            
            while not self.vec_env.is_terminal[0]:
                current_player = self.vec_env.current_players[0].item()
                
                if current_player == model_player:
                    # Model's turn using MCTS
                    from src.nn.environments.vectorized_c4_env import VectorizedC4State
                    state = VectorizedC4State(
                        boards=self.vec_env.boards[:1],
                        current_players=self.vec_env.current_players[:1],
                        move_counts=self.vec_env.move_counts[:1],
                        winners=self.vec_env.winners[:1],
                        is_terminal=self.vec_env.is_terminal[:1],
                        legal_moves=(self.vec_env.boards[:1, 0, :] == 0)
                    )
                    
                    action = self.mcts.select_action(state, deterministic=True)
                    
                else:
                    # Minimax player's turn
                    # Create board representation for minimax
                    board = minimax_player.create_board_from_env(self.vec_env, env_idx=0)
                    
                    # Get minimax move
                    action = minimax_player.get_best_move(board, current_player)
                    total_nodes += minimax_player.nodes_evaluated
                    
                    # Fallback to random if something goes wrong
                    if action == -1:
                        legal_moves = self.vec_env.boards[0, 0, :] == 0
                        legal_actions = legal_moves.nonzero(as_tuple=True)[0]
                        if len(legal_actions) > 0:
                            action = legal_actions[torch.randint(len(legal_actions), (1,))].item()
                
                # Make the move
                actions = torch.zeros(self.vec_env.n_envs, dtype=torch.long, device=self.device)
                actions[0] = action
                self.vec_env.step(actions)
                
                move_count += 1
                if move_count > 42:  # Safety check
                    break
            
            # Check winner
            winner = self.vec_env.winners[0].item()
            if winner == model_player:
                wins += 1
        
        # Calculate and log results
        win_rate = wins / self.hparams.eval_games_vs_baseline
        avg_nodes = total_nodes / self.hparams.eval_games_vs_baseline
        
        # Log metrics
        self.log(f'eval/win_rate_vs_minimax_d{minimax_depth}', win_rate, prog_bar=True)
        self.log(f'eval/minimax_avg_nodes', avg_nodes)
        
        # Track best performance
        if not hasattr(self, 'best_minimax_win_rate'):
            self.best_minimax_win_rate = 0.0
        
        if win_rate > self.best_minimax_win_rate:
            self.best_minimax_win_rate = win_rate
            import logging
            log = logging.getLogger(__name__)
            log.info(f"New best win rate vs minimax (depth={minimax_depth}): {win_rate:.1%}")
        
        return win_rate
        
    def evaluate_vs_random(self):
        """Evaluate model against random player"""
        wins = 0
        for game in range(self.hparams.eval_games_vs_baseline):
            # Reset first slot of vectorized environment
            self.vec_env.reset(env_ids=torch.tensor([0], device=self.device))
            
            model_player = 1 if game % 2 == 0 else 2
            
            while not self.vec_env.is_terminal[0]:
                current_player = self.vec_env.current_players[0].item()
                
                if current_player == model_player:
                    # Create state for MCTS
                    from src.nn.environments.vectorized_c4_env import VectorizedC4State
                    state = VectorizedC4State(
                        boards=self.vec_env.boards[:1],
                        current_players=self.vec_env.current_players[:1],
                        move_counts=self.vec_env.move_counts[:1],
                        winners=self.vec_env.winners[:1],
                        is_terminal=self.vec_env.is_terminal[:1],
                        legal_moves=(self.vec_env.boards[:1, 0, :] == 0)  # Top row empty = legal
                    )
                    action = self.mcts.select_action(state, deterministic=True)
                else:
                    # Random player
                    legal_moves = self.vec_env.boards[0, 0, :] == 0  # Top row empty = legal
                    legal_actions = legal_moves.nonzero(as_tuple=True)[0]
                    action = legal_actions[torch.randint(len(legal_actions), (1,))].item()
                
                # Make move
                actions = torch.zeros(self.vec_env.n_envs, dtype=torch.long, device=self.device)
                actions[0] = action
                self.vec_env.step(actions)
            
            # Check winner
            winner = self.vec_env.winners[0].item()
            if winner == model_player:
                wins += 1
        
        win_rate = wins / self.hparams.eval_games_vs_baseline
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
    
if __name__ == "__main__":
    # Simple test to instantiate the model
    model = TRMConnectFourModule(eval_games_vs_baseline=1)
    log.info("Model instantiated successfully")
    model.setup(stage="test")
    print(f"running a game")
    model.self_play_game(verbose=True)