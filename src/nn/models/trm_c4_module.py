"""
TRM Connect Four Module - Clean Implementation with Proper Optimizer Setup
Pure self-play training following base TRM patterns
"""

import torch
import torch.nn.functional as F
from lightning import LightningModule
from collections import deque
from typing import Tuple
import random
import time
import numpy as np
from src.nn.models.trm_module import TRMModule
from src.nn.environments.vectorized_c4_env import VectorizedConnectFour, VectorizedC4State
from src.nn.modules.batch_mcts import BatchMCTS
from src.nn.modules.utils import compute_lr
from src.nn.utils.constants import C4_EMPTY_CELL, C4_PLAYER1_CELL, C4_PLAYER2_CELL
from src.nn.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class TRMConnectFourModule(LightningModule):
    """
    TRM for Connect Four - using native TRM outputs with MCTS
    - Actions: First 7 logits from lm_head (one per column)
    - Value: 8th logit from lm_head as game outcome prediction
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
        learning_rate: float = 2e-3,
        learning_rate_emb: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        lr_min_ratio: float = 0.1,
        
        # Self-play
        games_per_epoch: int = 100,
        buffer_size: int = 100000,
        batch_size: int = 256,
        steps_per_epoch: int = 100,
        
        # MCTS
        mcts_simulations: int = 800,
        mcts_c_puct: float = 1.0,
        mcts_temperature: float = 1.0,
        mcts_dirichlet_alpha: float = 0.3,
        mcts_exploration_fraction: float = 0.25,
        
        # Evaluation
        eval_games_vs_minimax: int = 20,
        
        output_dir: str = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Manual optimization
        self.automatic_optimization = False
        
        # Get board dimensions
        temp_env = VectorizedConnectFour(n_envs=1)
        board_rows = temp_env.rows
        board_cols = temp_env.cols
        del temp_env
        
        # We use vocab_size=8: 
        # 0=padding, 1=empty, 2=player1, 3=player2
        # 4-7=unused (but we'll use position 7 for value prediction)
        vocab_size = 8  # Enough for board tokens + action/value outputs
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
            num_puzzles=1,
            batch_size=batch_size,
            seq_len=seq_len,
            puzzle_emb_dim=0,
            puzzle_emb_len=0,
            output_dir=output_dir,
        )
        
        # Store dimensions
        self.board_rows = board_rows
        self.board_cols = board_cols
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Components
        self.vec_env = None
        self.mcts = None
        
        # Tracking
        self.manual_step = 0
        self.games_played = 0
        self.total_steps = max_steps
    
    def forward(self, boards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: boards -> (policy, value)
        Uses TRM's lm_head directly:
        - First 7 outputs = action probabilities
        - 8th output = value
        
        Args:
            boards: (batch_size, 42) flattened board tensors
            
        Returns:
            policy: (batch_size, 7) move probabilities
            value: (batch_size,) position evaluations
        """
        if boards.dim() == 1:
            boards = boards.unsqueeze(0)
            
        batch_size = boards.size(0)
        
        # Create TRM batch
        batch = {
            'input': boards.long(),
            'output': torch.zeros_like(boards),
            'puzzle_identifiers': torch.zeros(batch_size, dtype=torch.long, device=self.device)
        }
        
        # Forward through TRM
        carry = self.trm.initial_carry(batch)
        carry, outputs = self.trm(carry, batch)
        
        # Get logits from lm_head: (batch_size, seq_len, vocab_size)
        logits = outputs['logits']
        
        # Use the first position's output for policy and value
        # Shape: (batch_size, vocab_size=8)
        first_pos_logits = logits[:, 0, :]
        
        # Policy: First 7 values (columns 0-6)
        policy_logits = first_pos_logits[:, :7]
        policy = F.softmax(policy_logits, dim=-1)
        
        # Value: Use the 8th logit as a value estimate
        # Apply tanh to bound it to [-1, 1]
        value_logit = first_pos_logits[:, 7]
        value = torch.tanh(value_logit)
        
        return policy, value
    
    def setup(self, stage: str):
        """Initialize components"""
        if stage == "fit":
            # Setup TRM
            self.trm.setup(stage)
            
            # Calculate total steps
            if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
                train_loader = self.trainer.datamodule.train_dataloader()
                steps_per_epoch = len(train_loader)
            else:
                steps_per_epoch = self.hparams.steps_per_epoch
            
            if self.trainer.max_epochs > 0:
                computed_total_steps = steps_per_epoch * self.trainer.max_epochs
            else:
                computed_total_steps = float("inf")
            
            if self.trainer.max_steps > 0:
                self.total_steps = min(self.trainer.max_steps, computed_total_steps)
            else:
                self.total_steps = computed_total_steps
            
            # Create environment
            self.vec_env = VectorizedConnectFour(
                n_envs=1,
                device=self.device
            )
            
            # Create MCTS
            self.mcts = BatchMCTS(
                model=self,
                c_puct=self.hparams.mcts_c_puct,
                num_simulations=self.hparams.mcts_simulations,
                dirichlet_alpha=self.hparams.mcts_dirichlet_alpha,
                exploration_fraction=self.hparams.mcts_exploration_fraction,
                device=self.device
            )
    
    def collect_self_play_games(self):
        """Collect self-play games using MCTS - fully vectorized version"""
        if self.mcts is None:
            return
        
        log.info("Collecting self-play games using batched MCTS...")
        # Run multiple games in parallel
        n_parallel = 8
        n_batches = max(1, self.hparams.games_per_epoch // n_parallel)
        n_positions_added = 0

        for batch_idx in range(n_batches):

            # Use a single vectorized environment
            vec_env = VectorizedConnectFour(n_envs=n_parallel, device=self.device)
            states = vec_env.reset()
            trajectories = [[] for _ in range(n_parallel)]
            
            # Play all games until completion
            while not states.is_terminal.all():
                # Prepare boards and legal moves for active games
                boards = []
                legal_moves_list = []
                active_indices = []
                
                for i in range(n_parallel):
                    if not states.is_terminal[i]:
                        boards.append(states.boards[i])
                        legal_moves_list.append(states.legal_moves[i])
                        active_indices.append(i)
                
                if not boards:
                    break
                
                # Determine temperature based on average move count
                avg_moves = sum(len(t) for t in trajectories) / n_parallel
                temperature = self.hparams.mcts_temperature if avg_moves < 30 else 0.1
                
                # Get MCTS policies for all active games in batch
                active_policies = self.mcts.get_action_probs_batch_parallel(
                    boards, 
                    legal_moves_list,
                    temperature=temperature
                )
                
                # Map back to full policy tensor
                policies = torch.zeros(n_parallel, 7, device=self.device)
                for idx, i in enumerate(active_indices):
                    policies[i] = active_policies[idx]
                    
                    # Store position
                    trajectories[i].append({
                        'board': states.boards[i].flatten(),
                        'policy': active_policies[idx],
                        'player': states.current_players[i].item()
                    })
                
                # Select actions for all active games
                actions = torch.zeros(n_parallel, dtype=torch.long, device=self.device)
                for idx, i in enumerate(active_indices):
                    if temperature > 0.1:
                        actions[i] = torch.multinomial(active_policies[idx], 1).item()
                    else:
                        actions[i] = active_policies[idx].argmax().item()
                
                # Step environment
                states = vec_env.step(actions)
            
            # Process completed games
            for i in range(n_parallel):
                if len(trajectories[i]) < 2:
                    continue
                
                winner = states.winners[i].item()
                
                for position in trajectories[i]:
                    if winner == 0:
                        value = 0.0
                    elif winner == position['player']:
                        value = 1.0
                    else:
                        value = -1.0
                    
                    self.replay_buffer.append({
                        'board': position['board'],
                        'policy': position['policy'],
                        'value': value
                    })
                    n_positions_added += 1
            
            self.games_played += n_parallel
        
        log.info(f"Collected {n_positions_added} positions using batched MCTS, replay buffer size: {len(self.replay_buffer)}")
    
    def training_step(self, batch, batch_idx):
        """Training step using TRM's native outputs"""
        # Skip if no data
        if batch['boards'].sum() == 0:
            return {'loss': torch.tensor(0.0, device=self.device)}
        
        if batch_idx % 100 == 0:
            self.debug_training_data(num_samples=15, output_file="debug_training_data.txt")
            self.debug_game_collection()

        boards = batch['boards'].to(self.device)
        target_policies = batch['policies'].to(self.device)
        target_values = batch['values'].to(self.device)
        
        # Forward pass through TRM
        pred_policies, pred_values = self.forward(boards)
        
        # Cross-entropy loss for policy
        policy_loss = -torch.mean(torch.sum(target_policies * torch.log(pred_policies + 1e-8), dim=1))
        
        # MSE loss for value
        value_loss = F.mse_loss(pred_values, target_values)
        
        # L2 regularization on TRM weights
        l2_reg = 0
        for name, param in self.trm.named_parameters():
            # Skip biases and layer norm parameters
            if 'bias' not in name and 'norm' not in name:
                l2_reg += torch.sum(param ** 2)
        

        # Total loss (no extra L2 since TRM already has weight decay)
        total_loss = policy_loss + value_loss + 1e-4 * l2_reg
        
        # Manual optimization
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        
        # Backward
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Update with learning rate schedule
        current_step = self.manual_step
        for opt in opts:
            if hasattr(opt, '_optimizer'):
                base_lr = self.hparams.learning_rate_emb
            else:
                base_lr = self.hparams.learning_rate
            
            lr = compute_lr(
                base_lr=base_lr,
                lr_warmup_steps=self.hparams.warmup_steps,
                lr_min_ratio=self.hparams.lr_min_ratio,
                current_step=current_step,
                total_steps=self.total_steps,
            )
            
            lr = base_lr

            # Update learning rate
            if hasattr(opt, '_optimizer'):
                for param_group in opt._optimizer.param_groups:
                    param_group['lr'] = lr
                opt._optimizer.step()
                opt._optimizer.zero_grad()
            else:
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
                opt.step()
                opt.zero_grad()
        
        self.manual_step += 1
        
        # Logging
        self.log('train/loss', total_loss, prog_bar=True)
        self.log('train/policy_loss', policy_loss, prog_bar=True)
        self.log('train/value_loss', value_loss, prog_bar=True)
        self.log('train/lr', lr, prog_bar=True)
        self.log('train/games_played', float(self.games_played))
        
        return {'loss': total_loss}
    
    def train_dataloader(self):
        """Create a dataloader that manages self-play data"""
        from torch.utils.data import DataLoader, Dataset
        
        class SelfPlayDataset(Dataset):
            def __init__(self, module, steps_per_epoch=100):
                self.module = module
                self.steps_per_epoch = steps_per_epoch
                
            def __len__(self):
                return self.steps_per_epoch
            
            def __getitem__(self, idx):
                assert self.module.mcts is not None, "MCTS not initialized!"

                # Ensure enough data
                min_buffer_size = self.module.hparams.batch_size * 2
                while len(self.module.replay_buffer) < min_buffer_size:                    
                    self.module.collect_self_play_games()
                    print(f"Buffer size: {len(self.module.replay_buffer)}, {min_buffer_size} needed")
                
                # Sample batch
                samples = random.sample(self.module.replay_buffer, self.module.hparams.batch_size)
                
                boards = torch.stack([s['board'] for s in samples])
                policies = torch.stack([s['policy'] for s in samples])
                values = torch.tensor([s['value'] for s in samples], device=self.module.device)
                
                return {
                    'boards': boards,
                    'policies': policies,
                    'values': values
                }
        
        dataset = SelfPlayDataset(self, steps_per_epoch=self.hparams.steps_per_epoch)
        
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
            shuffle=False,
        )
    
    def val_dataloader(self):
        """Validation dataloader"""
        from torch.utils.data import DataLoader, Dataset
        
        class ValidationDataset(Dataset):
            def __init__(self, module, num_batches=10):
                self.module = module
                self.num_batches = num_batches
                
            def __len__(self):
                return self.num_batches
            
            def __getitem__(self, idx):
                if len(self.module.replay_buffer) >= 32:
                    samples = random.sample(self.module.replay_buffer, 32)
                    
                    boards = torch.stack([s['board'] for s in samples])
                    policies = torch.stack([s['policy'] for s in samples])
                    values = torch.tensor([s['value'] for s in samples], device=self.module.device)
                    
                    return {
                        'boards': boards,
                        'policies': policies,
                        'values': values
                    }
                else:
                    return {
                        'boards': torch.zeros(32, 42, device=self.module.device),
                        'policies': torch.ones(32, 7, device=self.module.device) / 7,
                        'values': torch.zeros(32, device=self.module.device)
                    }
        
        dataset = ValidationDataset(self, num_batches=10)
        return DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        if batch['boards'].sum() == 0:
            return
        
        with torch.no_grad():
            boards = batch['boards']
            policy_targets = batch['policies']
            value_targets = batch['values']
            
            pred_policies, pred_values = self.forward(boards)
            
            policy_acc = (pred_policies.argmax(-1) == policy_targets.argmax(-1)).float().mean()
            value_acc = ((pred_values > 0) == (value_targets > 0)).float().mean()
            
            batch_size = boards.size(0)
            self.log('val/policy_accuracy', policy_acc, prog_bar=True, batch_size=batch_size)
            self.log('val/value_accuracy', value_acc, batch_size=batch_size)
    
    def on_train_epoch_start(self):
        """Collect initial games at the start of each epoch if buffer is low"""
        if self.mcts is not None:
            # Ensure minimum buffer size
            min_buffer_size = self.hparams.batch_size * 10
            
            # Clear old data periodically to keep fresh training distribution
            if self.trainer.current_epoch > 0 and self.trainer.current_epoch % 2 == 0:
                old_size = len(self.replay_buffer)
                self.replay_buffer.clear()
                log.info(f"Cleared replay buffer (was {old_size} samples) to refresh training distribution")

            if len(self.replay_buffer) < min_buffer_size:
                log.info(f"Epoch {self.trainer.current_epoch}: Buffer low ({len(self.replay_buffer)}), collecting games...")
                while len(self.replay_buffer) < min_buffer_size:
                    self.collect_self_play_games()
                    print(f"Buffer size: {len(self.replay_buffer)} / {min_buffer_size}")
                log.info(f"Buffer size: {len(self.replay_buffer)}")
    
    def debug_training_data(self, num_samples=10, output_file="debug_training_data.txt"):
        """Debug function to inspect training data quality"""
        import numpy as np
        from datetime import datetime
        
        if len(self.replay_buffer) == 0:
            print("Replay buffer is empty!")
            return
        
        # Sample some data
        import random
        samples = random.sample(self.replay_buffer, min(num_samples, len(self.replay_buffer)))
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Training Data Debug - {datetime.now()}\n")
            f.write(f"Total buffer size: {len(self.replay_buffer)}\n")
            f.write(f"Games played: {self.games_played}\n")
            f.write("=" * 80 + "\n\n")
            
            # Analyze value distribution
            all_values = [s['value'] for s in self.replay_buffer]
            f.write("VALUE DISTRIBUTION:\n")
            f.write(f"  Wins (value=1.0): {sum(1 for v in all_values if v > 0.5)} ({100*sum(1 for v in all_values if v > 0.5)/len(all_values):.1f}%)\n")
            f.write(f"  Draws (value=0.0): {sum(1 for v in all_values if -0.5 < v < 0.5)} ({100*sum(1 for v in all_values if -0.5 < v < 0.5)/len(all_values):.1f}%)\n")
            f.write(f"  Losses (value=-1.0): {sum(1 for v in all_values if v < -0.5)} ({100*sum(1 for v in all_values if v < -0.5)/len(all_values):.1f}%)\n")
            f.write(f"  Mean value: {np.mean(all_values):.3f}\n")
            f.write(f"  Std value: {np.std(all_values):.3f}\n")
            f.write("\n")
            
            # Display some samples
            for i, sample in enumerate(samples):
                f.write(f"SAMPLE {i+1}:\n")
                f.write("-" * 40 + "\n")
                
                # Reshape state to board
                state = sample['board'].cpu().numpy()
                board = state.reshape(6, 7)
                
                # Display board
                f.write("Board State:\n")
                symbols = {C4_EMPTY_CELL: '.', C4_PLAYER1_CELL: 'X', C4_PLAYER2_CELL: 'O'}
                f.write("  0 1 2 3 4 5 6\n")
                f.write("  -------------\n")
                for row in range(6):
                    f.write("| ")
                    for col in range(7):
                        f.write(symbols.get(board[row, col], '?') + " ")
                    f.write("|\n")
                f.write("  -------------\n")
                
                # Count pieces
                empty_count = np.sum(board == C4_EMPTY_CELL)
                p1_count = np.sum(board == C4_PLAYER1_CELL)
                p2_count = np.sum(board == C4_PLAYER2_CELL)
                f.write(f"Pieces: X={p1_count}, O={p2_count}, Empty={empty_count}\n")
                
                # Display policy
                policy = sample['policy'].cpu().numpy()
                f.write("\nPolicy (move probabilities):\n")
                f.write("  Col: ")
                for c in range(7):
                    f.write(f"{c:6d} ")
                f.write("\n")
                f.write("  Prob:")
                for c in range(7):
                    f.write(f"{policy[c]:6.3f} ")
                f.write("\n")
                f.write(f"  Entropy: {-np.sum(policy * np.log(policy + 1e-8)):.3f}\n")
                f.write(f"  Max prob column: {np.argmax(policy)}\n")
                
                # Display value
                value = sample['value']
                f.write(f"\nValue: {value:.3f}")
                if value > 0.5:
                    f.write(" (WIN)")
                elif value < -0.5:
                    f.write(" (LOSS)")
                else:
                    f.write(" (DRAW)")
                f.write("\n")
                
                # Check if board state makes sense
                f.write("\nSanity Checks:\n")
                
                # Check piece count difference
                if abs(p1_count - p2_count) > 1:
                    f.write(f"  ⚠️ WARNING: Piece count difference > 1 (X={p1_count}, O={p2_count})\n")
                else:
                    f.write(f"  ✓ Piece counts look reasonable\n")
                
                # Check if policy focuses on valid moves
                for col in range(7):
                    if board[0, col] != C4_EMPTY_CELL and policy[col] > 0.01:
                        f.write(f"  ⚠️ WARNING: Column {col} is full but has {policy[col]:.3f} probability\n")
                
                # Check for obvious wins that might be missed
                # (You could add more sophisticated win detection here)
                
                f.write("\n" + "=" * 80 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 40 + "\n")
            
            # Game length distribution (inferred from piece counts)
            piece_counts = []
            for s in self.replay_buffer:
                board = s['board'].cpu().numpy().reshape(6, 7)
                pieces = np.sum(board != C4_EMPTY_CELL)
                piece_counts.append(pieces)
            
            f.write(f"Average pieces on board: {np.mean(piece_counts):.1f}\n")
            f.write(f"Min pieces: {np.min(piece_counts)}\n")
            f.write(f"Max pieces: {np.max(piece_counts)}\n")
            
            # Policy entropy distribution
            entropies = []
            for s in self.replay_buffer:
                p = s['policy'].cpu().numpy()
                entropy = -np.sum(p * np.log(p + 1e-8))
                entropies.append(entropy)
            
            f.write(f"\nPolicy entropy:\n")
            f.write(f"  Mean: {np.mean(entropies):.3f}\n")
            f.write(f"  Std: {np.std(entropies):.3f}\n")
            f.write(f"  Min: {np.min(entropies):.3f}\n")
            f.write(f"  Max: {np.max(entropies):.3f}\n")
        
        print(f"Debug data written to {output_file}")

    def debug_game_collection(self, output_file="debug_game.txt"):
        """Debug a single game collection to see what's happening"""
        if self.vec_env is None or self.mcts is None:
            print("Environment not initialized")
            return
        
        with open(output_file, 'w') as f:
            f.write("DEBUGGING SINGLE GAME COLLECTION\n")
            f.write("=" * 80 + "\n\n")
            
            # Reset one environment
            states = self.vec_env.reset()
            game_trajectory = []
            
            move_num = 0
            while not states.is_terminal[0]:
                move_num += 1
                f.write(f"MOVE {move_num}:\n")
                f.write("-" * 40 + "\n")
                
                # Display board
                board = states.boards[0].cpu().numpy()
                f.write("Board:\n")
                symbols = {C4_EMPTY_CELL: '.', C4_PLAYER1_CELL: 'X', C4_PLAYER2_CELL: 'O'}
                for row in range(6):
                    for col in range(7):
                        f.write(symbols.get(board[row, col], '?') + " ")
                    f.write("\n")
                
                f.write(f"Current player: {states.current_players[0].item()}\n")
                f.write(f"Legal moves: {states.legal_moves[0].cpu().numpy()}\n")
                
                # Get policy from MCTS - updated to use new interface
                with torch.no_grad():
                    # Use the new MCTS interface with board and legal_moves separately
                    board_tensor = states.boards[0]
                    legal_moves = states.legal_moves[0]
                    
                    # Determine temperature based on move number
                    temperature = 1.0 if move_num < 30 else 0.1
                    
                    # Get MCTS policy
                    policy = self.mcts.get_action_probs(
                        board_tensor,
                        legal_moves,
                        temperature=temperature
                    )
                
                f.write(f"MCTS policy (temp={temperature:.1f}): {policy.cpu().numpy()}\n")
                f.write(f"  Max probability move: column {policy.argmax().item()} ({policy.max().item():.3f})\n")
                f.write(f"  Policy entropy: {-(policy * torch.log(policy + 1e-8)).sum().item():.3f}\n")
                
                # Select action
                if temperature > 0.1:
                    action = torch.multinomial(policy, 1).item()
                    f.write(f"Selected action (sampled): {action}\n\n")
                else:
                    action = policy.argmax().item()
                    f.write(f"Selected action (greedy): {action}\n\n")
                
                # Store for trajectory
                game_trajectory.append({
                    'board': board.copy(),
                    'policy': policy.cpu().numpy(),
                    'player': states.current_players[0].item(),
                    'action': action,
                    'move_num': move_num
                })
                
                # Make move
                actions = torch.zeros(self.vec_env.n_envs, dtype=torch.long, device=self.device)
                actions[0] = action
                states = self.vec_env.step(actions)
            
            # Final board
            f.write("FINAL BOARD:\n")
            f.write("-" * 40 + "\n")
            board = states.boards[0].cpu().numpy()
            for row in range(6):
                for col in range(7):
                    f.write(symbols.get(board[row, col], '?') + " ")
                f.write("\n")
            
            winner = states.winners[0].item()
            f.write(f"\nWinner: {winner}\n")
            f.write(f"Total moves: {move_num}\n")
            
            # Show what values would be assigned
            f.write("\nVALUES ASSIGNED TO TRAJECTORY:\n")
            f.write("-" * 40 + "\n")
            for i, step in enumerate(game_trajectory):
                if winner == 0:
                    value = 0.0
                    outcome_str = "DRAW"
                elif winner == step['player']:
                    value = 1.0
                    outcome_str = "WIN"
                else:
                    value = -1.0
                    outcome_str = "LOSS"
                
                f.write(f"Move {step['move_num']:2d}: Player {step['player']} played col {step['action']} -> Value {value:+.1f} ({outcome_str})\n")
            
            # Summary stats
            f.write("\n" + "=" * 80 + "\n")
            f.write("SUMMARY:\n")
            f.write(f"  Game length: {len(game_trajectory)} moves\n")
            f.write(f"  Winner: Player {winner if winner > 0 else 'None (draw)'}\n")
            
            # Analyze policy entropy over game
            entropies = [-(step['policy'] * np.log(step['policy'] + 1e-8)).sum() for step in game_trajectory]
            f.write(f"  Average policy entropy: {np.mean(entropies):.3f}\n")
            f.write(f"  Policy entropy range: {np.min(entropies):.3f} - {np.max(entropies):.3f}\n")
        
        print(f"Game debug written to {output_file}")
    
    def on_validation_epoch_end(self):
        """Periodic evaluation against minimax"""
        self.evaluate_vs_minimax()
    
    def evaluate_vs_minimax(self, depth: int = 2):
        """Evaluate against minimax baseline"""
        from src.nn.modules.minimax import ConnectFourMinimax
        
        minimax = ConnectFourMinimax(depth=depth)
        wins = 0
        draws = 0
        
        for game in range(self.hparams.eval_games_vs_minimax):
            # Reset environment
            self.vec_env.reset()
            
            # Alternate who plays first
            model_player = 1 if game % 2 == 0 else 2
            
            while not self.vec_env.is_terminal[0]:
                current_player = self.vec_env.current_players[0].item()
                
                if current_player == model_player:
                    # Model's turn using MCTS
                    board = self.vec_env.boards[0]
                    legal_moves = self.vec_env.boards[0, 0, :] == C4_EMPTY_CELL
                    
                    # Use lower temperature for evaluation (more deterministic)
                    with torch.no_grad():
                        policy = self.mcts.get_action_probs(
                            board,
                            legal_moves,
                            temperature=0.1  # Nearly deterministic
                        )
                    
                    # Select best action
                    action = policy.argmax().item()
                else:
                    # Minimax's turn
                    board_np = self.vec_env.boards[0].cpu().numpy()
                    action = minimax.get_best_move(board_np, current_player)
                
                # Execute action
                actions = torch.zeros(self.vec_env.n_envs, dtype=torch.long, device=self.device)
                actions[0] = action
                self.vec_env.step(actions)
            
            # Check result
            winner = self.vec_env.winners[0].item()
            if winner == model_player:
                wins += 1
            elif winner == 0:
                draws += 1
        
        # Calculate statistics
        total_games = self.hparams.eval_games_vs_minimax
        win_rate = wins / total_games
        draw_rate = draws / total_games
        loss_rate = 1 - win_rate - draw_rate
        
        # Log results
        self.log('eval/win_rate_vs_minimax', win_rate, prog_bar=True)
        self.log('eval/draw_rate_vs_minimax', draw_rate)
        self.log('eval/loss_rate_vs_minimax', loss_rate)
        
        log.info(f"vs Minimax (depth={depth}): Win={win_rate:.1%}, Draw={draw_rate:.1%}, Loss={loss_rate:.1%}")
    
    def configure_optimizers(self):
        """Configure optimizers - maintaining base TRM's special sparse embedding optimizer"""
        optimizers = []
        
        # Main optimizer for all regular parameters (including our new heads)
        all_regular_params = []
        
        # Get base TRM parameters EXCEPT sparse embeddings
        for name, param in self.trm.named_parameters():
            if 'puzzle_emb' not in name:  # Exclude sparse embeddings
                all_regular_params.append(param)
        
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