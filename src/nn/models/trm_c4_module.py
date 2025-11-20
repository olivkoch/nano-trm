"""
TRM Connect Four Module - Using SelfPlayTRMModule for AlphaZero-style training
"""

import torch
import torch.nn.functional as F
from lightning import LightningModule
from collections import deque
from typing import Tuple
import random
import numpy as np

from src.nn.environments.vectorized_c4_env import VectorizedConnectFour
from src.nn.modules.batch_mcts import BatchMCTS
from src.nn.modules.utils import compute_lr
from src.nn.utils.constants import C4_EMPTY_CELL, C4_PLAYER1_CELL, C4_PLAYER2_CELL
from src.nn.utils import RankedLogger

# Import our new SelfPlayTRMModule
from src.nn.models.selfplay_trm_module import SelfPlayTRMModule

log = RankedLogger(__name__, rank_zero_only=True)


class TRMConnectFourModule(LightningModule):
    """
    TRM for Connect Four using SelfPlayTRMModule architecture
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
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        lr_min_ratio: float = 0.1,
        
        # Loss weights
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        halt_weight: float = 0.5,
        
        # DataLoader settings
        num_workers: int = 0,
        max_epochs: int = 10,
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
        minimax_temperature: float = 0.5,
        minimax_depth: int = 2,
        
        output_dir: str = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Manual optimization
        self.automatic_optimization = False
        
        # Get board dimensions
        temp_env = VectorizedConnectFour(n_envs=1)
        self.board_rows = temp_env.rows
        self.board_cols = temp_env.cols
        del temp_env
        
        # Board configuration
        self.vocab_size = 3  # 0: padding, 1: empty, 2: player1
        seq_len = self.board_rows * self.board_cols  # 42
        action_space_size = self.board_cols  # 7
        
        max_steps = max_epochs * steps_per_epoch

        # Create SelfPlayTRMModule
        self.model = SelfPlayTRMModule(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_grid_size=seq_len,
            action_space_size=action_space_size,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            N_supervision=N_supervision,
            N_supervision_val=N_supervision_val,
            ffn_expansion=ffn_expansion,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            halt_exploration_prob=0.1,
            rope_theta=10000,
            lr_min_ratio=lr_min_ratio,
            vocab_size=self.vocab_size,
            seq_len=seq_len,
            policy_weight=policy_weight,
            value_weight=value_weight,
            halt_weight=halt_weight,
            output_dir=output_dir,
        )
        
        # Replay buffer
        self.n_games = self.hparams.steps_per_epoch * self.hparams.batch_size

        assert self.n_games % 8 == 0, "steps_per_epoch x batch_size must be multiple of 8 for parallel collection"

        self.replay_buffer = deque(maxlen=self.n_games)
        
        # Components (initialized in setup)
        self.vec_env = None
        self.mcts = None
        
        # Tracking
        self.manual_step = 0
        self.games_played = 0
        self.total_steps = max_steps
    
    def forward(self, boards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through SelfPlayTRM
        
        Args:
            boards: (batch_size, 42) flattened board tensors with tokens
            
        Returns:
            policy: (batch_size, 7) move probabilities
            value: (batch_size,) position evaluations in [-1, 1]
        """
        if boards.dim() == 1:
            boards = boards.unsqueeze(0)
        
        batch_size = boards.size(0)
        
        assert boards.min() >= 0 and boards.max() < self.vocab_size, \
            f"Board tokens out of range: [{boards.min()}, {boards.max()}], expected [0, {self.vocab_size-1}]"

        # Create valid actions mask (columns that aren't full)
        # Reshape to check top row
        boards_reshaped = boards.view(batch_size, self.board_rows, self.board_cols)
        valid_actions = boards_reshaped[:, 0, :] == C4_EMPTY_CELL  # Top row empty = valid
        
        # Create batch for SelfPlayTRM
        batch = {
            'board_state': boards.long(),
            'valid_actions': valid_actions,
            # Dummy targets for forward pass
            'mcts_policy': torch.zeros(batch_size, self.board_cols, device=boards.device),
            'game_outcome': torch.zeros(batch_size, device=boards.device)
        }
        
        # Initialize carry
        carry = self.model.initial_carry(batch)
        
        # Forward pass
        carry, outputs = self.model.forward(carry, batch)
        
        # Extract policy and value
        policy = outputs['policy']
        value = outputs['value']
        
        return policy, value
    
    def setup(self, stage: str):
        """Initialize components"""
        if stage == "fit":
            # Setup the internal model
            self.model.setup(stage)
            
            # Calculate total steps
            steps_per_epoch = self.hparams.steps_per_epoch
            self.total_steps = steps_per_epoch * self.hparams.max_epochs
            self.model.total_steps = self.total_steps 
            
            # Create environment
            self.vec_env = VectorizedConnectFour(
                n_envs=1,
                device=self.device
            )
            
            # Create MCTS with this module as the model
            self.mcts = BatchMCTS(
                model=self,  # Pass self, which implements forward()
                c_puct=self.hparams.mcts_c_puct,
                num_simulations=self.hparams.mcts_simulations,
                dirichlet_alpha=self.hparams.mcts_dirichlet_alpha,
                exploration_fraction=self.hparams.mcts_exploration_fraction,
                device=self.device
            )
    
    def collect_self_play_games(self, n_games: int):
        """Collect self-play games using MCTS"""
        if self.mcts is None:
            return
        
        log.info("Collecting self-play games using MCTS...")
        n_parallel = 8
        n_batches = max(1, n_games // n_parallel)
        n_positions_added = 0
        
        for batch_idx in range(n_batches):
            vec_env = VectorizedConnectFour(n_envs=n_parallel, device=self.device)
            states = vec_env.reset()
            trajectories = [[] for _ in range(n_parallel)]
            
            while not states.is_terminal.all():
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
                
                # Temperature scheduling
                avg_moves = sum(len(t) for t in trajectories) / n_parallel
                temperature = self.hparams.mcts_temperature if avg_moves < 30 else 0.1
                
                # Get MCTS policies
                active_policies = self.mcts.get_action_probs_batch_parallel(
                    boards, 
                    legal_moves_list,
                    temperature=temperature
                )
                
                # Store positions and select actions
                actions = torch.zeros(n_parallel, dtype=torch.long, device=self.device)
                for idx, i in enumerate(active_indices):
                    trajectories[i].append({
                        'board': states.boards[i].flatten(),
                        'policy': active_policies[idx],
                        'player': states.current_players[i].item()
                    })
                    
                    if temperature > 0.1:
                        actions[i] = torch.multinomial(active_policies[idx], 1).item()
                    else:
                        actions[i] = active_policies[idx].argmax().item()
                
                states = vec_env.step(actions)
            
            # Assign values based on game outcomes
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
        
        log.info(f"Collected {n_positions_added} positions, buffer size: {len(self.replay_buffer)}")
    
    def training_step(self, batch, batch_idx):
        """Training step using SelfPlayTRM"""
        if batch['boards'].sum() == 0:
            return {'loss': torch.tensor(0.0, device=self.device)}
        
        # Prepare batch for SelfPlayTRM
        boards = batch['boards'].to(self.device)
        target_policies = batch['policies'].to(self.device)
        target_values = batch['values'].to(self.device)
        
        if batch_idx % 100 == 0:
            self.debug_training_data(num_samples=15, output_file="debug_training_data.txt")
            self.debug_game_collection()
            
        # Get valid actions mask
        boards_reshaped = boards.view(-1, self.board_rows, self.board_cols)
        valid_actions = boards_reshaped[:, 0, :] == C4_EMPTY_CELL
        
        # Create proper batch format for SelfPlayTRM
        trm_batch = {
            'board_state': boards,
            'mcts_policy': target_policies,
            'game_outcome': target_values,
            'valid_actions': valid_actions
        }
        
        # Initialize or reuse carry
        if not hasattr(self.model, 'carry') or self.model.carry is None:
            self.model.carry = self.model.initial_carry(trm_batch)
        
        # Compute loss using SelfPlayTRM's method
        carry, total_loss, metrics, all_halted = self.model.compute_loss_and_metrics(
            self.model.carry, trm_batch
        )
        self.model.carry = carry
        
        # Scale loss
        batch_size = boards.size(0)
        scaled_loss = total_loss / batch_size
        
        # Manual optimization
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        
        # Backward
        scaled_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Update with learning rate schedule
        current_step = self.manual_step
        for opt in opts:
            # lr = compute_lr(
            #     base_lr=self.hparams.learning_rate,
            #     lr_warmup_steps=self.hparams.warmup_steps,
            #     lr_min_ratio=self.hparams.lr_min_ratio,
            #     current_step=current_step,
            #     total_steps=self.total_steps,
            # )
            
            lr = self.hparams.learning_rate
            for param_group in opt.param_groups:
                param_group['lr'] = lr
            
            opt.step()
            opt.zero_grad()
        
        self.manual_step += 1
        
        # Logging
        if metrics.get('count', 0) > 0:
            count = metrics['count']
            self.log('train/loss', total_loss / batch_size, prog_bar=True)
            self.log('train/policy_loss', metrics['policy_loss'] / batch_size)
            self.log('train/value_loss', metrics['value_loss'] / batch_size)
            self.log('train/policy_accuracy', metrics['policy_accuracy'] / count)
            self.log('train/value_mae', metrics['value_mae'] / count, prog_bar=True)
            self.log('train/halt_steps', metrics['steps'] / count)
            self.log('train/lr', lr)
            self.log('train/games_played', float(self.games_played))
        
        return {'loss': scaled_loss}
    
    def train_dataloader(self):
        """Create dataloader for self-play training"""
        from torch.utils.data import DataLoader, Dataset
        
        class SelfPlayDataset(Dataset):
            def __init__(self, module, steps_per_epoch):
                self.module = module
                self.steps_per_epoch = steps_per_epoch
                
            def __len__(self):
                return self.steps_per_epoch
            
            def __getitem__(self, idx):
                # Sample batch
                samples = random.sample(
                    self.module.replay_buffer, 
                    self.module.hparams.batch_size
                )
                
                boards = torch.stack([s['board'] for s in samples])
                policies = torch.stack([s['policy'] for s in samples])
                values = torch.tensor([s['value'] for s in samples])
                
                return {
                    'boards': boards,
                    'policies': policies,
                    'values': values
                }
        
        dataset = SelfPlayDataset(self, self.hparams.steps_per_epoch)
        return DataLoader(dataset, batch_size=None, num_workers=0, shuffle=False)
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        if batch['boards'].sum() == 0:
            return

        with torch.no_grad():
            boards = batch['boards'].to(self.device)
            policy_targets = batch['policies'].to(self.device)
            value_targets = batch['values'].to(self.device)
            
            # Get predictions
            pred_policies, pred_values = self.forward(boards)
            
            # Metrics
            policy_acc = (pred_policies.argmax(-1) == policy_targets.argmax(-1)).float().mean()
            value_mae = torch.abs(pred_values - value_targets).mean()
            
            self.log('val/policy_accuracy', policy_acc, prog_bar=True)
            self.log('val/value_mae', value_mae, prog_bar=True)

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
                assert(len(self.module.replay_buffer) >= 32)
                samples = random.sample(self.module.replay_buffer, 32)
                
                boards = torch.stack([s['board'] for s in samples])
                policies = torch.stack([s['policy'] for s in samples])
                values = torch.tensor([s['value'] for s in samples])
                
                return {'boards': boards, 'policies': policies, 'values': values}
        
        dataset = ValidationDataset(self)
        return DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
    
    def on_train_epoch_start(self):
        """Refresh training data at epoch start"""
        if self.mcts is not None:            
            self.replay_buffer.clear()
            self.collect_self_play_games_minimax(n_games=self.n_games, depth=2, temp_player1=0.2, temp_player2=0.8)
    
    def on_validation_epoch_end(self):
        """Evaluate against minimax baseline"""
        self.evaluate_vs_minimax_fast()
    
    def evaluate_vs_minimax_fast(self):
        """Fast evaluation using only neural network (no MCTS)"""
        from src.nn.modules.minimax import ConnectFourMinimax
        
        log.info(f"Fast eval vs Minimax (depth={self.hparams.minimax_depth}) and temperature={self.hparams.minimax_temperature} over {self.hparams.eval_games_vs_minimax} games...")
        
        minimax = ConnectFourMinimax(depth=self.hparams.minimax_depth)
        n_games = self.hparams.eval_games_vs_minimax
        n_parallel = 16  # Can handle more parallel games without MCTS
        n_batches = (n_games + n_parallel - 1) // n_parallel
        
        wins = 0
        draws = 0
        
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
                            temperature=self.hparams.minimax_temperature
                        )
                
                # Batch evaluate all model positions
                if active_model_positions:
                    with torch.no_grad():
                        boards_tensor = torch.stack(active_model_positions)
                        policies, _ = self.forward(boards_tensor)
                        
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
        
        self.log('eval/fast_win_rate_vs_minimax', win_rate, prog_bar=True)
        self.log('eval/fast_draw_rate_vs_minimax', draw_rate)
        
        log.info(f"Fast eval vs Minimax: W={win_rate:.1%}, D={draw_rate:.1%}, L={loss_rate:.1%}")
    
    def configure_optimizers(self):
        """Simple optimizer configuration without puzzle embeddings"""
        try:
            from adam_atan2 import AdamATan2
            optimizer = AdamATan2(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.95)
            )
        except ImportError:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.95)
            )
        
        return optimizer
    
    def debug_training_data(self, num_samples=15, output_file="debug_training_data.txt"):
        """Debug function to inspect training data quality"""
        import numpy as np
        from datetime import datetime
        
        if len(self.replay_buffer) == 0:
            print("Replay buffer is empty!")
            return
        
        # Sample some data
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
            
            # Display samples
            for i, sample in enumerate(samples):
                f.write(f"SAMPLE {i+1}:\n")
                f.write("-" * 40 + "\n")
                
                # Reshape board
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
                f.write("\nPolicy (MCTS probabilities):\n")
                f.write("  Col: ")
                for c in range(7):
                    f.write(f"{c:6d} ")
                f.write("\n")
                f.write("  Prob:")
                for c in range(7):
                    f.write(f"{policy[c]:6.3f} ")
                f.write("\n")
                
                # Check valid moves
                valid_cols = []
                for col in range(7):
                    if board[0, col] == C4_EMPTY_CELL:
                        valid_cols.append(col)
                f.write(f"  Valid columns: {valid_cols}\n")
                
                # Policy analysis
                f.write(f"  Entropy: {-np.sum(policy * np.log(policy + 1e-8)):.3f}\n")
                f.write(f"  Max prob column: {np.argmax(policy)}\n")
                
                # Value
                value = sample['value']
                f.write(f"\nValue: {value:.3f}")
                if value > 0.5:
                    f.write(" (WIN)")
                elif value < -0.5:
                    f.write(" (LOSS)")
                else:
                    f.write(" (DRAW)")
                f.write("\n")
                
                # Sanity checks
                f.write("\nSanity Checks:\n")
                
                # Check piece counts
                if abs(p1_count - p2_count) > 1:
                    f.write(f"  ⚠️ WARNING: Piece count difference > 1 (X={p1_count}, O={p2_count})\n")
                else:
                    f.write(f"  ✓ Piece counts look reasonable\n")
                
                # Check if policy assigns probability to invalid moves
                invalid_policy_mass = 0
                for col in range(7):
                    if board[0, col] != C4_EMPTY_CELL and policy[col] > 0.01:
                        f.write(f"  ⚠️ WARNING: Column {col} is full but has {policy[col]:.3f} probability\n")
                        invalid_policy_mass += policy[col]
                
                if invalid_policy_mass < 0.01:
                    f.write(f"  ✓ Policy respects valid moves\n")
                
                # Check for winning positions
                def check_win(board, player):
                    # Horizontal
                    for row in range(6):
                        for col in range(4):
                            if all(board[row, col+i] == player for i in range(4)):
                                return True
                    # Vertical
                    for row in range(3):
                        for col in range(7):
                            if all(board[row+i, col] == player for i in range(4)):
                                return True
                    # Diagonal
                    for row in range(3):
                        for col in range(4):
                            if all(board[row+i, col+i] == player for i in range(4)):
                                return True
                            if all(board[row+i, col+3-i] == player for i in range(4)):
                                return True
                    return False
                
                if check_win(board, C4_PLAYER1_CELL):
                    f.write("  ℹ️ Board shows X win\n")
                elif check_win(board, C4_PLAYER2_CELL):
                    f.write("  ℹ️ Board shows O win\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 40 + "\n")
            
            # Game length distribution
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
            
            # Check policy concentration
            max_probs = []
            for s in self.replay_buffer:
                p = s['policy'].cpu().numpy()
                max_probs.append(np.max(p))
            
            f.write(f"\nPolicy concentration (max probability):\n")
            f.write(f"  Mean: {np.mean(max_probs):.3f}\n")
            f.write(f"  Std: {np.std(max_probs):.3f}\n")
            f.write(f"  Min: {np.min(max_probs):.3f}\n")
            f.write(f"  Max: {np.max(max_probs):.3f}\n")
        
        print(f"Debug data written to {output_file}")

    def debug_game_collection(self, output_file="debug_game.txt"):
        """Debug a single game collection to see MCTS in action"""
        if self.vec_env is None or self.mcts is None:
            print("Environment or MCTS not initialized")
            return
        
        with open(output_file, 'w') as f:
            f.write("DEBUGGING SINGLE GAME WITH MCTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Play one game
            env = VectorizedConnectFour(n_envs=1, device=self.device)
            state = env.reset()
            game_trajectory = []
            
            move_num = 0
            while not state.is_terminal[0]:
                move_num += 1
                f.write(f"MOVE {move_num}:\n")
                f.write("-" * 40 + "\n")
                
                # Display board
                board = state.boards[0].cpu().numpy()
                f.write("Board:\n")
                symbols = {C4_EMPTY_CELL: '.', C4_PLAYER1_CELL: 'X', C4_PLAYER2_CELL: 'O'}
                for row in range(6):
                    for col in range(7):
                        f.write(symbols.get(board[row, col], '?') + " ")
                    f.write("\n")
                
                f.write(f"Current player: {state.current_players[0].item()}\n")
                f.write(f"Legal moves: {[i for i in range(7) if state.legal_moves[0][i].item()]}\n")
                
                # Get MCTS policy
                temperature = 1.0 if move_num < 30 else 0.1
                
                with torch.no_grad():
                    # Get raw network output
                    nn_policy, nn_value = self.forward(state.boards[0].flatten())
                    f.write(f"Neural network raw output:\n")
                    f.write(f"  NN Policy: {nn_policy.cpu().numpy()}\n")
                    f.write(f"  NN Value: {nn_value.item():.3f}\n")
                    
                    # Get MCTS-improved policy
                    mcts_policy = self.mcts.get_action_probs(
                        state.boards[0],
                        state.legal_moves[0],
                        temperature=temperature
                    )
                
                f.write(f"MCTS policy (temp={temperature:.1f}): {mcts_policy.cpu().numpy()}\n")
                f.write(f"  Max probability move: column {mcts_policy.argmax().item()} ({mcts_policy.max().item():.3f})\n")
                f.write(f"  Policy entropy: {-(mcts_policy * torch.log(mcts_policy + 1e-8)).sum().item():.3f}\n")
                
                # Compare NN vs MCTS
                nn_choice = nn_policy.argmax().item()
                mcts_choice = mcts_policy.argmax().item()
                if nn_choice != mcts_choice:
                    f.write(f"  📊 MCTS changed decision: NN chose {nn_choice}, MCTS chose {mcts_choice}\n")
                
                # Select action
                if temperature > 0.1:
                    action = torch.multinomial(mcts_policy, 1).item()
                    f.write(f"Selected action (sampled): {action}\n\n")
                else:
                    action = mcts_policy.argmax().item()
                    f.write(f"Selected action (greedy): {action}\n\n")
                
                # Store for trajectory
                game_trajectory.append({
                    'board': board.copy(),
                    'nn_policy': nn_policy.cpu().numpy(),
                    'mcts_policy': mcts_policy.cpu().numpy(),
                    'nn_value': nn_value.item(),
                    'player': state.current_players[0].item(),
                    'action': action,
                    'move_num': move_num
                })
                
                # Make move
                actions = torch.tensor([action], device=self.device)
                state = env.step(actions)
            
            # Final board
            f.write("FINAL BOARD:\n")
            f.write("-" * 40 + "\n")
            board = state.boards[0].cpu().numpy()
            for row in range(6):
                for col in range(7):
                    f.write(symbols.get(board[row, col], '?') + " ")
                f.write("\n")
            
            winner = state.winners[0].item()
            f.write(f"\nWinner: ")
            if winner == 1:
                f.write("X (Player 1)\n")
            elif winner == 2:
                f.write("O (Player 2)\n")
            else:
                f.write("Draw\n")
            f.write(f"Total moves: {move_num}\n")
            
            # Trajectory analysis
            f.write("\nTRAJECTORY ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            for i, step in enumerate(game_trajectory):
                if winner == 0:
                    value = 0.0
                    outcome = "DRAW"
                elif winner == step['player']:
                    value = 1.0
                    outcome = "WIN"
                else:
                    value = -1.0
                    outcome = "LOSS"
                
                f.write(f"Move {step['move_num']:2d}: P{step['player']} played {step['action']} ")
                f.write(f"-> True value {value:+.1f} ({outcome}), ")
                f.write(f"NN predicted {step['nn_value']:+.3f}\n")
                
                # Check if NN value prediction was accurate
                if abs(step['nn_value'] - value) < 0.3:
                    f.write(f"  ✓ Good value prediction\n")
                else:
                    f.write(f"  ⚠️ Value prediction error: {abs(step['nn_value'] - value):.3f}\n")
            
            # Summary
            f.write("\n" + "=" * 80 + "\n")
            f.write("SUMMARY:\n")
            f.write(f"  Game length: {len(game_trajectory)} moves\n")
            f.write(f"  Winner: ")
            if winner == 1:
                f.write("X\n")
            elif winner == 2:
                f.write("O\n")
            else:
                f.write("Draw\n")
            
            # Analyze improvements by MCTS
            nn_entropies = []
            mcts_entropies = []
            policy_changes = 0
            
            for step in game_trajectory:
                nn_ent = -(step['nn_policy'] * np.log(step['nn_policy'] + 1e-8)).sum()
                mcts_ent = -(step['mcts_policy'] * np.log(step['mcts_policy'] + 1e-8)).sum()
                nn_entropies.append(nn_ent)
                mcts_entropies.append(mcts_ent)
                
                if np.argmax(step['nn_policy']) != np.argmax(step['mcts_policy']):
                    policy_changes += 1
            
            f.write(f"\nMCTS Impact:\n")
            f.write(f"  Moves where MCTS changed decision: {policy_changes}/{len(game_trajectory)} ({100*policy_changes/len(game_trajectory):.1f}%)\n")
            f.write(f"  Avg NN policy entropy: {np.mean(nn_entropies):.3f}\n")
            f.write(f"  Avg MCTS policy entropy: {np.mean(mcts_entropies):.3f}\n")
            
            # Value prediction accuracy
            value_errors = []
            for step in game_trajectory:
                if winner == 0:
                    true_value = 0.0
                elif winner == step['player']:
                    true_value = 1.0
                else:
                    true_value = -1.0
                value_errors.append(abs(step['nn_value'] - true_value))
            
            f.write(f"\nValue Prediction Accuracy:\n")
            f.write(f"  Mean absolute error: {np.mean(value_errors):.3f}\n")
            f.write(f"  Max error: {np.max(value_errors):.3f}\n")
            f.write(f"  Correct predictions (error < 0.3): {sum(1 for e in value_errors if e < 0.3)}/{len(value_errors)}\n")
        
        print(f"Game debug written to {output_file}")

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
        
        log.info(f"Collecting self-play {n_games} games using Minimax (depth={depth}, P1_temp={temp_player1}, P2_temp={temp_player2})...")
        
        # Create minimax player
        minimax = ConnectFourMinimax(depth=depth)
        
        # Run multiple games in parallel
        n_parallel = 8
        n_batches = max(1, n_games // n_parallel)
        n_positions_added = 0
        
        for batch_idx in range(n_batches):
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
                    policy = self._create_minimax_policy(
                        board_np, 
                        current_player, 
                        states.legal_moves[i].cpu().numpy(),
                        minimax, 
                        temperature
                    )
                    
                    # Store position
                    trajectories[i].append({
                        'board': states.boards[i].flatten(),
                        'policy': torch.tensor(policy, device=self.device, dtype=torch.float32),
                        'player': current_player
                    })
                
                # Step environment
                states = vec_env.step(actions)
            
            # Process completed games and assign values
            for i in range(n_parallel):
                if len(trajectories[i]) < 2:  # Skip very short games
                    continue
                
                winner = states.winners[i].item()
                
                for position in trajectories[i]:
                    # Assign value based on game outcome from this player's perspective
                    if winner == 0:
                        value = 0.0  # Draw
                    elif winner == position['player']:
                        value = 1.0  # Win
                    else:
                        value = -1.0  # Loss
                    
                    self.replay_buffer.append({
                        'board': position['board'],
                        'policy': position['policy'],
                        'value': value
                    })
                    n_positions_added += 1
            
            self.games_played += n_parallel
        
        log.info(f"Collected {n_positions_added} positions using Minimax (depth={depth}), replay buffer size: {len(self.replay_buffer)}")

    def _create_minimax_policy(self, board_np, current_player, legal_moves, minimax, temperature):
        """
        Create a policy distribution from minimax evaluations.
        
        Instead of a one-hot vector, this creates a probability distribution
        where better moves (according to minimax) get higher probabilities.
        """
        import numpy as np
        
        policy = np.zeros(7)
        
        if temperature == 0.0:
            # Deterministic: one-hot for best move
            best_action = minimax.get_best_move(board_np, current_player, temperature=0.0)
            policy[best_action] = 1.0
        else:
            # Get scores for all valid moves
            move_scores = []
            valid_actions = []
            
            for col in range(7):
                if legal_moves[col]:
                    # Make the move
                    temp_board = board_np.copy()
                    for row in range(5, -1, -1):
                        if temp_board[row, col] == C4_EMPTY_CELL:
                            temp_board[row, col] = current_player
                            break
                    
                    # Evaluate position after this move
                    # Note: minimax returns score from current player's perspective
                    score = minimax.minimax(
                        temp_board,
                        depth=minimax.depth - 1,
                        alpha=-float('inf'),
                        beta=float('inf'),
                        maximizing=False,  # Opponent's turn next
                        player=current_player
                    )
                    
                    move_scores.append(score)
                    valid_actions.append(col)
            
            if move_scores:
                # Convert scores to probabilities using softmax with temperature
                move_scores = np.array(move_scores)
                
                # Normalize scores to prevent overflow in exp
                move_scores = move_scores - np.max(move_scores)
                
                # Apply temperature and softmax
                exp_scores = np.exp(move_scores / (temperature + 0.1))  # Add small value to prevent division by zero
                probabilities = exp_scores / np.sum(exp_scores)
                
                # Assign probabilities to valid actions
                for action, prob in zip(valid_actions, probabilities):
                    policy[action] = prob
        
        return policy