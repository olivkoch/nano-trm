"""
Base class for Connect Four models with self-play and evaluation capabilities
"""

import time
import pickle
from typing import Dict, Tuple
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.utils.data import Dataset, DataLoader

from src.nn.utils.constants import C4_EMPTY_CELL
from src.nn.modules.utils import CircularBuffer
from src.nn.modules.tensor_mcts import TensorMCTSWrapper
from src.nn.modules.minimax import ConnectFourMinimax
from src.nn.environments.vectorized_c4_env import VectorizedConnectFour

from src.nn.utils import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)


class MCTSModelWrapper:
    """Wrapper to make models compatible with MCTS interface"""
    
    def __init__(self, model):
        self.model = model
        
    def forward(self, boards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """MCTS expects (batch, 42) -> (batch, 7), (batch,)"""            
        batch = {
            'boards': boards,
            'puzzle_identifiers': torch.zeros(boards.shape[0], dtype=torch.long, device=boards.device)
        }
        
        with torch.no_grad():
            outputs = self.model.forward_for_mcts(batch)
        
        return outputs['policy'], outputs['value']


class SelfPlayDataset(Dataset):
    """Dataset for self-play training"""
    
    def __init__(self, module, batch_size, steps_per_epoch):
        self.module = module
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        
    def __len__(self):
        return self.steps_per_epoch
    
    def __getitem__(self, idx):
        # Sample from buffer
        buffer = self.module.replay_buffer
        samples = buffer.sample(self.batch_size, replace=len(buffer) < self.batch_size)

        boards = torch.stack([s['board'] for s in samples])
        policies = torch.stack([s['policy'] for s in samples])
        values = torch.tensor([s['value'] for s in samples], dtype=torch.float32)
        puzzle_identifiers = torch.zeros(self.batch_size, dtype=torch.long)
        
        return {
            'boards': boards,
            'policies': policies,
            'values': values,
            'puzzle_identifiers': puzzle_identifiers
        }


class C4BaseModule(LightningModule):
    """
    Base class for Connect Four models with shared self-play and evaluation functionality
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        # Store all hyperparameters
        self.save_hyperparameters()
        
        # Common attributes
        self.board_rows = 6
        self.board_cols = 7
        self.board_size = self.board_rows * self.board_cols
        
        # Manual optimization
        self.automatic_optimization = False
        self.manual_step = 0
        self.games_played = 0
        self.max_steps = self.hparams.steps_per_epoch * self.hparams.max_epochs
        
        # Self-play attributes
        if self.hparams.get('enable_selfplay', False):
            self.replay_buffer = CircularBuffer(maxlen=self.hparams.selfplay_buffer_size)
            self.games_generated = 0
            self.previous_model = None
            self.mcts = None
        else:
            self.replay_buffer = []
    
    def forward_for_mcts(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MCTS - to be overridden by child classes.
        Must return dict with 'policy' and 'value' keys.
        """
        raise NotImplementedError("Child class must implement forward_for_mcts")
    
    def compute_loss_and_metrics(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss and metrics - to be overridden by child classes.
        Must return (loss, metrics_dict)
        """
        raise NotImplementedError("Child class must implement compute_loss_and_metrics")
    
    def setup(self, stage: str):
        """Setup - model is still on CPU, only do non-tensor operations"""
        if stage == "fit":
            # Traditional training - load file data (no tensor operations needed)
            if not self.hparams.get('enable_selfplay', False):
                if len(self.replay_buffer) == 0:
                    # This is safe because it just loads data into buffer
                    self.load_games_from_file("minimax_games_.pkl")
    
    def generate_selfplay_games(self, num_games: int, verbose: bool = False) -> Tuple[int, int, int]:
        """Generate self-play games using MCTS"""
        if not self.hparams.get('enable_selfplay', False):
            raise RuntimeError("Self-play not enabled for this model")
        
        self.eval()
        
        # Update MCTS with current model
        self.mcts.model = MCTSModelWrapper(self)
                
        batch_size = min(512, num_games)
        num_batches = (num_games + batch_size - 1) // batch_size
        all_positions = []
        
        t0 = time.time()
        
        if verbose:
            print(f"Generating {num_games} self-play games... in {num_batches} batch(es)")

        for batch_idx in range(num_batches):

            current_batch_size = min(batch_size, num_games - batch_idx * batch_size)
            
            # Initialize games
            env = VectorizedConnectFour(n_envs=current_batch_size, device=self.device)
            states = env.reset()
            
            # Store trajectories
            trajectories = [[] for _ in range(current_batch_size)]
            move_count = 0
            
            while not states.is_terminal.all():
                # Temperature for exploration
                temperature = 1.0 if move_count < self.hparams.selfplay_temperature_moves else 0.1
                
                # Get active games
                active_games = ~states.is_terminal
                boards_list = []
                legal_moves_list = []
                
                for i in range(current_batch_size):
                    if active_games[i]:
                        boards_list.append(states.boards[i])
                        legal_moves_list.append(states.legal_moves[i])
                        assert states.legal_moves[i].device == states.boards[i].device
                if not boards_list:
                    break
                
                # Get MCTS policies
                visit_distributions, action_probs = self.mcts.get_action_probs_batch_parallel(
                    boards_list, legal_moves_list, temperature
                )
                
                # Store positions and select actions
                actions = torch.zeros(current_batch_size, dtype=torch.long, device=self.device)
                policy_idx = 0
                
                for i in range(current_batch_size):
                    if active_games[i]:
                        # Store position
                        trajectories[i].append({
                            'board': states.boards[i].clone().flatten(),
                            'policy': visit_distributions[policy_idx].clone(),
                            'player': states.current_players[i].item()
                        })
                        
                        # Sample action
                        if temperature > 0:
                            actions[i] = torch.multinomial(action_probs[policy_idx], 1).item()
                        else:
                            actions[i] = action_probs[policy_idx].argmax().item()
                        
                        policy_idx += 1
                
                states = env.step(actions)
                move_count += 1
            
            # Process completed games
            for i in range(current_batch_size):
                winner = states.winners[i].item()
                
                for position in trajectories[i]:
                    # Assign value based on outcome
                    if winner == 0:  # Draw
                        value = 0.0
                    elif winner == position['player']:  # Won
                        value = 1.0
                    else:  # Lost
                        value = -1.0
                    
                    all_positions.append({
                        'board': position['board'],
                        'policy': position['policy'],
                        'value': value
                    })
        
        # Add to replay buffer
        self.replay_buffer.extend(all_positions)
        self.games_generated += num_games
        
        if verbose:
            t1 = time.time()
            print(f"Generated {len(all_positions)} positions from {num_games} games")
            print(f"Buffer size: {len(self.replay_buffer)} positions")
            print(f"Generation took {t1 - t0:.3f} seconds. Games per second: {num_games / (t1 - t0):.2f}")
        
        self.train()
        return len(all_positions), len(self.replay_buffer), num_games
    
    def on_train_epoch_start(self):
        """Do all initialization here when model is on correct device"""
        if self.hparams.get('enable_selfplay', False):
        
            num_games = self.hparams.selfplay_games_per_iteration
            
            start_time = time.time()
            num_positions, _, _ = self.generate_selfplay_games(num_games, verbose=False)
            elapsed = time.time() - start_time
            
            self.log('selfplay/games_per_second', num_games / elapsed)
            self.log('selfplay/positions_generated', num_positions)
            self.log('selfplay/buffer_size', len(self.replay_buffer))
    
    def on_train_epoch_end(self):
        """Evaluate against various opponents at epoch end"""
        epoch = self.trainer.current_epoch if hasattr(self, 'trainer') else 0
        
        if epoch % self.hparams.eval_interval == 0:
            use_mcts = self.hparams.get('eval_use_mcts', True)
            mcts_sims = self.hparams.get('selfplay_eval_mcts_simulations', 100)
            mcts_parallel = self.hparams.get('selfplay_parallel_simulations', 100)
            
            # Evaluate vs random
            win_rate, draw_rate, _ = self.evaluate(
                opponent='random',
                n_games=self.hparams.eval_games_vs_random,
                use_mcts=use_mcts,
                mcts_simulations=mcts_sims,
                mcts_parallel_simulations=mcts_parallel,
            )
            self.log('eval/win_rate_vs_random', win_rate)
            self.log('eval/draw_rate_vs_random', draw_rate)
            
            # Evaluate vs previous
            if self.hparams.get('enable_selfplay', False) and self.previous_model is not None:
                win_rate, draw_rate, _ = self.evaluate(
                    opponent='previous',
                    n_games=50,
                    use_mcts=use_mcts,
                    mcts_simulations=mcts_sims,
                    mcts_parallel_simulations=mcts_parallel,
                )
                self.log('eval/win_rate_vs_previous', win_rate)
            
            # Evaluate vs minimax
            win_rate, draw_rate, _ = self.evaluate(
                opponent='minimax',
                n_games=self.hparams.eval_games_vs_minimax,
                use_mcts=use_mcts,
                mcts_simulations=mcts_sims,
                mcts_parallel_simulations=mcts_parallel,
                minimax_depth=self.hparams.eval_minimax_depth,
                minimax_temperature=self.hparams.eval_minimax_temperature,
            )
            self.log('eval/win_rate_vs_minimax', win_rate)
            self.log('eval/draw_rate_vs_minimax', draw_rate)
        
        # Update previous model checkpoint
        if self.hparams.get('enable_selfplay', False) and epoch > 0:
            if epoch % self.hparams.get('selfplay_update_interval', 10) == 0:
                self.previous_model = pickle.loads(pickle.dumps(self.state_dict()))
    

    def evaluate(
        self,
        opponent: str = 'random',  # 'random', 'previous', 'minimax'
        n_games: int = 50,
        use_mcts: bool = True,
        mcts_simulations: int = 100,
        mcts_parallel_simulations: int = 8,
        minimax_depth: int = 4,
        minimax_temperature: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Unified evaluation against various opponents.
        
        Returns:
            (win_rate, draw_rate, loss_rate)
        """
        self.eval()
        
        # Setup opponent
        prev_model = None
        prev_mcts = None
        minimax = None
        
        if opponent == 'previous':
            if self.previous_model is None:
                self.train()
                return 0.5, 0.0, 0.5
            prev_model = self.__class__(**dict(self.hparams))
            prev_model.load_state_dict(self.previous_model)
            prev_model.eval()
            prev_model.to(self.device)
        elif opponent == 'minimax':
            minimax = ConnectFourMinimax(depth=minimax_depth)
        
        # Setup MCTS for model
        model_mcts = None
        if use_mcts:
            model_mcts = TensorMCTSWrapper(
                model=MCTSModelWrapper(self),
                num_simulations=mcts_simulations,
                parallel_simulations=mcts_parallel_simulations,
                device=self.device
            )
            # Previous model also gets MCTS if applicable
            if opponent == 'previous':
                prev_mcts = TensorMCTSWrapper(
                    model=MCTSModelWrapper(prev_model),
                    num_simulations=mcts_simulations,
                    parallel_simulations=mcts_parallel_simulations,
                    device=self.device
                )
        
        # Run games
        env = VectorizedConnectFour(n_envs=n_games, device=self.device)
        states = env.reset()
        
        # Alternate who plays first
        model_players = [1 if i % 2 == 0 else 2 for i in range(n_games)]
        
        while not states.is_terminal.all():
            actions = torch.zeros(n_games, dtype=torch.long, device=self.device)
            
            # Collect positions by who's moving
            model_positions = []  # (index, board, legal_moves)
            opponent_positions = []
            
            for i in range(n_games):
                if states.is_terminal[i]:
                    continue
                
                current_player = states.current_players[i].item()
                board = states.boards[i]
                legal = states.legal_moves[i]
                
                if current_player == model_players[i]:
                    model_positions.append((i, board, legal))
                else:
                    opponent_positions.append((i, board, legal))
            
            # Model moves (batched)
            if model_positions:
                model_actions = self._get_actions_batch(
                    [p[1] for p in model_positions],
                    [p[2] for p in model_positions],
                    mcts=model_mcts
                )
                for idx, (game_idx, _, _) in enumerate(model_positions):
                    actions[game_idx] = model_actions[idx]
            
            # Opponent moves (batched where possible)
            if opponent_positions:
                if opponent == 'random':
                    for game_idx, _, legal in opponent_positions:
                        legal_actions = legal.nonzero(as_tuple=True)[0]
                        actions[game_idx] = legal_actions[
                            torch.randint(len(legal_actions), (1,))
                        ].item()
                
                elif opponent == 'previous':
                    opp_actions = self._get_actions_batch(
                        [p[1] for p in opponent_positions],
                        [p[2] for p in opponent_positions],
                        model=prev_model,
                        mcts=prev_mcts
                    )
                    for idx, (game_idx, _, _) in enumerate(opponent_positions):
                        actions[game_idx] = opp_actions[idx]
                
                elif opponent == 'minimax':
                    for game_idx, board, _ in opponent_positions:
                        current_player = states.current_players[game_idx].item()
                        board_np = board.cpu().numpy()
                        actions[game_idx] = minimax.get_best_move(
                            board_np, current_player,
                            temperature=minimax_temperature
                        )
            
            states = env.step(actions)
        
        # Count results
        wins = sum(1 for i in range(n_games) if states.winners[i].item() == model_players[i])
        draws = sum(1 for i in range(n_games) if states.winners[i].item() == 0)
        losses = n_games - wins - draws
        
        self.train()
        return wins / n_games, draws / n_games, losses / n_games


    def _get_actions_batch(
        self,
        boards: list,
        legal_moves: list,
        model=None,
        mcts=None,
    ) -> torch.Tensor:
        """
        Get actions for a batch of positions.
        Uses MCTS if provided, otherwise raw policy.
        """
        if not boards:
            return torch.tensor([], dtype=torch.long, device=self.device)
        
        model = model or self
        
        if mcts is not None:
            _, action_probs = mcts.get_action_probs_batch_parallel(
                boards, legal_moves, temperature=0  # Greedy for eval
            )
            return torch.tensor(
                [p.argmax().item() for p in action_probs],
                dtype=torch.long, device=self.device
            )
        else:
            # Raw policy (greedy)
            boards_tensor = torch.stack([b.flatten() for b in boards])
            batch = {
                'boards': boards_tensor,
                'puzzle_identifiers': torch.zeros(
                    len(boards), dtype=torch.long, device=self.device
                )
            }
            with torch.no_grad():
                outputs = model.forward_for_mcts(batch)
            
            actions = []
            for i, legal in enumerate(legal_moves):
                masked = outputs['policy'][i] * legal.float()
                actions.append(masked.argmax().item())
            
            return torch.tensor(actions, dtype=torch.long, device=self.device)
    
    def _bootstrap_random_games(self, num_games: int):
        """Generate initial games with random play"""
        env = VectorizedConnectFour(n_envs=num_games, device=self.device)
        states = env.reset()
        
        game_histories = [[] for _ in range(num_games)]
        
        while not states.is_terminal.all():
            for i in range(num_games):
                if not states.is_terminal[i]:
                    legal = states.legal_moves[i]
                    policy = legal.float() / legal.sum()
                    
                    game_histories[i].append({
                        'board': states.boards[i].clone().flatten(),
                        'policy': policy,
                        'player': states.current_players[i].item()
                    })
            
            # Random actions
            actions = torch.zeros(num_games, dtype=torch.long, device=self.device)
            for i in range(num_games):
                if not states.is_terminal[i]:
                    legal_actions = states.legal_moves[i].nonzero(as_tuple=True)[0]
                    actions[i] = legal_actions[torch.randint(len(legal_actions), (1,))].item()
            
            states = env.step(actions)
        
        # Process games
        for game_idx in range(num_games):
            winner = states.winners[game_idx].item()
            
            for move_data in game_histories[game_idx]:
                if winner == 0:
                    value = 0.0
                elif winner == move_data['player']:
                    value = 1.0
                else:
                    value = -1.0
                
                self.replay_buffer.append({
                    'board': move_data['board'],
                    'policy': move_data['policy'],
                    'value': value
                })
    
    def load_games_from_file(self, input_file: str):
        """Load games from file - matching interface"""
        print(f"Loading games from {input_file}...")
        
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        
        positions = data['positions']
        
        for pos in positions:
            board = torch.tensor(pos['board'], dtype=torch.float32)
            policy = torch.tensor(pos['policy'], dtype=torch.float32)
            value = pos['value']
            
            self.replay_buffer.append({
                'board': board,
                'policy': policy,
                'value': value
            })
        
        print(f"Loaded {len(positions)} positions from {data['num_games']} games")
        print(f"Replay buffer now contains {len(self.replay_buffer)} positions")
    
    def train_dataloader(self):
        """Create dataloader matching interface"""
        import random
        
        log.info(f"Building train dataloader...")
        
        if self.hparams.get('enable_selfplay', False):
            if len(self.replay_buffer) == 0:
                print("Buffer empty in dataloader, bootstrapping...")
                
                device = next(self.parameters()).device
                
                # Bootstrap with random games
                self._bootstrap_random_games(100)
                
                # Create MCTS if needed
                if not hasattr(self, 'mcts') or self.mcts is None:
                    log.info(f"Creating a MCTS with {self.hparams.selfplay_parallel_simulations} parallel simulations and {self.hparams.selfplay_mcts_simulations} simulations per move")
                    self.mcts = TensorMCTSWrapper(
                        model=MCTSModelWrapper(self),
                        c_puct=1.0,
                        num_simulations=self.hparams.selfplay_mcts_simulations,
                        parallel_simulations=self.hparams.selfplay_parallel_simulations,  # Can increase this now
                        virtual_loss_value=3.0,
                        device=device
                    )
                
                # Generate initial self-play games
                self.generate_selfplay_games(self.hparams.selfplay_games_per_iteration, verbose=True)
            
            # Now create dataset with populated buffer
            dataset = SelfPlayDataset(
                module=self,
                batch_size=self.hparams.batch_size,
                steps_per_epoch=self.hparams.steps_per_epoch
            )
        else:
            class SimpleDataset(Dataset):
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
                    values = torch.tensor([s['value'] for s in samples], dtype=torch.float32)
                    puzzle_identifiers = torch.zeros(
                        self.module.hparams.batch_size, dtype=torch.long
                    )
                    
                    return {
                        'boards': boards,
                        'policies': policies,
                        'values': values,
                        'puzzle_identifiers': puzzle_identifiers
                    }
            
            dataset = SimpleDataset(self, self.hparams.steps_per_epoch)
        
        return DataLoader(dataset, batch_size=None, num_workers=0, shuffle=False)