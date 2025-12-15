"""
Parallelized Minimax self-play data collection for Connect Four.
Uses multiprocessing to parallelize minimax evaluations across CPU cores.
"""

from src.nn.modules.minimax import ConnectFourMinimax
from src.nn.environments.vectorized_c4_env import VectorizedConnectFour
import numpy as np
import torch
import click
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial


# Global minimax instance per worker process (initialized lazily)
_worker_minimax = None
_worker_depth = None


def _init_worker(depth: int):
    """Initialize minimax instance for each worker process."""
    global _worker_minimax, _worker_depth
    _worker_minimax = ConnectFourMinimax(depth=depth)
    _worker_depth = depth


def _minimax_worker(args):
    """
    Worker function for parallel minimax evaluation.
    Args is a tuple: (board_flat, current_player, temperature)
    """
    global _worker_minimax
    board_flat, current_player, temperature = args
    board_np = np.array(board_flat).reshape(6, 7)
    action = _worker_minimax.get_best_move(board_np, current_player, temperature=temperature)
    return action


def collect_self_play_games_minimax_parallel(
    n_games: int,
    depth: int = 2,
    temp_player1: float = 0.0,
    temp_player2: float = 0.5,
    device: torch.device = torch.device("cpu"),
    n_workers: int = None,
    n_parallel: int = 64,
    discount_factor: float = 0.97,
    winners_only: bool = False,
):
    """
    Collect self-play games using Minimax players with different temperatures.
    Uses multiprocessing to parallelize minimax evaluations.
    
    Args:
        n_games: Number of games to collect
        depth: Minimax search depth for both players
        temp_player1: Temperature for player 1 (X). 0.0 = perfect play
        temp_player2: Temperature for player 2 (O). Different temp creates variety
        device: Device for tensor operations
        n_workers: Number of worker processes (defaults to CPU count)
        n_parallel: Number of games to run in parallel per batch
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    print(f"Collecting {n_games} self-play games using Minimax")
    print(f"  depth={depth}, P1_temp={temp_player1}, P2_temp={temp_player2}")
    print(f"  n_workers={n_workers}, n_parallel={n_parallel}")
    
    replay_buffer = []
    win_counts = [0, 0, 0]  # [draws, P1 wins, P2 wins]
    games_played = 0
    n_positions_added = 0
    trajectories_lengths = []
    value_stats = []  # Track value distribution
    
    n_batches = max(1, (n_games + n_parallel - 1) // n_parallel)
    
    # Create process pool with initialized workers
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(depth,)
    ) as executor:
        
        for batch_idx in range(n_batches):
            # Adjust n_parallel for last batch if needed
            games_this_batch = min(n_parallel, n_games - games_played)
            if games_this_batch <= 0:
                break
                
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"  Batch {batch_idx+1}/{n_batches} ({games_played}/{n_games} games)...")
            
            vec_env = VectorizedConnectFour(n_envs=games_this_batch, device=device)
            states = vec_env.reset()
            
            # Track trajectories for each game
            trajectories = [[] for _ in range(games_this_batch)]
            game_finished = [False] * games_this_batch
            
            # Play all games until completion
            while not all(game_finished):
                # Find active (non-terminal) games
                active_indices = [
                    i for i in range(games_this_batch)
                    if not states.is_terminal[i] and not game_finished[i]
                ]
                
                if not active_indices:
                    break
                
                # Prepare minimax arguments for all active games
                minimax_args = []
                for i in active_indices:
                    board_flat = states.boards[i].cpu().numpy().flatten().tolist()
                    current_player = states.current_players[i].item()
                    temperature = temp_player1 if current_player == 1 else temp_player2
                    minimax_args.append((board_flat, current_player, temperature))
                
                # Execute minimax in parallel across all active games
                results = list(executor.map(_minimax_worker, minimax_args))
                
                # Build actions tensor and store trajectory data
                actions = torch.zeros(games_this_batch, dtype=torch.long, device=device)
                
                for idx, i in enumerate(active_indices):
                    action = results[idx]
                    actions[i] = action
                    current_player = states.current_players[i].item()
                    
                    # Create one-hot policy target
                    policy = torch.zeros(7, device=device, dtype=torch.float32)
                    policy[action] = 1.0
                    
                    # Store position in trajectory
                    trajectories[i].append({
                        'board': states.boards[i].flatten().clone(),
                        'policy': policy,
                        'current_player': current_player
                    })
                
                # Step environment for all games
                states = vec_env.step(actions)
                
                # Mark completed games
                for i in active_indices:
                    if states.is_terminal[i]:
                        game_finished[i] = True
                        winner = states.winners[i].item()
                        win_counts[0 if winner == 0 else winner] += 1
            
            # Record trajectory lengths
            trajectories_lengths.extend([len(t) for t in trajectories])
            
            # Process completed games and assign values
            for i in range(games_this_batch):
                if len(trajectories[i]) < 2:
                    continue
                
                winner = states.winners[i].item()
                game_length = len(trajectories[i])
                
                for move_idx, position in enumerate(trajectories[i]):
                    moves_from_end = game_length - move_idx

                    # Assign value from current player's perspective
                    if winner == 0:
                        outcome = 0.0  # Draw
                    elif winner == position['current_player']:
                        outcome = 1.0  # Win
                    else:
                        outcome = -1.0  # Loss
                    
                    discounted_value = outcome * (discount_factor ** moves_from_end)

                    if winners_only and outcome <= 0:
                        continue

                    value_stats.append(discounted_value)

                    replay_buffer.append({
                        'board': position['board'],
                        'policy': position['policy'],
                        'value': discounted_value,
                        'current_player': position['current_player']
                    })
                    n_positions_added += 1
            
            games_played += games_this_batch
    
    # Compute statistics
    total_games = sum(win_counts)
    if total_games > 0:
        win_rates = [w / total_games for w in win_counts]
    else:
        win_rates = [0.0, 0.0, 0.0]
        
    print(f"\nCollected {n_positions_added} positions from {games_played} games")
    print(f"  Win rates: P1={win_rates[1]*100:.1f}%, P2={win_rates[2]*100:.1f}%, Draw={win_rates[0]*100:.1f}%")
    if trajectories_lengths:
        print(f"  Average game length: {np.mean(trajectories_lengths):.1f} moves")
    
    # Value distribution stats
    if value_stats:
        value_arr = np.array(value_stats)
        print(f"  Value distribution:")
        print(f"    min={value_arr.min():.3f}, max={value_arr.max():.3f}")
        print(f"    mean={value_arr.mean():.3f}, std={value_arr.std():.3f}")
        print(f"    unique values: {len(np.unique(value_arr.round(4)))}")


    return replay_buffer, games_played


@click.command()
@click.option('--n_games', type=int, default=100, help='Number of self-play games to collect')
@click.option('--depth', type=int, default=4, help='Minimax search depth')
@click.option('--temp_player1', type=float, default=0.1, help='Temperature for player 1 (X)')
@click.option('--temp_player2', type=float, default=0.2, help='Temperature for player 2 (O)')
@click.option('--n_workers', type=int, default=None, help='Number of worker processes (default: CPU count)')
@click.option('--n_parallel', type=int, default=64, help='Number of parallel games per batch')
@click.option('--discount', type=float, default=0.97, help='Discount factor for value targets')
@click.option('--winners_only', is_flag=True, default=True, help='Only include positions from winning side')
@click.option('--to_file', type=str, default=None, help='Output file to save collected positions (.pkl)')
def main(n_games, depth, temp_player1, temp_player2, n_workers, n_parallel, discount, winners_only, to_file):
    """Collect Connect Four training data using parallelized Minimax self-play."""
    import time
    
    device = torch.device("cpu")
    
    start_time = time.time()
    
    buffer, games_played = collect_self_play_games_minimax_parallel(
        n_games=n_games,
        depth=depth,
        temp_player1=temp_player1,
        temp_player2=temp_player2,
        device=device,
        n_workers=n_workers,
        n_parallel=n_parallel,
        discount_factor=discount,
        winners_only=winners_only
    )
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({games_played/elapsed:.1f} games/sec)")
    
    # Save to file if requested
    if to_file is not None:
        import pickle
        
        games_data = []
        for item in buffer:
            games_data.append({
                'board': item['board'].cpu().numpy().tolist(),
                'policy': item['policy'].cpu().numpy().tolist(),
                'value': float(item['value']),
                'current_player': int(item['current_player'])
            })
        
        data = {
            'positions': games_data,
            'num_games': games_played,
            'num_positions': len(games_data),
            'depth': depth,
            'temp_player1': temp_player1,
            'temp_player2': temp_player2,
        }
        
        with open(to_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Saved {len(games_data)} positions from {games_played} games to {to_file}")


if __name__ == "__main__":
    main()