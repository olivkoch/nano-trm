from src.nn.modules.minimax import ConnectFourMinimax
from src.nn.environments.vectorized_c4_env import VectorizedConnectFour
import numpy as np
import torch
import click

def collect_self_play_games_minimax(n_games: int, depth: int = 2, temp_player1: float = 0.0, temp_player2: float = 0.5, device: torch.device = torch.device("cpu")):
        """
        Collect self-play games using Minimax players with different temperatures.
        Lower temperature = stronger/more deterministic play
        Higher temperature = weaker/more exploratory play
        
        Args:
            depth: Minimax search depth for both players
            temp_player1: Temperature for player 1 (X). 0.0 = perfect play, 1.0 = more random
            temp_player2: Temperature for player 2 (O). Different temp creates variety
        """
        
        print(f"Collecting self-play {n_games} games using Minimax (depth={depth}, P1_temp={temp_player1}, P2_temp={temp_player2})...")
        
        # Create minimax player
        minimax = ConnectFourMinimax(depth=depth)
        games_played = 0
        replay_buffer = []
        win_rates = [.0, .0, .0]  # [draws, P1 wins, P2 wins]

        # Run multiple games in parallel
        n_parallel = 8
        n_batches = max(1, n_games // n_parallel)
        n_positions_added = 0
        trajectories_lengths = []

        for batch_idx in range(n_batches):
            print(f"  Batch {batch_idx+1}/{n_batches}...")
            vec_env = VectorizedConnectFour(n_envs=n_parallel, device=device)
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
                actions = torch.zeros(n_parallel, dtype=torch.long, device=device)
                
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
 
                    # Store position
                    trajectories[i].append({
                        'board': states.boards[i].flatten(),
                        'policy': torch.tensor(policy, device=device, dtype=torch.float32),
                        'current_player': current_player
                    })
                
                # Step environment
                states = vec_env.step(actions)

                # Check for completed games
                for i in range(n_parallel):
                    if states.is_terminal[i] and i in active_indices:
                        win_rates[0 if states.winners[i].item() == 0 else states.winners[i].item()] += 1
            
            trajectories_lengths.extend([len(t) for t in trajectories])

            # Process completed games and assign values
            for i in range(n_parallel):
                assert(len(trajectories[i]) == states.move_counts[i].item())
                assert(len(trajectories[i]) >= 7)  # Minimum moves to finish a game

                if len(trajectories[i]) < 2:  # Skip very short games
                    continue
                
                winner = states.winners[i].item()
                
                for position in trajectories[i]:
                    # Assign value based on game outcome from this player's perspective
                    if winner == 0:
                        value = 0.0  # Draw
                    elif winner == position['current_player']:
                        value = 1.0  # Win
                    else:
                        value = -1.0  # Loss
                    
                    replay_buffer.append({
                        'board': position['board'],
                        'policy': position['policy'],
                        'value': value,
                        'current_player': position['current_player']
                    })
                    n_positions_added += 1
            
            games_played += n_parallel
        
        win_rates = [w / games_played for w in win_rates]

        print(f"Collected {n_positions_added} positions using Minimax (depth={depth}), replay buffer size: {len(replay_buffer)}. Win rates = P1: {win_rates[1]*100:.1f}%, P2: {win_rates[2]*100:.1f}%, Draw: {win_rates[0]*100:.1f}%")
        print(f"Average game length: {np.mean(trajectories_lengths):.1f} moves")
        return replay_buffer, games_played


@click.command()
@click.option('--n_games', type=int, default=100, help='Number of self-play games to collect')
@click.option('--depth', type=int, default=4, help='Minimax search depth')
@click.option('--temp_player1', type=float, default=0.0, help='Temperature for player 1 (X)')
@click.option('--temp_player2', type=float, default=0.5, help='Temperature for player 2 (O)')
@click.option('--to_file', type=str, default=None, help='Optional output file to save collected positions')
def main(n_games, depth, temp_player1, temp_player2, to_file):
    """Evaluate Minimax self-play data collection"""
    device = "cpu"

    buffer, games_played = collect_self_play_games_minimax(
        n_games=n_games,
        depth=depth,
        temp_player1=temp_player1,
        temp_player2=temp_player2,
        device=device
    )

        # Save games to file
    if to_file is not None:
    
        games_data = []
        for item in buffer:
            games_data.append({
                'board': item['board'].cpu().numpy().tolist(),
                'policy': item['policy'].cpu().numpy().tolist(),
                'value': float(item['value']),
                'current_player': int(item['current_player'])
            })
        data = {}
        data['positions'] = games_data
        data['num_games'] = games_played
        data['num_positions'] = len(games_data)
        import pickle
        with open(to_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Saved {len(games_data)} positions from {games_played} games to {to_file}")

if __name__ == "__main__":
    main()
