#!/usr/bin/env python
"""
Interactive MCTS Debugger

Watch MCTS play Connect Four move by move.
Press SPACE to advance, 'q' to quit, 'v' for verbose tree info.

Supports different "models":
- dummy: Uniform policy, zero value
- minimax: Uses minimax evaluation for policy/value
"""

import torch
import sys
import os
import numpy as np

from src.nn.modules.tensor_mcts import TensorMCTSWrapper, TensorMCTSConfig
from src.nn.modules.minimax import ConnectFourMinimax
from src.nn.utils.constants import C4_EMPTY_CELL, C4_PLAYER1_CELL, C4_PLAYER2_CELL


class DummyModel:
    """Simple model - uniform policy, zero value"""
    def __init__(self, device):
        self.device = device
    
    def forward(self, boards, current_players):
        batch_size = boards.shape[0]
        policies = torch.ones(batch_size, 7, device=self.device) / 7
        values = torch.zeros(batch_size, device=self.device)
        return policies, values


class MinimaxModel:
    """Model that uses minimax for policy and value"""
    def __init__(self, device, depth=4, verbose=False):
        self.device = device
        self.minimax = ConnectFourMinimax(depth=depth)
        self.depth = depth
        self.verbose = verbose
        self.eval_count = 0  # Track evaluations to identify root
    
    def reset_eval_count(self):
        """Call before each MCTS run to reset counter"""
        self.eval_count = 0
    
    def forward(self, boards, current_players):
        """
        boards: (batch, 42) flattened boards using 0=empty, 1=P1, 2=P2
        current_players: (batch,) player to move (1 or 2)
        Returns: policies (batch, 7), values (batch,)
        """
        batch_size = boards.shape[0]
        policies = torch.zeros(batch_size, 7, device=self.device)
        values = torch.zeros(batch_size, device=self.device)
        
        is_root_eval = (self.eval_count == 0)
        self.eval_count += 1
        
        for i in range(batch_size):
            board_flat = boards[i].cpu().numpy()
            board_2d = board_flat.reshape(6, 7)
            
            current_player = int(current_players[i].item())
            
            if self.verbose and is_root_eval and i == 0:
                print(f"\n{'='*50}")
                print(f"[ROOT POSITION] Player {current_player} to move")
                print(f"Board:\n{board_2d}")
            
            # Get valid moves
            valid_moves = self.minimax.get_valid_moves(board_2d)
            
            if self.verbose and is_root_eval and i == 0:
                print(f"Valid moves: {valid_moves}")
            
            if not valid_moves:
                continue
            
            # Compute minimax scores for each move
            move_scores = {}
            for col in valid_moves:
                new_board = self.minimax.make_move(board_2d, col, current_player)
                if new_board is not None:
                    # Check for immediate win
                    if self.minimax.check_winner(new_board) == current_player:
                        move_scores[col] = 10000
                    else:
                        # Use minimax evaluation
                        score = self.minimax.minimax(
                            new_board, 
                            depth=self.depth - 1,
                            alpha=float('-inf'),
                            beta=float('inf'),
                            maximizing=False,
                            player=current_player
                        )
                        move_scores[col] = score
            
            if self.verbose and is_root_eval and i == 0:
                print(f"Minimax scores: {move_scores}")
            
            # Convert scores to policy (softmax with temperature)
            if move_scores:
                scores = torch.tensor([move_scores.get(c, -10000) for c in range(7)], 
                                     dtype=torch.float32)
                
                if self.verbose and is_root_eval and i == 0:
                    print(f"Raw scores: {scores.tolist()}")
                
                # Shift scores for softmax stability
                scores = scores - scores.max()
                
                # Use temperature based on score range
                score_range = scores.max() - scores.min()
                temp = max(1.0, score_range / 10)
                policy = torch.softmax(scores / temp, dim=0)
                
                if self.verbose and is_root_eval and i == 0:
                    print(f"After shift: {scores.tolist()}")
                    print(f"Temperature: {temp}")
                    print(f"Policy: {[f'{p:.3f}' for p in policy.tolist()]}")
                
                policies[i] = policy.to(self.device)
                
                # Value is normalized best score
                best_score = max(move_scores.values())
                values[i] = np.tanh(best_score / 1000)
                
                if self.verbose and is_root_eval and i == 0:
                    print(f"Best score: {best_score}, Value: {values[i]:.4f}")
                    print(f"{'='*50}\n")
        
        return policies, values


class InteractiveMCTSDebugger:
    def __init__(self, model_type="dummy", device="cpu", mcts_simulations=100, minimax_depth=4, verbose_model=False, setup=None):
        self.device = device
        self.mcts_simulations = mcts_simulations
        self.model_type = model_type
        self.verbose_model = verbose_model
        
        # Create model
        if model_type == "minimax":
            self.model = MinimaxModel(device, depth=minimax_depth, verbose=verbose_model)
            print(f"Using Minimax model (depth={minimax_depth}, verbose={verbose_model})")
        else:
            self.model = DummyModel(device)
            print("Using Dummy model (uniform policy)")
        
        self.config = TensorMCTSConfig(
            n_actions=7,
            max_nodes_per_tree=2000,
            c_puct=1.0,
            exploration_fraction=0.1,
            dirichlet_alpha=0.3,
        )
        
        # Board uses C4 constants
        self.board = torch.full((6, 7), C4_EMPTY_CELL, dtype=torch.float32, device=device)
        self.current_player = 1  # Player 1 or 2
        self.move_history = []
        self.game_over = False
        self.winner = None
        
        # Store the wrapper for tree inspection
        self.mcts_wrapper = None
        
        # Apply setup if provided
        if setup:
            self.setup_position(setup)
    
    def setup_position(self, name):
        """Set up a predefined position for testing"""
        positions = {
            # X has 3 vertical in column 3, O must block
            # Recent moves: P2:c5 → P1:c2 → P2:c1 → P1:c3 → P2:c2 → P1:c3
            "must_block_vertical": {
                "board": [
                    # (row, col, player_cell)
                    (5, 3, C4_PLAYER1_CELL),  # X
                    (5, 5, C4_PLAYER2_CELL),  # O
                    (5, 2, C4_PLAYER1_CELL),  # X
                    (5, 1, C4_PLAYER2_CELL),  # O
                    (4, 3, C4_PLAYER1_CELL),  # X
                    (4, 2, C4_PLAYER2_CELL),  # O
                    (3, 3, C4_PLAYER1_CELL),  # X
                ],
                "current_player": 2,  # O to move
                "history": [(1, 3), (2, 5), (1, 2), (2, 1), (1, 3), (2, 2), (1, 3)],
            },
            
            # X has 2 horizontal, O should block at 2 or 5
            "block_horizontal": {
                "board": [
                    (5, 3, C4_PLAYER1_CELL),  # X
                    (5, 4, C4_PLAYER1_CELL),  # X
                    (5, 1, C4_PLAYER2_CELL),  # O
                ],
                "current_player": 2,
                "history": [(1, 3), (2, 1), (1, 4)],
            },
            
            # X about to win diagonally
            "must_block_diagonal": {
                "board": [
                    (5, 0, C4_PLAYER1_CELL),  # X
                    (4, 1, C4_PLAYER1_CELL),  # X
                    (3, 2, C4_PLAYER1_CELL),  # X
                    (5, 1, C4_PLAYER2_CELL),  # O
                    (5, 2, C4_PLAYER2_CELL),  # O
                    (4, 2, C4_PLAYER2_CELL),  # O
                ],
                "current_player": 2,  # O must play (2, 3) to block
                "history": [],
            },
            
            # Empty board
            "empty": {
                "board": [],
                "current_player": 1,
                "history": [],
            },
        }
        
        if name not in positions:
            print(f"Unknown position: {name}")
            print(f"Available: {list(positions.keys())}")
            return
        
        pos = positions[name]
        
        # Reset board
        self.board = torch.full((6, 7), C4_EMPTY_CELL, dtype=torch.float32, device=self.device)
        
        # Place pieces
        for row, col, cell in pos["board"]:
            self.board[row, col] = cell
        
        self.current_player = pos["current_player"]
        self.move_history = pos["history"]
        self.game_over = False
        self.winner = None
        
        print(f"Set up position: {name}")
    
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_cell_char(self, cell_value):
        """Convert cell value to display character"""
        if cell_value == C4_EMPTY_CELL:
            return " "
        elif cell_value == C4_PLAYER1_CELL:
            return "X"
        elif cell_value == C4_PLAYER2_CELL:
            return "O"
        return "?"
    
    def print_board(self, highlight_col=None):
        """Print the board with optional column highlight"""
        print("\n  0   1   2   3   4   5   6")
        print("┌───┬───┬───┬───┬───┬───┬───┐")
        
        for row in range(6):
            print("│", end="")
            for col in range(7):
                cell = self.board[row, col].item()
                char = self.get_cell_char(cell)
                
                # Highlight selected column
                if highlight_col == col:
                    print(f" \033[92m{char}\033[0m │", end="")
                else:
                    print(f" {char} │", end="")
            print()
            
            if row < 5:
                print("├───┼───┼───┼───┼───┼───┼───┤")
        
        print("└───┴───┴───┴───┴───┴───┴───┘")
    
    def print_mcts_analysis(self, mcts, selected_action):
        """Print detailed MCTS analysis"""
        print("\n" + "=" * 60)
        print(f"MCTS Analysis ({self.mcts_simulations} simulations, model={self.model_type})")
        print("=" * 60)
        
        # Visit distribution
        dist = mcts._get_visit_distributions()[0]
        
        print("\nVisit Distribution:")
        print("┌─────┬────────┬────────┬────────┬─────────┐")
        print("│ Col │ Visits │  Prob  │   Q    │  Prior  │")
        print("├─────┼────────┼────────┼────────┼─────────┤")
        
        root_visits = mcts.visits[0, 0].item()
        
        for action in range(7):
            child_idx = mcts.children[0, 0, action].item()
            prior = mcts.priors[0, 0, action].item()
            
            if child_idx >= 0:
                visits = mcts.visits[0, child_idx].item()
                total_val = mcts.total_value[0, child_idx].item()
                # NEGATED: Show Q from root's perspective, not child's
                q = -total_val / visits if visits > 0 else 0
                prob = dist[action].item()
                is_term = mcts.is_terminal[0, child_idx].item()
                
                # Build column label with markers
                sel = "*" if action == selected_action else " "
                term = "T" if is_term else " "
                col_label = f"{action}{sel}{term}"
                
                print(f"│ {col_label} │ {visits:6.0f} │ {prob:6.1%} │ {q:+6.3f} │ {prior:7.3f} │")
            else:
                col_label = f"{action}  "
                print(f"│ {col_label} │      - │      - │      - │ {prior:7.3f} │")
        
        print("└─────┴────────┴────────┴────────┴─────────┘")
        print("(* = selected, T = terminal, Q = from root's perspective)")
        
        # Summary stats
        entropy = -(dist * (dist + 1e-8).log()).sum().item()
        max_prob = dist.max().item()
        
        print(f"\nRoot visits: {root_visits:.0f}")
        print(f"Entropy: {entropy:.3f} (low=confident, high=uncertain)")
        print(f"Max prob: {max_prob:.1%}")
        print(f"Selected: Column {selected_action}")
    
    def print_tree_structure(self, mcts, max_depth=2):
        """Print tree structure"""
        print("\n" + "-" * 40)
        print("Tree Structure (depth limited)")
        print("-" * 40)
        
        def print_node(node_idx, depth, action_from_parent):
            if depth > max_depth:
                return
            
            indent = "  " * depth
            visits = mcts.visits[0, node_idx].item()
            value = mcts.total_value[0, node_idx].item()
            # Q from this node's parent's perspective (negated)
            q = -value / visits if visits > 0 else 0
            is_term = mcts.is_terminal[0, node_idx].item()
            
            if action_from_parent == -1:
                label = "ROOT"
                # Root Q is from root's own perspective (not negated)
                q = value / visits if visits > 0 else 0
            else:
                label = f"a={action_from_parent}"
            
            term_str = " [TERM]" if is_term else ""
            print(f"{indent}├─ {label}: N={visits:.0f}, Q={q:+.3f}{term_str}")
            
            if depth < max_depth:
                children = mcts.children[0, node_idx]
                for action in range(7):
                    child = children[action].item()
                    if child >= 0 and mcts.visits[0, child].item() > 0:
                        print_node(child, depth + 1, action)
        
        print_node(0, 0, -1)
    
    def get_legal_moves(self):
        """Get legal moves (columns that aren't full)"""
        return (self.board[0, :] == C4_EMPTY_CELL)
    
    def make_move(self, col):
        """Make a move and return True if valid"""
        if self.board[0, col] != C4_EMPTY_CELL:
            return False
        
        player_token = C4_PLAYER1_CELL if self.current_player == 1 else C4_PLAYER2_CELL
        
        # Find lowest empty row
        for row in range(5, -1, -1):
            if self.board[row, col] == C4_EMPTY_CELL:
                self.board[row, col] = player_token
                self.move_history.append((self.current_player, col))
                break
        
        # Check for win
        if self.check_win(self.current_player):
            self.game_over = True
            self.winner = self.current_player
        elif self.get_legal_moves().sum() == 0:
            self.game_over = True
            self.winner = 0  # Draw
        else:
            self.current_player = 3 - self.current_player
        
        return True
    
    def check_win(self, player):
        """Check if player has won"""
        player_token = C4_PLAYER1_CELL if player == 1 else C4_PLAYER2_CELL
        b = self.board
        
        # Horizontal
        for row in range(6):
            for col in range(4):
                if (b[row, col:col+4] == player_token).all():
                    return True
        
        # Vertical
        for row in range(3):
            for col in range(7):
                if (b[row:row+4, col] == player_token).all():
                    return True
        
        # Diagonal down-right
        for row in range(3):
            for col in range(4):
                if all(b[row+i, col+i] == player_token for i in range(4)):
                    return True
        
        # Diagonal down-left
        for row in range(3):
            for col in range(3, 7):
                if all(b[row+i, col-i] == player_token for i in range(4)):
                    return True
        
        return False
    
    def run_mcts(self):
        """Run MCTS and return selected action using TensorMCTSWrapper (same as self-play)"""
        mcts = TensorMCTSWrapper(
            model=self.model,
            c_puct=self.config.c_puct,
            num_simulations=self.mcts_simulations,
            parallel_simulations=8,
            dirichlet_alpha=self.config.dirichlet_alpha,
            exploration_fraction=self.config.exploration_fraction,
            device=self.device
        )
        
        # Reset eval counter for verbose output (only affects MinimaxModel)
        if hasattr(self.model, 'reset_eval_count'):
            self.model.reset_eval_count()
        
        # Prepare inputs as lists (matching self-play usage)
        boards_list = [self.board.clone()]
        legal_moves_list = [self.get_legal_moves()]
        current_players = torch.tensor([self.current_player], dtype=torch.long, device=self.device)
        
        # Use the same function as self-play training
        visit_distributions, action_probs = mcts.get_action_probs_batch_parallel(
            boards_list,
            legal_moves_list,
            temperature=1.0,  # Use temperature for visit dist, we'll pick greedy from it
            current_players=current_players,
            verbose=False
        )
        
        # Get underlying TensorMCTS for tree inspection
        underlying_mcts = mcts._cached_mcts[1]
        
        # Get best action (greedy)
        dist = visit_distributions[0]
        action = dist.argmax().item()
        
        return underlying_mcts, action
    
    def wait_for_input(self):
        """Wait for user input"""
        print("\n[SPACE] Next move  [v] Verbose tree  [r] Reset  [q] Quit")
        
        # Cross-platform input handling
        try:
            import termios
            import tty
            
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
        except ImportError:
            # Windows fallback
            import msvcrt
            return msvcrt.getch().decode('utf-8')
    
    def run(self):
        """Main game loop"""
        verbose_tree = False
        
        while True:
            self.clear_screen()
            
            print("=" * 60)
            print("   INTERACTIVE MCTS DEBUGGER - Connect Four")
            print(f"   Model: {self.model_type} | Sims: {self.mcts_simulations}")
            print("=" * 60)
            
            # Show move history
            if self.move_history:
                history_str = " → ".join([f"P{p}:c{c}" for p, c in self.move_history[-6:]])
                print(f"\nRecent moves: {history_str}")
            
            player_char = 'X' if self.current_player == 1 else 'O'
            print(f"\nPlayer {player_char}'s turn (Player {self.current_player})")
            
            self.print_board()
            
            if self.game_over:
                if self.winner == 0:
                    print("\n🤝 DRAW!")
                else:
                    winner_char = 'X' if self.winner == 1 else 'O'
                    print(f"\n🎉 Player {self.winner} ({winner_char}) WINS!")
                
                print("\n[r] Reset  [q] Quit")
                key = self.wait_for_input()
                
                if key == 'r':
                    self.board = torch.full((6, 7), C4_EMPTY_CELL, dtype=torch.float32, device=self.device)
                    self.current_player = 1
                    self.move_history = []
                    self.game_over = False
                    self.winner = None
                    continue
                elif key == 'q':
                    break
                continue
            
            # Run MCTS
            print("\nRunning MCTS...")
            mcts, action = self.run_mcts()
            
            self.clear_screen()
            
            print("=" * 60)
            print("   INTERACTIVE MCTS DEBUGGER - Connect Four")
            print(f"   Model: {self.model_type} | Sims: {self.mcts_simulations}")
            print("=" * 60)
            
            if self.move_history:
                history_str = " → ".join([f"P{p}:c{c}" for p, c in self.move_history[-6:]])
                print(f"\nRecent moves: {history_str}")
            
            player_char = 'X' if self.current_player == 1 else 'O'
            print(f"\nPlayer {player_char}'s turn (Player {self.current_player})")
            
            self.print_board(highlight_col=action)
            self.print_mcts_analysis(mcts, action)
            
            if verbose_tree:
                self.print_tree_structure(mcts, max_depth=2)
            
            # Wait for input
            key = self.wait_for_input()
            
            if key == ' ':
                self.make_move(action)
            elif key == 'v':
                verbose_tree = not verbose_tree
            elif key == 'r':
                self.board = torch.full((6, 7), C4_EMPTY_CELL, dtype=torch.float32, device=self.device)
                self.current_player = 1
                self.move_history = []
                self.game_over = False
                self.winner = None
            elif key == 'q':
                break
        
        print("\nGoodbye!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive MCTS Debugger")
    parser.add_argument("--sims", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--model", type=str, default="dummy", choices=["dummy", "minimax"],
                       help="Model type: 'dummy' (uniform) or 'minimax'")
    parser.add_argument("--minimax-depth", type=int, default=4, 
                       help="Minimax search depth (only used with --model minimax)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show verbose model output (minimax scores, etc)")
    parser.add_argument("--setup", type=str, default=None,
                       choices=["must_block_vertical", "block_horizontal", "must_block_diagonal", "empty"],
                       help="Start with a predefined position")
    args = parser.parse_args()
    
    print("Starting Interactive MCTS Debugger...")
    print(f"Model: {args.model}")
    print(f"Simulations: {args.sims}")
    print(f"Device: {args.device}")
    if args.model == "minimax":
        print(f"Minimax depth: {args.minimax_depth}")
    print()
    
    debugger = InteractiveMCTSDebugger(
        model_type=args.model,
        device=args.device,
        mcts_simulations=args.sims,
        minimax_depth=args.minimax_depth,
        verbose_model=args.verbose,
        setup=args.setup
    )
    
    debugger.run()


if __name__ == "__main__":
    main()