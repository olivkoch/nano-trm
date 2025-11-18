"""
Simple minimax implementation for Connect Four evaluation.
No dependencies on vectorized environments - works with raw board states.
"""

import torch
import numpy as np
from typing import Optional, List


class SimpleMinimaxPlayer:
    """
    Simple minimax player for Connect Four.
    Board representation: 0 = empty, 1 = player 1, 2 = player 2
    """
    
    def __init__(self, depth: int = 4):
        self.depth = depth
        self.rows = 6
        self.cols = 7
        self.nodes_evaluated = 0
        
        # Precompute all winning patterns
        self.win_patterns = []
        
        # Horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                self.win_patterns.append([(row, col + i) for i in range(4)])
        
        # Vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                self.win_patterns.append([(row + i, col) for i in range(4)])
        
        # Diagonal (/)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                self.win_patterns.append([(row + i, col + i) for i in range(4)])
        
        # Diagonal (\)
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                self.win_patterns.append([(row - i, col + i) for i in range(4)])
    
    def create_board_from_env(self, vec_env, env_idx: int = 0) -> np.ndarray:
        """
        Create a simple board from the vectorized environment.
        vec_env.boards shape: [n_envs, 2, 6, 7] where [i, 0] is current player, [i, 1] is opponent
        Returns: 6x7 numpy array with 0=empty, 1=player1, 2=player2
        """
        board = np.zeros((self.rows, self.cols), dtype=int)
        current_player = vec_env.current_players[env_idx].item()
        
        # Current player's pieces
        board[vec_env.boards[env_idx, 0].cpu().numpy() == 1] = current_player
        # Opponent's pieces  
        board[vec_env.boards[env_idx, 1].cpu().numpy() == 1] = 3 - current_player
        
        return board
    
    def check_winner(self, board: np.ndarray) -> int:
        """Check if there's a winner. Returns 0 (none), 1 (player 1), or 2 (player 2)."""
        for pattern in self.win_patterns:
            values = [board[r, c] for r, c in pattern]
            if values[0] != 0 and all(v == values[0] for v in values):
                return values[0]
        return 0
    
    def is_draw(self, board: np.ndarray) -> bool:
        """Check if the game is a draw (board full)."""
        return np.all(board[0, :] != 0)  # Top row full = board full
    
    def get_valid_moves(self, board: np.ndarray) -> List[int]:
        """Get list of valid moves (columns with space)."""
        return [col for col in range(self.cols) if board[0, col] == 0]
    
    def make_move(self, board: np.ndarray, col: int, player: int) -> Optional[np.ndarray]:
        """Make a move on the board. Returns new board or None if invalid."""
        if board[0, col] != 0:
            return None
        
        new_board = board.copy()
        for row in range(self.rows - 1, -1, -1):
            if new_board[row, col] == 0:
                new_board[row, col] = player
                break
        
        return new_board
    
    def evaluate_position(self, board: np.ndarray, player: int) -> float:
        """
        Evaluate the board position from the perspective of 'player'.
        Positive scores favor 'player', negative scores favor opponent.
        """
        score = 0.0
        opponent = 3 - player
        
        # Check each potential winning line
        for pattern in self.win_patterns:
            values = [board[r, c] for r, c in pattern]
            
            player_count = sum(1 for v in values if v == player)
            opponent_count = sum(1 for v in values if v == opponent)
            empty_count = sum(1 for v in values if v == 0)
            
            # If line is still winnable by player
            if opponent_count == 0 and player_count > 0:
                if player_count == 3:
                    score += 50  # One move from winning
                elif player_count == 2:
                    score += 10
                elif player_count == 1:
                    score += 1
            
            # If line is still winnable by opponent
            if player_count == 0 and opponent_count > 0:
                if opponent_count == 3:
                    score -= 50  # Opponent one move from winning
                elif opponent_count == 2:
                    score -= 10
                elif opponent_count == 1:
                    score -= 1
        
        # Prefer center column (more opportunities)
        center_col = self.cols // 2
        for row in range(self.rows):
            if board[row, center_col] == player:
                score += 3
            elif board[row, center_col] == opponent:
                score -= 3
        
        return score
    
    def minimax(
        self, 
        board: np.ndarray, 
        depth: int, 
        alpha: float, 
        beta: float, 
        maximizing_player: bool,
        player: int
    ) -> float:
        """
        Minimax with alpha-beta pruning.
        Returns the evaluation score for the position.
        """
        self.nodes_evaluated += 1
        
        # Check terminal states
        winner = self.check_winner(board)
        if winner == player:
            return 10000 - (self.depth - depth)  # Prefer quicker wins
        elif winner == (3 - player):
            return -10000 + (self.depth - depth)  # Prefer slower losses
        elif self.is_draw(board):
            return 0
        
        # Depth limit reached
        if depth == 0:
            return self.evaluate_position(board, player)
        
        valid_moves = self.get_valid_moves(board)
        
        # Move ordering: try center columns first
        center = self.cols // 2
        valid_moves.sort(key=lambda col: abs(col - center))
        
        if maximizing_player:
            max_eval = float('-inf')
            for col in valid_moves:
                new_board = self.make_move(board, col, player)
                if new_board is not None:
                    eval_score = self.minimax(new_board, depth - 1, alpha, beta, False, player)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            opponent = 3 - player
            for col in valid_moves:
                new_board = self.make_move(board, col, opponent)
                if new_board is not None:
                    eval_score = self.minimax(new_board, depth - 1, alpha, beta, True, player)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha cutoff
            return min_eval
    
    def get_best_move(self, board: np.ndarray, player: int) -> int:
        """
        Get the best move for the given player.
        Returns column index of best move.
        """
        valid_moves = self.get_valid_moves(board)
        
        if not valid_moves:
            return -1
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # First, check for immediate wins
        for col in valid_moves:
            test_board = self.make_move(board, col, player)
            if test_board is not None and self.check_winner(test_board) == player:
                return col
        
        # Then check for immediate blocks
        opponent = 3 - player
        for col in valid_moves:
            test_board = self.make_move(board, col, opponent)
            if test_board is not None and self.check_winner(test_board) == opponent:
                return col  # Must block
        
        # Otherwise use minimax
        best_move = valid_moves[0]
        best_score = float('-inf')
        
        # Reset node counter
        self.nodes_evaluated = 0
        
        for col in valid_moves:
            new_board = self.make_move(board, col, player)
            if new_board is not None:
                # Evaluate position after this move
                score = self.minimax(
                    new_board, 
                    self.depth - 1, 
                    float('-inf'), 
                    float('inf'), 
                    False,  # Opponent's turn next
                    player
                )
                
                if score > best_score:
                    best_score = score
                    best_move = col
        
        return best_move


if __name__ == "__main__":
    # Test the minimax player standalone
    print("Testing SimpleMinimaxPlayer")
    print("-" * 40)
    
    # Create a test board
    minimax = SimpleMinimaxPlayer(depth=4)
    
    # Empty board
    board = np.zeros((6, 7), dtype=int)
    
    # Add some pieces for testing
    board[5, 3] = 1  # Player 1 in center
    board[5, 2] = 2  # Player 2 to the left
    board[5, 4] = 1  # Player 1 to the right
    
    print("Test board:")
    for row in board:
        print(" ".join(["." if x == 0 else str(x) for x in row]))
    
    # Get best move for player 2
    best_col = minimax.get_best_move(board, player=2)
    print(f"\nBest move for player 2: column {best_col}")
    print(f"Nodes evaluated: {minimax.nodes_evaluated}")
    
    # Test winning position
    print("\n" + "-" * 40)
    print("Testing winning position detection:")
    
    win_board = np.zeros((6, 7), dtype=int)
    win_board[5, 0] = 1
    win_board[5, 1] = 1
    win_board[5, 2] = 1
    # Player 1 can win by playing column 3
    
    print("Board where player 1 can win:")
    for row in win_board:
        print(" ".join(["." if x == 0 else str(x) for x in row]))
    
    best_col = minimax.get_best_move(win_board, player=1)
    print(f"Player 1 should play column 3 to win: got column {best_col}")
    
    # Test blocking
    best_col = minimax.get_best_move(win_board, player=2)
    print(f"Player 2 should block at column 3: got column {best_col}")