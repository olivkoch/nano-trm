"""
Minimax implementation for Connect Four using constants from src.nn.utils.constants
Works with VectorizedConnectFour environment that uses C4_EMPTY_CELL=1, C4_PLAYER1_CELL=2, C4_PLAYER2_CELL=3
"""

import torch
import numpy as np
from typing import List, Optional

from src.nn.utils.constants import C4_EMPTY_CELL, C4_PLAYER1_CELL, C4_PLAYER2_CELL


class ConnectFourMinimax:
    """
    Minimax player for Connect Four.
    Uses token constants: C4_EMPTY_CELL=1, C4_PLAYER1_CELL=2, C4_PLAYER2_CELL=3
    """
    
    def __init__(self, depth: int = 4):
        self.depth = depth
        self.rows = 6
        self.cols = 7
        self.nodes_evaluated = 0
        
        # Token constants
        self.EMPTY = C4_EMPTY_CELL      # 1
        self.PLAYER1 = C4_PLAYER1_CELL  # 2
        self.PLAYER2 = C4_PLAYER2_CELL  # 3
        
        # Precompute winning patterns
        self.win_patterns = []
        
        # Horizontal
        for r in range(6):
            for c in range(4):
                self.win_patterns.append([(r, c+i) for i in range(4)])
        
        # Vertical
        for r in range(3):
            for c in range(7):
                self.win_patterns.append([(r+i, c) for i in range(4)])
        
        # Diagonal /
        for r in range(3):
            for c in range(4):
                self.win_patterns.append([(r+i, c+i) for i in range(4)])
        
        # Diagonal \
        for r in range(3, 6):
            for c in range(4):
                self.win_patterns.append([(r-i, c+i) for i in range(4)])
    
    def get_board_from_env(self, vec_env, env_idx: int = 0) -> np.ndarray:
        """
        Get board from vectorized environment.
        Environment uses: C4_EMPTY_CELL=1, C4_PLAYER1_CELL=2, C4_PLAYER2_CELL=3
        """
        board = vec_env.boards[env_idx].cpu().numpy()
        return board
    
    def check_winner(self, board: np.ndarray) -> int:
        """Check for winner. Returns 0 (none), 1 (player 1), or 2 (player 2)."""
        for pattern in self.win_patterns:
            vals = [board[r, c] for r, c in pattern]
            # Check if all positions have the same non-empty value
            if vals[0] != self.EMPTY and all(v == vals[0] for v in vals):
                # Return player number (1 or 2) based on token
                if vals[0] == self.PLAYER1:
                    return 1
                elif vals[0] == self.PLAYER2:
                    return 2
        return 0
    
    def get_valid_moves(self, board: np.ndarray) -> List[int]:
        """Get list of valid moves (columns with space in top row)."""
        return [c for c in range(7) if board[0, c] == self.EMPTY]
    
    def make_move(self, board: np.ndarray, col: int, player: int) -> Optional[np.ndarray]:
        """Make a move on the board. Player is 1 or 2."""
        if col < 0 or col >= self.cols or board[0, col] != self.EMPTY:
            return None
        
        new_board = board.copy()
        # Find lowest empty row
        for row in range(self.rows - 1, -1, -1):
            if new_board[row, col] == self.EMPTY:
                # Place the appropriate token
                player_token = self.PLAYER1 if player == 1 else self.PLAYER2
                new_board[row, col] = player_token
                break
        
        return new_board
    
    def evaluate_position(self, board: np.ndarray, player: int) -> float:
        """Evaluate the board position from player's perspective (player is 1 or 2)."""
        score = 0
        opponent = 3 - player  # If player is 1, opponent is 2; if player is 2, opponent is 1
        player_token = self.PLAYER1 if player == 1 else self.PLAYER2
        opponent_token = self.PLAYER1 if opponent == 1 else self.PLAYER2
        
        # Check all potential winning lines
        for pattern in self.win_patterns:
            vals = [board[r, c] for r, c in pattern]
            player_count = sum(1 for v in vals if v == player_token)
            opponent_count = sum(1 for v in vals if v == opponent_token)
            
            # Lines available for player
            if opponent_count == 0 and player_count > 0:
                if player_count == 3:
                    score += 50
                elif player_count == 2:
                    score += 10
                elif player_count == 1:
                    score += 1
            
            # Lines available for opponent
            if player_count == 0 and opponent_count > 0:
                if opponent_count == 3:
                    score -= 50
                elif opponent_count == 2:
                    score -= 10
                elif opponent_count == 1:
                    score -= 1
        
        # Center column preference
        center = self.cols // 2
        for row in range(self.rows):
            if board[row, center] == player_token:
                score += 3
            elif board[row, center] == opponent_token:
                score -= 3
        
        return score
    
    def minimax(self, board: np.ndarray, depth: int, alpha: float, beta: float, 
                maximizing: bool, player: int) -> float:
        """Minimax with alpha-beta pruning. Player is 1 or 2."""
        self.nodes_evaluated += 1
        
        # Check terminal states
        winner = self.check_winner(board)
        if winner == player:
            return 10000 - (self.depth - depth)  # Prefer quicker wins
        elif winner == (3 - player):
            return -10000 + (self.depth - depth)  # Prefer slower losses
        
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return 0  # Draw
        
        if depth == 0:
            return self.evaluate_position(board, player)
        
        # Order moves (center first)
        valid_moves.sort(key=lambda c: abs(c - 3))
        
        if maximizing:
            max_eval = float('-inf')
            for col in valid_moves:
                new_board = self.make_move(board, col, player)
                if new_board is not None:
                    eval_score = self.minimax(new_board, depth-1, alpha, beta, False, player)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            opponent = 3 - player
            for col in valid_moves:
                new_board = self.make_move(board, col, opponent)
                if new_board is not None:
                    eval_score = self.minimax(new_board, depth-1, alpha, beta, True, player)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
            return min_eval
    
    def get_best_move(self, board: np.ndarray, player: int) -> int:
        """Get the best move for the given player (1 or 2)."""
        valid_moves = self.get_valid_moves(board)
        
        if not valid_moves:
            return -1
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Check for immediate wins
        for col in valid_moves:
            test_board = self.make_move(board, col, player)
            if test_board is not None and self.check_winner(test_board) == player:
                return col
        
        # Check for immediate blocks
        opponent = 3 - player
        for col in valid_moves:
            test_board = self.make_move(board, col, opponent)
            if test_board is not None and self.check_winner(test_board) == opponent:
                return col  # Must block
        
        # Use minimax for best move
        best_move = valid_moves[0]
        best_score = float('-inf')
        
        # Reset node counter for this move
        self.nodes_evaluated = 0
        
        for col in valid_moves:
            new_board = self.make_move(board, col, player)
            if new_board is not None:
                score = self.minimax(new_board, self.depth-1, float('-inf'), 
                                   float('inf'), False, player)
                if score > best_score:
                    best_score = score
                    best_move = col
        
        return best_move


if __name__ == "__main__":
    # Test the minimax player with new token values
    print("Testing ConnectFourMinimax with token constants")
    print(f"EMPTY={C4_EMPTY_CELL}, PLAYER1={C4_PLAYER1_CELL}, PLAYER2={C4_PLAYER2_CELL}")
    print("-" * 40)
    
    minimax = ConnectFourMinimax(depth=4)
    
    # Test with empty board (all cells = C4_EMPTY_CELL)
    board = np.full((6, 7), C4_EMPTY_CELL, dtype=int)
    print("Empty board - Player 1's turn:")
    best_move = minimax.get_best_move(board, player=1)
    print(f"Best move: column {best_move}")
    print(f"Nodes evaluated: {minimax.nodes_evaluated}")
    
    # Test with a specific position using proper tokens
    print("\n" + "-" * 40)
    board = np.full((6, 7), C4_EMPTY_CELL, dtype=int)
    board[5, 0] = C4_PLAYER1_CELL
    board[5, 1] = C4_PLAYER1_CELL
    board[5, 2] = C4_PLAYER2_CELL
    board[5, 3] = C4_PLAYER1_CELL
    board[5, 4] = C4_PLAYER2_CELL
    
    print("Test position:")
    symbols = {C4_EMPTY_CELL: ".", C4_PLAYER1_CELL: "X", C4_PLAYER2_CELL: "O"}
    for row in board:
        print(" ".join(symbols[v] for v in row))
    
    print("\nPlayer 2's turn:")
    best_move = minimax.get_best_move(board, player=2)
    print(f"Best move: column {best_move}")
    
    # Test win detection with proper tokens
    print("\n" + "-" * 40)
    win_board = np.full((6, 7), C4_EMPTY_CELL, dtype=int)
    win_board[5, 0] = C4_PLAYER1_CELL
    win_board[5, 1] = C4_PLAYER1_CELL
    win_board[5, 2] = C4_PLAYER1_CELL
    win_board[5, 4] = C4_PLAYER2_CELL
    win_board[5, 5] = C4_PLAYER2_CELL
    
    print("Win detection test (Player 1 can win):")
    for row in win_board:
        print(" ".join(symbols[v] for v in row))
    
    best_move = minimax.get_best_move(win_board, player=1)
    print(f"Player 1 should play column 3 to win: {best_move}")
    
    best_move = minimax.get_best_move(win_board, player=2)
    print(f"Player 2 should block at column 3: {best_move}")