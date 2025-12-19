"""
Connect Four Environment compatible with mcts_v2.py

Inherits from BoardGameEnv to work with the reference MCTS implementation.
"""

import numpy as np
from typing import Tuple, Optional
from collections import deque
from copy import copy

from src.nn.environments.base import BoardGameEnv


class ConnectFourEnv(BoardGameEnv):
    """
    Connect Four environment compatible with BoardGameEnv interface.
    
    Board is 6 rows x 7 columns.
    Actions are column indices (0-6).
    Player 1 (X) goes first, Player 2 (O) goes second.
    """
    
    def __init__(self, num_stack: int = 4):
        """
        Initialize Connect Four environment.
        
        Args:
            num_stack: Number of history frames to stack for observation
        """
        # Don't call super().__init__ directly since BoardGameEnv assumes square board
        # Instead, initialize the required attributes manually
        
        self.id = "ConnectFour"
        self.rows = 6
        self.cols = 7
        self.board_size = 7  # For compatibility, use cols as "board_size"
        self.num_stack = num_stack
        
        # Board state: 6x7, using int8 for compatibility with BoardGameEnv
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        
        # Player IDs
        self.black_player = 1  # Player 1 (X)
        self.white_player = 2  # Player 2 (O)
        
        # Action space: 7 columns
        self.action_dim = self.cols
        self.has_pass_move = False
        self.has_resign_move = False
        self.pass_move = None
        self.resign_move = -1
        
        # Legal actions mask
        self.legal_actions = np.ones(self.action_dim, dtype=np.float32)
        
        # Game state
        self.to_play = self.black_player
        self.steps = 0
        self.winner = None
        self.last_player = None
        self.last_move = None
        
        # History tracking
        self.board_deltas = self._get_empty_queue()
        self.history = []
    
    def _get_empty_queue(self) -> deque:
        """Returns empty queue with num_stack zeros planes."""
        return deque(
            [np.zeros((self.rows, self.cols), dtype=np.int8)] * self.num_stack,
            maxlen=self.num_stack
        )
    
    def reset(self, **kwargs) -> np.ndarray:
        """Reset game to initial state."""
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.legal_actions = np.ones(self.action_dim, dtype=np.float32)
        self.to_play = self.black_player
        self.steps = 0
        self.winner = None
        self.last_player = None
        self.last_move = None
        self.board_deltas = self._get_empty_queue()
        self.history.clear()
        
        return self.observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute a move (drop piece in column).
        
        Args:
            action: Column index (0-6)
            
        Returns:
            (observation, reward, done, info)
            Reward is from perspective of player who just moved.
        """
        if self.is_game_over():
            raise RuntimeError("Game is over, call reset before using step method.")
        
        if action == self.resign_move:
            # Handle resignation
            self.winner = self.opponent_player
            self.last_player = copy(self.to_play)
            reward = -1.0  # Resigning player loses
            return self.observation(), reward, True, {}
        
        if not 0 <= action < self.action_dim:
            raise ValueError(f"Invalid action: {action}. Must be 0-{self.action_dim-1}")
        
        if self.legal_actions[action] != 1:
            raise ValueError(f"Illegal action: column {action} is full")
        
        # Find lowest empty row in the column
        col = self.board[:, action]
        empty_rows = np.where(col == 0)[0]
        
        if len(empty_rows) == 0:
            raise ValueError(f"Column {action} is full")
        
        # Place piece in lowest empty row (highest index)
        row = empty_rows[-1]
        self.board[row, action] = self.to_play
        
        # Update game state
        self.last_move = copy(action)
        self.last_player = copy(self.to_play)
        self.steps += 1
        
        # Add to history
        self.history.append((self.last_player, self.last_move))
        
        # Update board history for observation stacking
        self.board_deltas.appendleft(np.copy(self.board))
        
        # Update legal actions (column full if top row is occupied)
        if self.board[0, action] != 0:
            self.legal_actions[action] = 0
        
        # Check for winner
        self.winner = self._check_winner()
        
        # Calculate reward from perspective of player who just moved
        reward = 0.0
        if self.winner == self.last_player:
            reward = 1.0
        elif self.winner is not None and self.winner != 0:
            reward = -1.0
        # Draw (winner == 0) gives reward 0
        
        # Switch player
        self.to_play = self.opponent_player
        
        done = self.is_game_over()
        
        return self.observation(), reward, done, {}
    
    def _check_winner(self) -> Optional[int]:
        """
        Check for a winner (4 in a row).
        
        Returns:
            Player ID (1 or 2) if winner, 0 if draw, None if game ongoing
        """
        # Check for each player
        for player in [self.black_player, self.white_player]:
            # Horizontal
            for row in range(self.rows):
                for col in range(self.cols - 3):
                    if np.all(self.board[row, col:col+4] == player):
                        return player
            
            # Vertical
            for row in range(self.rows - 3):
                for col in range(self.cols):
                    if np.all(self.board[row:row+4, col] == player):
                        return player
            
            # Diagonal (down-right)
            for row in range(self.rows - 3):
                for col in range(self.cols - 3):
                    if all(self.board[row+i, col+i] == player for i in range(4)):
                        return player
            
            # Diagonal (down-left)
            for row in range(self.rows - 3):
                for col in range(3, self.cols):
                    if all(self.board[row+i, col-i] == player for i in range(4)):
                        return player
        
        # Check for draw (board full)
        if np.all(self.board != 0):
            return 0  # Draw
        
        return None  # Game ongoing
    
    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        return self.winner is not None
    
    @property
    def opponent_player(self) -> int:
        """Get the opponent's player ID."""
        return self.white_player if self.to_play == self.black_player else self.black_player
    
    def observation(self) -> np.ndarray:
        """
        Get stacked observation planes.
        
        Returns:
            Array of shape (num_stack * 2 + 1, rows, cols)
            - Even indices: current player's pieces in history
            - Odd indices: opponent's pieces in history
            - Last plane: color to play (1 if black/P1, 0 if white/P2)
        """
        features = np.zeros((self.num_stack * 2, self.rows, self.cols), dtype=np.int8)
        
        deltas = np.array(list(self.board_deltas))
        
        # Current player's pieces, then opponent's
        features[::2] = (deltas == self.to_play).astype(np.int8)
        features[1::2] = (deltas == self.opponent_player).astype(np.int8)
        
        # Color to play plane
        color_to_play = np.ones((1, self.rows, self.cols), dtype=np.int8) if self.to_play == self.black_player else np.zeros((1, self.rows, self.cols), dtype=np.int8)
        
        return np.concatenate([features, color_to_play], axis=0)
    
    def render(self, mode: str = 'terminal') -> str:
        """Render the board as a string."""
        result = "\n"
        result += f"  {self.id} - "
        result += f"{'X' if self.to_play == self.black_player else 'O'} to play\n"
        result += f"  Steps: {self.steps}"
        if self.winner is not None:
            if self.winner == 0:
                result += " - Draw!"
            else:
                result += f" - {'X' if self.winner == self.black_player else 'O'} wins!"
        result += "\n\n"
        
        # Column labels
        result += "  " + " ".join(str(i) for i in range(self.cols)) + "\n"
        result += "  " + "-" * (self.cols * 2 - 1) + "\n"
        
        symbols = {0: ".", self.black_player: "X", self.white_player: "O"}
        
        for row in range(self.rows):
            result += "| "
            for col in range(self.cols):
                cell = symbols[self.board[row, col]]
                # Highlight last move
                if self.last_move == col and self._get_piece_row(col) == row:
                    cell = f"({cell})"
                else:
                    cell = f" {cell}"
                result += cell
            result += " |\n"
        
        result += "  " + "-" * (self.cols * 2 - 1) + "\n"
        
        return result
    
    def _get_piece_row(self, col: int) -> int:
        """Get the row of the topmost piece in a column, or -1 if empty."""
        for row in range(self.rows):
            if self.board[row, col] != 0:
                return row
        return -1
    
    def action_to_coords(self, action: int) -> Tuple[int, int]:
        """
        Convert action to board coordinates.
        For Connect Four, action is column, row is determined by gravity.
        Returns the position where a piece would land.
        """
        if action is None or action < 0 or action >= self.cols:
            return (-1, -1)
        
        # Find where piece would land
        col = self.board[:, action]
        empty_rows = np.where(col == 0)[0]
        
        if len(empty_rows) == 0:
            return (-1, action)  # Column full
        
        return (empty_rows[-1], action)
    
    def coords_to_action(self, coords: Tuple[int, int]) -> int:
        """Convert coordinates to action (just return the column)."""
        _, col = coords
        if 0 <= col < self.cols:
            return col
        return None
    
    def get_result_string(self) -> str:
        """Get game result as string."""
        if self.winner is None:
            return "ongoing"
        elif self.winner == 0:
            return "draw"
        elif self.winner == self.black_player:
            return "X+win"
        else:
            return "O+win"
    
    def clone(self) -> 'ConnectFourEnv':
        """Create a deep copy of the environment."""
        new_env = ConnectFourEnv(num_stack=self.num_stack)
        new_env.board = np.copy(self.board)
        new_env.legal_actions = np.copy(self.legal_actions)
        new_env.to_play = self.to_play
        new_env.steps = self.steps
        new_env.winner = self.winner
        new_env.last_player = self.last_player
        new_env.last_move = self.last_move
        new_env.board_deltas = deque(
            [np.copy(b) for b in self.board_deltas],
            maxlen=self.num_stack
        )
        new_env.history = list(self.history)
        return new_env
    
    def __deepcopy__(self, memo):
        """Support for copy.deepcopy() used by mcts_v2."""
        return self.clone()


# Convenience functions for testing
def create_board_from_string(board_str: str) -> np.ndarray:
    """
    Create a board from a string representation.
    
    Example:
        board_str = '''
        .......
        .......
        .......
        .......
        ....X..
        XXX.OO.
        '''
    """
    lines = [line.strip() for line in board_str.strip().split('\n') if line.strip()]
    board = np.zeros((6, 7), dtype=np.int8)
    
    for row, line in enumerate(lines):
        for col, char in enumerate(line):
            if char == 'X':
                board[row, col] = 1
            elif char == 'O':
                board[row, col] = 2
    
    return board


def create_env_from_board(board: np.ndarray, to_play: int = 1) -> ConnectFourEnv:
    """Create an environment with a specific board state."""
    env = ConnectFourEnv()
    env.board = np.copy(board).astype(np.int8)
    env.to_play = to_play
    
    # Count moves
    env.steps = np.sum(board != 0)
    
    # Update legal actions
    env.legal_actions = (board[0, :] == 0).astype(np.float32)
    
    # Update board deltas
    env.board_deltas.appendleft(np.copy(board))
    
    # Check if game is already over
    env.winner = env._check_winner()
    
    return env


if __name__ == "__main__":
    # Quick test
    env = ConnectFourEnv()
    print(env.render())
    
    # Play a few moves
    for col in [3, 3, 2, 4, 1, 5, 0]:  # P1 wins horizontally
        obs, reward, done, info = env.step(col)
        print(env.render())
        if done:
            print(f"Game over! Reward: {reward}")
            break
    
    # Test creating from board
    board = create_board_from_string('''
        .......
        .......
        .......
        .......
        ....O..
        XXX.OO.
    ''')
    env2 = create_env_from_board(board, to_play=1)
    print("\nCreated from string:")
    print(env2.render())
    print(f"Legal actions: {env2.legal_actions}")