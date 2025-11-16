"""
Connect Four Environment for TRM Self-Play
"""

import torch
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
import numpy as np


# Player constants
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2


@dataclass
class ConnectFourState:
    """State representation for Connect Four"""
    board: torch.Tensor  # (6, 7) tensor
    current_player: int  # 1 or 2
    move_count: int
    winner: Optional[int] = None
    is_terminal: bool = False
    legal_moves: Optional[torch.Tensor] = None  # Mask of legal columns
    
    def to_trm_input(self) -> torch.Tensor:
        """Convert board state to token sequence for TRM input.
        
        Encoding for TRM:
        - 0: padding token (vocab)
        - 1: empty cell
        - 2: player 1 piece  
        - 3: player 2 piece
        """
        # Flatten board and shift values (0->1, 1->2, 2->3)
        tokens = self.board.flatten() + 1
        return tokens.long()
    
    def get_legal_mask(self) -> torch.Tensor:
        """Get mask of legal moves (columns with space at top)"""
        return self.board[0] == EMPTY


class ConnectFourEnv:
    """Connect Four environment with TRM-compatible interface"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.rows = 6
        self.cols = 7
        self.win_length = 4
        self.reset()
    
    def reset(self) -> ConnectFourState:
        """Reset the game to initial state"""
        self.board = torch.zeros((self.rows, self.cols), dtype=torch.int32, device=self.device)
        self.current_player = PLAYER1
        self.move_count = 0
        return self._get_state()
    
    def _get_state(self) -> ConnectFourState:
        """Get current game state"""
        winner = self._check_winner()
        legal_moves = self.board[0] == EMPTY
        is_terminal = winner is not None or legal_moves.sum() == 0
        
        return ConnectFourState(
            board=self.board.clone(),
            current_player=self.current_player,
            move_count=self.move_count,
            winner=winner,
            is_terminal=is_terminal,
            legal_moves=legal_moves
        )
    
    def get_legal_moves(self) -> torch.Tensor:
        """Return mask of legal moves (columns with space)"""
        return self.board[0] == EMPTY
    
    def make_move(self, column: int) -> ConnectFourState:
        """Make a move in the specified column"""
        if not (0 <= column < self.cols):
            raise ValueError(f"Invalid column: {column}")
        
        # Find the lowest empty row in the column
        col_values = self.board[:, column]
        empty_rows = (col_values == EMPTY).nonzero(as_tuple=True)[0]
        
        if len(empty_rows) == 0:
            raise ValueError(f"Column {column} is full")
        
        # Place piece in the lowest empty row
        row = empty_rows[-1].item()
        self.board[row, column] = self.current_player
        
        # Switch players
        self.current_player = PLAYER2 if self.current_player == PLAYER1 else PLAYER1
        self.move_count += 1
        
        return self._get_state()
    
    def _check_winner(self) -> Optional[int]:
        """Check if there's a winner using efficient convolution"""
        for player in [PLAYER1, PLAYER2]:
            player_board = (self.board == player).float()
            
            # Check horizontal (kernel: 1x4)
            kernel_h = torch.ones(1, 1, 1, 4, device=self.device)
            conv_h = F.conv2d(player_board.unsqueeze(0).unsqueeze(0), kernel_h, padding=(0, 0))
            if (conv_h == 4).any():
                return player
            
            # Check vertical (kernel: 4x1)
            kernel_v = torch.ones(1, 1, 4, 1, device=self.device)
            conv_v = F.conv2d(player_board.unsqueeze(0).unsqueeze(0), kernel_v, padding=(0, 0))
            if (conv_v == 4).any():
                return player
            
            # Check diagonal \ (kernel: 4x4 identity)
            kernel_d1 = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0)
            conv_d1 = F.conv2d(player_board.unsqueeze(0).unsqueeze(0), kernel_d1)
            if (conv_d1 == 4).any():
                return player
            
            # Check diagonal / (kernel: 4x4 anti-identity)
            kernel_d2 = torch.flip(torch.eye(4, device=self.device), [0]).unsqueeze(0).unsqueeze(0)
            conv_d2 = F.conv2d(player_board.unsqueeze(0).unsqueeze(0), kernel_d2)
            if (conv_d2 == 4).any():
                return player
        
        return None
    
    def render(self) -> str:
        """Render the board as a string"""
        result = "\n"
        result += "  " + " ".join(str(i) for i in range(self.cols)) + "\n"
        result += "  " + "-" * (self.cols * 2 - 1) + "\n"
        
        symbols = {EMPTY: ".", PLAYER1: "X", PLAYER2: "O"}
        
        for row in range(self.rows):
            result += "| "
            for col in range(self.cols):
                result += symbols[self.board[row, col].item()] + " "
            result += "|\n"
        
        result += "  " + "-" * (self.cols * 2 - 1) + "\n"
        return result
    
    def get_reward(self, player: int) -> float:
        """Get reward for a player given current state"""
        state = self._get_state()
        if state.winner == player:
            return 1.0
        elif state.winner is not None:
            return -1.0
        elif state.is_terminal:
            return 0.0  # Draw
        else:
            return 0.0  # Game not finished
    
    def clone(self) -> 'ConnectFourEnv':
        """Create a deep copy of the environment"""
        new_env = ConnectFourEnv(device=self.device)
        new_env.board = self.board.clone()
        new_env.current_player = self.current_player
        new_env.move_count = self.move_count
        return new_env