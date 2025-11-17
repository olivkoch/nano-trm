"""
Vectorized Connect Four Environment for parallel game execution
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class VectorizedC4State:
    """Vectorized state for multiple Connect Four games"""
    boards: torch.Tensor  # (n_envs, 6, 7)
    current_players: torch.Tensor  # (n_envs,)
    move_counts: torch.Tensor  # (n_envs,)
    winners: torch.Tensor  # (n_envs,) - 0 if no winner, 1/2 for player wins
    is_terminal: torch.Tensor  # (n_envs,) bool
    legal_moves: torch.Tensor  # (n_envs, 7) bool mask
    
    def to_trm_input(self) -> torch.Tensor:
        """Convert board states to token sequences for TRM input.
        Returns: (n_envs, 42) tensor of tokens
        """
        # Flatten boards and shift values (0->1, 1->2, 2->3)
        tokens = self.boards.flatten(1) + 1
        return tokens.long()
    
    @property
    def n_envs(self):
        return self.boards.shape[0]


class VectorizedConnectFour:
    """Vectorized Connect Four for parallel game execution"""
    
    def __init__(self, n_envs: int = 8, device: str = "cpu"):
        self.n_envs = n_envs
        self.device = device
        self.rows = 6
        self.cols = 7
        self.win_length = 4
        
        # Pre-compute win checking kernels on device
        self._setup_win_kernels()
        
        self.reset()
    
    def _setup_win_kernels(self):
        """Pre-compute convolution kernels for win checking"""
        # Horizontal kernel (1x4)
        self.kernel_h = torch.ones(1, 1, 1, 4, device=self.device)
        
        # Vertical kernel (4x1)
        self.kernel_v = torch.ones(1, 1, 4, 1, device=self.device)
        
        # Diagonal \ kernel (4x4 identity)
        self.kernel_d1 = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0)
        
        # Diagonal / kernel (4x4 anti-identity)
        self.kernel_d2 = torch.flip(torch.eye(4, device=self.device), [0]).unsqueeze(0).unsqueeze(0)
    
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> VectorizedC4State:
        """Reset all or specified environments"""
        if env_ids is None:
            # Reset all
            self.boards = torch.zeros((self.n_envs, self.rows, self.cols), 
                                     dtype=torch.int32, device=self.device)
            self.current_players = torch.ones(self.n_envs, dtype=torch.int32, 
                                             device=self.device)
            self.move_counts = torch.zeros(self.n_envs, dtype=torch.int32, 
                                          device=self.device)
            self.winners = torch.zeros(self.n_envs, dtype=torch.int32, 
                                      device=self.device)
            self.is_terminal = torch.zeros(self.n_envs, dtype=torch.bool, 
                                          device=self.device)
        else:
            # Reset specific environments
            self.boards[env_ids] = 0
            self.current_players[env_ids] = 1
            self.move_counts[env_ids] = 0
            self.winners[env_ids] = 0
            self.is_terminal[env_ids] = False
        
        return self.get_state()
    
    def get_state(self) -> VectorizedC4State:
        """Get current state of all environments"""
        legal_moves = self.boards[:, 0, :] == 0  # Top row empty = legal
        
        return VectorizedC4State(
            boards=self.boards.clone(),
            current_players=self.current_players.clone(),
            move_counts=self.move_counts.clone(),
            winners=self.winners.clone(),
            is_terminal=self.is_terminal.clone(),
            legal_moves=legal_moves
        )
    
    def step(self, actions: torch.Tensor) -> VectorizedC4State:
        """Execute moves in all environments
        
        Args:
            actions: (n_envs,) tensor of column indices
        
        Returns:
            New state after moves
        """
        # Only process non-terminal games
        active = ~self.is_terminal
        
        if not active.any():
            return self.get_state()
        
        # Vectorized move making
        for env_id in range(self.n_envs):
            if not active[env_id]:
                continue
            
            col = actions[env_id].item()
            if col < 0 or col >= self.cols:
                continue
            
            # Find lowest empty row in column
            col_values = self.boards[env_id, :, col]
            empty_rows = (col_values == 0).nonzero(as_tuple=True)[0]
            
            if len(empty_rows) == 0:
                # Invalid move, skip
                continue
            
            row = empty_rows[-1].item()
            self.boards[env_id, row, col] = self.current_players[env_id]
            self.move_counts[env_id] += 1
        
        # Batch check for winners
        self._check_winners_batch()
        
        # Check for draws (board full)
        board_full = (self.boards != 0).all(dim=(1, 2))
        self.is_terminal = self.is_terminal | board_full | (self.winners != 0)
        
        # Switch players (only for non-terminal games)
        self.current_players[active] = 3 - self.current_players[active]
        
        return self.get_state()
    
    def _check_winners_batch(self):
        """Batch check for winners using convolution"""
        for player in [1, 2]:
            # Get player positions as float for convolution
            player_boards = (self.boards == player).float()
            
            # Reshape for conv2d: (n_envs, 1, rows, cols)
            player_boards = player_boards.unsqueeze(1)
            
            # Check all win conditions
            wins = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
            
            # Horizontal
            conv_h = F.conv2d(player_boards, self.kernel_h, padding=(0, 0))
            wins |= (conv_h == 4).any(dim=(1, 2, 3))
            
            # Vertical
            conv_v = F.conv2d(player_boards, self.kernel_v, padding=(0, 0))
            wins |= (conv_v == 4).any(dim=(1, 2, 3))
            
            # Diagonal \
            conv_d1 = F.conv2d(player_boards, self.kernel_d1)
            wins |= (conv_d1 == 4).any(dim=(1, 2, 3))
            
            # Diagonal /
            conv_d2 = F.conv2d(player_boards, self.kernel_d2)
            wins |= (conv_d2 == 4).any(dim=(1, 2, 3))
            
            # Update winners (only if not already won)
            new_wins = wins & (self.winners == 0)
            self.winners[new_wins] = player
    
    def get_rewards(self, player: int) -> torch.Tensor:
        """Get rewards for a specific player
        
        Returns:
            (n_envs,) tensor of rewards
        """
        rewards = torch.zeros(self.n_envs, device=self.device)
        rewards[self.winners == player] = 1.0
        rewards[self.winners == (3 - player)] = -1.0
        # Draws remain 0
        return rewards
    
    def render(self, env_id: int = 0) -> str:
        """Render a specific environment as string"""
        result = f"\nEnvironment {env_id}:\n"
        result += "  " + " ".join(str(i) for i in range(self.cols)) + "\n"
        result += "  " + "-" * (self.cols * 2 - 1) + "\n"
        
        symbols = {0: ".", 1: "X", 2: "O"}
        
        for row in range(self.rows):
            result += "| "
            for col in range(self.cols):
                result += symbols[self.boards[env_id, row, col].item()] + " "
            result += "|\n"
        
        result += "  " + "-" * (self.cols * 2 - 1) + "\n"
        
        if self.is_terminal[env_id]:
            if self.winners[env_id] > 0:
                result += f"Winner: Player {self.winners[env_id].item()}\n"
            else:
                result += "Draw!\n"
        else:
            result += f"Current player: {self.current_players[env_id].item()}\n"
        
        return result