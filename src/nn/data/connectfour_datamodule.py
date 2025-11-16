"""
Connect Four DataModule for Lightning
Generates Connect Four positions for training
"""

import torch
from typing import Optional, Dict
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, IterableDataset
import random
import numpy as np

from src.nn.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ConnectFourDataset(IterableDataset):
    """
    Dataset that generates Connect Four positions on-the-fly
    Used for bootstrapping before self-play data is available
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        vocab_size: int = 4,
        board_rows: int = 6,
        board_cols: int = 7,
        include_random_positions: bool = True,
        seed: Optional[int] = None
    ):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.seq_len = board_rows * board_cols
        self.include_random_positions = include_random_positions
        self.seed = seed
    
    def generate_random_position(self) -> Dict[str, torch.Tensor]:
        """Generate a random valid Connect Four position"""
        board = torch.zeros((self.board_rows, self.board_cols), dtype=torch.long)
        
        if self.include_random_positions:
            # Generate random number of moves
            num_moves = random.randint(0, self.seq_len - 7)
            
            # Simulate random valid moves
            column_heights = [0] * self.board_cols
            current_player = 1
            
            for _ in range(num_moves):
                # Find valid columns
                valid_cols = [c for c in range(self.board_cols) 
                             if column_heights[c] < self.board_rows]
                
                if not valid_cols:
                    break
                
                # Make random move
                col = random.choice(valid_cols)
                row = self.board_rows - 1 - column_heights[col]
                board[row, col] = current_player
                column_heights[col] += 1
                
                # Switch player
                current_player = 3 - current_player
        
        # Convert to TRM format (shift values: 0->1, 1->2, 2->3)
        input_seq = (board.flatten() + 1).long()
        
        # For self-play, output is same as input (will be overridden by game outcomes)
        output_seq = input_seq.clone()
        
        # Random puzzle ID
        puzzle_id = random.randint(0, 9999)
        
        return {
            'input': input_seq,
            'output': output_seq,
            'puzzle_identifiers': torch.tensor(puzzle_id, dtype=torch.long)
        }
    
    def __iter__(self):
        """Generate samples infinitely"""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        samples_generated = 0
        while samples_generated < self.num_samples:
            yield self.generate_random_position()
            samples_generated += 1
    
    def __len__(self):
        return self.num_samples





class ConnectFourDataModule(LightningDataModule):
    """
    Lightning DataModule for Connect Four
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        num_train_samples: int = 100000,
        num_val_samples: int = 1000,
        vocab_size: int = 4,
        board_rows: int = 6,
        board_cols: int = 7,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.batch_size = batch_size
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.vocab_size = vocab_size
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Properties needed by training script
        self.num_puzzles = 10000  # Max unique positions to track
        self.pad_value = 0
        self.max_grid_size = max(board_rows, board_cols)
        self.seq_len = board_rows * board_cols
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: str):
        """Setup datasets"""
        if stage == "fit" or stage is None:
            # Create Connect Four datasets
            self.train_dataset = ConnectFourDataset(
                num_samples=self.num_train_samples,
                vocab_size=self.vocab_size,
                board_rows=self.board_rows,
                board_cols=self.board_cols,
                include_random_positions=True,
                seed=42
            )
            
            self.val_dataset = ConnectFourDataset(
                num_samples=self.num_val_samples,
                vocab_size=self.vocab_size,
                board_rows=self.board_rows,
                board_cols=self.board_cols,
                include_random_positions=True,
                seed=123
            )
            
            log.info("Connect Four datasets created")
            log.info(f"  Train samples per epoch: {self.num_train_samples}")
            log.info(f"  Val samples: {self.num_val_samples}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )