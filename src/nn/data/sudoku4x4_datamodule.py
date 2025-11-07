"""
Sudoku 4x4 DataModule - Version 2
Matches the exact format from the reference implementation
"""

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
import os
import json
from src.nn.utils.constants import IGNORE_LABEL_ID

class Sudoku4x4Dataset(Dataset):
    """
    Dataset for 4x4 Sudoku puzzles - matches reference format exactly.
    Can either load from pre-generated .npy files or generate on-the-fly.
    """
    
    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        num_puzzles: int = 2000,
        min_givens: int = 6,
        max_givens: int = 10,
        max_grid_size: int = 6,
        seed: int = 42,
        generate_on_fly: bool = True
    ):
        """
        Args:
            data_dir: Directory with pre-generated data (if not generating on-the-fly)
            split: 'train' or 'test'
            num_puzzles: Number of puzzles to generate
            min_givens: Minimum numbers given
            max_givens: Maximum numbers given
            max_grid_size: Padding size (6 for 4x4 Sudoku)
            seed: Random seed
            generate_on_fly: If True, generate data. If False, load from data_dir.
        """
        self.split = split
        self.num_puzzles = num_puzzles
        self.min_givens = min_givens
        self.max_givens = max_givens
        self.max_grid_size = max_grid_size
        self.seed = seed + (0 if split == "train" else 1000)
        self.generate_on_fly = generate_on_fly
        
        if generate_on_fly:
            # Set up RNG for generation
            self.rng = np.random.RandomState(self.seed)
        else:
            # Load pre-generated data
            assert data_dir is not None, "data_dir required when not generating on-the-fly"
            self.load_from_files(data_dir, split)
    
    def load_from_files(self, data_dir: str, split: str):
        """Load pre-generated dataset from .npy files."""
        split_dir = os.path.join(data_dir, split)
        
        # Load arrays
        self.inputs = np.load(os.path.join(split_dir, "all__inputs.npy"))
        self.labels = np.load(os.path.join(split_dir, "all__labels.npy"))
        self.puzzle_identifiers = np.load(os.path.join(split_dir, "all__puzzle_identifiers.npy"))
        
        # Load metadata
        with open(os.path.join(split_dir, "dataset.json"), "r") as f:
            self.metadata = json.load(f)
        
        self.num_puzzles = len(self.inputs)
    
    def generate_sudoku_puzzle(self, num_givens: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a valid Sudoku puzzle with specified number of givens."""
        # Start with a valid complete grid
        base = np.array([
            [1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1]
        ], dtype=np.int32)
        
        # Shuffle to create variety
        # Permute numbers
        perm = self.rng.permutation(4) + 1
        complete = np.zeros_like(base)
        for i in range(4):
            complete[base == i+1] = perm[i]
        
        # Shuffle rows within bands
        if self.rng.rand() > 0.5:
            complete[[0, 1]] = complete[[1, 0]]
        if self.rng.rand() > 0.5:
            complete[[2, 3]] = complete[[3, 2]]
        
        # Shuffle columns within stacks
        if self.rng.rand() > 0.5:
            complete[:, [0, 1]] = complete[:, [1, 0]]
        if self.rng.rand() > 0.5:
            complete[:, [2, 3]] = complete[:, [3, 2]]
        
        # Create puzzle by removing cells
        puzzle = complete.copy()
        positions = [(r, c) for r in range(4) for c in range(4)]
        self.rng.shuffle(positions)
        
        cells_to_remove = 16 - num_givens
        for i, (r, c) in enumerate(positions[:cells_to_remove]):
            puzzle[r, c] = 0
        
        return puzzle, complete
    
    def pad_and_encode(self, puzzle: np.ndarray, solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad grids and encode with reference format:
        0=PAD, 1=EOS, 2=empty, 3-6=values 1-4
        """
        # Add 2 offset: 0->2 (empty), 1-4 -> 3-6 (values)
        puzzle_shifted = puzzle.copy()
        puzzle_shifted[puzzle == 0] = 2  # Empty cells
        puzzle_shifted[puzzle > 0] = puzzle[puzzle > 0] + 2  # Values 1-4 -> 3-6
        
        solution_shifted = solution + 2  # Values 1-4 -> 3-6
        
        # Create padded arrays
        inp_padded = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int32) + IGNORE_LABEL_ID
        labels_padded = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int32) + IGNORE_LABEL_ID
        
        # Fill in the actual data
        inp_padded[:4, :4] = puzzle_shifted
        labels_padded[:4, :4] = solution_shifted
        
        # Add EOS marker
        if self.max_grid_size > 4:
            inp_padded[4, 0] = 1  # EOS
            labels_padded[4, 0] = 1  # EOS in labels too
        
        return inp_padded.flatten(), labels_padded.flatten()
    
    def __len__(self):
        return self.num_puzzles
    
    def __getitem__(self, idx):
        """Get a single sample."""
        if self.generate_on_fly:
            # Use idx as seed for this specific sample
            sample_rng = np.random.RandomState(self.seed + idx * 1000)
            self.rng = sample_rng
            
            # Generate puzzle with varying difficulty
            num_givens = self.min_givens + (idx % (self.max_givens - self.min_givens + 1))
            puzzle, solution = self.generate_sudoku_puzzle(num_givens)
            
            # Encode and pad
            input_flat, labels_flat = self.pad_and_encode(puzzle, solution)
            
            # Reset RNG
            self.rng = np.random.RandomState(self.seed)
            
            puzzle_id = 0  # All Sudoku puzzles have ID 0
        else:
            # Load from pre-generated arrays
            input_flat = self.inputs[idx]
            labels_flat = self.labels[idx]
            puzzle_id = self.puzzle_identifiers[idx]
        
        # Reshape to grid format for compatibility with TRM
        input_grid = input_flat.reshape(self.max_grid_size, self.max_grid_size)
        labels_grid = labels_flat.reshape(self.max_grid_size, self.max_grid_size)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_grid).long()
        labels_tensor = torch.from_numpy(labels_grid).long()
        
        return {
            "input": input_tensor,
            "output": labels_tensor,  # Using 'output' for compatibility
            "puzzle_identifiers": puzzle_id
        }


def collate_fn_sudoku(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    # Stack inputs
    inputs = torch.stack([sample['input'] for sample in batch])
    
    # Stack outputs
    outputs = torch.stack([sample['output'] for sample in batch])
    
    # Stack puzzle identifiers
    puzzle_ids = torch.tensor(
        [sample['puzzle_identifiers'] for sample in batch], 
        dtype=torch.long
    )
    
    return {
        "input": inputs,
        "output": outputs,
        "puzzle_identifiers": puzzle_ids
    }


class Sudoku4x4DataModule(LightningDataModule):
    """
    Lightning DataModule for 4x4 Sudoku puzzles.
    Matches reference implementation format exactly.
    """
    
    def __init__(
        self,
        data_dir: str = None,
        batch_size: int = 128,
        num_train_puzzles: int = 2000,
        num_val_puzzles: int = 400,  # Using as test
        num_test_puzzles: int = 400,
        min_givens: int = 6,
        max_givens: int = 10,
        max_grid_size: int = 6,
        num_workers: int = 4,
        seed: int = 42,
        generate_on_fly: bool = True,
        pad_value: int = 0,
    ):
        """
        Args:
            data_dir: Directory with pre-generated data (if not generating)
            batch_size: Batch size
            num_train_puzzles: Number of training puzzles
            num_val_puzzles: Number of validation puzzles (uses test split)
            num_test_puzzles: Number of test puzzles
            min_givens: Minimum numbers given
            max_givens: Maximum numbers given
            max_grid_size: Padding size (6 for 4x4 Sudoku)
            num_workers: DataLoader workers
            seed: Random seed
            generate_on_fly: If True, generate data. If False, load from data_dir.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_train_puzzles = num_train_puzzles
        self.num_val_puzzles = num_val_puzzles
        self.num_test_puzzles = num_test_puzzles
        self.min_givens = min_givens
        self.max_givens = max_givens
        self.max_grid_size = max_grid_size
        self.num_workers = num_workers
        self.seed = seed
        self.generate_on_fly = generate_on_fly
        self.pad_value = pad_value

        num_colors = 4  # 4 values

        # Metadata matching reference
        self.num_puzzles = 1  # All Sudoku puzzles share ID 0
        self.vocab_size = 3 + num_colors   # -100=PAD, 1=EOS, 2=empty, 3-6=values 1-4
        self.seq_len = max_grid_size * max_grid_size
        
    def setup(self, stage: Optional[str] = None):
        """Create datasets for each split."""
        if stage == "fit" or stage is None:
            self.train_dataset = Sudoku4x4Dataset(
                data_dir=self.data_dir,
                split="train",
                num_puzzles=self.num_train_puzzles,
                min_givens=self.min_givens,
                max_givens=self.max_givens,
                max_grid_size=self.max_grid_size,
                seed=self.seed,
                generate_on_fly=self.generate_on_fly
            )
            
            # Validation uses test split (matching reference)
            self.val_dataset = Sudoku4x4Dataset(
                data_dir=self.data_dir,
                split="test",
                num_puzzles=self.num_val_puzzles,
                min_givens=self.min_givens,
                max_givens=self.max_givens,
                max_grid_size=self.max_grid_size,
                seed=self.seed,
                generate_on_fly=self.generate_on_fly
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = Sudoku4x4Dataset(
                data_dir=self.data_dir,
                split="test",
                num_puzzles=self.num_test_puzzles,
                min_givens=self.min_givens,
                max_givens=self.max_givens,
                max_grid_size=self.max_grid_size,
                seed=self.seed,
                generate_on_fly=self.generate_on_fly
            )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn_sudoku,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_sudoku,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_sudoku,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True
        )


if __name__ == "__main__":
    # Test the datamodule
    print("Testing Sudoku 4x4 DataModule (Reference Format)")
    print("="*60)
    
    # Test 1: Generate on-the-fly
    print("\nTest 1: Generating data on-the-fly")
    print("-"*40)
    dm = Sudoku4x4DataModule(
        batch_size=4,
        num_train_puzzles=10,
        num_val_puzzles=2,
        generate_on_fly=True
    )
    
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input'].shape}")
    print(f"Output shape: {batch['output'].shape}")
    print(f"Puzzle IDs: {batch['puzzle_identifiers'].unique()}")
    print(f"Vocab size: {dm.vocab_size} (0=PAD, 1=EOS, 2=empty, 3-6=values)")
    print(f"Sequence length: {dm.seq_len}")
    
    # Show first example
    print("\nFirst puzzle (6x6 padded):")
    print("Input encoding:")
    print(batch['input'][0].numpy())
    print("\nOutput (solution):")
    print(batch['output'][0].numpy())
    
    # Extract 4x4 and decode
    input_4x4 = batch['input'][0][:4, :4].numpy()
    output_4x4 = batch['output'][0][:4, :4].numpy()
    
    # Decode: 2=empty, 3-6 = numbers 1-4
    puzzle_decoded = input_4x4.copy()
    puzzle_decoded[input_4x4 == 2] = 0  # Empty
    puzzle_decoded[input_4x4 > 2] = input_4x4[input_4x4 > 2] - 2  # Numbers
    
    solution_decoded = output_4x4 - 2  # All should be 3-6 -> 1-4
    
    print("\nDecoded 4x4 puzzle (0=empty, 1-4=values):")
    print(puzzle_decoded)
    print("\nDecoded solution:")
    print(solution_decoded)
    
    # Check EOS marker
    if batch['input'][0, 4, 0] == 1:
        print("\n✓ EOS marker found at position [4, 0]")
    
    # Test 2: Would load from files (if they exist)
    print("\n" + "="*60)
    print("Test 2: Loading from files")
    print("-"*40)
    print("To use pre-generated data:")
    print("1. Run: python build_sudoku4x4_dataset.py")
    print("2. Then use: Sudoku4x4DataModule(data_dir='data/sudoku4x4', generate_on_fly=False)")
    
    print("\n✅ DataModule ready for nano-TRM!")