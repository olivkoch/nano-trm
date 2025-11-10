"""
Sudoku DataModule - Flexible Version
Supports 4x4, 6x6, and 9x9 Sudoku puzzles
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.nn.utils.constants import IGNORE_LABEL_ID


class SudokuDataset(Dataset):
    """
    Dataset for Sudoku puzzles of various sizes (4x4, 6x6, 9x9).
    Can either load from pre-generated .npy files or generate on-the-fly.
    """

    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        num_puzzles: int = 2000,
        min_givens: int = None,
        max_givens: int = None,
        grid_size: int = 4,
        max_grid_size: int = None,
        seed: int = 42,
        generate_on_fly: bool = True,
    ):
        """
        Args:
            data_dir: Directory with pre-generated data (if not generating on-the-fly)
            split: 'train' or 'test'
            num_puzzles: Number of puzzles to generate
            min_givens: Minimum numbers given (defaults to ~35% of cells)
            max_givens: Maximum numbers given (defaults to ~60% of cells)
            grid_size: Sudoku grid size (4, 6, or 9)
            max_grid_size: Padding size (defaults to grid_size + 2)
            seed: Random seed
            generate_on_fly: If True, generate data. If False, load from data_dir.
        """
        self.split = split
        self.num_puzzles = num_puzzles
        self.grid_size = grid_size
        self.max_grid_size = max_grid_size if max_grid_size is not None else grid_size + 2
        self.seed = seed + (0 if split == "train" else 1000)
        self.generate_on_fly = generate_on_fly
        
        # Determine box dimensions based on grid size
        if grid_size == 4:
            self.box_rows, self.box_cols = 2, 2
        elif grid_size == 6:
            self.box_rows, self.box_cols = 2, 3
        elif grid_size == 9:
            self.box_rows, self.box_cols = 3, 3
        else:
            raise ValueError(f"Unsupported grid_size: {grid_size}. Use 4, 6, or 9.")
        
        # Auto-scale min/max givens based on grid size if not provided
        total_cells = grid_size * grid_size
        if min_givens is None:
            self.min_givens = int(total_cells * 0.35)  # ~35% filled
        else:
            self.min_givens = min_givens
            
        if max_givens is None:
            self.max_givens = int(total_cells * 0.60)  # ~60% filled
        else:
            self.max_givens = max_givens

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
        with open(os.path.join(split_dir, "dataset.json")) as f:
            self.metadata = json.load(f)

        self.num_puzzles = len(self.inputs)

    def is_valid_sudoku(self, grid: np.ndarray) -> bool:
        """Check if a Sudoku grid is valid (no duplicates in rows/cols/boxes)."""
        n = self.grid_size
        
        # Check rows and columns
        for i in range(n):
            row = grid[i, :]
            col = grid[:, i]
            row_vals = row[row > 0]
            col_vals = col[col > 0]
            if len(row_vals) != len(set(row_vals)) or len(col_vals) != len(set(col_vals)):
                return False
        
        # Check boxes
        for box_r in range(0, n, self.box_rows):
            for box_c in range(0, n, self.box_cols):
                box = grid[box_r:box_r+self.box_rows, box_c:box_c+self.box_cols].flatten()
                box_vals = box[box > 0]
                if len(box_vals) != len(set(box_vals)):
                    return False
        
        return True

    def solve_sudoku(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """
        Simple backtracking solver for Sudoku.
        Returns solution if unique, None otherwise.
        """
        solution = grid.copy()
        
        def find_empty():
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if solution[i, j] == 0:
                        return (i, j)
            return None
        
        def is_valid_move(row, col, num):
            # Check row
            if num in solution[row, :]:
                return False
            # Check column
            if num in solution[:, col]:
                return False
            # Check box
            box_r = (row // self.box_rows) * self.box_rows
            box_c = (col // self.box_cols) * self.box_cols
            if num in solution[box_r:box_r+self.box_rows, box_c:box_c+self.box_cols]:
                return False
            return True
        
        def solve():
            pos = find_empty()
            if pos is None:
                return True  # Solved
            
            row, col = pos
            for num in range(1, self.grid_size + 1):
                if is_valid_move(row, col, num):
                    solution[row, col] = num
                    if solve():
                        return True
                    solution[row, col] = 0
            
            return False
        
        if solve():
            return solution
        return None

    def generate_sudoku_puzzle(self, num_givens: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a valid Sudoku puzzle with specified number of givens."""
        max_attempts = 100
        
        for attempt in range(max_attempts):
            # Start with a complete valid grid
            if self.grid_size == 4:
                complete = self._generate_complete_4x4()
            elif self.grid_size == 6:
                complete = self._generate_complete_6x6()
            else:  # 9x9
                complete = self._generate_complete_9x9()
            
            # Create puzzle by removing cells
            puzzle = complete.copy()
            positions = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
            self.rng.shuffle(positions)
            
            total_cells = self.grid_size * self.grid_size
            cells_to_remove = total_cells - num_givens
            
            for r, c in positions[:cells_to_remove]:
                puzzle[r, c] = 0
            
            # Verify puzzle is valid and has unique solution (for smaller grids)
            if self.grid_size <= 6:
                if self.is_valid_sudoku(puzzle):
                    solution = self.solve_sudoku(puzzle)
                    if solution is not None and np.array_equal(solution, complete):
                        return puzzle, complete
            else:
                # For 9x9, skip uniqueness check (too slow)
                if self.is_valid_sudoku(puzzle):
                    return puzzle, complete
        
        # Fallback: return what we have even if not perfectly validated
        return puzzle, complete

    def _generate_complete_4x4(self) -> np.ndarray:
        """Generate a complete valid 4x4 Sudoku grid."""
        base = np.array([[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]], dtype=np.int32)
        
        # Permute numbers
        perm = self.rng.permutation(4) + 1
        complete = np.zeros_like(base)
        for i in range(4):
            complete[base == i + 1] = perm[i]
        
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
        
        return complete

    def _generate_complete_6x6(self) -> np.ndarray:
        """Generate a complete valid 6x6 Sudoku grid."""
        # Start with a valid base pattern
        base = np.array([
            [1, 2, 3, 4, 5, 6],
            [4, 5, 6, 1, 2, 3],
            [2, 3, 4, 5, 6, 1],
            [5, 6, 1, 2, 3, 4],
            [3, 4, 5, 6, 1, 2],
            [6, 1, 2, 3, 4, 5],
        ], dtype=np.int32)
        
        # Permute numbers
        perm = self.rng.permutation(6) + 1
        complete = np.zeros_like(base)
        for i in range(6):
            complete[base == i + 1] = perm[i]
        
        # Shuffle rows within bands (2 rows per band)
        for band in range(3):
            r1, r2 = band * 2, band * 2 + 1
            if self.rng.rand() > 0.5:
                complete[[r1, r2]] = complete[[r2, r1]]
        
        # Shuffle column stacks (3 columns per stack)
        for stack in range(2):
            cols = [stack * 3, stack * 3 + 1, stack * 3 + 2]
            self.rng.shuffle(cols)
            complete[:, stack * 3:stack * 3 + 3] = complete[:, cols]
        
        return complete

    def _generate_complete_9x9(self) -> np.ndarray:
        """Generate a complete valid 9x9 Sudoku grid."""
        # Start with a valid base pattern (shifted by rows)
        base = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [4, 5, 6, 7, 8, 9, 1, 2, 3],
            [7, 8, 9, 1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7, 8, 9, 1],
            [5, 6, 7, 8, 9, 1, 2, 3, 4],
            [8, 9, 1, 2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8, 9, 1, 2],
            [6, 7, 8, 9, 1, 2, 3, 4, 5],
            [9, 1, 2, 3, 4, 5, 6, 7, 8],
        ], dtype=np.int32)
        
        # Permute numbers
        perm = self.rng.permutation(9) + 1
        complete = np.zeros_like(base)
        for i in range(9):
            complete[base == i + 1] = perm[i]
        
        # Shuffle rows within bands (3 rows per band)
        for band in range(3):
            rows = [band * 3, band * 3 + 1, band * 3 + 2]
            self.rng.shuffle(rows)
            complete[band * 3:band * 3 + 3] = complete[rows]
        
        # Shuffle column stacks (3 columns per stack)
        for stack in range(3):
            cols = [stack * 3, stack * 3 + 1, stack * 3 + 2]
            self.rng.shuffle(cols)
            complete[:, stack * 3:stack * 3 + 3] = complete[:, cols]
        
        return complete

    def pad_and_encode(
        self, puzzle: np.ndarray, solution: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad grids and encode with reference format:
        0=PAD, 1=EOS, 2=empty, 3-(grid_size+2)=values 1-grid_size
        """
        # Add 2 offset: 0->2 (empty), 1-n -> 3-(n+2) (values)
        puzzle_shifted = puzzle.copy()
        puzzle_shifted[puzzle == 0] = 2  # Empty cells
        puzzle_shifted[puzzle > 0] = puzzle[puzzle > 0] + 2  # Values shift by 2

        solution_shifted = solution + 2  # Values shift by 2

        # Create padded arrays
        inp_padded = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int32)
        labels_padded = (
            np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int32) + IGNORE_LABEL_ID
        )

        # Fill in the actual data
        inp_padded[:self.grid_size, :self.grid_size] = puzzle_shifted
        labels_padded[:self.grid_size, :self.grid_size] = solution_shifted

        # Add EOS marker
        if self.max_grid_size > self.grid_size:
            inp_padded[self.grid_size, 0] = 1  # EOS
            labels_padded[self.grid_size, 0] = 1  # EOS in labels too

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

        # Convert to tensors
        input_tensor = torch.from_numpy(input_flat).long()
        labels_tensor = torch.from_numpy(labels_flat).long()

        return {
            "input": input_tensor,
            "output": labels_tensor,
            "puzzle_identifiers": puzzle_id,
        }


def collate_fn_sudoku(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    # Stack inputs
    inputs = torch.stack([sample["input"] for sample in batch])

    # Stack outputs
    outputs = torch.stack([sample["output"] for sample in batch])

    # Stack puzzle identifiers
    puzzle_ids = torch.tensor([sample["puzzle_identifiers"] for sample in batch], dtype=torch.long)

    return {"input": inputs, "output": outputs, "puzzle_identifiers": puzzle_ids}


class SudokuDataModule(LightningDataModule):
    """
    Lightning DataModule for Sudoku puzzles of various sizes.
    Supports 4x4, 6x6, and 9x9 grids.
    """

    def __init__(
        self,
        data_dir: str = None,
        batch_size: int = 128,
        num_train_puzzles: int = 2000,
        num_val_puzzles: int = 400,
        num_test_puzzles: int = 400,
        min_givens: int = None,
        max_givens: int = None,
        grid_size: int = 4,
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
            num_val_puzzles: Number of validation puzzles
            num_test_puzzles: Number of test puzzles
            min_givens: Minimum numbers given (defaults to ~35% of cells)
            max_givens: Maximum numbers given (defaults to ~60% of cells)
            grid_size: Sudoku grid size (4, 6, or 9)
            max_grid_size: Padding size (defaults to grid_size + 2)
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
        self.grid_size = grid_size
        self.max_grid_size = max_grid_size
        self.num_workers = num_workers
        self.seed = seed
        self.generate_on_fly = generate_on_fly
        self.pad_value = pad_value
        
        # Auto-scale min/max givens based on grid size if not provided
        total_cells = grid_size * grid_size
        if min_givens is None:
            self.min_givens = int(total_cells * 0.35)  # ~35% filled
        else:
            self.min_givens = min_givens
            
        if max_givens is None:
            self.max_givens = int(total_cells * 0.60)  # ~60% filled
        else:
            self.max_givens = max_givens

        # Metadata matching reference
        self.num_puzzles = 1  # All Sudoku puzzles share ID 0
        self.vocab_size = 3 + grid_size  # 0=PAD, 1=EOS, 2=empty, 3+...=values
        self.seq_len = self.max_grid_size * self.max_grid_size

    def setup(self, stage: Optional[str] = None):
        """Create datasets for each split."""
        if stage == "fit" or stage is None:
            self.train_dataset = SudokuDataset(
                data_dir=self.data_dir,
                split="train",
                num_puzzles=self.num_train_puzzles,
                min_givens=self.min_givens,
                max_givens=self.max_givens,
                grid_size=self.grid_size,
                max_grid_size=self.max_grid_size,
                seed=self.seed,
                generate_on_fly=self.generate_on_fly,
            )

            self.val_dataset = SudokuDataset(
                data_dir=self.data_dir,
                split="test",
                num_puzzles=self.num_val_puzzles,
                min_givens=self.min_givens,
                max_givens=self.max_givens,
                grid_size=self.grid_size,
                max_grid_size=self.max_grid_size,
                seed=self.seed,
                generate_on_fly=self.generate_on_fly,
            )

        if stage == "test" or stage is None:
            self.test_dataset = SudokuDataset(
                data_dir=self.data_dir,
                split="test",
                num_puzzles=self.num_test_puzzles,
                min_givens=self.min_givens,
                max_givens=self.max_givens,
                grid_size=self.grid_size,
                max_grid_size=self.max_grid_size,
                seed=self.seed,
                generate_on_fly=self.generate_on_fly,
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
            drop_last=True,
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
            drop_last=True,
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
            drop_last=True,
        )


if __name__ == "__main__":
    # Test the datamodule with different grid sizes
    print("Testing Flexible Sudoku DataModule")
    print("=" * 60)

    for grid_size in [4, 6]:
        print(f"\n{'='*60}")
        print(f"Testing {grid_size}x{grid_size} Sudoku")
        print("-" * 40)
        
        dm = SudokuDataModule(
            batch_size=2,
            num_train_puzzles=5,
            num_val_puzzles=2,
            grid_size=grid_size,
            # min_givens and max_givens will auto-scale
            generate_on_fly=True,
        )

        dm.setup("fit")
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        print(f"Grid size: {grid_size}x{grid_size}")
        print(f"Batch input shape: {batch['input'].shape}")
        print(f"Vocab size: {dm.vocab_size} (0=PAD, 1=EOS, 2=empty, 3-{2+grid_size}=values)")
        print(f"Sequence length: {dm.seq_len}")
        print(f"Givens range (auto-scaled): {dm.min_givens}-{dm.max_givens}")

        # Show first puzzle
        print(f"\nFirst puzzle ({dm.max_grid_size}x{dm.max_grid_size} padded):")
        puzzle_grid = batch["input"][0].reshape(dm.max_grid_size, dm.max_grid_size)
        print(puzzle_grid.numpy())
        
        # Decode the actual puzzle
        puzzle_decoded = puzzle_grid[:grid_size, :grid_size].numpy()
        puzzle_decoded = np.where(puzzle_decoded == 2, 0, puzzle_decoded - 2)
        puzzle_decoded = np.where(puzzle_decoded < 0, 0, puzzle_decoded)
        print(f"\nDecoded {grid_size}x{grid_size} puzzle (0=empty):")
        print(puzzle_decoded)

    print("\n✅ Flexible DataModule ready!")