"""
Sudoku DataModule - Simplified with No Leakage Guarantee
Supports 4x4, 6x6, and 9x9 Sudoku puzzles

Two modes:
1. Generation mode (data_dir=None): Generate unique puzzle pool and split
2. Loading mode (data_dir="/path"): Load pre-generated data from disk
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.nn.utils.constants import IGNORE_LABEL_ID


def puzzle_hash(puzzle: np.ndarray) -> str:
    """Create a unique hash for a puzzle to detect duplicates."""
    return hashlib.md5(puzzle.tobytes()).hexdigest()


class SudokuDataset(Dataset):
    """
    Dataset for Sudoku puzzles of various sizes (4x4, 6x6, 9x9).

    Two modes:
    - Generation: Uses shared puzzle pool (passed from DataModule)
    - Loading: Loads from .npy files on disk
    """

    def __init__(
        self,
        split: str = "train",
        grid_size: int = 4,
        max_grid_size: int = None,
        min_givens: int = None,
        max_givens: int = None,
        # Generation mode parameters
        puzzle_pool: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        split_indices: Optional[List[int]] = None,
        # Loading mode parameters
        data_dir: Optional[str] = None,
    ):
        """
        Args:
            split: 'train', 'val', or 'test'
            grid_size: Sudoku grid size (4, 6, or 9)
            max_grid_size: Padding size (defaults to grid_size + 2)
            min_givens: Minimum numbers given (defaults to ~35% of cells)
            max_givens: Maximum numbers given (defaults to ~60% of cells)
            puzzle_pool: Shared pool of (puzzle, solution) pairs (generation mode)
            split_indices: Indices in the pool for this split (generation mode)
            data_dir: Directory with pre-generated data (loading mode)
        """
        self.split = split
        self.grid_size = grid_size
        self.max_grid_size = max_grid_size if max_grid_size is not None else grid_size + 2

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

        # Determine mode: generation or loading
        if data_dir is None:
            # Generation mode: use puzzle pool
            assert puzzle_pool is not None, "puzzle_pool required in generation mode"
            assert split_indices is not None, "split_indices required in generation mode"

            self.puzzle_pool = puzzle_pool
            self.split_indices = split_indices
            self.num_puzzles = len(split_indices)
            self.mode = "generation"
        else:
            # Loading mode: load from disk
            self.data_dir = data_dir
            self.mode = "loading"
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
        inp_padded[: self.grid_size, : self.grid_size] = puzzle_shifted
        labels_padded[: self.grid_size, : self.grid_size] = solution_shifted

        # Add EOS marker
        if self.max_grid_size > self.grid_size:
            inp_padded[self.grid_size, 0] = 1  # EOS
            labels_padded[self.grid_size, 0] = 1  # EOS in labels too

        return inp_padded.flatten(), labels_padded.flatten()

    def __len__(self):
        return self.num_puzzles

    def __getitem__(self, idx):
        """Get a single sample."""
        if self.mode == "generation":
            # Get puzzle from pool using split-specific index
            pool_idx = self.split_indices[idx]
            puzzle, solution = self.puzzle_pool[pool_idx]

            # Encode and pad
            input_flat, labels_flat = self.pad_and_encode(puzzle, solution)

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


class PuzzleGenerator:
    """Helper class for generating Sudoku puzzles."""

    def __init__(self, grid_size: int, min_givens: int, max_givens: int):
        self.grid_size = grid_size
        self.min_givens = min_givens
        self.max_givens = max_givens

        # Determine box dimensions
        if grid_size == 4:
            self.box_rows, self.box_cols = 2, 2
        elif grid_size == 6:
            self.box_rows, self.box_cols = 2, 3
        elif grid_size == 9:
            self.box_rows, self.box_cols = 3, 3
        else:
            raise ValueError(f"Unsupported grid_size: {grid_size}")

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
                box = grid[box_r : box_r + self.box_rows, box_c : box_c + self.box_cols].flatten()
                box_vals = box[box > 0]
                if len(box_vals) != len(set(box_vals)):
                    return False

        return True

    def solve_sudoku(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """Simple backtracking solver for Sudoku."""
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
            if num in solution[box_r : box_r + self.box_rows, box_c : box_c + self.box_cols]:
                return False
            return True

        def solve():
            pos = find_empty()
            if pos is None:
                return True

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

    def generate_complete_grid(self, rng: np.random.RandomState) -> np.ndarray:
        """Generate a complete valid Sudoku grid."""
        if self.grid_size == 4:
            return self._generate_complete_4x4(rng)
        elif self.grid_size == 6:
            return self._generate_complete_6x6(rng)
        else:
            return self._generate_complete_9x9(rng)

    def _generate_complete_4x4(self, rng: np.random.RandomState) -> np.ndarray:
        """Generate a complete valid 4x4 Sudoku grid."""
        base = np.array([[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]], dtype=np.int32)

        # Permute numbers
        perm = rng.permutation(4) + 1
        complete = np.zeros_like(base)
        for i in range(4):
            complete[base == i + 1] = perm[i]

        # Shuffle rows within bands
        if rng.rand() > 0.5:
            complete[[0, 1]] = complete[[1, 0]]
        if rng.rand() > 0.5:
            complete[[2, 3]] = complete[[3, 2]]

        # Shuffle columns within stacks
        if rng.rand() > 0.5:
            complete[:, [0, 1]] = complete[:, [1, 0]]
        if rng.rand() > 0.5:
            complete[:, [2, 3]] = complete[:, [3, 2]]

        return complete

    def _generate_complete_6x6(self, rng: np.random.RandomState) -> np.ndarray:
        """Generate a complete valid 6x6 Sudoku grid."""
        base = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [4, 5, 6, 1, 2, 3],
                [2, 3, 4, 5, 6, 1],
                [5, 6, 1, 2, 3, 4],
                [3, 4, 5, 6, 1, 2],
                [6, 1, 2, 3, 4, 5],
            ],
            dtype=np.int32,
        )

        # Permute numbers
        perm = rng.permutation(6) + 1
        complete = np.zeros_like(base)
        for i in range(6):
            complete[base == i + 1] = perm[i]

        # Shuffle rows within bands
        for band in range(3):
            r1, r2 = band * 2, band * 2 + 1
            if rng.rand() > 0.5:
                complete[[r1, r2]] = complete[[r2, r1]]

        # Shuffle column stacks
        for stack in range(2):
            cols = [stack * 3, stack * 3 + 1, stack * 3 + 2]
            rng.shuffle(cols)
            complete[:, stack * 3 : stack * 3 + 3] = complete[:, cols]

        return complete

    def _generate_complete_9x9(self, rng: np.random.RandomState) -> np.ndarray:
        """Generate a complete valid 9x9 Sudoku grid."""
        base = np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [4, 5, 6, 7, 8, 9, 1, 2, 3],
                [7, 8, 9, 1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7, 8, 9, 1],
                [5, 6, 7, 8, 9, 1, 2, 3, 4],
                [8, 9, 1, 2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 8, 9, 1, 2],
                [6, 7, 8, 9, 1, 2, 3, 4, 5],
                [9, 1, 2, 3, 4, 5, 6, 7, 8],
            ],
            dtype=np.int32,
        )

        # Permute numbers
        perm = rng.permutation(9) + 1
        complete = np.zeros_like(base)
        for i in range(9):
            complete[base == i + 1] = perm[i]

        # Shuffle rows within bands
        for band in range(3):
            rows = [band * 3, band * 3 + 1, band * 3 + 2]
            rng.shuffle(rows)
            complete[band * 3 : band * 3 + 3] = complete[rows]

        # Shuffle column stacks
        for stack in range(3):
            cols = [stack * 3, stack * 3 + 1, stack * 3 + 2]
            rng.shuffle(cols)
            complete[:, stack * 3 : stack * 3 + 3] = complete[:, cols]

        return complete

    def generate_puzzle(
        self, rng: np.random.RandomState, num_givens: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a valid Sudoku puzzle with specified number of givens."""
        max_attempts = 100

        for _ in range(max_attempts):
            # Generate complete grid
            complete = self.generate_complete_grid(rng)

            # Create puzzle by removing cells
            puzzle = complete.copy()
            positions = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
            rng.shuffle(positions)

            total_cells = self.grid_size * self.grid_size
            cells_to_remove = total_cells - num_givens

            for r, c in positions[:cells_to_remove]:
                puzzle[r, c] = 0

            # Verify puzzle is valid
            if self.grid_size <= 6:
                if self.is_valid_sudoku(puzzle):
                    solution = self.solve_sudoku(puzzle)
                    if solution is not None and np.array_equal(solution, complete):
                        return puzzle, complete
            else:
                # For 9x9, skip uniqueness check (too slow)
                if self.is_valid_sudoku(puzzle):
                    return puzzle, complete

        # Fallback
        return puzzle, complete


def collate_fn_sudoku(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    inputs = torch.stack([sample["input"] for sample in batch])
    outputs = torch.stack([sample["output"] for sample in batch])
    puzzle_ids = torch.tensor([sample["puzzle_identifiers"] for sample in batch], dtype=torch.long)
    return {"input": inputs, "output": outputs, "puzzle_identifiers": puzzle_ids}


class SudokuDataModule(LightningDataModule):
    """
    Lightning DataModule for Sudoku puzzles of various sizes.

    Two modes:
    - Generation (data_dir=None): Generate unique puzzle pool and split
    - Loading (data_dir="/path"): Load pre-generated data from disk
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        batch_size: int = 128,
        num_train_puzzles: int = 2000,
        num_val_puzzles: int = 400,
        num_test_puzzles: int = 400,
        min_givens: int = None,
        max_givens: int = None,
        grid_size: int = 4,
        max_grid_size: int = None,
        num_workers: int = 0,
        seed: int = 42,
        pad_value: int = 0,
    ):
        """
        Args:
            data_dir: Directory with pre-generated data. If None, generates on-the-fly.
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
        """
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pad_value = pad_value

        # Determine mode
        self.mode = "loading" if data_dir is not None else "generation"

        # If loading from pre-generated data, try to load metadata
        if self.mode == "loading":
            metadata = self._load_metadata(data_dir)

            if metadata is not None:
                # Override parameters with metadata from files
                self.grid_size = metadata["grid_size"]
                self.max_grid_size = metadata["max_grid_size"]
                self.vocab_size = metadata["vocab_size"]
                self.seq_len = metadata["seq_len"]
                self.min_givens = metadata.get("min_givens", min_givens)
                self.max_givens = metadata.get("max_givens", max_givens)
                self.num_train_puzzles = metadata.get("num_train", num_train_puzzles)
                self.num_val_puzzles = metadata.get("num_val", num_val_puzzles)
                self.num_test_puzzles = metadata.get("num_test", num_test_puzzles)

                print(f"✓ Loaded metadata from {data_dir}/metadata.json")
                print(f"  Grid size: {self.grid_size}x{self.grid_size}")
                print(f"  Max grid size: {self.max_grid_size}x{self.max_grid_size}")
                print(f"  Vocab size: {self.vocab_size}")
                print(f"  Sequence length: {self.seq_len}")
                print(
                    f"  Train/Val/Test: {self.num_train_puzzles}/{self.num_val_puzzles}/{self.num_test_puzzles}"
                )
            else:
                # Fallback to provided parameters
                print(f"⚠ Could not load metadata from {data_dir}, using provided parameters")
                self.grid_size = grid_size
                self.max_grid_size = max_grid_size if max_grid_size is not None else grid_size + 2
                self.num_train_puzzles = num_train_puzzles
                self.num_val_puzzles = num_val_puzzles
                self.num_test_puzzles = num_test_puzzles
                self.min_givens = min_givens
                self.max_givens = max_givens
                self._compute_derived_metadata()
        else:
            # Generation mode: use provided parameters
            self.grid_size = grid_size
            self.max_grid_size = max_grid_size if max_grid_size is not None else grid_size + 2
            self.num_train_puzzles = num_train_puzzles
            self.num_val_puzzles = num_val_puzzles
            self.num_test_puzzles = num_test_puzzles
            self.min_givens = min_givens
            self.max_givens = max_givens
            self._compute_derived_metadata()

        # Metadata matching reference
        self.num_puzzles = 1  # All Sudoku puzzles share ID 0

        # Initialize puzzle pool and split indices (will be populated in setup for generation mode)
        self.puzzle_pool = None
        self.split_indices = None

        # Save hyperparameters after everything is set
        self.save_hyperparameters()

    def _load_metadata(self, data_dir: str) -> Optional[dict]:
        """Load metadata from pre-generated data directory."""
        metadata_path = Path(data_dir) / "metadata.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            print(f"Warning: Failed to load metadata from {metadata_path}: {e}")
            return None

    def _compute_derived_metadata(self):
        """Compute derived metadata from grid_size."""
        total_cells = self.grid_size * self.grid_size

        if self.min_givens is None:
            self.min_givens = int(total_cells * 0.35)
        if self.max_givens is None:
            self.max_givens = int(total_cells * 0.60)

        self.vocab_size = 3 + self.grid_size  # 0=PAD, 1=EOS, 2=empty, 3+...=values
        self.seq_len = self.max_grid_size * self.max_grid_size

    def _generate_puzzle_pool(self):
        """
        Generate a global pool of unique puzzles and deterministically split them.
        This guarantees no overlap between train/val/test.
        """
        print(f"Generating puzzle pool for {self.grid_size}x{self.grid_size} Sudoku...")

        total_puzzles = self.num_train_puzzles + self.num_val_puzzles + self.num_test_puzzles

        # Initialize puzzle generator
        generator = PuzzleGenerator(self.grid_size, self.min_givens, self.max_givens)

        # Generate unique puzzles
        rng = np.random.RandomState(self.seed)
        self.puzzle_pool = []
        seen_hashes = set()

        attempts = 0
        max_attempts = total_puzzles * 10  # Safety limit

        while len(self.puzzle_pool) < total_puzzles and attempts < max_attempts:
            # Vary difficulty across the range
            puzzle_idx = len(self.puzzle_pool)
            num_givens = self.min_givens + (puzzle_idx % (self.max_givens - self.min_givens + 1))

            # Generate puzzle
            puzzle, solution = generator.generate_puzzle(rng, num_givens)

            # Check for uniqueness using hash
            h = puzzle_hash(puzzle)
            if h not in seen_hashes:
                seen_hashes.add(h)
                self.puzzle_pool.append((puzzle, solution))

                # Progress indicator
                if len(self.puzzle_pool) % 1000 == 0:
                    print(f"  Generated {len(self.puzzle_pool)}/{total_puzzles} puzzles...")

            attempts += 1

        if len(self.puzzle_pool) < total_puzzles:
            print(
                f"⚠ Warning: Only generated {len(self.puzzle_pool)}/{total_puzzles} unique puzzles"
            )
        else:
            print(f"✓ Generated {len(self.puzzle_pool)} unique puzzles")

        # Deterministically split the pool
        split_rng = np.random.RandomState(self.seed + 999999)
        all_indices = np.arange(len(self.puzzle_pool))
        split_rng.shuffle(all_indices)

        # Allocate indices to splits
        train_end = self.num_train_puzzles
        val_end = train_end + self.num_val_puzzles

        self.split_indices = {
            "train": all_indices[:train_end].tolist(),
            "val": all_indices[train_end:val_end].tolist(),
            "test": all_indices[val_end:].tolist(),
        }

        print("✓ Split allocation:")
        print(f"  Train: {len(self.split_indices['train'])} puzzles")
        print(f"  Val:   {len(self.split_indices['val'])} puzzles")
        print(f"  Test:  {len(self.split_indices['test'])} puzzles")

    def get_puzzle_pool(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the puzzle pool for saving to disk.
        Only available in generation mode after setup() is called.
        """
        if self.mode != "generation":
            raise RuntimeError("get_puzzle_pool() only available in generation mode")
        if self.puzzle_pool is None:
            raise RuntimeError("Call setup() first to generate puzzle pool")
        return self.puzzle_pool

    def get_split_indices(self) -> Dict[str, List[int]]:
        """
        Get the split indices for saving metadata.
        Only available in generation mode after setup() is called.
        """
        if self.mode != "generation":
            raise RuntimeError("get_split_indices() only available in generation mode")
        if self.split_indices is None:
            raise RuntimeError("Call setup() first to generate splits")
        return self.split_indices

    def setup(self, stage: Optional[str] = None):
        """Create datasets for each split."""
        # Generate puzzle pool once for all splits (only in generation mode)
        if self.mode == "generation" and self.puzzle_pool is None:
            self._generate_puzzle_pool()

        if stage == "fit" or stage is None:
            if self.mode == "generation":
                self.train_dataset = SudokuDataset(
                    split="train",
                    grid_size=self.grid_size,
                    max_grid_size=self.max_grid_size,
                    min_givens=self.min_givens,
                    max_givens=self.max_givens,
                    puzzle_pool=self.puzzle_pool,
                    split_indices=self.split_indices["train"],
                )

                self.val_dataset = SudokuDataset(
                    split="val",
                    grid_size=self.grid_size,
                    max_grid_size=self.max_grid_size,
                    min_givens=self.min_givens,
                    max_givens=self.max_givens,
                    puzzle_pool=self.puzzle_pool,
                    split_indices=self.split_indices["val"],
                )
            else:
                self.train_dataset = SudokuDataset(
                    split="train",
                    grid_size=self.grid_size,
                    max_grid_size=self.max_grid_size,
                    min_givens=self.min_givens,
                    max_givens=self.max_givens,
                    data_dir=self.data_dir,
                )

                self.val_dataset = SudokuDataset(
                    split="val",
                    grid_size=self.grid_size,
                    max_grid_size=self.max_grid_size,
                    min_givens=self.min_givens,
                    max_givens=self.max_givens,
                    data_dir=self.data_dir,
                )

        if stage == "test" or stage is None:
            if self.mode == "generation":
                self.test_dataset = SudokuDataset(
                    split="test",
                    grid_size=self.grid_size,
                    max_grid_size=self.max_grid_size,
                    min_givens=self.min_givens,
                    max_givens=self.max_givens,
                    puzzle_pool=self.puzzle_pool,
                    split_indices=self.split_indices["test"],
                )
            else:
                self.test_dataset = SudokuDataset(
                    split="test",
                    grid_size=self.grid_size,
                    max_grid_size=self.max_grid_size,
                    min_givens=self.min_givens,
                    max_givens=self.max_givens,
                    data_dir=self.data_dir,
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
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
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
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
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
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
            drop_last=True,
        )


if __name__ == "__main__":
    # Test the datamodule
    print("Testing Sudoku DataModule (Generation Mode)")
    print("=" * 60)

    dm = SudokuDataModule(
        data_dir=None,  # Generation mode
        batch_size=2,
        num_train_puzzles=10,
        num_val_puzzles=3,
        num_test_puzzles=3,
        grid_size=4,
    )

    dm.setup("fit")

    # Verify no overlap
    train_indices = set(dm.split_indices["train"])
    val_indices = set(dm.split_indices["val"])
    test_indices = set(dm.split_indices["test"])

    print("\n✓ Verification:")
    print(f"  Train: {len(train_indices)} puzzles")
    print(f"  Val:   {len(val_indices)} puzzles")
    print(f"  Test:  {len(test_indices)} puzzles")
    print(
        f"  Train ∩ Val: {len(train_indices & val_indices)} {'✓' if len(train_indices & val_indices) == 0 else '✗'}"
    )
    print(
        f"  Train ∩ Test: {len(train_indices & test_indices)} {'✓' if len(train_indices & test_indices) == 0 else '✗'}"
    )
    print(
        f"  Val ∩ Test: {len(val_indices & test_indices)} {'✓' if len(val_indices & test_indices) == 0 else '✗'}"
    )

    # Get batches
    train_batch = next(iter(dm.train_dataloader()))
    print(f"\n✓ Train batch shape: {train_batch['input'].shape}")

    print("\n✅ DataModule ready!")
