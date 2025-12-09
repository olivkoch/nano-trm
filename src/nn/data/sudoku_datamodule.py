"""
Sudoku DataModule - Simplified with No Leakage Guarantee
Supports 4x4, 6x6, and 9x9 Sudoku puzzles

Two modes:
1. Generation mode (data_dir=None): Generate unique puzzle pool and split
2. Loading mode (data_dir="/path"): Load pre-generated data from disk

Cross-size support:
- Train on one grid size (e.g., 6x6), evaluate on another (e.g., 9x9)
- Detected automatically by reading grid_size from per-split dataset.json
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

from src.nn.utils import RankedLogger
from src.nn.utils.constants import IGNORE_LABEL_ID

log = RankedLogger(__name__, rank_zero_only=True)


class GroupedBatchSampler:
    """
    Sampler that yields batches where each sample comes from a different group.
    
    This ensures augmented versions of the same base puzzle don't appear
    in the same batch, maximizing diversity.
    """
    
    def __init__(
        self, 
        group_indices: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            group_indices: Array of shape (num_groups + 1,) with boundaries
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle groups each epoch
            drop_last: Whether to drop the last incomplete batch
            seed: Random seed
        """
        self.group_indices = group_indices
        self.num_groups = len(group_indices) - 1
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        
    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across workers."""
        self.epoch = epoch
        
    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        
        # Shuffle group order
        if self.shuffle:
            group_order = rng.permutation(self.num_groups)
        else:
            group_order = np.arange(self.num_groups)
        
        # Build batches
        batch = []
        for group_id in group_order:
            # Sample one random puzzle from this group
            start = self.group_indices[group_id]
            end = self.group_indices[group_id + 1]
            puzzle_idx = rng.randint(start, end)
            
            batch.append(puzzle_idx)
            
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        # Handle last batch
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return self.num_groups // self.batch_size
        else:
            return (self.num_groups + self.batch_size - 1) // self.batch_size
        
def puzzle_hash(grid: np.ndarray) -> str:
    """Create a unique hash for a grid to detect duplicates."""
    return hashlib.md5(grid.tobytes()).hexdigest()


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

    def _remap_to_random_tokens(self, puzzle: np.ndarray, solution: np.ndarray):
        """Use random subset of tokens 1-max_grid_size to represent values.
        
        Ensures all token embeddings get trained, even on smaller grids.
        This is useful for e.g. training on 6x6 and validation on 9x9
        """
        n = self.grid_size
        max_val = self.max_grid_size
        
        if n >= max_val:
            return puzzle, solution  # No remapping needed
        
        rng = np.random.default_rng()
        
        # Pick n random tokens from {1, 2, ..., max_val}
        chosen_tokens = rng.choice(max_val, size=n, replace=False) + 1
        
        puzzle_remapped = np.zeros_like(puzzle)
        solution_remapped = np.zeros_like(solution)
        
        # Keep zeros (empty cells) as zeros
        for orig_val in range(1, n + 1):
            new_val = chosen_tokens[orig_val - 1]
            puzzle_remapped[puzzle == orig_val] = new_val
            solution_remapped[solution == orig_val] = new_val
        
        return puzzle_remapped, solution_remapped
    
    def load_from_files(self, data_dir: str, split: str):
        """Load pre-generated dataset from .npy files."""
        split_dir = os.path.join(data_dir, split)

        # Load arrays
        self.inputs = np.load(os.path.join(split_dir, "all__inputs.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(split_dir, "all__labels.npy"), mmap_mode="r")
        self.puzzle_identifiers = np.load(os.path.join(split_dir, "all__puzzle_identifiers.npy"))

        group_path = os.path.join(split_dir, "all__group_indices.npy")
        puzzle_idx_path = os.path.join(split_dir, "all__puzzle_indices.npy")
    
        if os.path.exists(group_path):
            self.group_indices = np.load(group_path)
            self.puzzle_indices = np.load(puzzle_idx_path)
            self.num_groups = len(self.group_indices) - 1
        else:
            # Fallback: each puzzle is its own group
            self.group_indices = np.arange(len(self.inputs) + 1, dtype=np.int32)
            self.puzzle_indices = np.arange(len(self.inputs) + 1, dtype=np.int32)
            self.num_groups = len(self.inputs)

        # Load split metadata
        with open(os.path.join(split_dir, "dataset.json")) as f:
            self.metadata = json.load(f)
        
        # Update grid_size from split metadata if available
        if "grid_size" in self.metadata:
            self.grid_size = self.metadata["grid_size"]
        if "max_grid_size" in self.metadata:
            self.max_grid_size = self.metadata["max_grid_size"]

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

            if self.split == "train":
                puzzle, solution = self._remap_to_random_tokens(puzzle, solution)
                
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
    """
    Helper class for generating Sudoku puzzles.
    
    Supports two modes:
    - Standard: Uses transformation-based generation from a single base grid
    - Full: Uses pre-computed base grids with full symmetry operations
    """

    def __init__(
        self, 
        grid_size: int, 
        min_givens: int, 
        max_givens: int,
        base_grids: Optional[np.ndarray] = None,
    ):
        """
        Args:
            grid_size: Size of the Sudoku grid (4, 6, or 9)
            min_givens: Minimum number of given cells
            max_givens: Maximum number of given cells
            base_grids: Optional array of canonical base grids for full enumeration mode
                        Shape: (num_classes, grid_size, grid_size)
        """
        self.grid_size = grid_size
        self.min_givens = min_givens
        self.max_givens = max_givens
        self.base_grids = base_grids
        self.full_mode = base_grids is not None

        # Determine box dimensions
        if grid_size == 4:
            self.box_rows, self.box_cols = 2, 2
        elif grid_size == 6:
            self.box_rows, self.box_cols = 2, 3
        elif grid_size == 9:
            self.box_rows, self.box_cols = 3, 3
        else:
            raise ValueError(f"Unsupported grid_size: {grid_size}")
        
        self.n_bands = grid_size // self.box_rows
        self.n_stacks = grid_size // self.box_cols

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
        if self.full_mode:
            return self._generate_complete_full(rng)
        elif self.grid_size == 4:
            return self._generate_complete_4x4(rng)
        elif self.grid_size == 6:
            return self._generate_complete_6x6(rng)
        else:
            return self._generate_complete_9x9(rng)

    def _generate_complete_full(self, rng: np.random.RandomState) -> np.ndarray:
        """
        Generate a complete grid using full symmetry operations.
        This provides uniform sampling over ALL valid grids.
        """
        # Pick a random base grid
        base_idx = rng.randint(len(self.base_grids))
        base = self.base_grids[base_idx].copy()
        
        # 1. Digit permutation (n!)
        perm = rng.permutation(self.grid_size) + 1
        complete = np.zeros_like(base)
        for i in range(self.grid_size):
            complete[base == i + 1] = perm[i]
        
        # 2. Band permutation (n_bands!)
        band_order = rng.permutation(self.n_bands)
        new_grid = np.zeros_like(complete)
        for new_band, old_band in enumerate(band_order):
            new_grid[new_band*self.box_rows:(new_band+1)*self.box_rows] = \
                complete[old_band*self.box_rows:(old_band+1)*self.box_rows]
        complete = new_grid
        
        # 3. Row swaps within bands (2^n_bands for 2-row bands)
        for band in range(self.n_bands):
            if self.box_rows == 2:
                # Simple swap
                if rng.rand() > 0.5:
                    r1, r2 = band * self.box_rows, band * self.box_rows + 1
                    complete[[r1, r2]] = complete[[r2, r1]]
            else:
                # Full permutation for 3-row bands
                rows = list(range(band * self.box_rows, (band + 1) * self.box_rows))
                rng.shuffle(rows)
                temp = complete[band * self.box_rows:(band + 1) * self.box_rows].copy()
                for new_pos, old_row in enumerate(rows):
                    complete[band * self.box_rows + new_pos] = temp[old_row - band * self.box_rows]
        
        # 4. Stack permutation (n_stacks!)
        stack_order = rng.permutation(self.n_stacks)
        new_grid = np.zeros_like(complete)
        for new_stack, old_stack in enumerate(stack_order):
            new_grid[:, new_stack*self.box_cols:(new_stack+1)*self.box_cols] = \
                complete[:, old_stack*self.box_cols:(old_stack+1)*self.box_cols]
        complete = new_grid
        
        # 5. Column shuffles within stacks (box_cols! per stack)
        for stack in range(self.n_stacks):
            cols = list(range(stack * self.box_cols, (stack + 1) * self.box_cols))
            rng.shuffle(cols)
            temp = complete[:, stack * self.box_cols:(stack + 1) * self.box_cols].copy()
            for new_pos, old_col in enumerate(cols):
                complete[:, stack * self.box_cols + new_pos] = temp[:, old_col - stack * self.box_cols]
        
        return complete

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
    Lightning DataModule for Sudoku puzzles.
    
    Modes:
    - Generation (data_dir=None): Generate puzzles on-the-fly
    - Loading (data_dir="/path"): Load pre-generated data
    
    Cross-size support:
    - Automatically detects when train and val/test have different grid sizes
    - Reads grid_size from per-split dataset.json files
    - Exposes train_grid_size, eval_grid_size, cross_size attributes
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
        base_grids: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pad_value = pad_value
        self.base_grids = base_grids

        self._train_sampler = None

        self.mode = "loading" if data_dir is not None else "generation"

        # Cross-size attributes (will be set in _load_metadata or _use_provided_params)
        self.train_grid_size = None
        self.eval_grid_size = None
        self.cross_size = False

        if self.mode == "loading":
            metadata = self._load_metadata(data_dir)

            if metadata is not None:
                # Global metadata
                self.max_grid_size = metadata["max_grid_size"]
                self.vocab_size = metadata["vocab_size"]
                self.seq_len = metadata["seq_len"]
                
                # Total puzzles (for reference)
                self.num_train_puzzles = metadata.get("num_train", num_train_puzzles)
                self.num_val_puzzles = metadata.get("num_val", num_val_puzzles)
                self.num_test_puzzles = metadata.get("num_test", num_test_puzzles)
                
                # Groups (for steps per epoch calculation)
                self.num_train_groups = metadata.get("num_train_groups", self.num_train_puzzles)
                self.num_val_groups = metadata.get("num_val_groups", self.num_val_puzzles)
                self.num_test_groups = metadata.get("num_test_groups", self.num_test_puzzles)
                
                # Infer grid sizes from per-split metadata
                self._load_split_grid_sizes(data_dir)
                
                # grid_size for backward compatibility (use eval_grid_size)
                self.grid_size = self.eval_grid_size

                log.info(f"✓ Loaded metadata from {data_dir}/metadata.json")
                log.info(f"  Train: {self.num_train_groups} groups, {self.num_train_puzzles} total puzzles")
                if self.cross_size:
                    log.info(f"  Cross-size: train={self.train_grid_size}x{self.train_grid_size}, "
                             f"eval={self.eval_grid_size}x{self.eval_grid_size}")
            else:
                log.warning(f"⚠ Could not load metadata, using provided parameters")
                self._use_provided_params(grid_size, max_grid_size, min_givens, max_givens,
                                          num_train_puzzles, num_val_puzzles, num_test_puzzles)
        else:
            self._use_provided_params(grid_size, max_grid_size, min_givens, max_givens,
                                      num_train_puzzles, num_val_puzzles, num_test_puzzles)

        self.num_puzzles = 1
        self.puzzle_pool = None
        self.split_indices = None

        self.save_hyperparameters(ignore=['base_grids'])

    def _load_split_grid_sizes(self, data_dir: str):
        """Load grid sizes from per-split dataset.json files."""
        data_path = Path(data_dir)
        
        # Read train grid size
        train_meta_path = data_path / "train" / "dataset.json"
        if not train_meta_path.exists():
            raise ValueError(f"Missing train metadata: {train_meta_path}")
        
        with open(train_meta_path) as f:
            train_meta = json.load(f)
        
        if "grid_size" not in train_meta:
            raise ValueError(f"Missing 'grid_size' in {train_meta_path}")
        
        self.train_grid_size = train_meta["grid_size"]
        
        # Read eval grid size (prefer val, fall back to test)
        eval_grid_size = None
        for split in ["val", "test"]:
            split_meta_path = data_path / split / "dataset.json"
            if split_meta_path.exists():
                with open(split_meta_path) as f:
                    split_meta = json.load(f)
                eval_grid_size = split_meta.get("grid_size")
                if eval_grid_size is not None:
                    break
        
        if eval_grid_size is None:
            raise ValueError(f"Missing 'grid_size' in val/dataset.json or test/dataset.json")
        
        self.eval_grid_size = eval_grid_size
        self.cross_size = (self.train_grid_size != self.eval_grid_size)

    def _use_provided_params(self, grid_size, max_grid_size, min_givens, max_givens,
                             num_train, num_val, num_test):
        """Set parameters from provided values (generation mode or fallback)."""
        self.grid_size = grid_size
        self.max_grid_size = max_grid_size if max_grid_size is not None else grid_size + 2
        self.num_train_puzzles = num_train
        self.num_val_puzzles = num_val
        self.num_test_puzzles = num_test
        
        total_cells = grid_size * grid_size
        self.min_givens = min_givens if min_givens is not None else int(total_cells * 0.35)
        self.max_givens = max_givens if max_givens is not None else int(total_cells * 0.60)
        
        self.vocab_size = 3 + grid_size
        self.seq_len = self.max_grid_size * self.max_grid_size
        
        # In generation mode, train and eval are same size
        self.train_grid_size = grid_size
        self.eval_grid_size = grid_size
        self.cross_size = False

    def _load_metadata(self, data_dir: str) -> Optional[dict]:
        """Load metadata from pre-generated data directory."""
        metadata_path = Path(data_dir) / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            with open(metadata_path) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load metadata: {e}")
            return None

    def _generate_puzzle_pool(self):
        """Generate puzzle pool for generation mode."""
        total_puzzles = self.num_train_puzzles + self.num_val_puzzles + self.num_test_puzzles
        
        generator = PuzzleGenerator(
            self.grid_size, self.min_givens, self.max_givens, base_grids=self.base_grids
        )

        rng = np.random.RandomState(self.seed)
        self.puzzle_pool = []
        seen_hashes = set()

        while len(self.puzzle_pool) < total_puzzles:
            puzzle_idx = len(self.puzzle_pool)
            num_givens = self.min_givens + (puzzle_idx % (self.max_givens - self.min_givens + 1))
            puzzle, solution = generator.generate_puzzle(rng, num_givens)

            h = hashlib.md5(solution.tobytes()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                self.puzzle_pool.append((puzzle, solution))

        # Split
        split_rng = np.random.RandomState(self.seed + 999999)
        all_indices = np.arange(len(self.puzzle_pool))
        split_rng.shuffle(all_indices)

        train_end = self.num_train_puzzles
        val_end = train_end + self.num_val_puzzles

        self.split_indices = {
            "train": all_indices[:train_end].tolist(),
            "val": all_indices[train_end:val_end].tolist(),
            "test": all_indices[val_end:].tolist(),
        }

    def get_puzzle_pool(self):
        if self.mode != "generation":
            raise RuntimeError("get_puzzle_pool() only available in generation mode")
        return self.puzzle_pool

    def get_split_indices(self):
        if self.mode != "generation":
            raise RuntimeError("get_split_indices() only available in generation mode")
        return self.split_indices

    def setup(self, stage: Optional[str] = None):
        if self.mode == "generation" and self.puzzle_pool is None:
            self._generate_puzzle_pool()

        if stage == "fit" or stage is None:
            if self.mode == "generation":
                self.train_dataset = SudokuDataset(
                    split="train", grid_size=self.grid_size, max_grid_size=self.max_grid_size,
                    min_givens=self.min_givens, max_givens=self.max_givens,
                    puzzle_pool=self.puzzle_pool, split_indices=self.split_indices["train"],
                )
                self.val_dataset = SudokuDataset(
                    split="val", grid_size=self.grid_size, max_grid_size=self.max_grid_size,
                    min_givens=self.min_givens, max_givens=self.max_givens,
                    puzzle_pool=self.puzzle_pool, split_indices=self.split_indices["val"],
                )
            else:
                self.train_dataset = SudokuDataset(split="train", data_dir=self.data_dir)
                self.val_dataset = SudokuDataset(split="val", data_dir=self.data_dir)

        if stage == "test" or stage is None:
            if self.mode == "generation":
                self.test_dataset = SudokuDataset(
                    split="test", grid_size=self.grid_size, max_grid_size=self.max_grid_size,
                    min_givens=self.min_givens, max_givens=self.max_givens,
                    puzzle_pool=self.puzzle_pool, split_indices=self.split_indices["test"],
                )
            else:
                self.test_dataset = SudokuDataset(split="test", data_dir=self.data_dir)

    def train_dataloader(self):
        if self.mode == "loading" and hasattr(self.train_dataset, 'group_indices'):
            # Use group-aware sampling
            self._train_sampler = GroupedBatchSampler(
                group_indices=self.train_dataset.group_indices,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                seed=self.seed,
            )
            
            return DataLoader(
                self.train_dataset,
                batch_sampler=self._train_sampler,
                num_workers=self.num_workers,
                collate_fn=collate_fn_sudoku,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
                multiprocessing_context="spawn" if self.num_workers > 0 else None,
            )
        else:
            # Fallback to standard shuffled sampling
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn_sudoku,
                pin_memory=True,
                drop_last=True,
                persistent_workers=self.num_workers > 0,
                multiprocessing_context="spawn" if self.num_workers > 0 else None,
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn_sudoku,
            pin_memory=True, drop_last=True,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn_sudoku,
            pin_memory=True, drop_last=True,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
        )

    def on_train_epoch_start(self, current_epoch: int):
        """Update sampler epoch for proper shuffling. Call from LightningModule."""
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(current_epoch)

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
    print(f"  Cross-size: {dm.cross_size}")
    print(f"  Train grid: {dm.train_grid_size}x{dm.train_grid_size}")
    print(f"  Eval grid: {dm.eval_grid_size}x{dm.eval_grid_size}")
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