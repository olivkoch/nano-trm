"""
Pre-generate Sudoku datasets and save to disk for fast training.

Modes:
  --full: Use multiple diverse base grids with full symmetry operations.
  --hybrid: Train in standard mode, val/test in full mode (OOD testing).
  --mixed-size: 50/50 mixture of 6x6 and 9x9 in all splits.
  --cross-size: Train on 6x6, val/test on 9x9.

Optimizations:
  - Vectorized encoding (no per-puzzle method calls)
  - Smart uniqueness: disables solution uniqueness check when num_puzzles > available solutions
  - Pre-allocated numpy arrays
  - Inlined pad_and_encode logic
"""

import json
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from src.nn.data.sudoku_datamodule import SudokuDataModule, SudokuDataset, PuzzleGenerator
from src.nn.utils.constants import IGNORE_LABEL_ID


# =============================================================================
# CONSTANTS
# =============================================================================

# Known counts of valid Sudoku solution grids
MAX_UNIQUE_SOLUTIONS = {
    4: 288,
    6: 28_200_960,  # ~28 million
    9: 6_670_903_752_021_072_936_960,  # ~6.67 × 10^21
}

# Box dimensions for each grid size: (box_rows, box_cols, num_bands, num_stacks)
BOX_DIMS = {
    4: (2, 2, 2, 2),
    6: (2, 3, 3, 2),
    9: (3, 3, 3, 3),
}


def get_max_unique_solutions(grid_size: int) -> int:
    """Return the theoretical maximum number of unique solution grids."""
    return MAX_UNIQUE_SOLUTIONS.get(grid_size, float('inf'))


# =============================================================================
# TRANSFORMATIONS
# =============================================================================

def shuffle_sudoku(board: np.ndarray, solution: np.ndarray, grid_size: int, rng: np.random.RandomState = None):
    """Apply Sudoku-preserving transformations (digit relabeling + band/stack shuffles).
    
    Optimized version with optional RNG parameter.
    """
    if rng is None:
        rng = np.random
    
    if grid_size not in BOX_DIMS:
        # Fallback: just do digit relabeling
        digit_map = np.zeros(grid_size + 1, dtype=np.int32)
        digit_map[1:] = rng.permutation(grid_size) + 1
        return digit_map[board], digit_map[solution]
    
    box_rows, box_cols, num_bands, num_stacks = BOX_DIMS[grid_size]
    
    # Create digit mapping (0 stays 0)
    digit_map = np.zeros(grid_size + 1, dtype=np.int32)
    digit_map[1:] = rng.permutation(grid_size) + 1
    
    # Generate row permutation (shuffle bands, then rows within bands)
    bands = rng.permutation(num_bands)
    row_perm = np.concatenate([b * box_rows + rng.permutation(box_rows) for b in bands])
    
    # Generate column permutation (shuffle stacks, then columns within stacks)
    stacks = rng.permutation(num_stacks)
    col_perm = np.concatenate([s * box_cols + rng.permutation(box_cols) for s in stacks])
    
    # Apply transformations using advanced indexing
    transpose_flag = rng.random() < 0.5
    
    if transpose_flag:
        new_board = digit_map[board.T[row_perm][:, col_perm]]
        new_solution = digit_map[solution.T[row_perm][:, col_perm]]
    else:
        new_board = digit_map[board[row_perm][:, col_perm]]
        new_solution = digit_map[solution[row_perm][:, col_perm]]
    
    return new_board, new_solution


def dihedral_transform(arr: np.ndarray, idx: int) -> np.ndarray:
    """Apply one of 8 dihedral group transformations (D4).
    
    idx 0-3: rotations (0°, 90°, 180°, 270°)
    idx 4-7: reflections (horizontal, vertical, diagonal, anti-diagonal)
    """
    if idx == 0:
        return arr
    elif idx == 1:
        return np.rot90(arr, 1)
    elif idx == 2:
        return np.rot90(arr, 2)
    elif idx == 3:
        return np.rot90(arr, 3)
    elif idx == 4:
        return np.fliplr(arr)
    elif idx == 5:
        return np.flipud(arr)
    elif idx == 6:
        return np.rot90(arr, 1).T  # diagonal flip
    elif idx == 7:
        return np.rot90(arr, 3).T  # anti-diagonal flip
    else:
        raise ValueError(f"Invalid dihedral index: {idx}")


# =============================================================================
# ENCODING (VECTORIZED)
# =============================================================================

def encode_puzzle_inline(
    puzzle: np.ndarray,
    solution: np.ndarray,
    grid_size: int,
    max_grid_size: int,
    inp_out: np.ndarray,
    lbl_out: np.ndarray,
):
    """Encode a puzzle directly into pre-allocated output arrays.
    
    This avoids creating intermediate arrays and method call overhead.
    """
    # Shift values: 0 -> 2 (blank token), 1-N -> 3 to N+2
    puzzle_shifted = puzzle.copy()
    puzzle_shifted[puzzle == 0] = 2
    puzzle_shifted[puzzle > 0] = puzzle[puzzle > 0] + 2
    solution_shifted = solution + 2
    
    # Write to output arrays
    seq_len = grid_size * grid_size
    inp_out[:seq_len] = puzzle_shifted.ravel()
    lbl_out[:seq_len] = solution_shifted.ravel()
    
    # Handle padding for smaller grids
    if max_grid_size > grid_size:
        max_seq = max_grid_size * max_grid_size
        inp_out[seq_len:max_seq] = 0  # padding token
        lbl_out[seq_len:max_seq] = IGNORE_LABEL_ID


# =============================================================================
# POOL GENERATION
# =============================================================================

def generate_single_size_pool(
    num_puzzles: int,
    grid_size: int,
    base_grids: np.ndarray,
    min_givens: int,
    max_givens: int,
    seed: int,
    exclude_solution_hashes: set = None,
) -> list:
    """Generate a pool of puzzles with a single grid size.
    
    Automatically disables uniqueness check when num_puzzles exceeds available unique solutions.
    """
    if exclude_solution_hashes is None:
        exclude_solution_hashes = set()
    
    # Determine if uniqueness enforcement is feasible
    max_unique = get_max_unique_solutions(grid_size)
    available_unique = max_unique - len(exclude_solution_hashes)
    enforce_unique = num_puzzles <= available_unique * 0.9  # 90% threshold
    
    if not enforce_unique:
        print(f"  ⚠ Requested {num_puzzles} puzzles but only ~{available_unique} unique solutions available.")
        print(f"    Disabling solution uniqueness - puzzles may share underlying solutions.")
        print(f"    (Augmentations will still provide diversity in presentation)")
    
    generator = PuzzleGenerator(
        grid_size=grid_size,
        min_givens=min_givens,
        max_givens=max_givens,
        base_grids=base_grids,
    )
    
    rng = np.random.RandomState(seed)
    pool = []
    seen_solution_hashes = set()
    
    attempts = 0
    max_attempts = num_puzzles * (50 if enforce_unique else 2)
    
    while len(pool) < num_puzzles and attempts < max_attempts:
        num_givens = rng.randint(min_givens, max_givens + 1)
        puzzle, solution = generator.generate_puzzle(rng, num_givens)
        
        if enforce_unique:
            sol_hash = solution.tobytes()
            
            if sol_hash in exclude_solution_hashes or sol_hash in seen_solution_hashes:
                attempts += 1
                continue
            
            seen_solution_hashes.add(sol_hash)
        
        pool.append((puzzle, solution))
        
        if len(pool) % 1000 == 0:
            print(f"    Generated {len(pool)}/{num_puzzles} puzzles...")
        
        attempts += 1
    
    if len(pool) < num_puzzles:
        print(f"  ⚠ Warning: Only generated {len(pool)}/{num_puzzles} unique puzzles")
        if enforce_unique:
            print(f"    Consider using --num-train <= {available_unique} for unique solutions")
    
    return pool


def generate_mixed_pool(
    num_puzzles: int,
    grid_sizes: list,
    base_grids_dict: dict,
    givens_dict: dict,
    seed: int,
    exclude_solution_hashes: set = None,
) -> list:
    """Generate a mixed pool of puzzles with different grid sizes."""
    if exclude_solution_hashes is None:
        exclude_solution_hashes = set()
    
    generators = {}
    for gs in grid_sizes:
        min_g, max_g = givens_dict[gs]
        generators[gs] = PuzzleGenerator(
            grid_size=gs,
            min_givens=min_g,
            max_givens=max_g,
            base_grids=base_grids_dict.get(gs),
        )
    
    # Check uniqueness feasibility per grid size
    puzzles_per_size = num_puzzles // len(grid_sizes)
    enforce_unique = {}
    for gs in grid_sizes:
        max_unique = get_max_unique_solutions(gs)
        # Count excluded hashes for this grid size
        excluded_for_size = sum(1 for h in exclude_solution_hashes if isinstance(h, tuple) and h[0] == gs)
        available = max_unique - excluded_for_size
        enforce_unique[gs] = puzzles_per_size <= available * 0.9
        
        if not enforce_unique[gs]:
            print(f"  ⚠ {gs}x{gs}: Requested ~{puzzles_per_size} but only ~{available} unique solutions.")
            print(f"    Disabling uniqueness for {gs}x{gs}.")
    
    rng = np.random.RandomState(seed)
    pool = []
    seen_solution_hashes = {gs: set() for gs in grid_sizes}
    
    remainder = num_puzzles % len(grid_sizes)
    targets = {gs: puzzles_per_size for gs in grid_sizes}
    for i, gs in enumerate(grid_sizes):
        if i < remainder:
            targets[gs] += 1
    
    counts = {gs: 0 for gs in grid_sizes}
    
    attempts = 0
    max_attempts = num_puzzles * 50
    
    while sum(counts.values()) < num_puzzles and attempts < max_attempts:
        available_sizes = [gs for gs in grid_sizes if counts[gs] < targets[gs]]
        if not available_sizes:
            break
        
        gs = rng.choice(available_sizes)
        generator = generators[gs]
        min_g, max_g = givens_dict[gs]
        
        num_givens = rng.randint(min_g, max_g + 1)
        puzzle, solution = generator.generate_puzzle(rng, num_givens)
        
        if enforce_unique[gs]:
            sol_hash = (gs, solution.tobytes())
            
            if sol_hash in exclude_solution_hashes or solution.tobytes() in seen_solution_hashes[gs]:
                attempts += 1
                continue
            
            seen_solution_hashes[gs].add(solution.tobytes())
        
        pool.append((puzzle, solution, gs))
        counts[gs] += 1
        
        total = sum(counts.values())
        if total % 1000 == 0:
            print(f"    Generated {total}/{num_puzzles} puzzles " + 
                  " ".join(f"({gs}x{gs}: {counts[gs]})" for gs in grid_sizes))
        
        attempts += 1
    
    if sum(counts.values()) < num_puzzles:
        print(f"  ⚠ Warning: Only generated {sum(counts.values())}/{num_puzzles} puzzles")
    
    return pool


def generate_ood_pool(
    num_puzzles: int,
    grid_size: int,
    min_givens: int,
    max_givens: int,
    seed: int,
    base_grids: np.ndarray,
    exclude_solution_hashes: set,
) -> list:
    """Generate puzzles using full mode, excluding specified solutions."""
    max_unique = get_max_unique_solutions(grid_size)
    available_unique = max_unique - len(exclude_solution_hashes)
    enforce_unique = num_puzzles <= available_unique * 0.9
    
    if not enforce_unique:
        print(f"  ⚠ Requested {num_puzzles} OOD puzzles but only ~{available_unique} unique solutions available.")
        print(f"    Disabling uniqueness check.")
    
    generator = PuzzleGenerator(
        grid_size=grid_size,
        min_givens=min_givens,
        max_givens=max_givens,
        base_grids=base_grids,
    )
    
    rng = np.random.RandomState(seed)
    pool = []
    seen_solution_hashes = set()
    
    attempts = 0
    max_attempts = num_puzzles * (50 if enforce_unique else 2)
    
    while len(pool) < num_puzzles and attempts < max_attempts:
        puzzle_idx = len(pool)
        num_givens = min_givens + (puzzle_idx % (max_givens - min_givens + 1))
        
        puzzle, solution = generator.generate_puzzle(rng, num_givens)
        
        if enforce_unique:
            sol_hash = solution.tobytes()
            
            if sol_hash in exclude_solution_hashes or sol_hash in seen_solution_hashes:
                attempts += 1
                continue
            
            seen_solution_hashes.add(sol_hash)
        
        pool.append((puzzle, solution))
        
        if len(pool) % 1000 == 0:
            print(f"    Generated {len(pool)}/{num_puzzles} puzzles...")
        
        attempts += 1
    
    if len(pool) < num_puzzles:
        print(f"  ⚠ Warning: Only generated {len(pool)}/{num_puzzles} puzzles")
        if enforce_unique:
            print(f"    (filtered out {attempts - len(pool)} duplicate solutions)")
    
    return pool


# =============================================================================
# SAVING TO DISK (VECTORIZED)
# =============================================================================

def save_split_to_disk(
    output_dir: Path,
    split: str,
    puzzle_pool: list,
    split_indices: list,
    grid_size: int,
    max_grid_size: int,
    min_givens: int,
    max_givens: int,
    vocab_size: int,
    num_aug: int = 0,
):
    """Save a split (train/val/test) to disk with vectorized encoding."""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    seq_len = max_grid_size * max_grid_size
    num_augments = num_aug if split == "train" else 0
    num_base = len(split_indices)
    num_total = num_base * (1 + num_augments)

    # Pre-allocate arrays
    all_inputs = np.zeros((num_total, seq_len), dtype=np.int32)
    all_labels = np.full((num_total, seq_len), IGNORE_LABEL_ID, dtype=np.int32)
    puzzle_identifiers = np.zeros(num_total, dtype=np.int32)
    puzzle_indices = [0]
    group_indices = [0]

    print(f"Encoding {split} split ({num_base} base puzzles, {1 + num_augments} variants each)...")
    
    rng = np.random.RandomState(42 + hash(split) % 10000)  # Deterministic per split
    idx = 0
    
    # Use tqdm only for larger datasets
    iterator = range(num_base)
    if num_base > 100:
        iterator = tqdm(iterator, desc=f"Encoding {split}")
    
    for i in iterator:
        pool_idx = split_indices[i]
        orig_puzzle, orig_solution = puzzle_pool[pool_idx]

        for aug_idx in range(1 + num_augments):
            if aug_idx == 0:
                puzzle, solution = orig_puzzle, orig_solution
            else:
                puzzle, solution = shuffle_sudoku(orig_puzzle, orig_solution, grid_size, rng)

            # Inline encoding for speed
            encode_puzzle_inline(
                puzzle, solution, grid_size, max_grid_size,
                all_inputs[idx], all_labels[idx]
            )
            
            idx += 1
            puzzle_indices.append(idx)
        
        group_indices.append(idx)

    num_groups = len(group_indices) - 1
    num_puzzles = idx

    # Trim arrays if we didn't fill them completely
    if idx < num_total:
        all_inputs = all_inputs[:idx]
        all_labels = all_labels[:idx]
        puzzle_identifiers = puzzle_identifiers[:idx]

    print(f"Saving to {split_dir}...")
    np.save(split_dir / "all__inputs.npy", all_inputs)
    np.save(split_dir / "all__labels.npy", all_labels)
    np.save(split_dir / "all__puzzle_identifiers.npy", puzzle_identifiers)
    np.save(split_dir / "all__puzzle_indices.npy", np.array(puzzle_indices, dtype=np.int32))
    np.save(split_dir / "all__group_indices.npy", np.array(group_indices, dtype=np.int32))

    # Save split metadata
    metadata = {
        "num_puzzles": num_puzzles,
        "num_groups": num_groups,
        "num_augmentations": num_augments,
        "mean_puzzles_per_group": num_puzzles / num_groups if num_groups > 0 else 1,
        "grid_size": grid_size,
        "max_grid_size": max_grid_size,
        "min_givens": min_givens,
        "max_givens": max_givens,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
    }

    with open(split_dir / "dataset.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved {split} split ({grid_size}x{grid_size}):")
    print(f"  - {num_groups} groups × {1 + num_augments} = {num_puzzles} puzzles")
    print(f"  - inputs: {all_inputs.shape}")
    print(f"  - labels: {all_labels.shape}")
    
    return num_groups, num_puzzles


def save_mixed_split_to_disk(
    output_dir: Path,
    split: str,
    puzzle_pool: list,
    split_indices: list,
    max_grid_size: int,
    vocab_size: int,
    num_aug: int = 0,
):
    """Save a split with mixed grid sizes to disk (vectorized)."""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    seq_len = max_grid_size * max_grid_size
    num_augments = num_aug if split == "train" else 0
    num_base = len(split_indices)
    num_total = num_base * (1 + num_augments)

    # Pre-allocate arrays
    all_inputs = np.zeros((num_total, seq_len), dtype=np.int32)
    all_labels = np.full((num_total, seq_len), IGNORE_LABEL_ID, dtype=np.int32)
    all_grid_sizes = np.zeros(num_total, dtype=np.int32)
    puzzle_identifiers = np.zeros(num_total, dtype=np.int32)
    puzzle_indices = [0]
    group_indices = [0]

    print(f"Encoding {split} split ({num_base} base puzzles)...")
    
    rng = np.random.RandomState(42 + hash(split) % 10000)
    idx = 0
    
    iterator = range(num_base)
    if num_base > 100:
        iterator = tqdm(iterator, desc=f"Encoding {split}")
    
    for i in iterator:
        pool_idx = split_indices[i]
        orig_puzzle, orig_solution, grid_size = puzzle_pool[pool_idx]

        for aug_idx in range(1 + num_augments):
            if aug_idx == 0:
                puzzle, solution = orig_puzzle, orig_solution
            else:
                puzzle, solution = shuffle_sudoku(orig_puzzle, orig_solution, grid_size, rng)

            # Inline encoding
            encode_puzzle_inline(
                puzzle, solution, grid_size, max_grid_size,
                all_inputs[idx], all_labels[idx]
            )
            all_grid_sizes[idx] = grid_size
            
            idx += 1
            puzzle_indices.append(idx)
        
        group_indices.append(idx)

    num_groups = len(group_indices) - 1
    num_puzzles = idx

    # Trim if needed
    if idx < num_total:
        all_inputs = all_inputs[:idx]
        all_labels = all_labels[:idx]
        all_grid_sizes = all_grid_sizes[:idx]
        puzzle_identifiers = puzzle_identifiers[:idx]

    print(f"Saving to {split_dir}...")
    np.save(split_dir / "all__inputs.npy", all_inputs)
    np.save(split_dir / "all__labels.npy", all_labels)
    np.save(split_dir / "all__puzzle_identifiers.npy", puzzle_identifiers)
    np.save(split_dir / "all__grid_sizes.npy", all_grid_sizes)
    np.save(split_dir / "all__puzzle_indices.npy", np.array(puzzle_indices, dtype=np.int32))
    np.save(split_dir / "all__group_indices.npy", np.array(group_indices, dtype=np.int32))

    # Count by grid size
    size_counts = {}
    for gs in all_grid_sizes[:idx]:
        size_counts[int(gs)] = size_counts.get(int(gs), 0) + 1

    metadata = {
        "num_puzzles": int(num_puzzles),
        "num_groups": int(num_groups),
        "num_augmentations": int(num_augments),
        "mean_puzzles_per_group": num_puzzles / num_groups if num_groups > 0 else 1,
        "mixed_size": True,
        "grid_sizes": [int(gs) for gs in sorted(size_counts.keys())],
        "grid_size_counts": {str(k): int(v) for k, v in size_counts.items()},
        "max_grid_size": int(max_grid_size),
        "seq_len": int(seq_len),
        "vocab_size": int(vocab_size),
    }

    with open(split_dir / "dataset.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved {split} split:")
    print(f"  - {num_groups} groups × {1 + num_augments} = {num_puzzles} puzzles")
    for gs, count in sorted(size_counts.items()):
        print(f"  - {gs}x{gs}: {count} puzzles ({100*count/num_puzzles:.1f}%)")
    
    return num_groups, num_puzzles


# =============================================================================
# BASE GRID ENUMERATION / LOADING
# =============================================================================

def enumerate_all_grids(grid_size: int) -> list[np.ndarray]:
    """
    Enumerate ALL valid Sudoku grids via backtracking.
    
    Supported sizes:
    - 4x4: 288 valid grids (instant)
    - 6x6: NOT USED - we use hardcoded base grids instead
    - 9x9: ~6.67 × 10^21 grids (NOT FEASIBLE)
    """
    if grid_size == 9:
        raise ValueError(
            "Full enumeration of 9x9 Sudoku is not feasible (~6.67 × 10^21 grids). "
            "Use the default transformation-based generation instead."
        )
    
    if grid_size == 6:
        raise ValueError(
            "Full enumeration of 6x6 is too slow. Use get_6x6_base_grids() instead."
        )
    
    if grid_size not in BOX_DIMS:
        raise ValueError(f"Full enumeration not supported for grid_size={grid_size}")
    
    box_rows, box_cols, _, _ = BOX_DIMS[grid_size]
    
    all_grids = []
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    total_cells = grid_size * grid_size
    
    def is_valid_placement(row, col, num):
        # Check row and column
        if num in grid[row, :] or num in grid[:, col]:
            return False
        # Check box
        br = (row // box_rows) * box_rows
        bc = (col // box_cols) * box_cols
        if num in grid[br:br+box_rows, bc:bc+box_cols]:
            return False
        return True
    
    def backtrack(pos):
        if pos == total_cells:
            all_grids.append(grid.copy())
            return
        
        row, col = pos // grid_size, pos % grid_size
        for num in range(1, grid_size + 1):
            if is_valid_placement(row, col, num):
                grid[row, col] = num
                backtrack(pos + 1)
                grid[row, col] = 0
    
    print(f"Enumerating all valid {grid_size}x{grid_size} Sudoku grids...")
    backtrack(0)
    print(f"✓ Found {len(all_grids):,} valid grids")
    
    return all_grids


def get_6x6_base_grids() -> np.ndarray:
    """Return diverse base grids for 6x6 Sudoku."""
    base_grids = []
    
    base_grids.append(np.array([
        [1,2,3,4,5,6],
        [4,5,6,1,2,3],
        [2,3,1,5,6,4],
        [5,6,4,2,3,1],
        [3,1,2,6,4,5],
        [6,4,5,3,1,2],
    ], dtype=np.int32))
    
    base_grids.append(np.array([
        [1,4,5,6,3,2],
        [2,3,6,4,1,5],
        [4,5,2,1,6,3],
        [3,6,1,2,5,4],
        [5,1,4,3,2,6],
        [6,2,3,5,4,1],
    ], dtype=np.int32))
    
    base_grids.append(np.array([
        [2,5,4,6,1,3],
        [6,3,1,5,2,4],
        [5,2,6,4,3,1],
        [4,1,3,2,5,6],
        [3,6,2,1,4,5],
        [1,4,5,3,6,2],
    ], dtype=np.int32))
    
    base_grids.append(np.array([
        [1,6,3,5,2,4],
        [2,5,4,1,3,6],
        [6,1,5,2,4,3],
        [4,3,2,6,1,5],
        [5,4,1,3,6,2],
        [3,2,6,4,5,1],
    ], dtype=np.int32))
    
    base_grids.append(np.array([
        [6,3,5,4,1,2],
        [1,4,2,6,3,5],
        [2,5,3,1,6,4],
        [4,1,6,2,5,3],
        [3,2,1,5,4,6],
        [5,6,4,3,2,1],
    ], dtype=np.int32))
    
    base_grids.append(np.array([
        [2,3,6,4,1,5],
        [1,4,5,3,6,2],
        [5,2,1,6,3,4],
        [3,6,4,2,5,1],
        [6,5,2,1,4,3],
        [4,1,3,5,2,6],
    ], dtype=np.int32))
    
    # Verify all grids are valid
    valid_grids = []
    for i, grid in enumerate(base_grids):
        if _is_valid_6x6(grid):
            valid_grids.append(grid)
        else:
            print(f"Warning: Base grid {i+1} is invalid, skipping")
    
    if len(valid_grids) == 0:
        raise RuntimeError("No valid base grids found!")
    
    print(f"Using {len(valid_grids)} verified base grids for 6x6")
    return np.stack(valid_grids)


def _is_valid_6x6(grid: np.ndarray) -> bool:
    """Check if a 6x6 Sudoku grid is valid and complete."""
    if not np.all((grid >= 1) & (grid <= 6)):
        return False
    
    for i in range(6):
        if len(set(grid[i, :])) != 6:
            return False
    
    for j in range(6):
        if len(set(grid[:, j])) != 6:
            return False
    
    for box_r in range(0, 6, 2):
        for box_c in range(0, 6, 3):
            box = grid[box_r:box_r+2, box_c:box_c+3].flatten()
            if len(set(box)) != 6:
                return False
    
    return True


def get_9x9_base_grids() -> np.ndarray:
    """Return diverse base grids for 9x9 Sudoku."""
    base_grids = []
    
    base_grids.append(np.array([
        [1,2,3,4,5,6,7,8,9],
        [4,5,6,7,8,9,1,2,3],
        [7,8,9,1,2,3,4,5,6],
        [2,3,1,5,6,4,8,9,7],
        [5,6,4,8,9,7,2,3,1],
        [8,9,7,2,3,1,5,6,4],
        [3,1,2,6,4,5,9,7,8],
        [6,4,5,9,7,8,3,1,2],
        [9,7,8,3,1,2,6,4,5],
    ], dtype=np.int32))
    
    base_grids.append(np.array([
        [1,2,3,4,5,6,7,8,9],
        [4,5,6,7,8,9,1,2,3],
        [7,8,9,1,2,3,4,5,6],
        [2,3,4,5,6,7,8,9,1],
        [5,6,7,8,9,1,2,3,4],
        [8,9,1,2,3,4,5,6,7],
        [3,4,5,6,7,8,9,1,2],
        [6,7,8,9,1,2,3,4,5],
        [9,1,2,3,4,5,6,7,8],
    ], dtype=np.int32))
    
    base_grids.append(np.array([
        [1,2,3,4,5,6,7,8,9],
        [4,5,6,7,8,9,1,2,3],
        [7,8,9,1,2,3,4,5,6],
        [3,1,2,6,4,5,9,7,8],
        [6,4,5,9,7,8,3,1,2],
        [9,7,8,3,1,2,6,4,5],
        [2,3,1,5,6,4,8,9,7],
        [5,6,4,8,9,7,2,3,1],
        [8,9,7,2,3,1,5,6,4],
    ], dtype=np.int32))
    
    base_grids.append(np.array([
        [1,2,3,4,5,6,7,8,9],
        [4,5,6,7,8,9,1,2,3],
        [7,8,9,1,2,3,4,5,6],
        [2,3,1,8,9,7,5,6,4],
        [8,9,7,5,6,4,2,3,1],
        [5,6,4,2,3,1,8,9,7],
        [3,1,2,9,7,8,6,4,5],
        [9,7,8,6,4,5,3,1,2],
        [6,4,5,3,1,2,9,7,8],
    ], dtype=np.int32))
    
    base_grids.append(np.array([
        [1,2,3,4,5,6,7,8,9],
        [4,5,6,7,8,9,1,2,3],
        [7,8,9,1,2,3,4,5,6],
        [5,6,4,2,3,1,8,9,7],
        [2,3,1,8,9,7,5,6,4],
        [8,9,7,5,6,4,2,3,1],
        [6,4,5,3,1,2,9,7,8],
        [3,1,2,9,7,8,6,4,5],
        [9,7,8,6,4,5,3,1,2],
    ], dtype=np.int32))
    
    # Verify all grids are valid
    valid_grids = []
    for i, grid in enumerate(base_grids):
        if _is_valid_9x9(grid):
            valid_grids.append(grid)
        else:
            print(f"Warning: Base grid {i+1} is invalid, skipping")
    
    if len(valid_grids) == 0:
        raise RuntimeError("No valid 9x9 base grids found!")
    
    print(f"Using {len(valid_grids)} verified base grids for 9x9")
    return np.stack(valid_grids)


def _is_valid_9x9(grid: np.ndarray) -> bool:
    """Check if a 9x9 Sudoku grid is valid and complete."""
    if not np.all((grid >= 1) & (grid <= 9)):
        return False
    
    for i in range(9):
        if len(set(grid[i, :])) != 9:
            return False
    
    for j in range(9):
        if len(set(grid[:, j])) != 9:
            return False
    
    for box_r in range(0, 9, 3):
        for box_c in range(0, 9, 3):
            box = grid[box_r:box_r+3, box_c:box_c+3].flatten()
            if len(set(box)) != 9:
                return False
    
    return True


def get_or_create_base_grids(grid_size: int, cache_dir: Path) -> np.ndarray:
    """Get base grids for full enumeration mode."""
    if grid_size == 6:
        return get_6x6_base_grids()
    
    if grid_size == 9:
        return get_9x9_base_grids()
    
    if grid_size == 4:
        cache_file = cache_dir / f"base_grids_{grid_size}x{grid_size}.npy"
        
        if cache_file.exists():
            print(f"Loading cached base grids from {cache_file}")
            return np.load(cache_file)
        
        all_grids = enumerate_all_grids(grid_size)
        cache_dir.mkdir(parents=True, exist_ok=True)
        base_grids = np.stack(all_grids)
        np.save(cache_file, base_grids)
        print(f"✓ Saved {len(all_grids)} base grids to {cache_file}")
        return base_grids
    
    raise ValueError(f"Full mode not supported for grid_size={grid_size}")


# =============================================================================
# MAIN CLI
# =============================================================================

@click.command()
@click.option("--output-dir", default="./data/sudoku_pregenerated", type=click.Path())
@click.option("--grid-size", default=4, type=click.IntRange(4, 9))
@click.option("--max-grid-size", default=None, type=int)
@click.option("--num-train", default=10000, type=int)
@click.option("--num-val", default=1000, type=int)
@click.option("--num-test", default=1000, type=int)
@click.option("--min-givens", default=None, type=int)
@click.option("--max-givens", default=None, type=int)
@click.option("--num-aug", default=0, type=int, help="Number of augmentations per training puzzle")
@click.option("--seed", default=42, type=int)
@click.option("--full", is_flag=True, default=False)
@click.option("--hybrid", is_flag=True, default=False)
@click.option("--mixed-size", is_flag=True, default=False)
@click.option("--cross-size", is_flag=True, default=False)
@click.option("--cache-dir", default="./data/sudoku_cache", type=click.Path())
def main(
    output_dir, grid_size, max_grid_size, num_train, num_val, num_test,
    min_givens, max_givens, num_aug, seed, full, hybrid, mixed_size, cross_size, cache_dir,
):
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    
    np.random.seed(seed)

    mode_flags = sum([full, hybrid, mixed_size, cross_size])
    if mode_flags > 1:
        raise click.UsageError("Cannot combine --full, --hybrid, --mixed-size, and --cross-size.")

    # Print theoretical limits for the requested grid size
    max_unique = get_max_unique_solutions(grid_size)
    total_requested = num_train + num_val + num_test
    click.echo(f"\n{'='*60}")
    click.echo(f"Grid size: {grid_size}x{grid_size}")
    click.echo(f"Theoretical unique solutions: {max_unique:,}")
    click.echo(f"Requested puzzles: {total_requested:,} (train={num_train}, val={num_val}, test={num_test})")
    if total_requested > max_unique:
        click.echo(f"⚠ Note: Requesting more puzzles than unique solutions exist.")
        click.echo(f"  Solution uniqueness will be disabled; augmentations provide diversity.")
    click.echo(f"{'='*60}\n")

    # =========================================================================
    # MIXED-SIZE MODE
    # =========================================================================
    if mixed_size:
        grid_sizes = [6, 9]
        max_grid_size = 10
        vocab_size = 3 + max(grid_sizes)  # 12
        
        givens_dict = {}
        for gs in grid_sizes:
            total_cells = gs * gs
            givens_dict[gs] = (int(total_cells * 0.35), int(total_cells * 0.60))
        
        click.echo("MIXED-SIZE MODE (50/50 mixture of 6x6 and 9x9)")
        click.echo("=" * 60)
        
        # Load base grids
        base_grids_dict = {}
        for gs in grid_sizes:
            base_grids_dict[gs] = get_or_create_base_grids(gs, cache_dir)
            click.echo(f"  ✓ {gs}x{gs}: {len(base_grids_dict[gs])} base grids")
        
        # Generate pools
        click.echo(f"\nGenerating train split ({num_train} puzzles)...")
        train_pool = generate_mixed_pool(num_train, grid_sizes, base_grids_dict, givens_dict, seed)
        
        train_hashes = set((gs, sol.tobytes()) for _, sol, gs in train_pool)
        
        click.echo(f"\nGenerating val split ({num_val} puzzles)...")
        val_pool = generate_mixed_pool(num_val, grid_sizes, base_grids_dict, givens_dict, 
                                        seed + 1000000, train_hashes)
        
        val_hashes = set((gs, sol.tobytes()) for _, sol, gs in val_pool)
        
        click.echo(f"\nGenerating test split ({num_test} puzzles)...")
        test_pool = generate_mixed_pool(num_test, grid_sizes, base_grids_dict, givens_dict,
                                         seed + 2000000, train_hashes | val_hashes)
        
        puzzle_pool = train_pool + val_pool + test_pool
        split_indices = {
            "train": list(range(len(train_pool))),
            "val": list(range(len(train_pool), len(train_pool) + len(val_pool))),
            "test": list(range(len(train_pool) + len(val_pool), len(puzzle_pool))),
        }
        
        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {}
        for split_name in ["train", "val", "test"]:
            num_groups, num_puzzles = save_mixed_split_to_disk(
                output_dir, split_name, puzzle_pool, 
                split_indices[split_name], max_grid_size, vocab_size, 
                num_aug if split_name == "train" else 0
            )
            stats[split_name] = (num_groups, num_puzzles)
        
        # Global metadata
        overall_meta = {
            "mode": "mixed-size",
            "max_grid_size": max_grid_size,
            "vocab_size": vocab_size,
            "seq_len": max_grid_size * max_grid_size,
            "num_train": stats["train"][1],
            "num_val": stats["val"][1],
            "num_test": stats["test"][1],
            "num_train_groups": stats["train"][0],
            "num_val_groups": stats["val"][0],
            "num_test_groups": stats["test"][0],
            "num_augmentations": num_aug,
            "seed": seed,
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(overall_meta, f, indent=2)

        click.echo(f"\n✓ Saved to {output_dir}")
        click.echo(f"  Train: {stats['train'][0]} groups × {1 + num_aug} = {stats['train'][1]} puzzles")
        click.echo(f"  Val:   {stats['val'][0]} groups = {stats['val'][1]} puzzles")
        click.echo(f"  Test:  {stats['test'][0]} groups = {stats['test'][1]} puzzles")
        return

    # =========================================================================
    # CROSS-SIZE MODE
    # =========================================================================
    if cross_size:
        train_grid_size = 6
        eval_grid_size = 9
        max_grid_size = 9
        vocab_size = 3 + eval_grid_size  # 12
        
        train_cells = train_grid_size * train_grid_size
        eval_cells = eval_grid_size * eval_grid_size
        
        train_min_givens = int(train_cells * 0.35)
        train_max_givens = int(train_cells * 0.60)
        eval_min_givens = int(eval_cells * 0.35)
        eval_max_givens = int(eval_cells * 0.60)
        
        click.echo("CROSS-SIZE MODE (train: 6x6, val/test: 9x9)")
        click.echo("=" * 60)
        
        base_grids_6x6 = get_or_create_base_grids(6, cache_dir)
        base_grids_9x9 = get_or_create_base_grids(9, cache_dir)
        
        click.echo(f"\nGenerating train split ({num_train} puzzles, 6x6)...")
        train_pool = generate_single_size_pool(
            num_train, train_grid_size, base_grids_6x6,
            train_min_givens, train_max_givens, seed
        )
        
        click.echo(f"\nGenerating val split ({num_val} puzzles, 9x9)...")
        val_pool = generate_single_size_pool(
            num_val, eval_grid_size, base_grids_9x9,
            eval_min_givens, eval_max_givens, seed + 1000000
        )
        
        val_hashes = set(sol.tobytes() for _, sol in val_pool)
        
        click.echo(f"\nGenerating test split ({num_test} puzzles, 9x9)...")
        test_pool = generate_single_size_pool(
            num_test, eval_grid_size, base_grids_9x9,
            eval_min_givens, eval_max_givens, seed + 2000000, val_hashes
        )
        
        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_train_groups, num_train_puzzles = save_split_to_disk(
            output_dir, "train", train_pool, list(range(len(train_pool))),
            train_grid_size, max_grid_size, train_min_givens, train_max_givens, vocab_size, num_aug
        )
        num_val_groups, num_val_puzzles = save_split_to_disk(
            output_dir, "val", val_pool, list(range(len(val_pool))),
            eval_grid_size, max_grid_size, eval_min_givens, eval_max_givens, vocab_size, 0
        )
        num_test_groups, num_test_puzzles = save_split_to_disk(
            output_dir, "test", test_pool, list(range(len(test_pool))),
            eval_grid_size, max_grid_size, eval_min_givens, eval_max_givens, vocab_size, 0
        )
        
        # Global metadata
        overall_meta = {
            "mode": "cross-size",
            "max_grid_size": max_grid_size,
            "vocab_size": vocab_size,
            "seq_len": max_grid_size * max_grid_size,
            "num_train": num_train_puzzles,
            "num_val": num_val_puzzles,
            "num_test": num_test_puzzles,
            "num_train_groups": num_train_groups,
            "num_val_groups": num_val_groups,
            "num_test_groups": num_test_groups,
            "num_augmentations": num_aug,
            "seed": seed,
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(overall_meta, f, indent=2)

        click.echo(f"\n✓ Saved to {output_dir}")
        click.echo(f"  Train: {num_train_groups} groups × {1 + num_aug} = {num_train_puzzles} puzzles")
        click.echo(f"  Val:   {num_val_groups} groups = {num_val_puzzles} puzzles")
        click.echo(f"  Test:  {num_test_groups} groups = {num_test_puzzles} puzzles")
        return

    # =========================================================================
    # STANDARD / FULL / HYBRID MODES
    # =========================================================================
    if max_grid_size is None:
        max_grid_size = grid_size + 2

    total_cells = grid_size * grid_size
    if min_givens is None:
        min_givens = int(total_cells * 0.35)
    if max_givens is None:
        max_givens = int(total_cells * 0.60)
    
    vocab_size = 3 + grid_size

    if hybrid:
        click.echo("HYBRID MODE")
        click.echo("=" * 60)
        
        dm_train = SudokuDataModule(
            data_dir=None, num_train_puzzles=num_train, num_val_puzzles=0, num_test_puzzles=0,
            grid_size=grid_size, max_grid_size=max_grid_size,
            min_givens=min_givens, max_givens=max_givens, seed=seed, base_grids=None,
        )
        dm_train.setup("fit")
        
        train_pool = dm_train.get_puzzle_pool()
        train_indices = dm_train.get_split_indices()
        
        train_solution_hashes = set()
        for idx in train_indices["train"]:
            _, solution = train_pool[idx]
            train_solution_hashes.add(solution.tobytes())
        
        base_grids = get_or_create_base_grids(grid_size, cache_dir)
        
        val_pool = generate_ood_pool(num_val, grid_size, min_givens, max_givens,
                                      seed + 1000000, base_grids, train_solution_hashes)
        
        val_solution_hashes = set(sol.tobytes() for _, sol in val_pool)
        
        test_pool = generate_ood_pool(num_test, grid_size, min_givens, max_givens,
                                       seed + 2000000, base_grids, 
                                       train_solution_hashes | val_solution_hashes)
        
        puzzle_pool = train_pool + val_pool + test_pool
        split_indices = {
            "train": train_indices["train"],
            "val": list(range(len(train_pool), len(train_pool) + len(val_pool))),
            "test": list(range(len(train_pool) + len(val_pool), len(puzzle_pool))),
        }
    else:
        # Standard or Full mode
        base_grids = get_or_create_base_grids(grid_size, cache_dir) if full else None
        
        mode_name = "FULL" if full else "STANDARD"
        click.echo(f"{mode_name} MODE")
        click.echo("=" * 60)
        
        # Generate pools directly using our optimized function
        click.echo(f"\nGenerating train split ({num_train} puzzles)...")
        train_pool = generate_single_size_pool(
            num_train, grid_size, base_grids,
            min_givens, max_givens, seed
        )
        
        train_hashes = set(sol.tobytes() for _, sol in train_pool)
        
        click.echo(f"\nGenerating val split ({num_val} puzzles)...")
        val_pool = generate_single_size_pool(
            num_val, grid_size, base_grids,
            min_givens, max_givens, seed + 1000000, train_hashes
        )
        
        val_hashes = set(sol.tobytes() for _, sol in val_pool)
        
        click.echo(f"\nGenerating test split ({num_test} puzzles)...")
        test_pool = generate_single_size_pool(
            num_test, grid_size, base_grids,
            min_givens, max_givens, seed + 2000000, train_hashes | val_hashes
        )
        
        puzzle_pool = train_pool + val_pool + test_pool
        split_indices = {
            "train": list(range(len(train_pool))),
            "val": list(range(len(train_pool), len(train_pool) + len(val_pool))),
            "test": list(range(len(train_pool) + len(val_pool), len(puzzle_pool))),
        }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    for split_name in ["train", "val", "test"]:
        num_groups, num_puzzles = save_split_to_disk(
            output_dir, split_name, puzzle_pool, split_indices[split_name],
            grid_size, max_grid_size, min_givens, max_givens, vocab_size,
            num_aug if split_name == "train" else 0
        )
        stats[split_name] = (num_groups, num_puzzles)

    # Global metadata
    overall_meta = {
        "mode": "hybrid" if hybrid else ("full" if full else "standard"),
        "grid_size": grid_size,
        "max_grid_size": max_grid_size,
        "vocab_size": vocab_size,
        "seq_len": max_grid_size * max_grid_size,
        "min_givens": min_givens,
        "max_givens": max_givens,
        "num_train": stats["train"][1],
        "num_val": stats["val"][1],
        "num_test": stats["test"][1],
        "num_train_groups": stats["train"][0],
        "num_val_groups": stats["val"][0],
        "num_test_groups": stats["test"][0],
        "num_augmentations": num_aug,
        "seed": seed,
        "theoretical_max_unique_solutions": int(max_unique) if max_unique < float('inf') else "unlimited",
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(overall_meta, f, indent=2)

    click.echo(f"\n✓ Saved to {output_dir}")
    click.echo(f"  Train: {stats['train'][0]} groups × {1 + num_aug} = {stats['train'][1]} puzzles")
    click.echo(f"  Val:   {stats['val'][0]} groups = {stats['val'][1]} puzzles")
    click.echo(f"  Test:  {stats['test'][0]} groups = {stats['test'][1]} puzzles")


if __name__ == "__main__":
    main()