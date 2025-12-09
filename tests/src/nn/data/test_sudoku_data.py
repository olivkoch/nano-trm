#!/usr/bin/env python3
"""
Independent test script to validate pre-generated Sudoku datasets.

Reads a dataset from disk, prints samples, and validates correctness.
No dependencies on the main codebase - just numpy and click.

Supports all modes:
- standard/full/hybrid: Single grid_size for all splits
- cross-size: Different grid_size per split (train vs val/test)
- mixed-size: Multiple grid sizes within each split

Also validates group/augmentation structure.

Usage:
    python test_sudoku_data.py ./data/sudoku_pregenerated
    python test_sudoku_data.py ./data/sudoku_pregenerated --num-samples 10
    python test_sudoku_data.py ./data/sudoku_pregenerated --validate-all
    python test_sudoku_data.py ./data/sudoku_pregenerated --show-groups
"""

import json
from pathlib import Path

import click
import numpy as np


def load_dataset(data_dir: Path, split: str = "train"):
    """Load a dataset split from disk."""
    split_dir = data_dir / split
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    inputs = np.load(split_dir / "all__inputs.npy")
    labels = np.load(split_dir / "all__labels.npy")
    
    # Load optional arrays
    grid_sizes = None
    puzzle_indices = None
    group_indices = None
    
    grid_sizes_path = split_dir / "all__grid_sizes.npy"
    if grid_sizes_path.exists():
        grid_sizes = np.load(grid_sizes_path)
    
    puzzle_indices_path = split_dir / "all__puzzle_indices.npy"
    if puzzle_indices_path.exists():
        puzzle_indices = np.load(puzzle_indices_path)
    
    group_indices_path = split_dir / "all__group_indices.npy"
    if group_indices_path.exists():
        group_indices = np.load(group_indices_path)
    
    with open(split_dir / "dataset.json") as f:
        metadata = json.load(f)
    
    return {
        "inputs": inputs,
        "labels": labels,
        "grid_sizes": grid_sizes,
        "puzzle_indices": puzzle_indices,
        "group_indices": group_indices,
        "metadata": metadata,
    }


def load_global_metadata(data_dir: Path):
    """Load global metadata from the dataset."""
    metadata_path = data_dir / "metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path) as f:
        return json.load(f)


def decode_grid(encoded: np.ndarray, grid_size: int, max_grid_size: int):
    """
    Decode an encoded sequence back to puzzle and solution grids.
    
    Encoding: 0=PAD, 1=EOS, 2=empty, 3-(grid_size+2)=values 1-grid_size
    """
    seq = encoded.reshape(max_grid_size, max_grid_size)
    
    # Extract the actual grid (without padding)
    grid = seq[:grid_size, :grid_size].copy()
    
    # Decode: 2 -> 0 (empty), 3+ -> 1+ (values)
    decoded = np.zeros_like(grid)
    decoded[grid == 2] = 0  # Empty cells
    decoded[grid >= 3] = grid[grid >= 3] - 2  # Values
    
    return decoded


def get_box_dims(grid_size: int):
    """Get box dimensions for a given grid size."""
    if grid_size == 4:
        return 2, 2
    elif grid_size == 6:
        return 2, 3
    elif grid_size == 9:
        return 3, 3
    else:
        raise ValueError(f"Unsupported grid size: {grid_size}")


def is_valid_sudoku(grid: np.ndarray, grid_size: int, check_complete: bool = False):
    """
    Check if a Sudoku grid is valid.
    
    Args:
        grid: The Sudoku grid
        grid_size: Size of the grid (4, 6, or 9)
        check_complete: If True, also check that all cells are filled
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    box_rows, box_cols = get_box_dims(grid_size)
    
    # Check for completeness if requested
    if check_complete:
        if np.any(grid == 0):
            return False, "Grid has empty cells"
        if np.any((grid < 1) | (grid > grid_size)):
            return False, f"Grid has values outside range [1, {grid_size}]"
    
    # Check rows
    for i in range(grid_size):
        row = grid[i, :]
        row_vals = row[row > 0]
        if len(row_vals) != len(set(row_vals)):
            return False, f"Duplicate in row {i}: {row}"
    
    # Check columns
    for j in range(grid_size):
        col = grid[:, j]
        col_vals = col[col > 0]
        if len(col_vals) != len(set(col_vals)):
            return False, f"Duplicate in column {j}: {col}"
    
    # Check boxes
    for box_r in range(0, grid_size, box_rows):
        for box_c in range(0, grid_size, box_cols):
            box = grid[box_r:box_r+box_rows, box_c:box_c+box_cols].flatten()
            box_vals = box[box > 0]
            if len(box_vals) != len(set(box_vals)):
                return False, f"Duplicate in box ({box_r//box_rows}, {box_c//box_cols}): {box}"
    
    return True, "Valid"


def solution_matches_puzzle(puzzle: np.ndarray, solution: np.ndarray):
    """Check that the solution preserves all given cells from the puzzle."""
    given_mask = puzzle > 0
    if not np.all(solution[given_mask] == puzzle[given_mask]):
        return False, "Solution doesn't match puzzle givens"
    return True, "Matches"


def print_grid(grid: np.ndarray, grid_size: int, title: str = ""):
    """Pretty print a Sudoku grid."""
    box_rows, box_cols = get_box_dims(grid_size)
    
    if title:
        print(title)
    
    h_line = "+" + "+".join(["-" * (box_cols * 2 + 1)] * (grid_size // box_cols)) + "+"
    
    for i in range(grid_size):
        if i % box_rows == 0:
            print(h_line)
        
        row_str = "|"
        for j in range(grid_size):
            if j % box_cols == 0 and j > 0:
                row_str += " |"
            val = grid[i, j]
            row_str += f" {val if val > 0 else '.'}"
        row_str += " |"
        print(row_str)
    
    print(h_line)


def print_sample(idx: int, puzzle: np.ndarray, solution: np.ndarray, grid_size: int, 
                 group_idx: int = None, aug_idx: int = None):
    """Print a single sample with puzzle and solution side by side."""
    print(f"\n{'='*60}")
    header = f"Sample {idx} (grid_size={grid_size}x{grid_size})"
    if group_idx is not None:
        header += f" [Group {group_idx}"
        if aug_idx is not None:
            header += f", Aug {aug_idx}"
        header += "]"
    print(header)
    print(f"{'='*60}")
    
    num_givens = np.sum(puzzle > 0)
    num_empty = np.sum(puzzle == 0)
    print(f"Givens: {num_givens}, Empty: {num_empty}")
    
    print_grid(puzzle, grid_size, "\nPuzzle:")
    print_grid(solution, grid_size, "\nSolution:")


def validate_sample(puzzle: np.ndarray, solution: np.ndarray, grid_size: int):
    """Validate a single sample. Returns list of errors (empty if valid)."""
    errors = []
    
    # Check puzzle is valid (may have empty cells)
    valid, msg = is_valid_sudoku(puzzle, grid_size, check_complete=False)
    if not valid:
        errors.append(f"Invalid puzzle: {msg}")
    
    # Check solution is valid and complete
    valid, msg = is_valid_sudoku(solution, grid_size, check_complete=True)
    if not valid:
        errors.append(f"Invalid solution: {msg}")
    
    # Check solution matches puzzle
    valid, msg = solution_matches_puzzle(puzzle, solution)
    if not valid:
        errors.append(f"Mismatch: {msg}")
    
    return errors


def validate_group_structure(data: dict):
    """Validate the group/puzzle index structure."""
    errors = []
    
    puzzle_indices = data.get("puzzle_indices")
    group_indices = data.get("group_indices")
    num_puzzles = len(data["inputs"])
    metadata = data["metadata"]
    
    if puzzle_indices is None or group_indices is None:
        return ["No group indices found (old format?)"]
    
    # Check puzzle_indices
    expected_puzzle_indices = list(range(num_puzzles + 1))
    if not np.array_equal(puzzle_indices, expected_puzzle_indices):
        errors.append(f"puzzle_indices mismatch: expected [0..{num_puzzles}], got range [{puzzle_indices[0]}..{puzzle_indices[-1]}]")
    
    # Check group_indices
    num_groups = metadata.get("num_groups", 0)
    if len(group_indices) != num_groups + 1:
        errors.append(f"group_indices length mismatch: expected {num_groups + 1}, got {len(group_indices)}")
    
    if group_indices[0] != 0:
        errors.append(f"group_indices should start at 0, got {group_indices[0]}")
    
    if group_indices[-1] != num_puzzles:
        errors.append(f"group_indices should end at {num_puzzles}, got {group_indices[-1]}")
    
    # Check monotonicity
    if not np.all(np.diff(group_indices) > 0):
        errors.append("group_indices is not strictly monotonic")
    
    # Check mean_puzzles_per_group
    expected_mean = num_puzzles / num_groups if num_groups > 0 else 1
    actual_mean = metadata.get("mean_puzzles_per_group", 0)
    if abs(expected_mean - actual_mean) > 0.01:
        errors.append(f"mean_puzzles_per_group mismatch: expected {expected_mean:.2f}, got {actual_mean:.2f}")
    
    return errors


def get_group_for_sample(idx: int, group_indices: np.ndarray):
    """Get the group index and within-group index for a sample."""
    if group_indices is None:
        return None, None
    
    # Binary search for group
    group_idx = np.searchsorted(group_indices[1:], idx, side='right')
    aug_idx = idx - group_indices[group_idx]
    
    return group_idx, aug_idx


@click.command()
@click.argument(
    "data_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test"]),
    default="train",
    help="Which split to test (default: train)",
)
@click.option(
    "--num-samples",
    type=int,
    default=5,
    help="Number of samples to print (default: 5)",
)
@click.option(
    "--validate-all",
    is_flag=True,
    default=False,
    help="Validate all samples (not just printed ones)",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Only print summary, not individual samples",
)
@click.option(
    "--show-groups",
    is_flag=True,
    default=False,
    help="Show samples from different groups to visualize augmentation",
)
def main(data_dir: Path, split: str, num_samples: int, validate_all: bool, quiet: bool, show_groups: bool):
    """
    Test and validate pre-generated Sudoku datasets.
    
    DATA_DIR is the path to the pre-generated dataset directory.
    """
    # Load global metadata
    click.echo(f"Loading dataset from: {data_dir}")
    metadata = load_global_metadata(data_dir)
    
    # Detect mode from global metadata
    mode = metadata.get("mode", "standard")
    
    click.echo(f"\nGlobal metadata ({mode.upper()}):")
    click.echo(f"  Max grid size: {metadata['max_grid_size']}x{metadata['max_grid_size']}")
    click.echo(f"  Vocab size: {metadata['vocab_size']}")
    click.echo(f"  Sequence length: {metadata['seq_len']}")
    
    # Show puzzle counts
    click.echo(f"\n  Puzzle counts (total):")
    click.echo(f"    Train: {metadata['num_train']}")
    click.echo(f"    Val:   {metadata['num_val']}")
    click.echo(f"    Test:  {metadata['num_test']}")
    
    # Show group counts if available
    if "num_train_groups" in metadata:
        click.echo(f"\n  Group counts (base puzzles):")
        click.echo(f"    Train: {metadata['num_train_groups']}")
        click.echo(f"    Val:   {metadata['num_val_groups']}")
        click.echo(f"    Test:  {metadata['num_test_groups']}")
        
        num_aug = metadata.get("num_augmentations", 0)
        click.echo(f"\n  Augmentations: {num_aug} per training puzzle")
    
    # Load split
    click.echo(f"\nLoading {split} split...")
    data = load_dataset(data_dir, split)
    
    inputs = data["inputs"]
    labels = data["labels"]
    grid_sizes_array = data["grid_sizes"]
    group_indices = data["group_indices"]
    split_meta = data["metadata"]
    
    total_samples = len(inputs)
    max_grid_size = split_meta["max_grid_size"]
    is_mixed = split_meta.get("mixed_size", False) or grid_sizes_array is not None
    
    click.echo(f"\nSplit metadata:")
    click.echo(f"  Total puzzles: {total_samples}")
    
    # Show group info
    num_groups = split_meta.get("num_groups", total_samples)
    num_aug = split_meta.get("num_augmentations", 0)
    mean_per_group = split_meta.get("mean_puzzles_per_group", 1)
    
    click.echo(f"  Groups: {num_groups}")
    click.echo(f"  Augmentations: {num_aug}")
    click.echo(f"  Mean puzzles/group: {mean_per_group:.2f}")
    
    if is_mixed:
        grid_sizes_list = split_meta.get("grid_sizes", [])
        size_counts = split_meta.get("grid_size_counts", {})
        click.echo(f"  Mixed sizes: {grid_sizes_list}")
        for gs in grid_sizes_list:
            count = size_counts.get(str(gs), "?")
            click.echo(f"    {gs}x{gs}: {count} puzzles")
    else:
        grid_size = split_meta["grid_size"]
        click.echo(f"  Grid size: {grid_size}x{grid_size}")
        click.echo(f"  Givens: {split_meta.get('min_givens', '?')}-{split_meta.get('max_givens', '?')}")
    
    # Validate group structure
    click.echo(f"\n{'='*60}")
    click.echo("Validating group structure...")
    click.echo(f"{'='*60}")
    
    group_errors = validate_group_structure(data)
    if group_errors:
        click.echo("❌ Group structure errors:")
        for err in group_errors:
            click.echo(f"   - {err}")
    else:
        click.echo("✓ Group structure valid")
    
    # Show groups if requested
    if show_groups and group_indices is not None and num_groups > 0:
        click.echo(f"\n{'='*60}")
        click.echo("Showing augmentation groups...")
        click.echo(f"{'='*60}")
        
        # Show first 2 groups
        num_groups_to_show = min(2, num_groups)
        for g in range(num_groups_to_show):
            start_idx = group_indices[g]
            end_idx = group_indices[g + 1]
            group_size = end_idx - start_idx
            
            click.echo(f"\n--- Group {g} ({group_size} puzzles: indices {start_idx}-{end_idx-1}) ---")
            
            # Show first and last in group
            indices_to_show = [start_idx]
            if group_size > 1:
                indices_to_show.append(start_idx + 1)
            if group_size > 2:
                indices_to_show.append(end_idx - 1)
            
            for idx in indices_to_show:
                if is_mixed and grid_sizes_array is not None:
                    sample_grid_size = int(grid_sizes_array[idx])
                else:
                    sample_grid_size = split_meta["grid_size"]
                
                puzzle = decode_grid(inputs[idx], sample_grid_size, max_grid_size)
                solution = decode_grid(labels[idx], sample_grid_size, max_grid_size)
                
                aug_idx = idx - start_idx
                print_sample(idx, puzzle, solution, sample_grid_size, g, aug_idx)
    
    # Print some samples
    elif not quiet:
        print_indices = np.linspace(0, total_samples - 1, min(num_samples, total_samples), dtype=int)
        
        for idx in print_indices:
            # Get grid_size for this sample
            if is_mixed and grid_sizes_array is not None:
                sample_grid_size = int(grid_sizes_array[idx])
            else:
                sample_grid_size = split_meta["grid_size"]
            
            puzzle = decode_grid(inputs[idx], sample_grid_size, max_grid_size)
            solution = decode_grid(labels[idx], sample_grid_size, max_grid_size)
            
            group_idx, aug_idx = get_group_for_sample(idx, group_indices)
            print_sample(idx, puzzle, solution, sample_grid_size, group_idx, aug_idx)
            
            errors = validate_sample(puzzle, solution, sample_grid_size)
            if errors:
                click.echo("❌ ERRORS:")
                for err in errors:
                    click.echo(f"   - {err}")
            else:
                click.echo("✓ Valid")
    
    # Validate all samples if requested
    if validate_all:
        click.echo(f"\n{'='*60}")
        click.echo(f"Validating ALL {total_samples} samples...")
        click.echo(f"{'='*60}")
        
        invalid_count = 0
        error_details = []
        
        for idx in range(total_samples):
            # Get grid_size for this sample
            if is_mixed and grid_sizes_array is not None:
                sample_grid_size = int(grid_sizes_array[idx])
            else:
                sample_grid_size = split_meta["grid_size"]
            
            puzzle = decode_grid(inputs[idx], sample_grid_size, max_grid_size)
            solution = decode_grid(labels[idx], sample_grid_size, max_grid_size)
            
            errors = validate_sample(puzzle, solution, sample_grid_size)
            if errors:
                invalid_count += 1
                if len(error_details) < 10:  # Keep first 10 errors
                    error_details.append((idx, sample_grid_size, errors))
            
            if (idx + 1) % 1000 == 0:
                click.echo(f"  Validated {idx + 1}/{total_samples}...")
        
        click.echo(f"\n{'='*60}")
        click.echo(f"VALIDATION SUMMARY")
        click.echo(f"{'='*60}")
        click.echo(f"Total samples: {total_samples}")
        click.echo(f"Valid samples: {total_samples - invalid_count}")
        click.echo(f"Invalid samples: {invalid_count}")
        
        if invalid_count == 0:
            click.echo("\n✅ ALL SAMPLES VALID!")
        else:
            click.echo(f"\n❌ {invalid_count} INVALID SAMPLES")
            click.echo("\nFirst few errors:")
            for idx, gs, errors in error_details:
                click.echo(f"\n  Sample {idx} ({gs}x{gs}):")
                for err in errors:
                    click.echo(f"    - {err}")
    
    # Check for duplicates
    click.echo(f"\n{'='*60}")
    click.echo("Checking for duplicates...")
    click.echo(f"{'='*60}")
    
    # Hash all puzzles - track by group
    puzzle_hashes = {}  # hash -> list of (idx, group_idx)
    
    for idx in range(total_samples):
        h = inputs[idx].tobytes()
        group_idx, _ = get_group_for_sample(idx, group_indices)
        
        if h not in puzzle_hashes:
            puzzle_hashes[h] = []
        puzzle_hashes[h].append((idx, group_idx))
    
    # Count duplicates
    cross_group_dupes = 0
    within_group_dupes = 0
    
    for h, occurrences in puzzle_hashes.items():
        if len(occurrences) > 1:
            groups = set(g for _, g in occurrences)
            if len(groups) > 1:
                # Same puzzle in different groups
                cross_group_dupes += len(occurrences) - 1
            else:
                # Same puzzle within same group (unexpected for Sudoku augmentation)
                within_group_dupes += len(occurrences) - 1
    
    unique_puzzles = len(puzzle_hashes)
    click.echo(f"Unique puzzles: {unique_puzzles}")
    click.echo(f"Total puzzles: {total_samples}")
    
    if cross_group_dupes > 0:
        click.echo(f"⚠ Cross-group duplicates: {cross_group_dupes}")
    else:
        click.echo(f"✓ No cross-group duplicates")
    
    if within_group_dupes > 0:
        click.echo(f"⚠ Within-group duplicates: {within_group_dupes} (unexpected)")
    
    # Check solution uniqueness across groups
    click.echo(f"\n{'='*60}")
    click.echo("Checking solution uniqueness across groups...")
    click.echo(f"{'='*60}")
    
    solution_to_groups = {}  # solution_hash -> set of group_indices
    
    for idx in range(total_samples):
        sol_h = labels[idx].tobytes()
        group_idx, _ = get_group_for_sample(idx, group_indices)
        
        if sol_h not in solution_to_groups:
            solution_to_groups[sol_h] = set()
        solution_to_groups[sol_h].add(group_idx)
    
    shared_solutions = sum(1 for groups in solution_to_groups.values() if len(groups) > 1)
    
    if shared_solutions > 0:
        click.echo(f"⚠ {shared_solutions} solutions shared across groups")
    else:
        click.echo(f"✓ All groups have unique solutions")
    
    click.echo("\n✅ Test complete!")


if __name__ == "__main__":
    main()