#!/usr/bin/env python3
"""
Independent test script to validate pre-generated Sudoku datasets.

Reads a dataset from disk, prints samples, and validates correctness.
No dependencies on the main codebase - just numpy.

Usage:
    python test_sudoku_dataset.py ./data/sudoku_pregenerated
    python test_sudoku_dataset.py ./data/sudoku_pregenerated --num-samples 10
    python test_sudoku_dataset.py ./data/sudoku_pregenerated --validate-all
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_dataset(data_dir: Path, split: str = "train"):
    """Load a dataset split from disk."""
    split_dir = data_dir / split
    
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    inputs = np.load(split_dir / "all__inputs.npy")
    labels = np.load(split_dir / "all__labels.npy")
    
    with open(split_dir / "dataset.json") as f:
        metadata = json.load(f)
    
    return inputs, labels, metadata


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
                row_str += "|"
            val = grid[i, j]
            row_str += f" {val if val > 0 else '.'}"
        row_str += " |"
        print(row_str)
    
    print(h_line)


def print_sample(idx: int, puzzle: np.ndarray, solution: np.ndarray, grid_size: int):
    """Print a single sample with puzzle and solution side by side."""
    print(f"\n{'='*60}")
    print(f"Sample {idx}")
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


def main():
    parser = argparse.ArgumentParser(
        description="Test and validate pre-generated Sudoku datasets"
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the pre-generated dataset directory"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which split to test (default: train)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to print (default: 5)"
    )
    parser.add_argument(
        "--validate-all",
        action="store_true",
        help="Validate all samples (not just printed ones)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary, not individual samples"
    )
    
    args = parser.parse_args()
    
    # Load metadata
    print(f"Loading dataset from: {args.data_dir}")
    metadata = load_global_metadata(args.data_dir)
    
    grid_size = metadata["grid_size"]
    max_grid_size = metadata["max_grid_size"]
    
    print(f"\nDataset metadata:")
    print(f"  Grid size: {grid_size}x{grid_size}")
    print(f"  Max grid size: {max_grid_size}x{max_grid_size}")
    print(f"  Vocab size: {metadata['vocab_size']}")
    print(f"  Sequence length: {metadata['seq_len']}")
    print(f"  Train/Val/Test: {metadata['num_train']}/{metadata['num_val']}/{metadata['num_test']}")
    print(f"  Givens range: {metadata.get('min_givens', '?')}-{metadata.get('max_givens', '?')}")
    if metadata.get('full_enumeration'):
        print(f"  Generation mode: FULL ({metadata.get('num_base_grids', '?')} base grids)")
    else:
        print(f"  Generation mode: STANDARD")
    
    # Load split
    print(f"\nLoading {args.split} split...")
    inputs, labels, split_meta = load_dataset(args.data_dir, args.split)
    
    num_samples = len(inputs)
    print(f"Loaded {num_samples} samples")
    
    # Print some samples
    if not args.quiet:
        print_indices = np.linspace(0, num_samples - 1, min(args.num_samples, num_samples), dtype=int)
        
        for idx in print_indices:
            puzzle = decode_grid(inputs[idx], grid_size, max_grid_size)
            solution = decode_grid(labels[idx], grid_size, max_grid_size)
            print_sample(idx, puzzle, solution, grid_size)
            
            errors = validate_sample(puzzle, solution, grid_size)
            if errors:
                print("❌ ERRORS:")
                for err in errors:
                    print(f"   - {err}")
            else:
                print("✓ Valid")
    
    # Validate all samples if requested
    if args.validate_all:
        print(f"\n{'='*60}")
        print(f"Validating ALL {num_samples} samples...")
        print(f"{'='*60}")
        
        invalid_count = 0
        error_details = []
        
        for idx in range(num_samples):
            puzzle = decode_grid(inputs[idx], grid_size, max_grid_size)
            solution = decode_grid(labels[idx], grid_size, max_grid_size)
            
            errors = validate_sample(puzzle, solution, grid_size)
            if errors:
                invalid_count += 1
                if len(error_details) < 10:  # Keep first 10 errors
                    error_details.append((idx, errors))
            
            if (idx + 1) % 1000 == 0:
                print(f"  Validated {idx + 1}/{num_samples}...")
        
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples: {num_samples}")
        print(f"Valid samples: {num_samples - invalid_count}")
        print(f"Invalid samples: {invalid_count}")
        
        if invalid_count == 0:
            print("\n✅ ALL SAMPLES VALID!")
        else:
            print(f"\n❌ {invalid_count} INVALID SAMPLES")
            print("\nFirst few errors:")
            for idx, errors in error_details:
                print(f"\n  Sample {idx}:")
                for err in errors:
                    print(f"    - {err}")
    
    # Check for duplicates
    print(f"\n{'='*60}")
    print("Checking for duplicates...")
    print(f"{'='*60}")
    
    # Hash all puzzles
    puzzle_hashes = set()
    duplicate_count = 0
    
    for idx in range(num_samples):
        h = inputs[idx].tobytes()
        if h in puzzle_hashes:
            duplicate_count += 1
        else:
            puzzle_hashes.add(h)
    
    if duplicate_count == 0:
        print(f"✓ No duplicates found in {args.split} split")
    else:
        print(f"⚠ Found {duplicate_count} duplicate puzzles in {args.split} split")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()