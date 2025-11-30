#!/usr/bin/env python3
"""
Verify that --full mode generates solutions that standard mode cannot produce.

This script:
1. Enumerates ALL solutions reachable by standard mode (single base grid + limited symmetry)
2. Loads solutions from a full-mode dataset
3. Reports how many solutions are outside standard mode's reach

Usage:
    python verify_full_mode.py ./data/sudoku_6x6_full
    python verify_full_mode.py ./data/sudoku_6x6_full --num-samples 1000
"""

import json
import math
from itertools import permutations
from pathlib import Path

import click
import numpy as np


def get_standard_base_grid(grid_size: int) -> np.ndarray:
    """Get the single base grid used by standard mode."""
    if grid_size == 4:
        return np.array([
            [1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1]
        ], dtype=np.int32)
    elif grid_size == 6:
        return np.array([
            [1, 2, 3, 4, 5, 6],
            [4, 5, 6, 1, 2, 3],
            [2, 3, 4, 5, 6, 1],
            [5, 6, 1, 2, 3, 4],
            [3, 4, 5, 6, 1, 2],
            [6, 1, 2, 3, 4, 5],
        ], dtype=np.int32)
    elif grid_size == 9:
        return np.array([
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
    else:
        raise ValueError(f"Unsupported grid size: {grid_size}")


def enumerate_standard_mode_solutions(grid_size: int) -> set:
    """
    Enumerate ALL solutions reachable by standard mode.
    
    Standard mode uses:
    - Single base grid
    - Digit permutation (n!)
    - Row swaps within bands (2^n_bands for 2-row bands, or (3!)^n_bands for 3-row bands)
    - Column shuffles within stacks ((box_cols!)^n_stacks)
    
    Note: Standard mode does NOT use band permutation or stack permutation.
    """
    base = get_standard_base_grid(grid_size)
    
    if grid_size == 4:
        box_rows, box_cols = 2, 2
        n_bands, n_stacks = 2, 2
    elif grid_size == 6:
        box_rows, box_cols = 2, 3
        n_bands, n_stacks = 3, 2
    elif grid_size == 9:
        box_rows, box_cols = 3, 3
        n_bands, n_stacks = 3, 3
    else:
        raise ValueError(f"Unsupported grid size: {grid_size}")
    
    all_solutions = set()
    
    # Iterate over all digit permutations
    for digit_perm in permutations(range(1, grid_size + 1)):
        # Apply digit permutation
        grid = np.zeros_like(base)
        for old_val, new_val in enumerate(digit_perm, 1):
            grid[base == old_val] = new_val
        
        # Iterate over row swap patterns within bands
        if box_rows == 2:
            # 2 rows per band: swap or not (2^n_bands patterns)
            row_patterns = range(2 ** n_bands)
        else:
            # 3 rows per band: all permutations ((3!)^n_bands patterns)
            row_patterns = range(6 ** n_bands)
        
        for row_pattern in row_patterns:
            grid2 = grid.copy()
            
            if box_rows == 2:
                # Apply row swaps for 2-row bands
                for band in range(n_bands):
                    if (row_pattern >> band) & 1:
                        r1, r2 = band * 2, band * 2 + 1
                        grid2[[r1, r2]] = grid2[[r2, r1]]
            else:
                # Apply row permutations for 3-row bands
                temp_pattern = row_pattern
                for band in range(n_bands):
                    perm_idx = temp_pattern % 6
                    temp_pattern //= 6
                    row_perms = list(permutations(range(3)))
                    row_perm = row_perms[perm_idx]
                    rows = [band * 3 + r for r in row_perm]
                    grid2[band * 3:band * 3 + 3] = grid2[rows]
            
            # Iterate over column shuffle patterns within stacks
            col_patterns = range(math.factorial(box_cols) ** n_stacks)
            
            for col_pattern in col_patterns:
                grid3 = grid2.copy()
                
                temp_pattern = col_pattern
                for stack in range(n_stacks):
                    perm_idx = temp_pattern % math.factorial(box_cols)
                    temp_pattern //= math.factorial(box_cols)
                    col_perms = list(permutations(range(box_cols)))
                    col_perm = col_perms[perm_idx]
                    cols = [stack * box_cols + c for c in col_perm]
                    grid3[:, stack * box_cols:stack * box_cols + box_cols] = grid3[:, cols]
                
                # Add solution hash
                all_solutions.add(grid3.tobytes())
    
    return all_solutions


def load_split_metadata(data_dir: Path, split: str) -> dict:
    """Load split-specific metadata."""
    split_dir = data_dir / split
    with open(split_dir / "dataset.json") as f:
        return json.load(f)


def load_solutions_from_dataset(data_dir: Path, split: str = "train") -> tuple:
    """Load solutions (labels) from a pre-generated dataset."""
    split_dir = data_dir / split
    labels = np.load(split_dir / "all__labels.npy")
    
    # Load optional grid_sizes array for mixed-size mode
    grid_sizes_path = split_dir / "all__grid_sizes.npy"
    if grid_sizes_path.exists():
        grid_sizes_array = np.load(grid_sizes_path)
    else:
        grid_sizes_array = None
    
    # Use split-specific metadata for correct grid_size
    split_meta = load_split_metadata(data_dir, split)
    
    return labels, grid_sizes_array, split_meta


def decode_solution(encoded: np.ndarray, grid_size: int, max_grid_size: int) -> np.ndarray:
    """Decode an encoded label back to a solution grid."""
    seq = encoded.reshape(max_grid_size, max_grid_size)
    grid = seq[:grid_size, :grid_size].copy()
    
    # Decode: values are shifted by 2 (3+ -> 1+)
    decoded = np.zeros_like(grid)
    decoded[grid >= 3] = grid[grid >= 3] - 2
    
    return decoded


@click.command()
@click.argument(
    "data_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test"]),
    default="train",
    help="Which split to check (default: train)",
)
@click.option(
    "--num-samples",
    type=int,
    default=None,
    help="Number of samples to check (default: all)",
)
def main(data_dir: Path, split: str, num_samples: int):
    """
    Verify that full mode generates solutions unreachable by standard mode.
    
    DATA_DIR is the path to a full-mode generated dataset.
    """
    
    # Load global dataset metadata
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    mode = metadata.get("mode", "standard")
    max_grid_size = metadata["max_grid_size"]
    
    click.echo(f"Dataset: {data_dir}")
    click.echo(f"Mode: {mode.upper()}")
    click.echo(f"Max grid size: {max_grid_size}x{max_grid_size}")
    click.echo()
    
    # Handle cross-size and mixed-size modes
    if mode == "cross-size":
        click.echo("⚠ Cross-size datasets have different grid sizes for train vs eval.")
        click.echo("  The 'standard mode' OOD check doesn't apply here.")
        click.echo("  Use test_sudoku_data.py to validate the data instead.")
        return
    
    if mode == "mixed-size":
        click.echo("⚠ Mixed-size datasets have multiple grid sizes per split.")
        click.echo("  The 'standard mode' OOD check would need to run per grid size.")
        click.echo("  Use test_sudoku_data.py to validate the data instead.")
        return
    
    # Load split metadata to get grid_size
    split_meta = load_split_metadata(data_dir, split)
    grid_size = split_meta["grid_size"]
    
    click.echo(f"Split: {split}")
    click.echo(f"Grid size: {grid_size}x{grid_size}")
    
    if mode == "hybrid":
        click.echo(f"  (Hybrid mode: train=standard, val/test=full)")
    click.echo()
    
    # Enumerate all standard mode solutions
    click.echo("Enumerating all solutions reachable by standard mode...")
    click.echo("(This may take a moment for 6x6...)")
    
    standard_solutions = enumerate_standard_mode_solutions(grid_size)
    click.echo(f"✓ Standard mode can generate {len(standard_solutions):,} unique solutions")
    click.echo()
    
    # Load solutions from dataset
    click.echo(f"Loading solutions from {split} split...")
    labels, grid_sizes_array, _ = load_solutions_from_dataset(data_dir, split)
    
    total_samples = len(labels)
    if num_samples is not None:
        total_samples = min(num_samples, total_samples)
    
    click.echo(f"Checking {total_samples:,} solutions...")
    
    # Check each solution
    in_standard = 0
    outside_standard = 0
    outside_examples = []
    
    for i in range(total_samples):
        # Get grid_size for this sample (for future mixed-size support)
        sample_grid_size = grid_size
        if grid_sizes_array is not None:
            sample_grid_size = int(grid_sizes_array[i])
        
        solution = decode_solution(labels[i], sample_grid_size, max_grid_size)
        solution_hash = solution.tobytes()
        
        if solution_hash in standard_solutions:
            in_standard += 1
        else:
            outside_standard += 1
            if len(outside_examples) < 3:
                outside_examples.append(solution)
        
        if (i + 1) % 10000 == 0:
            click.echo(f"  Checked {i + 1:,}/{total_samples:,}...")
    
    # Report results
    click.echo()
    click.echo("=" * 60)
    click.echo("RESULTS")
    click.echo("=" * 60)
    click.echo(f"Total solutions checked: {total_samples:,}")
    click.echo(f"Reachable by standard mode: {in_standard:,} ({100*in_standard/total_samples:.1f}%)")
    click.echo(f"Outside standard mode: {outside_standard:,} ({100*outside_standard/total_samples:.1f}%)")
    click.echo()
    
    if outside_standard > 0:
        click.echo("✓ VERIFIED: Dataset contains solutions that standard mode cannot generate!")
        if mode == "hybrid" and split in ("val", "test"):
            click.echo(f"  This confirms the {split} set is truly out-of-distribution (OOD).")
        click.echo()
        click.echo("Example solutions outside standard mode's reach:")
        for idx, sol in enumerate(outside_examples):
            click.echo(f"\n  Example {idx + 1}:")
            for row in sol:
                click.echo(f"    {row}")
    else:
        click.echo("⚠ All solutions are reachable by standard mode.")
        if mode == "hybrid" and split in ("val", "test"):
            click.echo(f"  WARNING: {split} set should contain OOD solutions in hybrid mode!")
        click.echo("  This dataset may not have been generated with --full or --hybrid mode,")
        click.echo("  or you're just unlucky with the random sample.")


if __name__ == "__main__":
    main()