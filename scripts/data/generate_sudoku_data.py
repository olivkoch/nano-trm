"""
Pre-generate Sudoku datasets and save to disk for fast training.
This avoids the overhead of generating puzzles on-the-fly during training.

Updated to use the new clean datamodule design with guaranteed no-leakage.

Options:
  --full: Use complete enumeration of all valid grids (6x6 only for now)
          This ensures uniform sampling over ALL possible Sudoku grids.
"""

import json
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from src.nn.data.sudoku_datamodule import SudokuDataModule, SudokuDataset, PuzzleGenerator


def save_split_to_disk(
    output_dir: Path,
    split: str,
    puzzle_pool: list,
    split_indices: list,
    grid_size: int,
    max_grid_size: int,
    min_givens: int,
    max_givens: int,
):
    """Save a split (train/val/test) to disk."""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    num_puzzles = len(split_indices)
    seq_len = max_grid_size * max_grid_size

    # Pre-allocate arrays
    inputs = np.zeros((num_puzzles, seq_len), dtype=np.int32)
    labels = np.zeros((num_puzzles, seq_len), dtype=np.int32)
    puzzle_identifiers = np.zeros(num_puzzles, dtype=np.int32)

    # Create a temporary dataset to access pad_and_encode
    temp_ds = SudokuDataset(
        split=split,
        grid_size=grid_size,
        max_grid_size=max_grid_size,
        min_givens=min_givens,
        max_givens=max_givens,
        puzzle_pool=puzzle_pool,
        split_indices=split_indices,
    )

    # Generate encoded data
    print(f"Encoding {split} split ({num_puzzles} puzzles)...")
    for i in tqdm(range(num_puzzles), desc=f"Encoding {split}"):
        pool_idx = split_indices[i]
        puzzle, solution = puzzle_pool[pool_idx]

        inp, lbl = temp_ds.pad_and_encode(puzzle, solution)
        inputs[i] = inp
        labels[i] = lbl
        puzzle_identifiers[i] = 0  # All Sudoku puzzles have ID 0

    # Save arrays
    print(f"Saving to {split_dir}...")
    np.save(split_dir / "all__inputs.npy", inputs)
    np.save(split_dir / "all__labels.npy", labels)
    np.save(split_dir / "all__puzzle_identifiers.npy", puzzle_identifiers)

    # Save split metadata
    metadata = {
        "num_puzzles": num_puzzles,
        "grid_size": grid_size,
        "max_grid_size": max_grid_size,
        "min_givens": min_givens,
        "max_givens": max_givens,
        "seq_len": seq_len,
        "vocab_size": 3 + grid_size,  # 0=PAD, 1=EOS, 2=empty, 3+...=values
    }

    with open(split_dir / "dataset.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved {split} split:")
    print(f"  - inputs: {inputs.shape}")
    print(f"  - labels: {labels.shape}")
    print(f"  - puzzle_identifiers: {puzzle_identifiers.shape}")


def enumerate_all_grids(grid_size: int) -> list[np.ndarray]:
    """
    Enumerate ALL valid Sudoku grids via backtracking.
    
    Supported sizes:
    - 4x4: 288 valid grids (instant)
    - 6x6: 28,200,960 valid grids (few minutes)
    - 9x9: ~6.67 × 10^21 grids (NOT FEASIBLE)
    """
    if grid_size == 9:
        raise ValueError(
            "Full enumeration of 9x9 Sudoku is not feasible (~6.67 × 10^21 grids). "
            "Use the default transformation-based generation instead."
        )
    
    # Determine box dimensions
    if grid_size == 4:
        box_rows, box_cols = 2, 2
    elif grid_size == 6:
        box_rows, box_cols = 2, 3
    else:
        raise ValueError(f"Full enumeration not supported for grid_size={grid_size}")
    
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


def get_canonical_form(grid: np.ndarray, box_rows: int, box_cols: int) -> np.ndarray:
    """
    Get the canonical (lexicographically smallest) form of a grid
    under all symmetry transformations.
    
    Symmetries for 6x6 (2x3 boxes):
    - Digit relabeling: 6!
    - Band permutation: 3!
    - Row swaps within bands: (2!)^3
    - Stack permutation: 2!
    - Column swaps within stacks: (3!)^2
    """
    grid_size = grid.shape[0]
    n_bands = grid_size // box_rows
    n_stacks = grid_size // box_cols
    
    min_form = grid.flatten().tobytes()
    
    # Generate all symmetry transformations
    from itertools import permutations
    
    def apply_transforms(g):
        """Generator yielding all symmetric forms of grid g."""
        # Digit relabeling
        for digit_perm in permutations(range(1, grid_size + 1)):
            g1 = np.zeros_like(g)
            for old_val, new_val in enumerate(digit_perm, 1):
                g1[g == old_val] = new_val
            
            # Band permutation
            for band_perm in permutations(range(n_bands)):
                g2 = np.zeros_like(g1)
                for new_band, old_band in enumerate(band_perm):
                    g2[new_band*box_rows:(new_band+1)*box_rows] = \
                        g1[old_band*box_rows:(old_band+1)*box_rows]
                
                # Row swaps within bands
                for row_swaps in range(2**n_bands):
                    g3 = g2.copy()
                    for band in range(n_bands):
                        if (row_swaps >> band) & 1:
                            r1, r2 = band * box_rows, band * box_rows + 1
                            g3[[r1, r2]] = g3[[r2, r1]]
                    
                    # Stack permutation
                    for stack_perm in permutations(range(n_stacks)):
                        g4 = np.zeros_like(g3)
                        for new_stack, old_stack in enumerate(stack_perm):
                            g4[:, new_stack*box_cols:(new_stack+1)*box_cols] = \
                                g3[:, old_stack*box_cols:(old_stack+1)*box_cols]
                        
                        # Column swaps within stacks
                        for col_pattern in range(6**n_stacks):  # 3! = 6 per stack
                            g5 = g4.copy()
                            temp_pattern = col_pattern
                            for stack in range(n_stacks):
                                col_perm_idx = temp_pattern % 6
                                temp_pattern //= 6
                                col_perms = list(permutations(range(box_cols)))
                                col_perm = col_perms[col_perm_idx]
                                base = stack * box_cols
                                old_cols = [base + c for c in col_perm]
                                g5[:, base:base+box_cols] = g4[:, old_cols]
                            
                            yield g5
    
    for transformed in apply_transforms(grid):
        form = transformed.flatten().tobytes()
        if form < min_form:
            min_form = form
    
    return np.frombuffer(min_form, dtype=np.int32).reshape(grid_size, grid_size).copy()


def find_canonical_representatives(
    all_grids: list[np.ndarray], 
    grid_size: int
) -> list[np.ndarray]:
    """
    Find one representative grid from each equivalence class.
    These are the minimal set of base grids needed to generate all others via symmetry.
    """
    if grid_size == 4:
        box_rows, box_cols = 2, 2
    elif grid_size == 6:
        box_rows, box_cols = 2, 3
    else:
        raise ValueError(f"Unsupported grid_size: {grid_size}")
    
    print(f"Finding canonical representatives from {len(all_grids):,} grids...")
    print("(This may take a while for 6x6...)")
    
    canonical_map = {}
    
    for i, grid in enumerate(tqdm(all_grids, desc="Computing canonical forms")):
        canon = get_canonical_form(grid, box_rows, box_cols)
        key = canon.tobytes()
        if key not in canonical_map:
            canonical_map[key] = grid.copy()
    
    representatives = list(canonical_map.values())
    print(f"✓ Found {len(representatives)} equivalence classes")
    
    return representatives


def get_or_create_base_grids(grid_size: int, cache_dir: Path) -> np.ndarray:
    """
    Get base grids from cache or compute them.
    Returns array of shape (num_classes, grid_size, grid_size).
    """
    cache_file = cache_dir / f"base_grids_{grid_size}x{grid_size}.npy"
    
    if cache_file.exists():
        print(f"Loading cached base grids from {cache_file}")
        return np.load(cache_file)
    
    # Enumerate all grids
    all_grids = enumerate_all_grids(grid_size)
    
    # Find canonical representatives
    representatives = find_canonical_representatives(all_grids, grid_size)
    
    # Save to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_grids = np.stack(representatives)
    np.save(cache_file, base_grids)
    print(f"✓ Saved {len(representatives)} base grids to {cache_file}")
    
    return base_grids


@click.command()
@click.option(
    "--output-dir",
    default="./data",
    help="Output directory for generated data",
    type=click.Path(),
)
@click.option(
    "--grid-size",
    default=4,
    help="Sudoku grid size (4, 6, or 9)",
    type=click.IntRange(4, 9),
)
@click.option(
    "--max-grid-size",
    default=None,
    help="Padded grid size (default: grid_size + 2)",
    type=int,
)
@click.option(
    "--num-train",
    default=10000,
    help="Number of training samples",
    type=int,
)
@click.option(
    "--num-val",
    default=1000,
    help="Number of validation samples",
    type=int,
)
@click.option(
    "--num-test",
    default=1000,
    help="Number of test samples",
    type=int,
)
@click.option(
    "--min-givens",
    default=None,
    help="Minimum givens (default: 35%% of cells)",
    type=int,
)
@click.option(
    "--max-givens",
    default=None,
    help="Maximum givens (default: 60%% of cells)",
    type=int,
)
@click.option(
    "--seed",
    default=42,
    help="Random seed",
    type=int,
)
@click.option(
    "--full",
    is_flag=True,
    default=False,
    help="Use full enumeration for uniform sampling over ALL valid grids (4x4, 6x6 only)",
)
@click.option(
    "--cache-dir",
    default="./data/sudoku_cache",
    help="Directory to cache base grids for --full mode",
    type=click.Path(),
)
def main(
    output_dir,
    grid_size,
    max_grid_size,
    num_train,
    num_val,
    num_test,
    min_givens,
    max_givens,
    seed,
    full,
    cache_dir,
):
    """
    Pre-generate Sudoku datasets for fast training.

    This script uses the new clean datamodule design which guarantees
    no leakage between train/val/test splits by:
    1. Generating a single pool of unique puzzles
    2. Deterministically splitting the pool
    3. Saving each split to disk
    
    With --full flag:
    - Enumerates ALL valid Sudoku grids (4x4: 288, 6x6: 28M)
    - Finds canonical base grids (one per equivalence class)
    - Uses full symmetry operations for uniform sampling
    - Caches base grids for future use
    """

    # Set defaults
    if max_grid_size is None:
        max_grid_size = grid_size + 2

    total_cells = grid_size * grid_size
    if min_givens is None:
        min_givens = int(total_cells * 0.35)
    if max_givens is None:
        max_givens = int(total_cells * 0.60)

    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)

    click.echo("\n" + "=" * 60)
    click.echo("SUDOKU DATA GENERATION")
    click.echo("=" * 60)
    click.echo(f"Grid size: {grid_size}x{grid_size}")
    click.echo(f"Max grid size (padded): {max_grid_size}x{max_grid_size}")
    click.echo(f"Givens range: {min_givens}-{max_givens}")
    click.echo(f"Train samples: {num_train}")
    click.echo(f"Val samples: {num_val}")
    click.echo(f"Test samples: {num_test}")
    click.echo(f"Total samples: {num_train + num_val + num_test}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Seed: {seed}")
    click.echo(f"Full enumeration mode: {full}")
    click.echo()

    # Handle --full mode
    base_grids = None
    if full:
        if grid_size == 9:
            click.echo("⚠ ERROR: --full mode not supported for 9x9 Sudoku")
            click.echo("  (There are ~6.67 × 10^21 valid 9x9 grids)")
            click.echo("  Use the default transformation-based generation instead.")
            return
        
        click.echo("=" * 60)
        click.echo("FULL ENUMERATION MODE")
        click.echo("=" * 60)
        base_grids = get_or_create_base_grids(grid_size, cache_dir)
        click.echo(f"✓ Using {len(base_grids)} base grids for full coverage")
        click.echo()

    # Create datamodule in generation mode (data_dir=None)
    click.echo("Creating SudokuDataModule in generation mode...")
    dm = SudokuDataModule(
        data_dir=None,  # Generation mode - creates unique puzzle pool
        num_train_puzzles=num_train,
        num_val_puzzles=num_val,
        num_test_puzzles=num_test,
        grid_size=grid_size,
        max_grid_size=max_grid_size,
        min_givens=min_givens,
        max_givens=max_givens,
        seed=seed,
        base_grids=base_grids,  # Pass base grids for full mode
    )

    # Generate puzzle pool and splits
    click.echo("\nGenerating puzzle pool and splits...")
    click.echo("(This will take a few minutes for large datasets)")
    dm.setup("fit")  # Creates train and val datasets
    dm.setup("test")  # Creates test dataset

    # Get the generated data
    puzzle_pool = dm.get_puzzle_pool()
    split_indices = dm.get_split_indices()

    click.echo(f"\n✓ Generated {len(puzzle_pool)} unique puzzles")
    click.echo(f"  Train indices: {len(split_indices['train'])}")
    click.echo(f"  Val indices:   {len(split_indices['val'])}")
    click.echo(f"  Test indices:  {len(split_indices['test'])}")

    # Verify no overlap (sanity check)
    train_set = set(split_indices["train"])
    val_set = set(split_indices["val"])
    test_set = set(split_indices["test"])

    train_val_overlap = len(train_set & val_set)
    train_test_overlap = len(train_set & test_set)
    val_test_overlap = len(val_set & test_set)

    click.echo("\n✓ Verification:")
    click.echo(
        f"  Train ∩ Val:  {train_val_overlap} {'✓ PASS' if train_val_overlap == 0 else '✗ FAIL'}"
    )
    click.echo(
        f"  Train ∩ Test: {train_test_overlap} {'✓ PASS' if train_test_overlap == 0 else '✗ FAIL'}"
    )
    click.echo(
        f"  Val ∩ Test:   {val_test_overlap} {'✓ PASS' if val_test_overlap == 0 else '✗ FAIL'}"
    )

    if train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0:
        click.echo("\n✗ ERROR: Overlap detected between splits!")
        raise RuntimeError("Data leakage detected!")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each split to disk
    click.echo("\n" + "=" * 60)
    click.echo("SAVING TO DISK")
    click.echo("=" * 60)

    save_split_to_disk(
        output_dir=output_dir,
        split="train",
        puzzle_pool=puzzle_pool,
        split_indices=split_indices["train"],
        grid_size=grid_size,
        max_grid_size=max_grid_size,
        min_givens=min_givens,
        max_givens=max_givens,
    )

    save_split_to_disk(
        output_dir=output_dir,
        split="val",
        puzzle_pool=puzzle_pool,
        split_indices=split_indices["val"],
        grid_size=grid_size,
        max_grid_size=max_grid_size,
        min_givens=min_givens,
        max_givens=max_givens,
    )

    save_split_to_disk(
        output_dir=output_dir,
        split="test",
        puzzle_pool=puzzle_pool,
        split_indices=split_indices["test"],
        grid_size=grid_size,
        max_grid_size=max_grid_size,
        min_givens=min_givens,
        max_givens=max_givens,
    )

    # Save global metadata
    overall_meta = {
        "grid_size": grid_size,
        "max_grid_size": max_grid_size,
        "vocab_size": 3 + grid_size,
        "seq_len": max_grid_size * max_grid_size,
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        "min_givens": min_givens,
        "max_givens": max_givens,
        "seed": seed,
        "full_enumeration": full,
        "num_base_grids": len(base_grids) if base_grids is not None else None,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(overall_meta, f, indent=2)

    click.echo("\n" + "=" * 60)
    click.echo("✓ DATA GENERATION COMPLETE!")
    click.echo("=" * 60)
    click.echo(f"All data saved to: {output_dir}")
    click.echo("\nDataset statistics:")
    click.echo(f"  Total unique puzzles: {len(puzzle_pool)}")
    click.echo(f"  Train: {num_train} puzzles")
    click.echo(f"  Val:   {num_val} puzzles")
    click.echo(f"  Test:  {num_test} puzzles")
    click.echo(f"  Vocab size: {3 + grid_size}")
    click.echo(f"  Sequence length: {max_grid_size * max_grid_size}")
    if full:
        click.echo(f"  Generation mode: FULL (uniform over all {grid_size}x{grid_size} grids)")
        click.echo(f"  Base grids used: {len(base_grids)}")
    else:
        click.echo(f"  Generation mode: STANDARD (transformation-based)")
    click.echo("\n✓ Guaranteed no leakage between splits!")
    click.echo("\nTo use this data in training:")
    click.echo(f"  dm = SudokuDataModule(data_dir='{output_dir}')")


if __name__ == "__main__":
    main()