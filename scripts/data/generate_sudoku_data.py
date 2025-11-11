"""
Pre-generate Sudoku datasets and save to disk for fast training.
This avoids the overhead of generating puzzles on-the-fly during training.

Updated to use the new clean datamodule design with guaranteed no-leakage.
"""

import json
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from src.nn.data.sudoku_datamodule import SudokuDataModule, SudokuDataset


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


@click.command()
@click.option(
    "--output-dir",
    default="./data/sudoku_pregenerated",
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
):
    """
    Pre-generate Sudoku datasets for fast training.

    This script uses the new clean datamodule design which guarantees
    no leakage between train/val/test splits by:
    1. Generating a single pool of unique puzzles
    2. Deterministically splitting the pool
    3. Saving each split to disk
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
    click.echo("\n✓ Guaranteed no leakage between splits!")
    click.echo("\nTo use this data in training:")
    click.echo(f"  dm = SudokuDataModule(data_dir='{output_dir}')")


if __name__ == "__main__":
    main()
