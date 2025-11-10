"""
Pre-generate Sudoku datasets and save to disk for fast training.
This avoids the overhead of generating puzzles on-the-fly during training.
"""

import json
import os
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

# Import your dataset class
import sys
sys.path.append('/mnt/user-data/uploads')  # Adjust if needed
from src.nn.data.sudoku_datamodule import SudokuDataset


def generate_and_save_split(
    split: str,
    output_dir: Path,
    num_puzzles: int,
    grid_size: int,
    max_grid_size: int,
    min_givens: int,
    max_givens: int,
    seed: int,
):
    """Generate a dataset split and save to disk."""
    print(f"\n{'='*60}")
    print(f"Generating {split} split: {num_puzzles} puzzles")
    print(f"Grid size: {grid_size}x{grid_size}, max_grid_size: {max_grid_size}")
    print(f"Givens range: {min_givens}-{max_givens}")
    print('='*60)
    
    # Create the dataset with on-the-fly generation
    dataset = SudokuDataset(
        data_dir=None,
        split=split,
        num_puzzles=num_puzzles,
        min_givens=min_givens,
        max_givens=max_givens,
        grid_size=grid_size,
        max_grid_size=max_grid_size,
        seed=seed,
        generate_on_fly=True,  # Generate on-the-fly for saving
    )
    
    # Pre-allocate arrays
    seq_len = max_grid_size * max_grid_size
    inputs = np.zeros((num_puzzles, seq_len), dtype=np.int32)
    labels = np.zeros((num_puzzles, seq_len), dtype=np.int32)
    puzzle_identifiers = np.zeros(num_puzzles, dtype=np.int32)
    
    # Generate all samples
    print(f"Generating {num_puzzles} samples...")
    for idx in tqdm(range(num_puzzles)):
        sample = dataset[idx]
        inputs[idx] = sample['input'].numpy()
        labels[idx] = sample['output'].numpy()
        puzzle_identifiers[idx] = sample['puzzle_identifiers']
    
    # Create output directory
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    print(f"Saving to {split_dir}...")
    np.save(split_dir / "all__inputs.npy", inputs)
    np.save(split_dir / "all__labels.npy", labels)
    np.save(split_dir / "all__puzzle_identifiers.npy", puzzle_identifiers)
    
    # Save metadata
    metadata = {
        "num_puzzles": num_puzzles,
        "grid_size": grid_size,
        "max_grid_size": max_grid_size,
        "min_givens": min_givens,
        "max_givens": max_givens,
        "seq_len": seq_len,
        "vocab_size": 3 + grid_size,  # 0=PAD, 1=EOS, 2=empty, 3+...=values
        "seed": seed,
    }
    
    with open(split_dir / "dataset.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved {split} split:")
    print(f"  - inputs: {inputs.shape}")
    print(f"  - labels: {labels.shape}")
    print(f"  - puzzle_identifiers: {puzzle_identifiers.shape}")
    
    return metadata


@click.command()
@click.option(
    '--output-dir',
    default='./data/sudoku_pregenerated',
    help='Output directory for generated data',
    type=click.Path(),
)
@click.option(
    '--grid-size',
    default=4,
    help='Sudoku grid size (4, 6, or 9)',
    type=click.IntRange(4, 9),
)
@click.option(
    '--max-grid-size',
    default=None,
    help='Padded grid size (default: grid_size + 2)',
    type=int,
)
@click.option(
    '--num-train',
    default=10000,
    help='Number of training samples',
    type=int,
)
@click.option(
    '--num-val',
    default=1000,
    help='Number of validation samples',
    type=int,
)
@click.option(
    '--num-test',
    default=1000,
    help='Number of test samples',
    type=int,
)
@click.option(
    '--min-givens',
    default=None,
    help='Minimum givens (default: 35% of cells)',
    type=int,
)
@click.option(
    '--max-givens',
    default=None,
    help='Maximum givens (default: 60% of cells)',
    type=int,
)
@click.option(
    '--seed',
    default=42,
    help='Random seed',
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
    """Pre-generate Sudoku datasets for fast training."""
    
    # Set defaults
    if max_grid_size is None:
        max_grid_size = grid_size + 2
    
    total_cells = grid_size * grid_size
    if min_givens is None:
        min_givens = int(total_cells * 0.35)
    if max_givens is None:
        max_givens = int(total_cells * 0.60)
    
    output_dir = Path(output_dir)
    
    click.echo("\n" + "="*60)
    click.echo("SUDOKU DATA GENERATION")
    click.echo("="*60)
    click.echo(f"Grid size: {grid_size}x{grid_size}")
    click.echo(f"Max grid size (padded): {max_grid_size}x{max_grid_size}")
    click.echo(f"Givens range: {min_givens}-{max_givens}")
    click.echo(f"Train samples: {num_train}")
    click.echo(f"Val samples: {num_val}")
    click.echo(f"Test samples: {num_test}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Seed: {seed}")
    
    # Generate all splits
    train_meta = generate_and_save_split(
        split='train',
        output_dir=output_dir,
        num_puzzles=num_train,
        grid_size=grid_size,
        max_grid_size=max_grid_size,
        min_givens=min_givens,
        max_givens=max_givens,
        seed=seed,
    )
    
    val_meta = generate_and_save_split(
        split='test',  # Note: your code uses 'test' for validation
        output_dir=output_dir,
        num_puzzles=num_val,
        grid_size=grid_size,
        max_grid_size=max_grid_size,
        min_givens=min_givens,
        max_givens=max_givens,
        seed=seed,
    )
    
    # Save overall metadata
    overall_meta = {
        'grid_size': grid_size,
        'max_grid_size': max_grid_size,
        'vocab_size': train_meta['vocab_size'],
        'seq_len': train_meta['seq_len'],
        'num_train': num_train,
        'num_val': num_val,
        'num_test': num_test,
        'min_givens': min_givens,
        'max_givens': max_givens,
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(overall_meta, f, indent=2)
    
    click.echo("\n" + "="*60)
    click.echo("✓ DATA GENERATION COMPLETE!")
    click.echo("="*60)
    click.echo(f"All data saved to: {output_dir}")
    click.echo("\nTo use this data in training, set:")
    click.echo(f"  data_dir='{output_dir}'")
    click.echo(f"  generate_on_fly=False")
    click.echo("\nExample:")
    click.echo(f"  python train.py --data-dir {output_dir} --no-generate-on-fly")
    

if __name__ == '__main__':
    main()