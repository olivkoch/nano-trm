from typing import Optional, Tuple
import os
import csv
import json
import numpy as np

import click
from tqdm import tqdm
from huggingface_hub import hf_hub_download


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    
    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def convert_subset(set_name: str, source_repo: str, output_dir: str, 
                   subsample_size: Optional[int], min_difficulty: Optional[int], num_aug: int,
                   preloaded_data: Optional[Tuple] = None) -> Tuple[int, int, Optional[Tuple]]:
    """Convert a subset and return the number of puzzles generated and optionally the remaining data."""
    # Read CSV or use preloaded data
    if preloaded_data is not None:
        inputs, labels = preloaded_data
    else:
        inputs = []
        labels = []
        
        # Determine source file (val uses test.csv)
        source_file = "test" if set_name == "val" else set_name
        
        with open(hf_hub_download(source_repo, f"{source_file}.csv", repo_type="dataset"), newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for source, q, a, rating in reader:
                if (min_difficulty is None) or (int(rating) >= min_difficulty):
                    assert len(q) == 81 and len(a) == 81
                    
                    inputs.append(np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
                    labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))

    # If subsample_size is specified, randomly sample the desired number of examples.
    remaining_data = None
    if subsample_size is not None:
        print(f"Subsampling {subsample_size} examples from {set_name} set of size {len(inputs)}")
        total_samples = len(inputs)
        if subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=subsample_size, replace=False)
            mask = np.ones(total_samples, dtype=bool)
            mask[indices] = False
            remaining_indices = np.where(mask)[0]
            
            # Keep remaining data for potential next split
            remaining_data = ([inputs[i] for i in remaining_indices], [labels[i] for i in remaining_indices])
            
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset
    num_augments = num_aug if set_name == "train" else 0

    all_inputs = []
    all_labels = []
    puzzle_identifiers = []
    puzzle_indices = [0]  # Start at 0
    group_indices = [0]   # Start at 0
    puzzle_id = 0

    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            all_inputs.append(inp)
            all_labels.append(out)

            puzzle_identifiers.append(0)
            
            puzzle_id += 1
            puzzle_indices.append(puzzle_id)
        group_indices.append(puzzle_id)

    num_groups = len(group_indices) - 1
    num_puzzles = len(all_inputs)

    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        assert np.all((arr >= 0) & (arr <= 9))
        # Encoding: 0 (blank) -> 2, digits 1-9 -> 3-11
        # (0=PAD, 1=separator, 2=blank, 3-11=digits 1-9)
        return arr + 2
    
    inputs_arr = _seq_to_numpy(all_inputs)
    labels_arr = _seq_to_numpy(all_labels)
    puzzle_identifiers = np.zeros(len(all_inputs), dtype=np.int32)

    num_puzzles = len(all_inputs)
    grid_size = 9
    seq_len = 81
    vocab_size = 3 + grid_size  # PAD + EOS + BLANK + 1-9

    # Calculate min/max givens from the data (count non-zero cells in inputs)
    givens_counts = [np.count_nonzero(inp) for inp in all_inputs]
    min_givens = int(min(givens_counts))
    max_givens = int(max(givens_counts))

    # Metadata matching the other generator's format
    metadata = {
        "num_puzzles": num_puzzles,           # Total including augmentations
        "num_groups": num_groups,              # Unique base puzzles
        "num_augmentations": num_aug,          # Augmentations per puzzle
        "mean_puzzles_per_group": num_puzzles / num_groups if num_groups > 0 else 1,
        "grid_size": grid_size,
        "max_grid_size": grid_size,
        "min_givens": min_givens,
        "max_givens": max_givens,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
    }

    # Save
    save_dir = os.path.join(output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    np.save(os.path.join(save_dir, "all__inputs.npy"), inputs_arr)
    np.save(os.path.join(save_dir, "all__labels.npy"), labels_arr)
    np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)
    np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), 
            np.array(puzzle_indices, dtype=np.int32))
    np.save(os.path.join(save_dir, "all__group_indices.npy"), 
            np.array(group_indices, dtype=np.int32))
        
    return num_groups, num_puzzles, remaining_data

@click.command()
@click.option("--source-repo", default="sapientinc/sudoku-extreme", help="Source HuggingFace repository")
@click.option("--output-dir", default="data/sudoku-extreme-full", help="Output directory")
@click.option("--subsample-size", type=int, default=None, help="Subsample size for training set")
@click.option("--min-difficulty", type=int, default=None, help="Minimum difficulty rating")
@click.option("--num-aug", type=int, default=0, help="Number of augmentations per puzzle")
@click.option("--eval-ratio", type=float, default=None, help="Test set size as ratio of training size")
@click.option("--seed", type=int, default=42, help="Random seed")
def preprocess_data(source_repo: str, output_dir: str, subsample_size: Optional[int], 
                    min_difficulty: Optional[int], num_aug: int, eval_ratio: Optional[float], seed: int):
    np.random.seed(seed)
    
    num_train_groups, num_train, _ = convert_subset("train", source_repo, output_dir, 
                                                      subsample_size, min_difficulty, num_aug)
    
    # Val and test (no augmentation, so groups = puzzles)
    eval_subsample_size = None
    if eval_ratio is not None:
        with open(hf_hub_download(source_repo, "test.csv", repo_type="dataset"), newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            original_test_size = sum(1 for _ in reader)
        eval_subsample_size = int(original_test_size * eval_ratio)
        print(f"Original test.csv has {original_test_size} samples, using {eval_subsample_size} for each eval split")
    
    num_val_groups, num_val, remaining_data = convert_subset("val", source_repo, output_dir, 
                                                               eval_subsample_size, min_difficulty, num_aug=0)
    
    num_test_groups, num_test, _ = convert_subset("test", source_repo, output_dir, 
                                                    eval_subsample_size, min_difficulty, num_aug=0,
                                                    preloaded_data=remaining_data)
    
    # Save global metadata
    overall_meta = {
        "mode": "extreme",
        "max_grid_size": 9,
        "vocab_size": 12,
        "seq_len": 81,
        # Puzzle counts (including augmentations)
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        # Group counts (unique base puzzles) - USE THIS FOR STEPS/EPOCH
        "num_train_groups": num_train_groups,
        "num_val_groups": num_val_groups,
        "num_test_groups": num_test_groups,
        # Augmentation info
        "num_augmentations": num_aug,
        "seed": seed,
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(overall_meta, f, indent=2)
    
    print(f"\n✓ Dataset saved to {output_dir}")
    print(f"  Train: {num_train_groups} groups × {1 + num_aug} = {num_train} puzzles")
    print(f"  Val:   {num_val_groups} groups = {num_val} puzzles")
    print(f"  Test:  {num_test_groups} groups = {num_test} puzzles")


if __name__ == "__main__":
    preprocess_data()