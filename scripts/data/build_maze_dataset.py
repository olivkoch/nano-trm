from typing import Optional, Tuple
import os
import csv
import json
import numpy as np

import click
from tqdm import tqdm
from huggingface_hub import hf_hub_download


CHARSET = "# SGo"  # wall, space, start, goal, path(solution)


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


def convert_subset(set_name: str, source_repo: str, output_dir: str,
                   subsample_size: Optional[int], num_aug: int,
                   preloaded_data: Optional[Tuple] = None) -> Tuple[int, int, Optional[Tuple]]:
    """Convert a subset and return (num_groups, num_puzzles, remaining_data)."""
    
    # Read CSV or use preloaded data
    if preloaded_data is not None:
        inputs, labels = preloaded_data
        grid_size = inputs[0].shape[0]
    else:
        inputs = []
        labels = []
        grid_size = None
        
        # Determine source file (val uses test.csv)
        source_file = "test" if set_name == "val" else set_name
        
        with open(hf_hub_download(source_repo, f"{source_file}.csv", repo_type="dataset"), newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for source, q, a, rating in reader:
                if grid_size is None:
                    n = int(len(q) ** 0.5)
                    grid_size = n
                    
                inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid_size, grid_size))
                labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid_size, grid_size))

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

    # Generate dataset with augmentation
    num_augments = num_aug if set_name == "train" else 0

    all_inputs = []
    all_labels = []
    puzzle_identifiers = []
    puzzle_indices = [0]  # Start at 0
    group_indices = [0]   # Start at 0
    puzzle_id = 0
    
    for orig_inp, orig_out in zip(tqdm(inputs, desc=f"Processing {set_name}"), labels):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented, rest use dihedral transforms
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                # Use dihedral transforms 1-7 for augmentation
                transform_idx = (aug_idx - 1) % 7 + 1
                inp = dihedral_transform(orig_inp, transform_idx)
                out = dihedral_transform(orig_out, transform_idx)

            all_inputs.append(inp)
            all_labels.append(out)
            puzzle_identifiers.append(0)
            
            puzzle_id += 1
            puzzle_indices.append(puzzle_id)
        
        # Close the group after all augmentations of this puzzle
        group_indices.append(puzzle_id)

    num_groups = len(group_indices) - 1
    num_puzzles = len(all_inputs)

    # Build char to id mapping
    char2id = np.zeros(256, dtype=np.uint8)
    for i, c in enumerate(CHARSET):
        char2id[ord(c)] = i + 1  # 0 is PAD
    
    # To Numpy - apply char mapping
    def _seq_to_numpy(seq):
        arr = np.vstack([char2id[s.reshape(-1)] for s in seq])
        return arr
    
    inputs_arr = _seq_to_numpy(all_inputs)
    labels_arr = _seq_to_numpy(all_labels)
    puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)

    seq_len = grid_size * grid_size
    vocab_size = len(CHARSET) + 1  # PAD + charset

    # Metadata matching Sudoku format
    metadata = {
        "num_puzzles": num_puzzles,           # Total including augmentations
        "num_groups": num_groups,              # Unique base puzzles
        "num_augmentations": num_aug if set_name == "train" else 0,
        "mean_puzzles_per_group": num_puzzles / num_groups if num_groups > 0 else 1,
        "grid_size": grid_size,
        "max_grid_size": grid_size,
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
        
    print(f"✓ Saved {set_name} split:")
    print(f"  - {num_groups} groups × {1 + (num_aug if set_name == 'train' else 0)} = {num_puzzles} puzzles")
    print(f"  - inputs: {inputs_arr.shape}")
    print(f"  - labels: {labels_arr.shape}")
    
    return num_groups, num_puzzles, remaining_data


@click.command()
@click.option("--source-repo", default="sapientinc/maze-30x30-hard-1k", help="Source HuggingFace repository")
@click.option("--output-dir", default="data/maze-30x30-hard-1k", help="Output directory")
@click.option("--subsample-size", type=int, default=None, help="Subsample size for training set")
@click.option("--num-aug", type=int, default=7, help="Number of augmentations per puzzle (max 7 for dihedral)")
@click.option("--eval-ratio", type=float, default=None, help="Test set size as ratio of training size")
@click.option("--seed", type=int, default=42, help="Random seed")
def preprocess_data(source_repo: str, output_dir: str, subsample_size: Optional[int],
                    num_aug: int, eval_ratio: Optional[float], seed: int):
    np.random.seed(seed)
    
    # Dihedral group has 8 elements (indices 0-7), so max meaningful num_aug is 7
    if num_aug > 7:
        print(f"Warning: num_aug={num_aug} clamped to 7 (dihedral group has 8 unique transforms)")
        num_aug = 7
    
    num_train_groups, num_train, _ = convert_subset("train", source_repo, output_dir, 
                                                     subsample_size, num_aug)
    
    # Val and test sets are taken from test.csv (no leakage with training)
    eval_subsample_size = None
    if eval_ratio is not None:
        with open(hf_hub_download(source_repo, "test.csv", repo_type="dataset"), newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            original_test_size = sum(1 for _ in reader)
        eval_subsample_size = int(original_test_size * eval_ratio)
        print(f"Original test.csv has {original_test_size} samples, using {eval_subsample_size} for each eval split")
    
    # Generate val set, keeping remaining data for test
    num_val_groups, num_val, remaining_data = convert_subset("val", source_repo, output_dir, 
                                                              eval_subsample_size, num_aug=0)
    
    # Generate test set from remaining pool
    num_test_groups, num_test, _ = convert_subset("test", source_repo, output_dir, 
                                                   eval_subsample_size, num_aug=0,
                                                   preloaded_data=remaining_data)
    
    # Infer grid size from first training file
    train_meta_path = os.path.join(output_dir, "train", "dataset.json")
    with open(train_meta_path) as f:
        train_meta = json.load(f)
    grid_size = train_meta["grid_size"]
    
    # Save global metadata
    overall_meta = {
        "mode": "maze",
        "max_grid_size": grid_size,
        "vocab_size": len(CHARSET) + 1,
        "seq_len": grid_size * grid_size,
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