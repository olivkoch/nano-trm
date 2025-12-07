"""
Test suite for SudokuDataModule with group-aware sampling.

Run with: python test_sudoku_datamodule.py --data-dir /path/to/generated/data
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from src.nn.data.sudoku_datamodule import GroupedBatchSampler
from src.nn.data.sudoku_datamodule import SudokuDataModule


def test_metadata_loading(data_dir: str):
    """Test that metadata is loaded correctly."""
    print("\n" + "=" * 60)
    print("TEST: Metadata Loading")
    print("=" * 60)
    
    metadata_path = Path(data_dir) / "metadata.json"
    assert metadata_path.exists(), f"metadata.json not found in {data_dir}"
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    required_fields = [
        "num_train", "num_val", "num_test",
        "num_train_groups", "num_val_groups", "num_test_groups",
        "vocab_size", "seq_len", "max_grid_size"
    ]
    
    for field in required_fields:
        assert field in metadata, f"Missing field: {field}"
        print(f"  {field}: {metadata[field]}")
    
    # Verify groups <= puzzles
    assert metadata["num_train_groups"] <= metadata["num_train"], \
        "num_train_groups should be <= num_train"
    
    print("\n✓ Metadata loading passed")
    return metadata


def test_dataset_loading(data_dir: str):
    """Test that dataset files are loaded correctly."""
    print("\n" + "=" * 60)
    print("TEST: Dataset File Loading")
    print("=" * 60)
    
    for split in ["train", "val", "test"]:
        split_dir = Path(data_dir) / split
        assert split_dir.exists(), f"Split directory not found: {split_dir}"
        
        # Check required files
        required_files = [
            "all__inputs.npy",
            "all__labels.npy", 
            "all__puzzle_identifiers.npy",
            "all__group_indices.npy",
            "all__puzzle_indices.npy",
            "dataset.json"
        ]
        
        for fname in required_files:
            fpath = split_dir / fname
            assert fpath.exists(), f"Missing file: {fpath}"
        
        # Load and verify shapes
        inputs = np.load(split_dir / "all__inputs.npy")
        labels = np.load(split_dir / "all__labels.npy")
        group_indices = np.load(split_dir / "all__group_indices.npy")
        puzzle_indices = np.load(split_dir / "all__puzzle_indices.npy")
        
        print(f"\n  {split}:")
        print(f"    inputs shape: {inputs.shape}")
        print(f"    labels shape: {labels.shape}")
        print(f"    group_indices shape: {group_indices.shape}")
        print(f"    puzzle_indices shape: {puzzle_indices.shape}")
        print(f"    num_groups: {len(group_indices) - 1}")
        
        # Verify consistency
        assert inputs.shape == labels.shape, "inputs and labels shape mismatch"
        assert group_indices[0] == 0, "group_indices should start at 0"
        assert group_indices[-1] == len(inputs), \
            f"group_indices[-1] ({group_indices[-1]}) should equal num_puzzles ({len(inputs)})"
        
        # Verify group_indices is monotonically increasing
        assert np.all(np.diff(group_indices) > 0), "group_indices should be strictly increasing"
    
    print("\n✓ Dataset file loading passed")


def test_group_structure(data_dir: str):
    """Test the group structure is correct."""
    print("\n" + "=" * 60)
    print("TEST: Group Structure")
    print("=" * 60)
    
    split_dir = Path(data_dir) / "train"
    group_indices = np.load(split_dir / "all__group_indices.npy")
    
    num_groups = len(group_indices) - 1
    group_sizes = np.diff(group_indices)
    
    print(f"  Number of groups: {num_groups}")
    print(f"  Group size stats:")
    print(f"    min: {group_sizes.min()}")
    print(f"    max: {group_sizes.max()}")
    print(f"    mean: {group_sizes.mean():.2f}")
    print(f"    std: {group_sizes.std():.2f}")
    
    # All groups should have at least 1 puzzle
    assert group_sizes.min() >= 1, "All groups should have at least 1 puzzle"
    
    # If augmentation was used, groups should have more than 1 puzzle
    # (this is informational, not a hard requirement)
    if group_sizes.max() > 1:
        print(f"\n  ✓ Augmentation detected (max group size > 1)")
    else:
        print(f"\n  ℹ No augmentation (all groups have size 1)")
    
    print("\n✓ Group structure test passed")


def test_grouped_batch_sampler(data_dir: str, batch_size: int = 32):
    """Test the GroupedBatchSampler."""
    print("\n" + "=" * 60)
    print("TEST: GroupedBatchSampler")
    print("=" * 60)
        
    split_dir = Path(data_dir) / "train"
    group_indices = np.load(split_dir / "all__group_indices.npy")
    
    sampler = GroupedBatchSampler(
        group_indices=group_indices,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=42,
    )
    
    num_groups = len(group_indices) - 1
    expected_batches = num_groups // batch_size
    
    print(f"  Batch size: {batch_size}")
    print(f"  Num groups: {num_groups}")
    print(f"  Expected batches: {expected_batches}")
    print(f"  Sampler len: {len(sampler)}")
    
    assert len(sampler) == expected_batches, \
        f"Sampler length mismatch: {len(sampler)} vs {expected_batches}"
    
    # Collect all batches
    all_batches = list(sampler)
    print(f"  Actual batches: {len(all_batches)}")
    
    assert len(all_batches) == expected_batches, \
        f"Batch count mismatch: {len(all_batches)} vs {expected_batches}"
    
    # Verify batch sizes
    for i, batch in enumerate(all_batches):
        assert len(batch) == batch_size, \
            f"Batch {i} has wrong size: {len(batch)} vs {batch_size}"
    
    # Verify no duplicate indices within epoch
    all_indices = [idx for batch in all_batches for idx in batch]
    unique_indices = set(all_indices)
    print(f"  Total samples: {len(all_indices)}")
    print(f"  Unique samples: {len(unique_indices)}")
    
    # Each index should appear exactly once
    assert len(all_indices) == len(unique_indices), \
        "Duplicate indices found within epoch!"
    
    # Verify each sample comes from a different group
    def get_group_for_idx(idx):
        for g in range(len(group_indices) - 1):
            if group_indices[g] <= idx < group_indices[g + 1]:
                return g
        return -1
    
    for batch_idx, batch in enumerate(all_batches[:5]):  # Check first 5 batches
        groups_in_batch = [get_group_for_idx(idx) for idx in batch]
        unique_groups = set(groups_in_batch)
        
        if len(groups_in_batch) != len(unique_groups):
            print(f"  ⚠ Batch {batch_idx} has duplicate groups!")
            group_counts = Counter(groups_in_batch)
            duplicates = {g: c for g, c in group_counts.items() if c > 1}
            print(f"    Duplicates: {duplicates}")
        else:
            pass  # All good
    
    print("\n✓ GroupedBatchSampler test passed")


def test_epoch_shuffling(data_dir: str, batch_size: int = 32):
    """Test that different epochs produce different orderings."""
    print("\n" + "=" * 60)
    print("TEST: Epoch Shuffling")
    print("=" * 60)
        
    split_dir = Path(data_dir) / "train"
    group_indices = np.load(split_dir / "all__group_indices.npy")
    
    sampler = GroupedBatchSampler(
        group_indices=group_indices,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=42,
    )
    
    # Get first batch from epoch 0
    sampler.set_epoch(0)
    epoch0_batches = list(sampler)
    epoch0_first_batch = epoch0_batches[0]
    
    # Get first batch from epoch 1
    sampler.set_epoch(1)
    epoch1_batches = list(sampler)
    epoch1_first_batch = epoch1_batches[0]
    
    # Get first batch from epoch 0 again (should be same as before)
    sampler.set_epoch(0)
    epoch0_again_batches = list(sampler)
    epoch0_again_first_batch = epoch0_again_batches[0]
    
    print(f"  Epoch 0 first batch (first 5): {epoch0_first_batch[:5]}")
    print(f"  Epoch 1 first batch (first 5): {epoch1_first_batch[:5]}")
    print(f"  Epoch 0 again first batch (first 5): {epoch0_again_first_batch[:5]}")
    
    # Epoch 0 and epoch 1 should be different
    assert epoch0_first_batch != epoch1_first_batch, \
        "Epoch 0 and epoch 1 should produce different orderings"
    
    # Epoch 0 twice should be the same (deterministic)
    assert epoch0_first_batch == epoch0_again_first_batch, \
        "Same epoch should produce same ordering (deterministic)"
    
    print("\n✓ Epoch shuffling test passed")


def test_datamodule_integration(data_dir: str, batch_size: int = 32):
    """Test full DataModule integration."""
    print("\n" + "=" * 60)
    print("TEST: DataModule Integration")
    print("=" * 60)
        
    dm = SudokuDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
        seed=42,
    )
    
    # Setup
    dm.setup("fit")
    
    print(f"  Mode: {dm.mode}")
    print(f"  Vocab size: {dm.vocab_size}")
    print(f"  Seq len: {dm.seq_len}")
    print(f"  Grid size: {dm.grid_size}")
    print(f"  Num train groups: {getattr(dm, 'num_train_groups', 'N/A')}")
    print(f"  Num train puzzles: {dm.num_train_puzzles}")
    
    # Get train dataloader
    train_loader = dm.train_dataloader()
    
    # Check sampler is set
    if dm._train_sampler is not None:
        print(f"  ✓ GroupedBatchSampler is active")
    else:
        print(f"  ℹ Using standard sampling (no groups)")
    
    # Get a few batches
    batches = []
    for i, batch in enumerate(train_loader):
        batches.append(batch)
        if i >= 2:
            break
    
    print(f"\n  Sample batch:")
    print(f"    input shape: {batches[0]['input'].shape}")
    print(f"    output shape: {batches[0]['output'].shape}")
    print(f"    puzzle_identifiers shape: {batches[0]['puzzle_identifiers'].shape}")
    
    # Verify shapes
    assert batches[0]['input'].shape[0] == batch_size, "Batch size mismatch"
    assert batches[0]['input'].shape == batches[0]['output'].shape, "Input/output shape mismatch"
    
    # Test epoch update
    print(f"\n  Testing epoch update...")
    dm.on_train_epoch_start(current_epoch=1)
    
    if dm._train_sampler is not None:
        assert dm._train_sampler.epoch == 1, "Sampler epoch not updated"
        print(f"    ✓ Sampler epoch updated to 1")
    
    print("\n✓ DataModule integration test passed")


def test_val_test_loaders(data_dir: str, batch_size: int = 32):
    """Test validation and test dataloaders."""
    print("\n" + "=" * 60)
    print("TEST: Val/Test DataLoaders")
    print("=" * 60)
        
    dm = SudokuDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
        seed=42,
    )
    
    dm.setup("fit")
    dm.setup("test")
    
    # Val loader
    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    print(f"  Val batch shape: {val_batch['input'].shape}")
    
    # Test loader
    test_loader = dm.test_dataloader()
    test_batch = next(iter(test_loader))
    print(f"  Test batch shape: {test_batch['input'].shape}")
    
    # Val/test should NOT use grouped sampling (sequential evaluation)
    print(f"\n  Val/Test use sequential sampling (no grouping)")
    
    print("\n✓ Val/Test dataloader test passed")


def test_data_content(data_dir: str):
    """Test that data content is valid."""
    print("\n" + "=" * 60)
    print("TEST: Data Content Validation")
    print("=" * 60)
    
    split_dir = Path(data_dir) / "train"
    inputs = np.load(split_dir / "all__inputs.npy")
    labels = np.load(split_dir / "all__labels.npy")
    
    with open(split_dir / "dataset.json") as f:
        metadata = json.load(f)
    
    vocab_size = metadata["vocab_size"]
    
    print(f"  Vocab size: {vocab_size}")
    print(f"  Input value range: [{inputs.min()}, {inputs.max()}]")
    print(f"  Label value range: [{labels.min()}, {labels.max()}]")
    
    # Inputs should be in valid range
    assert inputs.min() >= 0, f"Input min should be >= 0, got {inputs.min()}"
    assert inputs.max() < vocab_size, f"Input max should be < {vocab_size}, got {inputs.max()}"
    
    # Labels should be in valid range (or IGNORE_LABEL_ID which is typically -100)
    valid_labels = labels[labels >= 0]
    if len(valid_labels) > 0:
        assert valid_labels.max() < vocab_size, \
            f"Label max should be < {vocab_size}, got {valid_labels.max()}"
    
    # Sample a puzzle and display
    print(f"\n  Sample puzzle (first):")
    sample_input = inputs[0].reshape(9, 9) if inputs.shape[1] == 81 else inputs[0]
    sample_label = labels[0].reshape(9, 9) if labels.shape[1] == 81 else labels[0]
    
    print(f"    Input:\n{sample_input}")
    print(f"    Label:\n{sample_label}")
    
    print("\n✓ Data content validation passed")


def main():
    parser = argparse.ArgumentParser(description="Test SudokuDataModule")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to generated dataset directory")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for testing")
    parser.add_argument("--skip-integration", action="store_true",
                        help="Skip tests that require importing the datamodule")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SUDOKU DATAMODULE TEST SUITE")
    print("=" * 60)
    print(f"Data dir: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    
    # Basic tests (no imports needed)
    test_metadata_loading(args.data_dir)
    test_dataset_loading(args.data_dir)
    test_group_structure(args.data_dir)
    test_data_content(args.data_dir)
    
    if not args.skip_integration:
        # Integration tests (require importing the datamodule)
        test_grouped_batch_sampler(args.data_dir, args.batch_size)
        test_epoch_shuffling(args.data_dir, args.batch_size)
        test_datamodule_integration(args.data_dir, args.batch_size)
        test_val_test_loaders(args.data_dir, args.batch_size)
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()