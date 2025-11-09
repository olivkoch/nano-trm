"""
XOR DataModule that generates data on the fly, compatible with TRM
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.nn.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class XORDataset(Dataset):
    """Dataset for XOR puzzles generated on the fly."""

    def __init__(
        self,
        num_puzzles: int = 1000,
        num_examples_per_puzzle: int = 3,
        grid_size: int = 3,
        max_grid_size: int = 6,
        seed: int = 42,
        split: str = "train",
    ):
        self.num_puzzles = num_puzzles
        self.num_examples_per_puzzle = num_examples_per_puzzle
        self.grid_size = grid_size
        self.max_grid_size = max_grid_size
        self.seed = seed + (0 if split == "train" else 1000)
        self.split = split

        # Total number of samples
        self.num_samples = num_puzzles * num_examples_per_puzzle

        # Set up RNG
        self.rng = np.random.RandomState(self.seed)

    def __len__(self):
        return self.num_samples

    def generate_simple_xor_example(self):
        """Generate a single XOR example - matches your generator exactly."""
        # Generate input with all 4 possible combinations
        input_grid = self.rng.randint(0, 4, (self.grid_size, self.grid_size), dtype=np.uint8)

        # Compute XOR for each cell
        output_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        output_grid[input_grid == 1] = 1  # (1,0) -> 1
        output_grid[input_grid == 2] = 1  # (0,1) -> 1
        # (0,0) and (1,1) remain 0

        return input_grid, output_grid

    def pad_grid(self, grid: np.ndarray) -> np.ndarray:
        """Pad and transform grid - matches your pad_and_flatten but returns 2D."""
        # Add 2 to all values to reserve 0 for PAD and 1 for EOS
        grid = grid + 2

        # Pad to max_size x max_size
        h, w = grid.shape
        padded = np.pad(
            grid, ((0, self.max_grid_size - h), (0, self.max_grid_size - w)), constant_values=0
        )

        # Add simple EOS marker at the actual data boundary
        if h < self.max_grid_size and w < self.max_grid_size:
            padded[h, 0] = 1  # Single EOS marker

        return padded

    def __getitem__(self, idx):
        # Seed based on index for reproducibility
        self.rng.seed(self.seed + idx)

        # Generate example
        input_grid, output_grid = self.generate_simple_xor_example()

        # Pad grids
        input_padded = self.pad_grid(input_grid)
        output_padded = self.pad_grid(output_grid)

        # All XOR puzzles have puzzle_identifier = 1 (0 is reserved)
        return {
            "input": torch.from_numpy(input_padded).long(),
            "output": torch.from_numpy(output_padded).long(),
            "puzzle_identifier": 1,  # All XOR puzzles have ID 1
        }


def collate_fn_xor(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function matching TRM's expected format."""
    return {
        "input": torch.stack([sample["input"] for sample in batch]),
        "output": torch.stack([sample["output"] for sample in batch]),
        "puzzle_identifiers": torch.tensor(
            [sample["puzzle_identifier"] for sample in batch], dtype=torch.long
        ),
    }


class XORDataModule(LightningDataModule):
    """DataModule for XOR task compatible with TRM."""

    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 4,
        num_train_puzzles: int = 1000,
        num_val_puzzles: int = 200,
        num_examples_per_puzzle: int = 3,
        grid_size: int = 3,
        max_grid_size: int = 6,
        pad_value: int = 0,
        seed: int = 42,
        drop_last: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train_puzzles = num_train_puzzles
        self.num_val_puzzles = num_val_puzzles
        self.num_examples_per_puzzle = num_examples_per_puzzle
        self.grid_size = grid_size
        self.max_grid_size = max_grid_size
        self.seed = seed
        self.drop_last = drop_last
        self.pad_value = pad_value

        # Metadata matching your generator
        self.seq_len = max_grid_size * max_grid_size  # 36
        num_colors = 4  # 4 possible values
        self.vocab_size = 2 + num_colors  # PAD(0) + EOS(1) + actual values (2-5)
        self.num_puzzles = 2  # 0 (reserved) + 1 (XOR puzzle type)

    def setup(self, stage: Optional[str] = None):
        """Create datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = XORDataset(
                num_puzzles=self.num_train_puzzles,
                num_examples_per_puzzle=self.num_examples_per_puzzle,
                grid_size=self.grid_size,
                max_grid_size=self.max_grid_size,
                seed=self.seed,
                split="train",
            )

            self.val_dataset = XORDataset(
                num_puzzles=self.num_val_puzzles,
                num_examples_per_puzzle=self.num_examples_per_puzzle,
                grid_size=self.grid_size,
                max_grid_size=self.max_grid_size,
                seed=self.seed,
                split="test",
            )

            print("✓ Created XOR dataset:")
            print(
                f"  - Grid size: {self.grid_size}x{self.grid_size} padded to {self.max_grid_size}x{self.max_grid_size}"
            )
            print(f"  - Sequence length: {self.seq_len}")
            print(f"  - Vocab size: {self.vocab_size}")
            print(f"  - Train samples: {len(self.train_dataset)}")
            print(f"  - Val samples: {len(self.val_dataset)}")

        if stage == "test":
            self.test_dataset = XORDataset(
                num_puzzles=self.num_val_puzzles,
                num_examples_per_puzzle=self.num_examples_per_puzzle,
                grid_size=self.grid_size,
                max_grid_size=self.max_grid_size,
                seed=self.seed,
                split="test",
            )

    def train_dataloader(self):
        log.info(
            f"Creating train dataloader with batch_size={self.batch_size}, num_workers={self.num_workers}"
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_xor,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_xor,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return self.val_dataloader()


# Test to verify it matches your generator
if __name__ == "__main__":
    dm = XORDataModule(batch_size=4)
    dm.setup("fit")

    batch = next(iter(dm.train_dataloader()))
    print("Batch shapes:")
    print(f"  input: {batch['input'].shape}")  # [4, 6, 6]
    print(f"  output: {batch['output'].shape}")  # [4, 6, 6]

    print("\nSample input[0] (should have values 0-6):")
    print(batch["input"][0])
    print("\nSample output[0] (should have values 0,2,3 with maybe 1 for EOS):")
    print(batch["output"][0])
