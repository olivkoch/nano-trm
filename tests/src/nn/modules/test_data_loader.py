import time

import numpy as np
import torch

from src.nn.data.sudoku4x4_datamodule import Sudoku4x4DataModule


# Test DataLoader speed
def test_dataloader_speed():
    # Create your datamodule
    dm = Sudoku4x4DataModule(
        batch_size=8,
        num_workers=4,  # Test with different values
    )
    dm.setup("fit")

    train_loader = dm.train_dataloader()

    # Time data loading only
    print("Testing DataLoader speed...")
    times = []

    for i, batch in enumerate(train_loader):
        if i == 0:
            # Skip first batch (warmup)
            continue

        start = time.time()
        # Force data to GPU to measure full pipeline
        batch = {k: v.to("mps") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elapsed = time.time() - start
        times.append(elapsed)

        if i >= 10:  # Test 10 batches
            break

    print(f"Average batch load time: {np.mean(times) * 1000:.2f}ms")
    print(f"Min/Max: {np.min(times) * 1000:.2f}ms / {np.max(times) * 1000:.2f}ms")


if __name__ == "__main__":
    # Test with different worker counts
    for num_workers in [0, 1, 2, 4]:
        print(f"\n--- Testing with num_workers={num_workers} ---")
        dm = Sudoku4x4DataModule(batch_size=8, num_workers=num_workers)
        dm.setup("fit")

        start = time.time()
        for i, _ in enumerate(dm.train_dataloader()):
            if i >= 20:
                break
        elapsed = time.time() - start
        print(f"20 batches took: {elapsed:.2f}s ({elapsed / 20 * 1000:.2f}ms per batch)")
