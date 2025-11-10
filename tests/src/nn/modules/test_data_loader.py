import time

import numpy as np
import torch
import click

from src.nn.data.sudoku_datamodule import SudokuDataModule


# Test DataLoader speed
def test_dataloader_speed(num_batches: int, device: torch.device, data_dir: str, batch_size: int) -> float:

    # Create your datamodule
    dm = SudokuDataModule(
        batch_size=batch_size,
        num_workers=4,  # Test with different values
        data_dir=data_dir,
        generate_on_fly=data_dir is None,
    )
    dm.setup("fit")

    train_loader = dm.train_dataloader()

    # Time data loading only

    times = []

    for i, batch in enumerate(train_loader):
        if i == 0:
            # Skip first batch (warmup)
            continue

        start = time.perf_counter()
        # Force data to GPU to measure full pipeline
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed)

        if i >= num_batches:
            break

    return np.mean(times)

@click.command()
@click.option("--num-batches", type=int, default=100, help="Number of batches to time")
@click.option("--data-dir", type=str, default=None, help="Data directory")
@click.option("--batch-size", type=int, default=512, help="Batch size")

def main(num_batches: int, data_dir: str, batch_size: int):
    # Test with different worker counts
    num_runs = 1

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Loading data to device {device}")

    for num_workers in [0, 1, 2, 4]:
        ans = []
        for _ in range(num_runs):
            elapsed_ms = test_dataloader_speed(num_batches=num_batches, device=device, data_dir=data_dir, batch_size=batch_size)
            ans.append(elapsed_ms)
        print(
            f"num_workers={num_workers} => avg data loading time per batch: {np.mean(ans):.3f} ms (+- {np.std(ans):.3f} ms over {num_runs} runs)"
        )

if __name__ == "__main__":
    main()