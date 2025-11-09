# Test without Lightning to confirm it's the framework

import time

import torch

from src.nn.data.xor_datamodule import XORDataModule
from src.nn.models.trm_module import TRMModule


def test_raw_training():
    dm = XORDataModule(batch_size=8, num_workers=0)
    dm.setup("fit")

    model = TRMModule(
        hidden_size=128,
        num_layers=2,
        puzzle_emb_dim=128,
        puzzle_emb_len=4,
        N_supervision=3,
        n_latent_recursions=1,
        T_deep_recursions=1,
        num_puzzles=dm.num_puzzles,
        batch_size=dm.batch_size,
        pad_value=dm.pad_value,
    )
    model.to("mps")
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i, batch in enumerate(dm.train_dataloader()):
        if i >= 20:
            break

        start = time.time()

        # Move to device
        batch = {k: v.to("mps") for k, v in batch.items()}

        # Your training logic
        if model.carry is None:
            model.carry = model.initial_carry(batch)

        model.carry, loss, metrics, _ = model.compute_loss_and_metrics(model.carry, batch)

        scaled_loss = loss / 8
        scaled_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        elapsed = time.time() - start
        print(f"Step {i}: {elapsed * 1000:.1f}ms")  # Should be ~15-225ms


if __name__ == "__main__":
    test_raw_training()
