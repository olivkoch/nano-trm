import time

import torch
from lightning import LightningModule, Trainer


class MinimalModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)
        self.automatic_optimization = False
        self.last_time = None

    def training_step(self, batch, batch_idx):
        if self.last_time:
            print(f"Gap: {time.time() - self.last_time:.3f}s")

        # Fix: use the batch data and ensure it's on the right device
        x = batch[0].to(self.device)  # Use actual batch data
        loss = self.layer(x).sum()
        loss.backward()

        opt = self.optimizers()
        opt.step()
        opt.zero_grad()

        self.last_time = time.time()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


# Test
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(torch.randn(100, 10))
dataloader = DataLoader(dataset, batch_size=8)

model = MinimalModule()
trainer = Trainer(
    max_epochs=1,
    accelerator="mps",  # Explicit CPU
    callbacks=[],
    logger=False,
    enable_progress_bar=False,
    enable_checkpointing=False,
    enable_model_summary=False,
)
trainer.fit(model, dataloader)
