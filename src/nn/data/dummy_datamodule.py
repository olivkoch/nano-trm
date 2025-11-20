# src/nn/data/dummy_datamodule.py
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    """Dummy dataset for self-play training."""
    def __init__(self, length=100):
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {}  # Return empty dict


class DummyDataModule(LightningDataModule):
    """Dummy datamodule for pure self-play training."""
    
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Properties needed by model
        self.num_puzzles = 1  # Single game type
        self.pad_value = 0
        self.max_grid_size = 7
        self.vocab_size = 25  # Enough for board + move + outcome tokens
        self.seq_len = 42
    
    def setup(self, stage=None):
        self.train_dataset = DummyDataset()
        self.val_dataset = DummyDataset(length=100)
        self.test_dataset = DummyDataset(length=100)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,  # Batch size handled in model
            num_workers=0,
            shuffle=False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=0
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=0
        )