import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from lightning import LightningModule

class MLPModule(LightningModule):
    def __init__(self,                  
                 num_colors,
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers=2, 
                 lr=1e-3, 
                 output_dir: str=None
                 ):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self.lr = lr
        self.input_embedding = nn.Embedding(
            num_colors + 1,  # +1 for padding
            hidden_dim
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x = batch['input']  # [batch, H, W]
        y_true = batch['output']  # [batch, H, W]
        batch_size, height, width = x.shape
        
        # Embed input
        x_emb = self.input_embedding(x)
        x_emb = x_emb.view(batch_size, -1, self.hparams.hidden_size)
        logits = self(x_emb)
        loss = F.cross_entropy(logits, y_true.view(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['input']
        y_true = batch['output']
        batch_size, height, width = x.shape

        x_emb = self.input_embedding(x)
        x_emb = x_emb.view(batch_size, -1, self.hparams.hidden_size)
        logits = self(x_emb)
        loss = F.cross_entropy(logits, y_true.view(-1))
        acc = (logits.argmax(dim=1) == y_true.view(-1)).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)