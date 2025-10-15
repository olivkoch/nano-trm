import torch
import torch.nn as nn

def stable_s(x: torch.Tensor) -> torch.Tensor:
    # piecewise function from the StableMax paper
    return torch.where(x >= 0, x + 1, 1.0 / (1.0 - x))

def stable_max(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    s_vals = stable_s(logits)
    denom = s_vals.sum(dim=dim, keepdim=True)
    return s_vals / denom.clamp_min(1e-12) # avoid div-by-zero

class StableMaxCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: Tensor of shape (batch, num_classes)
        targets: LongTensor of shape (batch,) with class indices
        """
        probs = stable_max(logits, dim=-1)
        # select probability of the correct class
        batch_size = logits.size(0)
        pt = probs[torch.arange(batch_size, device=logits.device), targets]
        # compute loss = -log(pt)
        loss = -torch.log(pt.clamp_min(1e-12))
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index)
            loss = loss[mask]
        return loss.mean()
    