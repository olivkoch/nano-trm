import torch
import torch.nn as nn


class StableMaxCrossEntropyLoss(nn.Module):
    """
    StableMax cross-entropy loss with numerical stability improvements.
    """

    def __init__(self, ignore_index: int = -100, epsilon: float = 1e-8):
        super().__init__()
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def stable_s(self, x: torch.Tensor) -> torch.Tensor:
        """Piecewise stable function with gradient clipping."""
        # Clamp input to prevent extreme values
        x = torch.clamp(x, min=-10, max=10)

        # Compute piecewise function
        positive_part = x + 1.0
        negative_part = 1.0 / (1.0 - x + self.epsilon)

        return torch.where(x >= 0, positive_part, negative_part)

    def stable_max(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Compute stable softmax-like probabilities."""
        # Subtract max for numerical stability (like standard softmax)
        logits_max = logits.max(dim=dim, keepdim=True)[0]
        logits_shifted = logits - logits_max

        # Apply stable_s function
        s_vals = self.stable_s(logits_shifted)

        # Normalize
        denom = s_vals.sum(dim=dim, keepdim=True)
        probs = s_vals / (denom + self.epsilon)

        return probs

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, num_classes] - raw logits
            targets: [N] - class indices
        Returns:
            loss: scalar
        """
        # Get probabilities
        probs = self.stable_max(logits, dim=-1)

        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            if not mask.any():
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            logits = logits[mask]
            targets = targets[mask]
            probs = probs[mask]

        # Select probability of correct class
        batch_size = probs.size(0)
        pt = probs[torch.arange(batch_size, device=logits.device), targets]

        # Compute negative log likelihood with stability
        loss = -torch.log(pt + self.epsilon)

        # Check for NaN/Inf
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("WARNING: NaN/Inf in loss!")
            print(f"  logits range: [{logits.min():.4f}, {logits.max():.4f}]")
            print(f"  probs range: [{probs.min():.4f}, {probs.max():.4f}]")
            print(f"  pt range: [{pt.min():.4f}, {pt.max():.4f}]")
            # Return a fallback value
            return torch.tensor(10.0, device=logits.device, requires_grad=True)

        return loss.mean()
