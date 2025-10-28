import torch


def trunc_normal_init(shape, std=1.0):
    """Truncated normal initialization."""
    t = torch.randn(shape) * std
    # Truncate to [-2*std, 2*std]
    t = torch.clamp(t, -2 * std, 2 * std)
    return t
