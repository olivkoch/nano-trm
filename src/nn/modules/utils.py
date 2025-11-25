import math
import torch
import numpy as np

# def trunc_normal_init(shape, std=1.0):
#     """Truncated normal initialization."""
#     t = torch.randn(shape) * std
#     # Truncate to [-2*std, 2*std]
#     t = torch.clamp(t, -2 * std, 2 * std)
#     return t


def trunc_normal_init_(
    tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0
):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower**2)
            pdf_l = c * math.exp(-0.5 * upper**2)
            comp_std = std / math.sqrt(
                1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
            )

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x, dim=-1):
    # Ensure we handle the dtype properly
    original_dtype = x.dtype
    if x.device.type == "mps" and x.dtype == torch.float64:
        x = x.to(torch.float32)

    s_x = s(x)
    result = torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))

    # Convert back to original dtype if possible
    if x.device.type != "mps" and original_dtype == torch.float64:
        result = result.to(original_dtype)

    return result


def stablemax_cross_entropy(logits, labels, ignore_index: int = 0, valid_mask=None):
    # Detect device type and use appropriate dtype
    if logits.device.type == "mps":
        # MPS doesn't support float64, use float32
        logprobs = log_stablemax(logits.to(torch.float32), dim=-1)
    else:
        # CUDA and CPU support float64
        logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def compute_lr(
    base_lr: float, lr_warmup_steps: int, lr_min_ratio: float, current_step: int, total_steps: int
) -> float:
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=current_step,
        base_lr=base_lr,
        num_warmup_steps=round(lr_warmup_steps),
        num_training_steps=total_steps,
        min_ratio=lr_min_ratio,
    )


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return base_lr * (
        min_ratio
        + max(
            0.0,
            (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
    )

def robust_kl_div(pred_probs: torch.Tensor, target_probs: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute KL divergence robustly, handling zeros on both sides
    KL(target || pred) = sum(target * log(target/pred))
    """
    # Add epsilon to avoid log(0) but preserve zero structure
    pred_safe = pred_probs + epsilon
    target_safe = target_probs + epsilon
    
    # Renormalize to ensure they sum to 1
    pred_safe = pred_safe / pred_safe.sum(dim=-1, keepdim=True)
    target_safe = target_safe / target_safe.sum(dim=-1, keepdim=True)
    
    # Compute KL divergence
    # Where target is 0, the contribution should be 0 (0 * log(...) = 0)
    # We mask these out explicitly
    kl_per_element = target_safe * (torch.log(target_safe) - torch.log(pred_safe))
    
    # Mask out contributions where original target was 0
    mask = target_probs > epsilon
    kl_per_element = kl_per_element * mask
    
    # Sum over action dimension, mean over batch
    kl_div = kl_per_element.sum(dim=-1).mean()
    
    return kl_div

class CircularBuffer:
    """Efficient circular buffer with O(1) random access"""
    
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buffer = []
        self.position = 0
    
    def append(self, item):
        if len(self.buffer) < self.maxlen:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.position = (self.position + 1) % self.maxlen
    
    def extend(self, items):
        for item in items:
            self.append(item)
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        return self.buffer[idx]
    
    def sample(self, n: int, replace: bool = False) -> list:
        """Sample n items efficiently"""
        if replace or n > len(self.buffer):
            indices = np.random.randint(0, len(self.buffer), size=n)
        else:
            indices = np.random.choice(len(self.buffer), n, replace=False)
        return [self.buffer[i] for i in indices]