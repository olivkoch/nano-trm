import math

import torch

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
