import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.nn.utils import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS normalization
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / norm


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len: int):
        # x: [batch, seq_len, dim]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        cos = emb.cos()[None, :, :]  # [1, seq_len, dim]
        sin = emb.sin()[None, :, :]  # [1, seq_len, dim]
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary position embedding."""
    return (x * cos) + (rotate_half(x) * sin)


class SwiGLU(nn.Module):
    """SwiGLU activation function (Gated Linear Unit with Swish)."""
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        # SwiGLU(x, W, V, W2) = (Swish(xW) ⊗ xV)W2
        # where Swish(x) = x * sigmoid(x)
        swish = F.silu(self.w1(x))  # SiLU is Swish
        x_V = self.w2(x)
        return self.w3(swish * x_V)


class TransformerBlock(nn.Module):
    """Single Transformer block with specifications from paper."""
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Pre-attention norm
        self.norm1 = RMSNorm(hidden_size)
        
        # Self-attention (no bias as per paper)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # Pre-FFN norm
        self.norm2 = RMSNorm(hidden_size)
        
        # SwiGLU FFN (hidden_dim = 4 * hidden_size as standard)
        self.ffn = SwiGLU(hidden_size, hidden_size * 4, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        
        # Multi-head attention
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings to q and k
        cos, sin = self.rotary_emb(x, seq_len)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        x = residual + attn_output
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        
        return x