from typing import List, Tuple, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

from src.nn.modules.utils import trunc_normal_init_
from src.nn.utils import RankedLogger

CosSin = Tuple[torch.Tensor, torch.Tensor]

log = RankedLogger(__name__, rank_zero_only=True)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS normalization
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / norm


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((out_features, in_features)), std=1.0 / (in_features**0.5)
            )
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input,
            self.weight.to(input.dtype),
            bias=self.bias.to(input.dtype) if self.bias is not None else None,
        )


class CastedEmbedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype
    ):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        self.enabled = base > 0
        if not self.enabled:
            return
        # RoPE
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        if not self.enabled:
            return None, None
        return self.cos_cached, self.sin_cached

class RotaryEmbedding2D(nn.Module):
    """2D RoPE using same frequency basis as 1D, with prefix support."""
    
    def __init__(self, dim, prefix_len, max_grid_size, base=10000, device=None):
        super().__init__()
        self.prefix_len = prefix_len
        self.max_grid_size = max_grid_size
        self.dim = dim
        
        # SAME frequency formula as 1D RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self._build_cache(device)
    
    def _build_cache(self, device=None):
        if device is None:
            device = self.inv_freq.device
        
        n_freq = self.inv_freq.shape[0]  # dim // 2 (32 for dim=64)
        quarter = n_freq // 2            # dim // 4 (16 for dim=64)
        
        # Prefix: standard 1D positions
        if self.prefix_len > 0:
            prefix_pos = torch.arange(self.prefix_len, dtype=torch.float32, device=device)
            prefix_freqs = torch.outer(prefix_pos, self.inv_freq)  # [prefix, 32]
            prefix_emb = torch.cat((prefix_freqs, prefix_freqs), dim=-1)  # [prefix, 64]
        
        # Grid: 2D positions
        grid_len = self.max_grid_size ** 2
        indices = torch.arange(grid_len, dtype=torch.float32, device=device)
        rows = indices // self.max_grid_size
        cols = indices % self.max_grid_size
        
        # Row and col BOTH use the same frequencies (first quarter of inv_freq)
        # This gives them equal expressiveness
        row_freqs = torch.outer(rows, self.inv_freq[:quarter])  # [grid, 16]
        col_freqs = torch.outer(cols, self.inv_freq[:quarter])  # [grid, 16]
        
        # Structure: [row, col, row, col] to match rotate_half pattern
        grid_emb = torch.cat([row_freqs, col_freqs, row_freqs, col_freqs], dim=-1)  # [grid, 64]
        
        # Combine prefix + grid
        if self.prefix_len > 0:
            full_emb = torch.cat([prefix_emb, grid_emb], dim=0)
        else:
            full_emb = grid_emb
        
        self.cos_cached = nn.Buffer(full_emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(full_emb.sin(), persistent=False)
    
    def forward(self):
        return self.cos_cached, self.sin_cached
    
def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

class CastedConv1d(nn.Conv1d):
    """Conv1d that automatically casts weights/bias to input dtype."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.constant_(self.bias, 0) if self.bias is not None else None
        trunc_normal_init_(self.weight, std=0.02)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            input,
            self.weight.to(input.dtype),
            self.bias.to(input.dtype) if self.bias is not None else None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

class CastedConv2d(nn.Conv2d):
    """Conv2d that automatically casts weights/bias to input dtype."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.constant_(self.bias, 0) if self.bias is not None else None
        trunc_normal_init_(self.weight, std=0.02)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input,
            self.weight.to(input.dtype),
            self.bias.to(input.dtype) if self.bias is not None else None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
    
class ConvSwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        conv_kernel: int = 3, # Changed default to 3 (Odd is better for alignment)
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()

        inter = intermediate_size if intermediate_size is not None else _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)

        self.dwconv = CastedConv1d(
            in_channels=inter * 2,
            out_channels=inter * 2,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=inter * 2,
            bias=True,
        )

        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, timer: Optional[object] = None, prefix: str = ""):
        
        x_expanded = self.gate_up_proj(x)
        x_conv = self.dwconv(x_expanded.transpose(1, 2))
        if x_conv.size(-1) != x.size(1):
             x_conv = x_conv[..., :x.size(1)]
        x_conv = x_conv.transpose(1, 2)
        gate, up = x_conv.chunk(2, dim=-1)
        x_out = F.silu(gate) * up
        return self.down_proj(x_out)
    
class BoardAwareSwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        rows: int = None,
        cols: int = None,
        puzzle_emb_len: int = 0,
        conv_kernel: int = 3,
    ):
        super().__init__()
        assert rows is not None and cols is not None, "Rows and Cols must be specified for BoardAwareSwiGLU."
        assert rows > 0 and cols > 0, "Rows and Cols must be positive integers."
        self.rows = rows
        self.cols = cols
        self.puzzle_len = puzzle_emb_len
        
        inter = _find_multiple(int(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.board_conv = CastedConv2d(
            inter * 2, inter * 2, 
            kernel_size=conv_kernel, 
            padding=conv_kernel // 2,
            groups=inter * 2,
            bias=True
        )
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        x_expanded = self.gate_up_proj(x)
        x_puzzle = x_expanded[:, :self.puzzle_len, :]
        x_board = x_expanded[:, self.puzzle_len:, :]
        x_board_img = x_board.transpose(1, 2).view(B, -1, self.rows, self.cols)
        x_board_conv = self.board_conv(x_board_img)
        x_board_out = x_board_conv.flatten(2).transpose(1, 2)
        x_out = torch.cat([x_puzzle, x_board_out], dim=1)
        gate, value = x_out.chunk(2, dim=-1)
        x_act = F.silu(gate) * value
        return self.down_proj(x_act)
     
class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(
            batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim
        )
        query = qkv[:, :, : self.num_heads]
        key = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        query, key, value = map(
            lambda t: einops.rearrange(t, "B S H D -> B H S D"), (query, key, value)
        )  # needed for scaled_dot_product_attention but not flash_attn_func
        
        attn_output = scaled_dot_product_attention(
            query=query, key=key, value=value, is_causal=self.causal
        )
        attn_output = einops.rearrange(attn_output, "B H S D -> B S H D")

        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


class ReasoningBlockConfig:
    """
    Configuration A: The "Standard Transformer"
        mlp_t=False (Use Attention)
        use_convswiglu/use_boardswiglu=False (Use Standard MLP)
        Result: Classic powerful reasoning. Attention handles global context; MLP handles logic.

    Configuration B: The "Spatial-Inductive Transformer"
        mlp_t=False (Use Attention)
        use_convswiglu/use_boardswiglu=True (Use Conv MLP)
        Result: Strongest. Attention sees the whole board ("I can win in column 7"), while ConvSwiGLU recognizes patterns immediately ("I have 3-in-a-row here"). This gives the best of both worlds.

    Configuration C: The "MLP-Mixer" (Pure MLP)
        mlp_t=True (Use Token MLP)
        use_convswiglu/use_boardswiglu=False (Use Standard MLP)
        Result: Very fast, very stable, but no Attention. The model mixes information globally using a fixed matrix. It might struggle with "dynamic" reasoning.

    Configuration D: The "ConvMixer"
        mlp_t=True
        use_convswiglu/use_boardswiglu=True
        Result: A fully convolutional/MLP network. It has zero attention mechanisms: mlp_t mixes the board globally (fixed weights), convswiglu mixes neighbors locally.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: int,
        rms_norm_eps: float,
        mlp_t: bool = False,
        seq_len: int = 0,
        cols: int = None,
        rows: int = None,
        puzzle_emb_ndim: int = 0,
        puzzle_emb_len: int = 0,
        use_conv_swiglu: bool = False,
        use_board_swiglu: bool = False,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expansion = expansion
        self.rms_norm_eps = rms_norm_eps
        self.mlp_t = mlp_t
        self.puzzle_emb_ndim = puzzle_emb_ndim
        self.puzzle_emb_len = puzzle_emb_len
        self.seq_len = seq_len
        self.cols = cols
        self.rows = rows
        self.use_conv_swiglu = use_conv_swiglu
        self.use_board_swiglu = use_board_swiglu


class ReasoningBlock(nn.Module):
    def __init__(self, config: ReasoningBlockConfig) -> None:
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps

        # 1. Calculate Effective Length
        # If config is 0 (auto), infer from dimensions. Otherwise use config.
        # This handles the case where puzzle_emb_ndim > 0 but puzzle_emb_len was not manually set.
        self.puzzle_emb_len = (
            -(config.puzzle_emb_ndim // -config.hidden_size)
            if config.puzzle_emb_len == 0
            else config.puzzle_emb_len
        )

        if self.config.mlp_t:
            self.mlp_t = SwiGLU(
                hidden_size=config.seq_len + self.puzzle_emb_len, 
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )

        if self.config.use_board_swiglu:
            self.mlp = BoardAwareSwiGLU(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
                rows=config.rows,
                cols=config.cols,
                puzzle_emb_len=self.puzzle_emb_len, 
                conv_kernel=3
            )
            
        elif self.config.use_conv_swiglu:
            self.mlp = ConvSwiGLU(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
                conv_kernel=3
            )
            
        else:
            self.mlp = SwiGLU(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
            )

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:

        if self.config.mlp_t:
            residual = hidden_states
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = out.transpose(1, 2)
            
            hidden_states = rms_norm(residual + hidden_states, variance_epsilon=self.norm_eps)
        else:
            attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
            hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)

        mlp_out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_out, variance_epsilon=self.norm_eps)
            
        return hidden_states

class ReasoningModule(nn.Module):
    def __init__(self, layers: List[ReasoningBlock]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


def _find_multiple(a, b):
    return (-(a // -b)) * b
