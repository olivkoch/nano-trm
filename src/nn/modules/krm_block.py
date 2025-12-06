# src/nn/modules/kme.py
"""
Full Kernel Mean Embedding (KME) architecture for TRM.

All representations are distributions in a base space, encoded via Random Fourier Features.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn.modules.trm_block import (
    apply_rotary_pos_emb,
)
from src.nn.modules.utils import trunc_normal_init_

@dataclass
class KMEState:
    """
    Distributional state represented as weighted atoms.
    
    atoms: [batch, seq, num_atoms, d_base] - points in base space
    log_weights: [batch, seq, num_atoms] - unnormalized log mixture weights
    """
    atoms: torch.Tensor
    log_weights: torch.Tensor
    
    @property
    def weights(self) -> torch.Tensor:
        return F.softmax(self.log_weights, dim=-1)
    
    @property
    def device(self) -> torch.device:
        return self.atoms.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self.atoms.dtype
    
    def detach(self) -> 'KMEState':
        return KMEState(self.atoms.detach(), self.log_weights.detach())
    
    def clone(self) -> 'KMEState':
        return KMEState(self.atoms.clone(), self.log_weights.clone())
    
    def to(self, device: torch.device) -> 'KMEState':
        return KMEState(self.atoms.to(device), self.log_weights.to(device))
    
    @staticmethod
    def where(condition: torch.Tensor, state_true: 'KMEState', state_false: 'KMEState') -> 'KMEState':
        """Conditional selection, like torch.where for KMEState."""
        # condition: [batch] or [batch, 1, ...]
        cond_atoms = condition.view(-1, 1, 1, 1)
        cond_weights = condition.view(-1, 1, 1)
        return KMEState(
            atoms=torch.where(cond_atoms, state_true.atoms, state_false.atoms),
            log_weights=torch.where(cond_weights, state_true.log_weights, state_false.log_weights),
        )


class RFFEncoder(nn.Module):
    """
    Random Fourier Feature encoder: KME state → vector.
    
    For RBF kernel k(x,y) = exp(-||x-y||²/2σ²):
        φ(x) = [cos(ωᵀx), sin(ωᵀx)] / √D
    
    KME of distribution μ:
        z = Σᵢ wᵢ φ(aᵢ)
    """
    
    def __init__(self, d_base: int, d_out: int, bandwidth: float = 1.0):
        super().__init__()
        self.d_base = d_base
        self.d_out = d_out
        
        assert d_out % 2 == 0, "d_out must be even for cos/sin pairs"
        n_freq = d_out // 2
        
        # ω ~ N(0, I/σ²) for RBF with bandwidth σ
        frequencies = torch.randn(d_base, n_freq) / bandwidth
        self.register_buffer('frequencies', frequencies)
        self.scale = 1.0 / math.sqrt(n_freq)
    
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """RFF map: [..., d_base] → [..., d_out]"""
        proj = x @ self.frequencies
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) * self.scale
    
    def forward(self, state: KMEState) -> torch.Tensor:
        """Encode KME to vector: [batch, seq, d_out]"""
        phi_atoms = self.phi(state.atoms)  # [B, S, M, d_out]
        return torch.einsum('bsmh,bsm->bsh', phi_atoms, state.weights)


class KMECategoricalEmbedding(nn.Module):
    """
    Embed tokens as KME distributions.
    
    - Special tokens (pad, EOS, empty): learnable embeddings
    - Digit tokens: orthogonal categorical embeddings for generalization
    """
    
    def __init__(
        self,
        d_base: int,
        num_atoms: int,
        vocab_size: int = 12,
        pad_token: int = 0,
        eos_token: int = 1,
        empty_token: int = 2,
        digit_token_start: int = 3,  # Token 3 = digit 1
    ):
        super().__init__()
        self.d_base = d_base
        self.num_atoms = num_atoms
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.empty_token = empty_token
        self.digit_token_start = digit_token_start
        
        # Number of digit tokens (tokens 3-11 = digits 1-9)
        self.num_digits = vocab_size - digit_token_start
        
        # Orthogonal encodings for digits (equidistant, categorical)
        digit_enc = self._create_orthogonal_encoding(self.num_digits, d_base)
        self.register_buffer('digit_encoding', digit_enc)
        
        # Learnable embeddings for special tokens
        self.pad_embedding = nn.Parameter(torch.zeros(d_base))  # Zero for padding
        self.eos_embedding = nn.Parameter(torch.randn(d_base) * 0.5)
        self.empty_embedding = nn.Parameter(torch.randn(d_base) * 0.5)
        
        # Shared learnable atom offsets: [num_atoms, d_base]
        self.atom_offsets = nn.Parameter(torch.randn(num_atoms, d_base) * 1.0)
        
        # Per-token log weights: [vocab_size, num_atoms]
        self.log_weights = nn.Parameter(torch.zeros(vocab_size, num_atoms))
    
    def _create_orthogonal_encoding(self, num_tokens: int, d_base: int) -> torch.Tensor:
        """Create maximally separated encodings."""
        generator = torch.Generator().manual_seed(42)
        
        if num_tokens <= d_base:
            random_matrix = torch.randn(d_base, d_base, generator=generator)
            Q, _ = torch.linalg.qr(random_matrix)
            encoding = Q[:num_tokens]
        else:
            encoding = torch.randn(num_tokens, d_base, generator=generator)
            encoding = F.normalize(encoding, dim=-1)
        
        return encoding
    
    def forward(self, token_ids: torch.Tensor) -> KMEState:
        """
        token_ids: [B, S] integers in range [0, vocab_size)
        Returns: KMEState with atoms [B, S, num_atoms, d_base]
        """
        B, S = token_ids.shape
        device = token_ids.device
        
        # Start with zeros
        base = torch.zeros(B, S, self.d_base, device=device)
        
        # Pad token (0) - keep as zeros
        # (no action needed, already zeros)
        
        # EOS token (1)
        eos_mask = (token_ids == self.eos_token)
        base = torch.where(
            eos_mask.unsqueeze(-1),
            self.eos_embedding.expand(B, S, -1),
            base
        )
        
        # Empty token (2)
        empty_mask = (token_ids == self.empty_token)
        base = torch.where(
            empty_mask.unsqueeze(-1),
            self.empty_embedding.expand(B, S, -1),
            base
        )
        
        # Digit tokens (3+)
        digit_mask = (token_ids >= self.digit_token_start)
        if digit_mask.any():
            # Map token_id to digit index: token 3 -> index 0, token 4 -> index 1, etc.
            digit_indices = (token_ids - self.digit_token_start).clamp(min=0, max=self.num_digits - 1)
            digit_base = self.digit_encoding[digit_indices]  # [B, S, d_base]
            base = torch.where(
                digit_mask.unsqueeze(-1),
                digit_base,
                base
            )
        
        # Atoms = base + shared offsets: [B, S, num_atoms, d_base]
        atoms = base.unsqueeze(2) + self.atom_offsets.unsqueeze(0).unsqueeze(0)
        
        # Per-token weights: [B, S, num_atoms]
        log_weights = F.embedding(token_ids, self.log_weights)
        
        return KMEState(atoms, log_weights)


class KMERMSNorm(nn.Module):
    """RMSNorm for KME states - normalize atoms."""
    
    def __init__(self, d_base: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_base))
    
    def forward(self, state: KMEState) -> KMEState:
        rms = torch.sqrt(state.atoms.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        new_atoms = self.weight * state.atoms / rms
        return KMEState(new_atoms, state.log_weights)


class KMEMLP(nn.Module):
    """
    MLP that operates on KME states.
    Transforms atoms in base space while preserving distributional structure.
    """
    
    def __init__(self, d_base: int, num_atoms: int = 16, expansion: int = 4):
        super().__init__()
        hidden = d_base * expansion
        
        # SwiGLU in base space
        self.gate_up = nn.Linear(d_base, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, d_base, bias=False)
        
        # NEW: Weight update path
        self.w_proj = nn.Linear(d_base, num_atoms, bias=False)
        
        with torch.no_grad():
            self.down.weight.mul_(0.1)
            self.w_proj.weight.mul_(0.1)
    
    def forward(self, state: KMEState) -> KMEState:
        """Transform atoms through MLP with residual."""
        gate, up = self.gate_up(state.atoms).chunk(2, dim=-1)
        delta = self.down(F.silu(gate) * up)
        new_atoms = state.atoms + delta

        # Update weights based on atom features
        atom_features = new_atoms.mean(dim=2)  # [B, S, d_base]
        log_weight_delta = self.w_proj(atom_features)
        
        return KMEState(new_atoms, state.log_weights + log_weight_delta)

class KMEAttention(nn.Module):
    """
    Full KME attention with RoPE support.
    Batched across heads for memory efficiency.
    """
    
    def __init__(
        self,
        d_base: int,
        num_atoms: int,
        num_heads: int,
        bandwidth: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_base = d_base
        self.num_atoms = num_atoms
        self.num_heads = num_heads
        self.head_dim = d_base
        
        assert self.head_dim % 2 == 0, "head_dim must be even for RFF"
        
        self.q_proj = nn.Linear(d_base, num_heads * d_base, bias=False)
        self.k_proj = nn.Linear(d_base, num_heads * d_base, bias=False)
        self.v_proj = nn.Linear(d_base, num_heads * d_base, bias=False)
        self.o_proj = nn.Linear(num_heads * d_base, d_base, bias=False)
        
        # n_freq = self.head_dim // 2
        # frequencies = torch.randn(num_heads, d_base, n_freq) / bandwidth
        # self.register_buffer('frequencies', frequencies)
        # self.rff_scale = 1.0 / math.sqrt(n_freq)
        
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Project to update log_weights
        self.w_proj = nn.Linear(d_base, num_atoms, bias=False)
        with torch.no_grad():
            self.w_proj.weight.mul_(0.1)  # Small init
    
    def encode_batched(
        self,
        atoms: torch.Tensor,    # [B, S, M, num_heads, d_base]
        weights: torch.Tensor,  # [B, S, M]
    ) -> torch.Tensor:
        """Direct weighted mean instead of RFF."""
        # Weighted mean over atoms: [B, S, H, d_base]
        z = torch.einsum('bsmhd,bsm->bshd', atoms, weights)
        # Transpose to [B, H, S, d_base] for attention
        return z.transpose(1, 2)

    def rff_encode_batched(
        self,
        atoms: torch.Tensor,    # [B, S, M, num_heads, d_base]
        weights: torch.Tensor,  # [B, S, M]
    ) -> torch.Tensor:
        """
        Batched RFF encoding across all heads.
        Returns: [B, num_heads, S, head_dim]
        """
        # atoms: [B, S, M, H, D] -> proj: [B, S, M, H, n_freq]
        proj = torch.einsum('bsmhd,hdf->bsmhf', atoms, self.frequencies)
        
        # phi: [B, S, M, H, head_dim]
        phi = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) * self.rff_scale
        
        # Weighted sum over atoms: [B, S, H, head_dim]
        z = torch.einsum('bsmhf,bsm->bshf', phi, weights)
        
        # Transpose to [B, H, S, head_dim] for attention
        return z.transpose(1, 2)
    
    def forward(
        self,
        query: KMEState,
        key: KMEState,
        value: KMEState,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> KMEState:
        """
        KME attention with optional RoPE - fully batched.
        """
        B, S_q, M, _ = query.atoms.shape
        S_kv = key.atoms.shape[1]
        
        q_weights = query.weights
        k_weights = key.weights
        
        # Project atoms: [B, S, M, d_base] → [B, S, M, num_heads, d_base]
        q_atoms = self.q_proj(query.atoms).view(B, S_q, M, self.num_heads, self.d_base)
        k_atoms = self.k_proj(key.atoms).view(B, S_kv, M, self.num_heads, self.d_base)
        v_atoms = self.v_proj(value.atoms).view(B, S_kv, M, self.num_heads, self.d_base)
        
        # Batched RFF encode for Q and K: [B, num_heads, S, head_dim]
        # q_enc = self.rff_encode_batched(q_atoms, q_weights)
        # k_enc = self.rff_encode_batched(k_atoms, k_weights)
        q_enc = self.encode_batched(q_atoms, q_weights)
        k_enc = self.encode_batched(k_atoms, k_weights)

        # Apply RoPE if provided
        if cos_sin is not None:
            cos, sin = cos_sin
            # Reshape for apply_rotary_pos_emb: [B, S, H, head_dim]
            q_enc = q_enc.transpose(1, 2)
            k_enc = k_enc.transpose(1, 2)
            
            q_enc, k_enc = apply_rotary_pos_emb(q_enc, k_enc, cos, sin)
            
            q_enc = q_enc.transpose(1, 2)  # Back to [B, H, S, head_dim]
            k_enc = k_enc.transpose(1, 2)
        
        # Compute attention weights (WITH gradients!)
        # [B, H, S_q, head_dim] @ [B, H, head_dim, S_kv] -> [B, H, S_q, S_kv]
        attn_weights = torch.matmul(q_enc, k_enc.transpose(-2, -1)) * self.attn_scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Aggregate value atoms weighted by attention
        v_atoms_t = v_atoms.permute(0, 3, 1, 2, 4)  # [B, H, S_kv, M, d_base]
        out_atoms = torch.einsum('bhqk,bhkmd->bhqmd', attn_weights, v_atoms_t)
        
        # Transpose back: [B, S_q, M, H, d_base]
        out_atoms = out_atoms.permute(0, 2, 3, 1, 4)
        
        # Merge heads: [B, S_q, M, num_heads * d_base] -> [B, S_q, M, d_base]
        out_atoms = out_atoms.reshape(B, S_q, M, self.num_heads * self.d_base)
        out_atoms = self.o_proj(out_atoms)
        
        # Compute weight updates from output atoms
        out_features = out_atoms.mean(dim=2)
        log_weight_delta = self.w_proj(out_features)
        
        new_log_weights = query.log_weights + log_weight_delta

        return KMEState(out_atoms, new_log_weights)
        # ============================================================
        # # Original code with detached attention weights (commented out)
        # # Aggregate value atoms weighted by attention
        # # v_atoms: [B, S_kv, M, H, d_base] -> [B, H, S_kv, M, d_base]
        # v_atoms_t = v_atoms.permute(0, 3, 1, 2, 4)
        
        # # [B, H, S_q, S_kv] @ [B, H, S_kv, M, d_base] -> [B, H, S_q, M, d_base]
        # out_atoms = torch.einsum('bhqk,bhkmd->bhqmd', attn_weights, v_atoms_t)
        
        # # Transpose back: [B, S_q, M, H, d_base]
        # out_atoms = out_atoms.permute(0, 2, 3, 1, 4)
        
        # # Merge heads: [B, S_q, M, num_heads * d_base] -> [B, S_q, M, d_base]
        # out_atoms = out_atoms.reshape(B, S_q, M, self.num_heads * self.d_base)
        # out_atoms = self.o_proj(out_atoms)
        
        # # Compute weight updates from output atoms
        # # Average across atoms to get per-position features
        # out_features = out_atoms.mean(dim=2)  # [B, S_q, d_base]
        # log_weight_delta = self.w_proj(out_features)  # [B, S_q, num_atoms]
        
        # new_log_weights = query.log_weights + log_weight_delta

        # return KMEState(out_atoms, new_log_weights)


class KMEReasoningBlock(nn.Module):
    """KME reasoning block: attention + MLP with residuals and normalization."""
    
    def __init__(
        self,
        d_base: int,
        num_atoms: int,
        num_heads: int,
        ffn_expansion: int = 4,
        bandwidth: float = 1.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        
        self.attn = KMEAttention(
            d_base=d_base,
            num_atoms=num_atoms,
            num_heads=num_heads,
            bandwidth=bandwidth,
        )
        self.mlp = KMEMLP(d_base=d_base, num_atoms=num_atoms, expansion=ffn_expansion)
        
        self.norm1 = KMERMSNorm(d_base, norm_eps)
        self.norm2 = KMERMSNorm(d_base, norm_eps)
    
    def forward(
        self,
        state: KMEState,
        context: Optional[KMEState] = None,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> KMEState:
        if context is None:
            context = state
        
        # Attention with residual
        attn_out = self.attn(query=state, key=context, value=context, cos_sin=cos_sin)
        state = KMEState(
            atoms=state.atoms + attn_out.atoms,
            log_weights=attn_out.log_weights,
        )
        state = self.norm1(state)
        
        # MLP (has internal residual)
        state = self.mlp(state)
        state = self.norm2(state)
        
        return state


class KMEReasoningModule(nn.Module):
    """Stack of KME reasoning blocks."""
    
    def __init__(
        self,
        num_layers: int,
        d_base: int,
        num_atoms: int,
        num_heads: int,
        ffn_expansion: int = 4,
        bandwidth: float = 1.0,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            KMEReasoningBlock(
                d_base=d_base,
                num_atoms=num_atoms,
                num_heads=num_heads,
                ffn_expansion=ffn_expansion,
                bandwidth=bandwidth,
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        state: KMEState,
        context: KMEState,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> KMEState:
        for layer in self.layers:
            state = layer(state, context=context, cos_sin=cos_sin)
        
        return state


class KMEOutputHead(nn.Module):
    """Decode KME state to vocabulary logits."""
    
    def __init__(
        self,
        d_base: int,
        num_atoms: int,
        vocab_size: int,
        hidden_size: int,
        bandwidth: float = 1.0,
    ):
        super().__init__()
        self.atom_proj = nn.Linear(d_base, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, vocab_size, bias=False)
        # self.encoder = RFFEncoder(d_base, hidden_size, bandwidth)
        # self.proj = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, state: KMEState) -> torch.Tensor:
        """state → [B, S, vocab_size] logits"""
        # [B, S, M, d_base] → [B, S, M, hidden]
        atom_hidden = self.atom_proj(state.atoms)
        # Weighted sum: [B, S, hidden]
        z = torch.einsum('bsmh,bsm->bsh', atom_hidden, state.weights)
        return self.out_proj(z)
        # z = self.encoder(state)
        # return self.proj(z)


class KMEQHead(nn.Module):
    """Q-value head for halting decision."""
    
    def __init__(
        self,
        d_base: int,
        num_atoms: int,
        hidden_size: int,
        bandwidth: float = 1.0,
    ):
        super().__init__()
        
        self.encoder = RFFEncoder(d_base, hidden_size, bandwidth)
        self.proj = nn.Linear(hidden_size, 1, bias=True)
        
        # Initialize to not halt early
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.fill_(-5.0)
    
    def forward(self, state: KMEState, position: int = 0) -> torch.Tensor:
        """
        Get halt logit from a specific position.
        Returns: [B, 1]
        """
        z = self.encoder(state)  # [B, S, hidden]
        return self.proj(z[:, position])
    
# Add to src/nn/modules/kme.py

# class KernelKMEAttention(nn.Module):
#     """
#     True kernel-based KME attention.
#     score(q, k) = ⟨μ_q, μ_k⟩_H = Σᵢⱼ wᵢ^q wⱼ^k k(aᵢ^q, aⱼ^k)
#     """
    
#     def __init__(
#         self,
#         d_base: int,
#         num_atoms: int,
#         num_heads: int,
#         init_bandwidth: float = 1.0,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
#         self.d_base = d_base
#         self.num_atoms = num_atoms
#         self.num_heads = num_heads
        
#         self.q_proj = nn.Linear(d_base, num_heads * d_base, bias=False)
#         self.k_proj = nn.Linear(d_base, num_heads * d_base, bias=False)
#         self.v_proj = nn.Linear(d_base, num_heads * d_base, bias=False)
#         self.o_proj = nn.Linear(num_heads * d_base, d_base, bias=False)
        
#         # Learnable bandwidth per head
#         self.log_bandwidth = nn.Parameter(
#             torch.full((num_heads,), math.log(init_bandwidth))
#         )
#         # self.log_kernel_scale = nn.Parameter(torch.zeros(num_heads))
#         self.log_kernel_scale = nn.Parameter(
#            torch.full((num_heads,), math.log(math.sqrt(d_base)))
#         )
        
#         self.w_proj = nn.Linear(d_base, num_atoms, bias=False)
#         with torch.no_grad():
#             self.w_proj.weight.mul_(0.1)
        
#         self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
#     def kernel_inner_product_fast_approx(
#         self,
#         q_atoms: torch.Tensor,  # [B, H, S_q, M, D]
#         k_atoms: torch.Tensor,  # [B, H, S_k, M, D]
#         q_weights: torch.Tensor,  # [B, S_q, M]
#         k_weights: torch.Tensor,  # [B, S_k, M]
#     ) -> torch.Tensor:
#         """
#         Memory-efficient kernel inner product.
#         Returns LOG kernel scores (not kernel values) for proper softmax behavior.
#         """
#         B, H, S_q, M, D = q_atoms.shape
#         S_k = k_atoms.shape[2]
        
#         bandwidth_sq = (self.log_bandwidth.exp() ** 2).view(1, H, 1, 1)
        
#         # Weighted mean atoms
#         q_w = q_weights.unsqueeze(1).unsqueeze(-1)
#         k_w = k_weights.unsqueeze(1).unsqueeze(-1)
        
#         q_mean = (q_atoms * q_w).sum(dim=3)  # [B, H, S_q, D]
#         k_mean = (k_atoms * k_w).sum(dim=3)  # [B, H, S_k, D]
        
#         # Weighted variance
#         q_var = ((q_atoms - q_mean.unsqueeze(3)) ** 2 * q_w).sum(dim=(3, 4))
#         k_var = ((k_atoms - k_mean.unsqueeze(3)) ** 2 * k_w).sum(dim=(3, 4))
        
#         # Squared distance between means
#         q_sq = (q_mean ** 2).sum(dim=-1)
#         k_sq = (k_mean ** 2).sum(dim=-1)
#         cross = torch.einsum('bhqd,bhkd->bhqk', q_mean, k_mean)
        
#         sq_dist = q_sq.unsqueeze(-1) + k_sq.unsqueeze(-2) - 2 * cross
#         total_var = q_var.unsqueeze(-1) + k_var.unsqueeze(-2)
        
#         # Return LOG kernel scores (negative values), NOT exp()!
#         # log(exp(-x/2σ²)) = -x/2σ²
#         log_scores = -(sq_dist + total_var) / (2 * bandwidth_sq)
        
#         return log_scores  # Pass these directly to softmax
    
#     def apply_rope_to_atoms(
#         self,
#         atoms: torch.Tensor,  # [B, H, S, M, D]
#         cos: torch.Tensor,
#         sin: torch.Tensor,
#     ) -> torch.Tensor:
#         """Apply RoPE to each atom."""
#         B, H, S, M, D = atoms.shape
        
#         # Reshape for RoPE: [B, H, S, M, D] -> [B*M, S, H, D]
#         atoms_r = atoms.permute(0, 3, 2, 1, 4).reshape(B * M, S, H, D)
        
#         # Apply standard RoPE
#         cos = cos[:S].unsqueeze(0).unsqueeze(2)  # [1, S, 1, D]
#         sin = sin[:S].unsqueeze(0).unsqueeze(2)
        
#         x1 = atoms_r[..., : D // 2]
#         x2 = atoms_r[..., D // 2 :]
#         rotated = torch.cat((-x2, x1), dim=-1)
#         atoms_r = atoms_r * cos + rotated * sin
        
#         # Reshape back: [B*M, S, H, D] -> [B, H, S, M, D]
#         return atoms_r.view(B, M, S, H, D).permute(0, 3, 2, 1, 4)
    
#     def forward(
#         self,
#         query: KMEState,
#         key: KMEState,
#         value: KMEState,
#         cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#     ) -> KMEState:
#         B, S_q, M, D = query.atoms.shape
#         S_kv = key.atoms.shape[1]
#         H = self.num_heads
        
#         # Project atoms
#         q_atoms = self.q_proj(query.atoms).view(B, S_q, M, H, D).permute(0, 3, 1, 2, 4)
#         k_atoms = self.k_proj(key.atoms).view(B, S_kv, M, H, D).permute(0, 3, 1, 2, 4)
#         v_atoms = self.v_proj(value.atoms).view(B, S_kv, M, H, D).permute(0, 3, 1, 2, 4)
        
#         # Apply RoPE to atoms
#         if cos_sin is not None:
#             cos, sin = cos_sin
#             q_atoms = self.apply_rope_to_atoms(q_atoms, cos, sin)
#             k_atoms = self.apply_rope_to_atoms(k_atoms, cos, sin)
        
#         # Get LOG kernel scores (not kernel values!)
#         log_scores = self.kernel_inner_product_fast_approx(
#             q_atoms, k_atoms, query.weights, key.weights
#         )
        
#         # Scale like standard attention (optional temperature control)
#         # log_kernel_scale acts as inverse temperature here
#         log_scores = log_scores * self.log_kernel_scale.exp().view(1, -1, 1, 1)
        
#         # Softmax on log-scores gives proper attention distribution
#         attn_weights = F.softmax(log_scores, dim=-1)
#         attn_weights = self.dropout(attn_weights)
        
#         # Aggregate values
#         out_atoms = torch.einsum('bhqk,bhkmd->bhqmd', attn_weights, v_atoms)
#         out_atoms = out_atoms.permute(0, 2, 3, 1, 4).reshape(B, S_q, M, H * D)
#         out_atoms = self.o_proj(out_atoms)
        
#         # Weight updates
#         log_weight_delta = self.w_proj(out_atoms.mean(dim=2))
        
#         return KMEState(out_atoms, query.log_weights + log_weight_delta)

class KernelKMEAttention(nn.Module):
    """
    True MMD attention via Random Fourier Features.
    
    ⟨μ_q, μ_k⟩_H ≈ φ̄_qᵀ φ̄_k  where φ̄ = Σᵢ wᵢ φ(aᵢ)
    """
    
    def __init__(
        self,
        d_base: int,
        num_atoms: int,
        num_heads: int,
        n_rff_features: int = 64,  # Number of RFF features per head
        init_bandwidth: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_base = d_base
        self.num_atoms = num_atoms
        self.num_heads = num_heads
        self.n_rff = n_rff_features
        
        self.q_proj = nn.Linear(d_base, num_heads * d_base, bias=False)
        self.k_proj = nn.Linear(d_base, num_heads * d_base, bias=False)
        self.v_proj = nn.Linear(d_base, num_heads * d_base, bias=False)
        self.o_proj = nn.Linear(num_heads * d_base, d_base, bias=False)
        
        # Learnable bandwidth per head
        self.log_bandwidth = nn.Parameter(
            torch.full((num_heads,), math.log(init_bandwidth))
        )
        
        # RFF frequencies: ω ~ N(0, I/σ²)
        # We'll scale by bandwidth at runtime
        # Shape: [num_heads, d_base, n_rff]
        self.register_buffer(
            'rff_freq_base',
            torch.randn(num_heads, d_base, n_rff_features)
        )
        
        self.rff_scale = 1.0 / math.sqrt(n_rff_features)
        
        # Standard attention scale
        self.attn_scale = 1.0 / math.sqrt(n_rff_features * 2)  # *2 for cos+sin
        
        self.w_proj = nn.Linear(d_base, num_atoms, bias=False)
        with torch.no_grad():
            self.w_proj.weight.mul_(0.1)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def rff_encode(
        self,
        atoms: torch.Tensor,    # [B, H, S, M, D]
        weights: torch.Tensor,  # [B, S, M]
    ) -> torch.Tensor:
        """
        Encode weighted atoms to RFF mean embedding.
        
        Returns: [B, H, S, 2*n_rff] (cos and sin features)
        """
        B, H, S, M, D = atoms.shape
        
        # Scale frequencies by bandwidth: ω/σ
        bandwidth = self.log_bandwidth.exp().view(H, 1, 1)  # [H, 1, 1]
        rff_freq = self.rff_freq_base / bandwidth  # [H, D, n_rff]
        
        # Project atoms: [B, H, S, M, D] @ [H, D, n_rff] -> [B, H, S, M, n_rff]
        proj = torch.einsum('bhsmd,hdf->bhsmf', atoms, rff_freq)
        
        # RFF features: φ(x) = [cos(ωᵀx), sin(ωᵀx)]
        phi = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # [B, H, S, M, 2*n_rff]
        phi = phi * self.rff_scale
        
        # Weighted mean in RFF space: φ̄ = Σᵢ wᵢ φ(aᵢ)
        # weights: [B, S, M] -> [B, 1, S, M, 1]
        w = weights.unsqueeze(1).unsqueeze(-1)
        rff_mean = (phi * w).sum(dim=3)  # [B, H, S, 2*n_rff]
        
        return rff_mean
    
    def apply_rope_to_rff(
        self,
        rff: torch.Tensor,  # [B, H, S, 2*n_rff]
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE to RFF embeddings."""
        B, H, S, F = rff.shape
        
        # RoPE expects [B, S, H, D]
        rff = rff.permute(0, 2, 1, 3)  # [B, S, H, F]
        
        # Standard RoPE (only on first d_base dimensions, or adapt)
        # For simplicity, apply to all features
        cos = cos[:S].unsqueeze(0).unsqueeze(2)  # [1, S, 1, D]
        sin = sin[:S].unsqueeze(0).unsqueeze(2)
        
        # Pad cos/sin to match F if needed
        if cos.shape[-1] < F:
            repeats = (F + cos.shape[-1] - 1) // cos.shape[-1]
            cos = cos.repeat(1, 1, 1, repeats)[..., :F]
            sin = sin.repeat(1, 1, 1, repeats)[..., :F]
        
        x1 = rff[..., :F // 2]
        x2 = rff[..., F // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        rff = rff * cos + rotated * sin
        
        return rff.permute(0, 2, 1, 3)  # [B, H, S, F]
    
    def forward(
        self,
        query: KMEState,
        key: KMEState,
        value: KMEState,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> KMEState:
        B, S_q, M, D = query.atoms.shape
        S_kv = key.atoms.shape[1]
        H = self.num_heads
        
        # Project atoms
        q_atoms = self.q_proj(query.atoms).view(B, S_q, M, H, D).permute(0, 3, 1, 2, 4)
        k_atoms = self.k_proj(key.atoms).view(B, S_kv, M, H, D).permute(0, 3, 1, 2, 4)
        v_atoms = self.v_proj(value.atoms).view(B, S_kv, M, H, D).permute(0, 3, 1, 2, 4)
        
        # Apply RoPE to atoms BEFORE RFF encoding
        if cos_sin is not None:
            cos, sin = cos_sin
            q_atoms = self.apply_rope_to_atoms(q_atoms, cos, sin)
            k_atoms = self.apply_rope_to_atoms(k_atoms, cos, sin)
        
        # Encode to RFF space (position is now baked into atoms)
        q_rff = self.rff_encode(q_atoms, query.weights)
        k_rff = self.rff_encode(k_atoms, key.weights)
        
        # NO RoPE here - already applied to atoms
        
        # Attention scores
        scores = torch.einsum('bhqf,bhkf->bhqk', q_rff, k_rff)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Aggregate values
        out_atoms = torch.einsum('bhqk,bhkmd->bhqmd', attn_weights, v_atoms)
        out_atoms = out_atoms.permute(0, 2, 3, 1, 4).reshape(B, S_q, M, H * D)
        out_atoms = self.o_proj(out_atoms)
        
        log_weight_delta = self.w_proj(out_atoms.mean(dim=2))
        
        return KMEState(out_atoms, query.log_weights + log_weight_delta)

    def apply_rope_to_atoms(
        self,
        atoms: torch.Tensor,  # [B, H, S, M, D]
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE to each atom."""
        B, H, S, M, D = atoms.shape
        
        # Reshape: [B, H, S, M, D] -> [B*M, S, H, D]
        atoms_r = atoms.permute(0, 3, 2, 1, 4).reshape(B * M, S, H, D)
        
        cos = cos[:S].unsqueeze(0).unsqueeze(2)  # [1, S, 1, D]
        sin = sin[:S].unsqueeze(0).unsqueeze(2)
        
        x1 = atoms_r[..., :D // 2]
        x2 = atoms_r[..., D // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        atoms_r = atoms_r * cos + rotated * sin
        
        # Reshape back: [B*M, S, H, D] -> [B, H, S, M, D]
        return atoms_r.view(B, M, S, H, D).permute(0, 3, 2, 1, 4)
    
class KernelKMEReasoningBlock(nn.Module):
    """Reasoning block with true kernel attention."""
    
    def __init__(
        self,
        d_base: int,
        num_atoms: int,
        num_heads: int,
        ffn_expansion: int = 4,
        init_bandwidth: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.attn = KernelKMEAttention(
            d_base=d_base,
            num_atoms=num_atoms,
            num_heads=num_heads,
            init_bandwidth=init_bandwidth,
            dropout=dropout,
        )
        self.mlp = KMEMLP(d_base=d_base, num_atoms=num_atoms, expansion=ffn_expansion)
        self.norm1 = KMERMSNorm(d_base)
        self.norm2 = KMERMSNorm(d_base)
    
    def forward(
        self,
        state: KMEState,
        context: Optional[KMEState] = None,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> KMEState:
        if context is None:
            context = state
        
        attn_out = self.attn(query=state, key=context, value=context, cos_sin=cos_sin)
        state = KMEState(state.atoms + attn_out.atoms, attn_out.log_weights)
        state = self.norm1(state)
        state = self.mlp(state)
        state = self.norm2(state)
        return state


class KernelKMEReasoningModule(nn.Module):
    """Stack of kernel reasoning blocks."""
    
    def __init__(
        self,
        num_layers: int,
        d_base: int,
        num_atoms: int,
        num_heads: int,
        ffn_expansion: int = 4,
        init_bandwidth: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            KernelKMEReasoningBlock(
                d_base=d_base,
                num_atoms=num_atoms,
                num_heads=num_heads,
                ffn_expansion=ffn_expansion,
                init_bandwidth=init_bandwidth,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        state: KMEState,
        context: KMEState,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> KMEState:
        for layer in self.layers:
            state = layer(state, context=context, cos_sin=cos_sin)
        return state