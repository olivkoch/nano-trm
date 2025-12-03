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
    RotaryEmbedding,
    RotaryEmbedding2D,
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


class KMETokenEmbedding(nn.Module):
    """
    Embed discrete tokens as KME distributions.
    
    Each token has num_atoms learned atoms and weights in base space.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_base: int,
        num_atoms: int,
        init_std: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_base = d_base
        self.num_atoms = num_atoms
        
        # Store as 2D for F.embedding: [vocab, num_atoms * d_base]
        self.atom_embeddings = nn.Parameter(
            trunc_normal_init_(
                torch.empty(vocab_size, num_atoms * d_base),
                std=init_std
            )
        )
        
        # Weight embeddings: [vocab, num_atoms]
        self.log_weight_embeddings = nn.Parameter(
            torch.zeros(vocab_size, num_atoms)
        )
    
    def forward(self, token_ids: torch.Tensor) -> KMEState:
        """
        token_ids: [batch, seq] of integers
        Returns: KMEState with atoms [batch, seq, num_atoms, d_base]
        """
        B, S = token_ids.shape
        
        # Lookup and reshape: [B, S, num_atoms * d_base] -> [B, S, num_atoms, d_base]
        atoms_flat = F.embedding(token_ids, self.atom_embeddings)
        atoms = atoms_flat.view(B, S, self.num_atoms, self.d_base)
        
        log_weights = F.embedding(token_ids, self.log_weight_embeddings)
        
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

        # NEW: Update weights based on atom features
        atom_features = state.atoms.mean(dim=2)  # [B, S, d_base]
        log_weight_delta = self.w_proj(atom_features)  # [B, S, num_atoms]
        
        return KMEState(
            state.atoms + delta,
            state.log_weights + log_weight_delta
        )

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
        
        n_freq = self.head_dim // 2
        frequencies = torch.randn(num_heads, d_base, n_freq) / bandwidth
        self.register_buffer('frequencies', frequencies)
        self.rff_scale = 1.0 / math.sqrt(n_freq)
        
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Project to update log_weights
        self.w_proj = nn.Linear(d_base, num_atoms, bias=False)
        with torch.no_grad():
            self.w_proj.weight.mul_(0.1)  # Small init
    
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
        q_enc = self.rff_encode_batched(q_atoms, q_weights)
        k_enc = self.rff_encode_batched(k_atoms, k_weights)
        
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
        # v_atoms: [B, S_kv, M, H, d_base] -> [B, H, S_kv, M, d_base]
        v_atoms_t = v_atoms.permute(0, 3, 1, 2, 4)
        
        # [B, H, S_q, S_kv] @ [B, H, S_kv, M, d_base] -> [B, H, S_q, M, d_base]
        out_atoms = torch.einsum('bhqk,bhkmd->bhqmd', attn_weights, v_atoms_t)
        
        # Transpose back: [B, S_q, M, H, d_base]
        out_atoms = out_atoms.permute(0, 2, 3, 1, 4)
        
        # Merge heads: [B, S_q, M, num_heads * d_base] -> [B, S_q, M, d_base]
        out_atoms = out_atoms.reshape(B, S_q, M, self.num_heads * self.d_base)
        out_atoms = self.o_proj(out_atoms)
        
        # Compute weight updates from output atoms
        # Average across atoms to get per-position features
        out_features = out_atoms.mean(dim=2)  # [B, S_q, d_base]
        log_weight_delta = self.w_proj(out_features)  # [B, S_q, num_atoms]
        
        new_log_weights = query.log_weights + log_weight_delta

        return KMEState(out_atoms, new_log_weights)


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
        # Input injection: combine state with context
        state = KMEState(
            atoms=state.atoms + context.atoms,
            log_weights=state.log_weights,
        )
        
        for layer in self.layers:
            state = layer(state, cos_sin=cos_sin)
        
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
        
        self.encoder = RFFEncoder(d_base, hidden_size, bandwidth)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, state: KMEState) -> torch.Tensor:
        """state → [B, S, vocab_size] logits"""
        z = self.encoder(state)
        return self.proj(z)


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