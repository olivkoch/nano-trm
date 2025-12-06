"""
General KME Attention - No RoPE needed.

Position information is embedded directly in the atoms via GeneralKMEEmbedding.
This allows cross-size generalization since attention is purely content-based.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn.modules.krm_block import KMEState, KMERMSNorm, KMEMLP


class GeneralKernelKMEAttention(nn.Module):
    """
    Kernel-based KME attention via Random Fourier Features.
    
    No RoPE - position is encoded in the atoms themselves.
    
    ⟨μ_q, μ_k⟩_H ≈ φ̄_qᵀ φ̄_k  where φ̄ = Σᵢ wᵢ φ(aᵢ)
    """
    
    def __init__(
        self,
        d_base: int,
        num_atoms: int,
        num_heads: int,
        n_rff_features: int = 64,
        init_bandwidth: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_base = d_base
        self.num_atoms = num_atoms
        self.num_heads = num_heads
        self.n_rff = n_rff_features
        
        # Projections
        self.q_proj = nn.Linear(d_base, num_heads * d_base, bias=False)
        self.k_proj = nn.Linear(d_base, num_heads * d_base, bias=False)
        self.v_proj = nn.Linear(d_base, num_heads * d_base, bias=False)
        self.o_proj = nn.Linear(num_heads * d_base, d_base, bias=False)
        
        # Learnable bandwidth per head (for RBF kernel)
        self.log_bandwidth = nn.Parameter(
            torch.full((num_heads,), math.log(init_bandwidth))
        )
        
        # Learnable temperature for attention scores
        self.log_temperature = nn.Parameter(torch.full((num_heads,), math.log(5.0))) 
        
        # RFF frequencies
        self.register_buffer(
            'rff_freq_base',
            torch.randn(num_heads, d_base, n_rff_features)
        )
        
        self.rff_scale = 1.0 / math.sqrt(n_rff_features)
        
        # Weight update projection
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
        
        Returns: [B, H, S, 2*n_rff]
        """
        B, H, S, M, D = atoms.shape
        
        # Scale frequencies by bandwidth
        bandwidth = self.log_bandwidth.exp().view(H, 1, 1)  # [H, 1, 1]
        rff_freq = self.rff_freq_base / bandwidth  # [H, D, n_rff]
        
        # Project atoms: [B, H, S, M, D] @ [H, D, n_rff] -> [B, H, S, M, n_rff]
        proj = torch.einsum('bhsmd,hdf->bhsmf', atoms, rff_freq)
        
        # RFF features: φ(x) = [cos(ωᵀx), sin(ωᵀx)]
        phi = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        phi = phi * self.rff_scale
        
        # Weighted mean: φ̄ = Σᵢ wᵢ φ(aᵢ)
        w = weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, S, M, 1]
        rff_mean = (phi * w).sum(dim=3)  # [B, H, S, 2*n_rff]
        
        return rff_mean
    
    def forward(
        self,
        query: KMEState,
        key: KMEState,
        value: KMEState,
    ) -> KMEState:
        """
        KME attention - purely content-based, no position encoding here.
        """
        B, S_q, M, D = query.atoms.shape
        S_kv = key.atoms.shape[1]
        H = self.num_heads
        
        # Project atoms
        q_atoms = self.q_proj(query.atoms).view(B, S_q, M, H, D).permute(0, 3, 1, 2, 4)
        k_atoms = self.k_proj(key.atoms).view(B, S_kv, M, H, D).permute(0, 3, 1, 2, 4)
        v_atoms = self.v_proj(value.atoms).view(B, S_kv, M, H, D).permute(0, 3, 1, 2, 4)
        
        # RFF encode (approximate MMD)
        q_rff = self.rff_encode(q_atoms, query.weights)
        k_rff = self.rff_encode(k_atoms, key.weights)
        
        # Attention scores with learnable temperature
        # scores ∈ (0, 1] approximately, temperature controls sharpness
        temperature = self.log_temperature.exp().view(1, H, 1, 1)
        scores = torch.einsum('bhqf,bhkf->bhqk', q_rff, k_rff) * temperature
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Aggregate values
        out_atoms = torch.einsum('bhqk,bhkmd->bhqmd', attn_weights, v_atoms)
        out_atoms = out_atoms.permute(0, 2, 3, 1, 4).reshape(B, S_q, M, H * D)
        out_atoms = self.o_proj(out_atoms)
        
        # Weight updates
        log_weight_delta = self.w_proj(out_atoms.mean(dim=2))
        
        return KMEState(out_atoms, query.log_weights + log_weight_delta)


class GeneralKMEReasoningBlock(nn.Module):
    """Reasoning block without RoPE."""
    
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
        
        self.attn = GeneralKernelKMEAttention(
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
    ) -> KMEState:
        if context is None:
            context = state
        
        # Attention with residual
        attn_out = self.attn(query=state, key=context, value=context)
        state = KMEState(state.atoms + attn_out.atoms, attn_out.log_weights)
        state = self.norm1(state)
        
        # MLP
        state = self.mlp(state)
        state = self.norm2(state)
        
        return state


class GeneralKMEReasoningModule(nn.Module):
    """Stack of reasoning blocks without RoPE."""
    
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
            GeneralKMEReasoningBlock(
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
    ) -> KMEState:
        for layer in self.layers:
            state = layer(state, context=context)
        return state