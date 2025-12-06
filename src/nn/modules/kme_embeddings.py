"""
General KME Embedding for cross-domain generalization.

Key insight: Position is content, not structure.
The model should learn "this is X at position (r, c)" as a unified entity.

Supports:
- Sudoku (any size)
- Connect Four
- ARC-AGI
- Any 2D grid task
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn.modules.krm_block import KMEState


class FourierPositionEncoder(nn.Module):
    """
    Encode continuous coordinates using Fourier features.
    
    Like NeRF's positional encoding, but for 2D grid positions.
    Maps (x, y) ∈ [0, 1]² → high-dimensional features.
    
    This allows the model to learn functions of position at multiple scales.
    """
    
    def __init__(self, d_out: int, num_frequencies: int = 8, learnable: bool = False):
        super().__init__()
        self.d_out = d_out
        self.num_frequencies = num_frequencies
        
        # Frequencies: 2^0, 2^1, ..., 2^(L-1) for each coordinate
        # Output: [sin(2^0 πx), cos(2^0 πx), ..., sin(2^0 πy), cos(2^0 πy), ...]
        freqs = 2.0 ** torch.arange(num_frequencies).float() * math.pi
        
        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer('freqs', freqs)
        
        # Project from Fourier features to desired dimension
        fourier_dim = 2 * 2 * num_frequencies  # sin + cos, for x and y
        self.proj = nn.Linear(fourier_dim, d_out, bias=False)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [..., 2] normalized coordinates in [0, 1]
        Returns: [..., d_out]
        """
        # coords: [..., 2] -> [..., 2, num_freq]
        scaled = coords.unsqueeze(-1) * self.freqs  # [..., 2, num_freq]
        
        # Fourier features: [..., 2, num_freq, 2] for sin and cos
        fourier = torch.stack([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        
        # Flatten: [..., 2 * num_freq * 2]
        fourier = fourier.flatten(-3)
        
        return self.proj(fourier)


class DiscretePositionEncoder(nn.Module):
    """
    Encode discrete grid coordinates using learned embeddings.
    
    Simpler than Fourier, but requires max_coord to be known.
    Better for tasks where grid positions have distinct meanings.
    """
    
    def __init__(self, d_out: int, max_coord: int = 16):
        super().__init__()
        self.d_out = d_out
        self.max_coord = max_coord
        
        # Separate embeddings for row and column
        self.row_emb = nn.Embedding(max_coord, d_out // 2)
        self.col_emb = nn.Embedding(max_coord, d_out // 2)
        
        # Initialize with small values
        nn.init.normal_(self.row_emb.weight, std=0.5)
        nn.init.normal_(self.col_emb.weight, std=0.5)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [..., 2] integer coordinates
        Returns: [..., d_out]
        """
        row = coords[..., 0].clamp(0, self.max_coord - 1)
        col = coords[..., 1].clamp(0, self.max_coord - 1)
        
        row_enc = self.row_emb(row)  # [..., d_out//2]
        col_enc = self.col_emb(col)  # [..., d_out//2]
        
        return torch.cat([row_enc, col_enc], dim=-1)


class GeneralKMEEmbedding(nn.Module):
    """
    General-purpose KME embedding for 2D grid tasks.
    
    Combines:
    - Content embedding (what is at this position)
    - Position embedding (where is this position)
    
    Into unified atoms that can be compared across different grid sizes.
    
    Args:
        d_base: Base dimension for atoms
        num_atoms: Number of atoms per position
        vocab_size: Number of content tokens
        position_encoder: 'fourier' or 'discrete'
        max_coord: Maximum grid coordinate (for discrete encoder)
        num_frequencies: Number of Fourier frequencies (for fourier encoder)
        special_tokens: Dict mapping token names to ids (e.g., {'pad': 0, 'empty': 2})
    """
    
    def __init__(
        self,
        d_base: int,
        num_atoms: int,
        vocab_size: int,
        position_encoder: str = 'fourier',
        max_coord: int = 16,
        num_frequencies: int = 8,
        special_tokens: Optional[dict] = None,
    ):
        super().__init__()
        self.d_base = d_base
        self.num_atoms = num_atoms
        self.vocab_size = vocab_size
        self.position_encoder_type = position_encoder
        
        # Default special tokens for Sudoku-like tasks
        if special_tokens is None:
            special_tokens = {'pad': 0, 'eos': 1, 'empty': 2}
        self.special_tokens = special_tokens
        
        # Content dimension and position dimension
        d_content = d_base // 2
        d_position = d_base - d_content  # Handles odd d_base
        
        # Content encoding - orthogonal for generalization
        content_enc = self._create_orthogonal_encoding(vocab_size, d_content)
        self.register_buffer('content_encoding', content_enc)
        
        # Learnable special token overrides (can learn better representations)
        self.special_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.randn(d_content) * 0.5)
            for name in special_tokens.keys()
        })
        
        # Position encoder
        if position_encoder == 'fourier':
            self.pos_encoder = FourierPositionEncoder(
                d_out=d_position,
                num_frequencies=num_frequencies,
                learnable=True,
            )
        elif position_encoder == 'discrete':
            self.pos_encoder = DiscretePositionEncoder(
                d_out=d_position,
                max_coord=max_coord,
            )
        else:
            raise ValueError(f"Unknown position encoder: {position_encoder}")
        
        # Atom offsets - create diversity in the distribution
        self.atom_offsets = nn.Parameter(torch.randn(num_atoms, d_base) * 1.0)
        
        # Per-token log weights
        self.log_weights = nn.Parameter(torch.zeros(vocab_size, num_atoms))
    
    def _create_orthogonal_encoding(self, num_tokens: int, dim: int) -> torch.Tensor:
        """Create maximally separated encodings for tokens."""
        generator = torch.Generator().manual_seed(42)
        
        if num_tokens <= dim:
            # Use orthogonal vectors
            random_matrix = torch.randn(dim, dim, generator=generator)
            Q, _ = torch.linalg.qr(random_matrix)
            encoding = Q[:num_tokens]
        else:
            # More tokens than dimensions - use normalized random
            encoding = torch.randn(num_tokens, dim, generator=generator)
            encoding = F.normalize(encoding, dim=-1)
        
        return encoding
    
    def forward(
        self,
        content: torch.Tensor,                    # [B, S] token ids
        coords: torch.Tensor,                     # [B, S, 2] (row, col)
        grid_size: Optional[Tuple[int, int]] = None,  # (H, W) for normalization
    ) -> KMEState:
        """
        Embed content + coordinates into KME state.
        
        Args:
            content: Token ids [B, S]
            coords: Grid coordinates [B, S, 2], integers or floats
            grid_size: (height, width) for normalizing coordinates to [0, 1]
                       If None, coords are assumed already normalized or discrete
        
        Returns:
            KMEState with atoms [B, S, num_atoms, d_base]
        """
        B, S = content.shape
        device = content.device
        
        # === Content Encoding ===
        content_enc = self.content_encoding[content]  # [B, S, d_content]
        
        # Override special tokens with learnable embeddings
        for name, token_id in self.special_tokens.items():
            mask = (content == token_id)
            if mask.any():
                content_enc = torch.where(
                    mask.unsqueeze(-1),
                    self.special_embeddings[name].expand(B, S, -1),
                    content_enc
                )
        
        # === Position Encoding ===
        if self.position_encoder_type == 'fourier':
            # Normalize coordinates to [0, 1] if grid_size provided
            if grid_size is not None:
                H, W = grid_size
                norm_coords = coords.float()
                norm_coords = norm_coords / torch.tensor([H, W], device=device, dtype=torch.float)
            else:
                norm_coords = coords.float()
            pos_enc = self.pos_encoder(norm_coords)  # [B, S, d_position]
        else:
            # Discrete - use integer coordinates directly
            pos_enc = self.pos_encoder(coords.long())  # [B, S, d_position]
        
        # === Combine ===
        base = torch.cat([content_enc, pos_enc], dim=-1)  # [B, S, d_base]
        
        # === Create Atoms ===
        atoms = base.unsqueeze(2) + self.atom_offsets  # [B, S, M, d_base]
        
        # === Weights ===
        log_weights = F.embedding(content, self.log_weights)  # [B, S, M]
        
        return KMEState(atoms, log_weights)


class GeneralKMEEmbeddingWithPrefix(nn.Module):
    """
    GeneralKMEEmbedding with support for a learnable prefix (like puzzle embeddings).
    
    The prefix positions don't have coordinates - they're global context.
    """
    
    def __init__(
        self,
        d_base: int,
        num_atoms: int,
        vocab_size: int,
        prefix_len: int = 0,
        position_encoder: str = 'fourier',
        max_coord: int = 16,
        num_frequencies: int = 8,
        special_tokens: Optional[dict] = None,
    ):
        super().__init__()
        self.prefix_len = prefix_len
        
        # Main embedding
        self.embedding = GeneralKMEEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
            position_encoder=position_encoder,
            max_coord=max_coord,
            num_frequencies=num_frequencies,
            special_tokens=special_tokens,
        )
        
        # Prefix (if used)
        if prefix_len > 0:
            self.prefix_atoms = nn.Parameter(
                torch.randn(prefix_len, num_atoms, d_base) * 0.5
            )
            self.prefix_log_weights = nn.Parameter(
                torch.zeros(prefix_len, num_atoms)
            )
    
    def forward(
        self,
        content: torch.Tensor,
        coords: torch.Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
        prefix_scale: float = 1.0,
    ) -> KMEState:
        """
        Embed with optional prefix.
        
        Returns KMEState with shape [B, prefix_len + S, num_atoms, d_base]
        """
        B = content.shape[0]
        device = content.device
        
        # Main content + coords embedding
        main_state = self.embedding(content, coords, grid_size)
        
        if self.prefix_len > 0:
            # Expand prefix to batch
            prefix_atoms = self.prefix_atoms.unsqueeze(0).expand(B, -1, -1, -1)
            prefix_log_weights = self.prefix_log_weights.unsqueeze(0).expand(B, -1, -1)
            
            # Scale prefix (can be modulated by external signal)
            prefix_atoms = prefix_atoms * prefix_scale
            
            # Concatenate
            return KMEState(
                atoms=torch.cat([prefix_atoms, main_state.atoms], dim=1),
                log_weights=torch.cat([prefix_log_weights, main_state.log_weights], dim=1),
            )
        else:
            return main_state


def create_coords_for_grid(grid_size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    """
    Create coordinate tensor for a grid.
    
    Returns: [H*W, 2] tensor of (row, col) coordinates
    """
    H, W = grid_size
    rows = torch.arange(H, device=device).unsqueeze(1).expand(H, W)
    cols = torch.arange(W, device=device).unsqueeze(0).expand(H, W)
    coords = torch.stack([rows, cols], dim=-1)  # [H, W, 2]
    return coords.reshape(-1, 2)  # [H*W, 2]


def create_coords_batch(
    batch_size: int,
    grid_size: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Create coordinate tensor for a batch of grids.
    
    Returns: [B, H*W, 2]
    """
    coords = create_coords_for_grid(grid_size, device)
    return coords.unsqueeze(0).expand(batch_size, -1, -1)


# ============================================================
# Example usage and testing
# ============================================================

if __name__ == "__main__":
    # Test the embedding
    d_base = 32
    num_atoms = 8
    vocab_size = 12  # Sudoku: pad, eos, empty, 1-9
    
    emb = GeneralKMEEmbedding(
        d_base=d_base,
        num_atoms=num_atoms,
        vocab_size=vocab_size,
        position_encoder='fourier',
    )
    
    # Simulate 6x6 Sudoku
    B, H, W = 4, 6, 6
    S = H * W
    
    content = torch.randint(0, vocab_size, (B, S))
    coords = create_coords_batch(B, (H, W), content.device)
    
    state = emb(content, coords, grid_size=(H, W))
    
    print(f"Content shape: {content.shape}")
    print(f"Coords shape: {coords.shape}")
    print(f"Atoms shape: {state.atoms.shape}")
    print(f"Log weights shape: {state.log_weights.shape}")
    
    # Now test with 9x9 - should work without changes!
    H, W = 9, 9
    S = H * W
    
    content_9x9 = torch.randint(0, vocab_size, (B, S))
    coords_9x9 = create_coords_batch(B, (H, W), content.device)
    
    state_9x9 = emb(content_9x9, coords_9x9, grid_size=(H, W))
    
    print(f"\n9x9 test:")
    print(f"Content shape: {content_9x9.shape}")
    print(f"Coords shape: {coords_9x9.shape}")
    print(f"Atoms shape: {state_9x9.shape}")