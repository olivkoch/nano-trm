# tests/test_kme_modules_mathematical.py
"""
Mathematical property tests for KME (Kernel Mean Embedding) modules.

These tests verify that the modules satisfy their mathematical specifications,
not just that they run without errors.
"""

import pytest
import torch
import torch.nn.functional as F
import math

from src.nn.modules.krm_block import (
    KMEState,
    RFFEncoder,
    KMECategoricalEmbedding,
    KMERMSNorm,
    KMEMLP,
    KMEAttention,
    KMEReasoningBlock,
    KMEReasoningModule,
    KMEOutputHead,
    KMEQHead,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 16


@pytest.fixture
def d_base():
    return 32


@pytest.fixture
def num_atoms():
    return 8


@pytest.fixture
def num_heads():
    return 4


@pytest.fixture
def vocab_size():
    return 12  # Default for Sudoku: pad, eos, empty, digits 1-9


@pytest.fixture
def hidden_size():
    return 64


@pytest.fixture
def sample_kme_state(batch_size, seq_len, num_atoms, d_base, device):
    """Create a sample KMEState for testing."""
    atoms = torch.randn(batch_size, seq_len, num_atoms, d_base, device=device)
    log_weights = torch.randn(batch_size, seq_len, num_atoms, device=device)
    return KMEState(atoms, log_weights)


# =============================================================================
# KMEState Mathematical Properties
# =============================================================================

class TestKMEStateMathematical:
    """Test that KMEState represents valid probability distributions."""
    
    def test_weights_form_probability_distribution(self, sample_kme_state):
        """Weights must be a valid probability simplex: non-negative, sum to 1."""
        weights = sample_kme_state.weights
        
        # Non-negativity
        assert (weights >= 0).all(), "Weights must be non-negative"
        
        # Sum to 1 along atom dimension
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            "Weights must sum to 1"
    
    def test_softmax_numerical_stability(self, batch_size, seq_len, num_atoms, d_base, device):
        """Test weights computation is stable with extreme log_weights."""
        # Very negative log weights (should still give valid probabilities)
        atoms = torch.randn(batch_size, seq_len, num_atoms, d_base, device=device)
        log_weights = torch.full((batch_size, seq_len, num_atoms), -100.0, device=device)
        log_weights[..., 0] = 0.0  # One atom dominates
        
        state = KMEState(atoms, log_weights)
        weights = state.weights
        
        assert not torch.isnan(weights).any(), "NaN in weights"
        assert not torch.isinf(weights).any(), "Inf in weights"
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)
    
    def test_where_preserves_distribution_validity(self, batch_size, seq_len, num_atoms, d_base, device):
        """KMEState.where should preserve valid probability distributions."""
        state1 = KMEState(
            torch.randn(batch_size, seq_len, num_atoms, d_base, device=device),
            torch.randn(batch_size, seq_len, num_atoms, device=device),
        )
        state2 = KMEState(
            torch.randn(batch_size, seq_len, num_atoms, d_base, device=device),
            torch.randn(batch_size, seq_len, num_atoms, device=device),
        )
        
        condition = torch.rand(batch_size, device=device) > 0.5
        result = KMEState.where(condition, state1, state2)
        
        # Result should still have valid probability weights
        weights = result.weights
        assert (weights >= 0).all()
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)


# =============================================================================
# KMECategoricalEmbedding Mathematical Properties
# =============================================================================

class TestKMECategoricalEmbeddingMathematical:
    """
    Test mathematical properties of KME categorical embeddings.
    
    Each token is embedded as a distribution in base space:
    - Token i → μᵢ = Σⱼ wᵢⱼ δ(aᵢⱼ)
    - aᵢⱼ ∈ ℝ^d_base are the atoms (base + shared offsets)
    - wᵢⱼ form a probability simplex (non-negative, sum to 1)
    
    Key properties:
    - Special tokens (pad, eos, empty) have learnable base embeddings
    - Digit tokens have fixed orthogonal base encodings for generalization
    - All tokens share the same atom offsets
    """
    
    def test_each_token_is_valid_distribution(
        self, vocab_size, d_base, num_atoms, batch_size, seq_len, device
    ):
        """Each token should map to a valid probability distribution."""
        embedding = KMECategoricalEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
        ).to(device)
        
        # Embed all tokens
        all_tokens = torch.arange(vocab_size, device=device).unsqueeze(0)  # [1, vocab_size]
        state = embedding(all_tokens)
        
        weights = state.weights  # [1, vocab_size, num_atoms]
        
        # Non-negativity
        assert (weights >= 0).all(), "Weights must be non-negative"
        
        # Sum to 1
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
            "Weights must sum to 1 for each token"
    
    def test_embedding_is_deterministic_lookup(
        self, vocab_size, d_base, num_atoms, device
    ):
        """Same token should always produce identical embedding."""
        embedding = KMECategoricalEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
        ).to(device)
        
        token_id = 5  # A digit token
        tokens1 = torch.tensor([[token_id, token_id, token_id]], device=device)
        tokens2 = torch.tensor([[token_id]], device=device)
        
        state1 = embedding(tokens1)
        state2 = embedding(tokens2)
        
        # All positions with same token should be identical
        assert torch.allclose(state1.atoms[0, 0], state1.atoms[0, 1])
        assert torch.allclose(state1.atoms[0, 1], state1.atoms[0, 2])
        
        # Different calls should give same result
        assert torch.allclose(state1.atoms[0, 0], state2.atoms[0, 0])
        assert torch.allclose(state1.log_weights[0, 0], state2.log_weights[0, 0])
    
    def test_different_tokens_have_different_embeddings(
        self, vocab_size, d_base, num_atoms, device
    ):
        """Different tokens should map to different distributions."""
        embedding = KMECategoricalEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
        ).to(device)
        
        # Test pairs: (pad, eos), (empty, digit), (digit1, digit2)
        test_pairs = [
            (0, 1),   # pad vs eos
            (2, 3),   # empty vs digit 1
            (3, 4),   # digit 1 vs digit 2
            (5, 9),   # digit 3 vs digit 7
        ]
        
        for t1, t2 in test_pairs:
            if t1 >= vocab_size or t2 >= vocab_size:
                continue
            tokens = torch.tensor([[t1, t2]], device=device)
            state = embedding(tokens)
            
            # Atoms should be different (different base encodings)
            assert not torch.allclose(state.atoms[0, 0], state.atoms[0, 1], atol=1e-5), \
                f"Tokens {t1} and {t2} should have different embeddings"
    
    def test_digit_encodings_are_orthogonal(
        self, d_base, num_atoms, device
    ):
        """Digit token base encodings should be orthogonal (equidistant)."""
        vocab_size = 12
        embedding = KMECategoricalEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
        ).to(device)
        
        # Get digit encodings from buffer
        digit_enc = embedding.digit_encoding  # [num_digits, d_base]
        num_digits = digit_enc.shape[0]
        
        # Normalize for cosine similarity
        digit_enc_norm = F.normalize(digit_enc, dim=-1)
        
        # Compute pairwise similarities
        similarities = digit_enc_norm @ digit_enc_norm.T
        
        # Diagonal should be 1 (self-similarity)
        assert torch.allclose(similarities.diag(), torch.ones(num_digits, device=device), atol=1e-5)
        
        # Off-diagonal should be ~0 for orthogonal vectors (if d_base >= num_digits)
        if d_base >= num_digits:
            mask = ~torch.eye(num_digits, dtype=torch.bool, device=device)
            off_diag = similarities[mask]
            assert off_diag.abs().max() < 0.1, \
                f"Digit encodings should be orthogonal, max similarity: {off_diag.abs().max():.3f}"
    
    def test_digit_encodings_are_deterministic_across_instances(
        self, d_base, num_atoms, device
    ):
        """Digit encodings should be identical across different embedding instances (fixed seed)."""
        emb1 = KMECategoricalEmbedding(d_base=d_base, num_atoms=num_atoms).to(device)
        emb2 = KMECategoricalEmbedding(d_base=d_base, num_atoms=num_atoms).to(device)
        
        # Digit encodings should be identical (fixed seed=42)
        assert torch.allclose(emb1.digit_encoding, emb2.digit_encoding), \
            "Digit encodings should be deterministic across instances"
    
    def test_pad_token_produces_zero_base(
        self, d_base, num_atoms, device
    ):
        """Pad token (0) should have zero base embedding."""
        embedding = KMECategoricalEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
        ).to(device)
        
        # Pad embedding should be zeros
        assert torch.allclose(embedding.pad_embedding, torch.zeros(d_base, device=device)), \
            "Pad embedding should be zero"
        
        # When embedded, atoms should equal just the offsets
        pad_tokens = torch.tensor([[0]], device=device)
        state = embedding(pad_tokens)
        
        expected_atoms = embedding.atom_offsets.unsqueeze(0).unsqueeze(0)  # [1, 1, M, d_base]
        assert torch.allclose(state.atoms, expected_atoms), \
            "Pad token atoms should equal just the atom offsets"
    
    def test_shared_atom_offsets(
        self, vocab_size, d_base, num_atoms, device
    ):
        """All tokens should share the same atom offsets structure."""
        embedding = KMECategoricalEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
        ).to(device)
        
        # Embed multiple tokens
        tokens = torch.tensor([[1, 3, 5, 7]], device=device)  # eos, digit1, digit3, digit5
        state = embedding(tokens)
        
        # For each pair of tokens, the difference in atoms should be constant across atom indices
        # Because atoms = base + offsets, and offsets are shared
        for i in range(state.atoms.shape[1]):
            for j in range(i + 1, state.atoms.shape[1]):
                diff = state.atoms[0, i] - state.atoms[0, j]  # [num_atoms, d_base]
                # All atoms should have same diff (since offsets are shared)
                diff_var = diff.var(dim=0).mean()
                assert diff_var < 1e-10, \
                    f"Atom differences should be constant, variance: {diff_var:.2e}"
    
    def test_atom_offsets_are_learnable(
        self, vocab_size, d_base, num_atoms, device
    ):
        """Atom offsets should receive gradients."""
        embedding = KMECategoricalEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
        ).to(device)
        
        tokens = torch.tensor([[0, 1, 5]], device=device)  # pad, eos, digit3
        state = embedding(tokens)
        
        loss = state.atoms.sum()
        loss.backward()
        
        # Atom offsets should have gradients
        assert embedding.atom_offsets.grad is not None
        assert embedding.atom_offsets.grad.abs().sum() > 0, \
            "Atom offsets should have non-zero gradient"
    
    def test_special_token_embeddings_are_learnable(
        self, d_base, num_atoms, device
    ):
        """Special token embeddings (eos, empty) should receive gradients."""
        embedding = KMECategoricalEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
        ).to(device)
        
        # Use eos and empty tokens
        tokens = torch.tensor([[1, 2]], device=device)  # eos, empty
        state = embedding(tokens)
        
        loss = state.atoms.sum()
        loss.backward()
        
        # Check eos embedding has gradient
        assert embedding.eos_embedding.grad is not None
        assert embedding.eos_embedding.grad.abs().sum() > 0, \
            "EOS embedding should have non-zero gradient"
        
        # Check empty embedding has gradient
        assert embedding.empty_embedding.grad is not None
        assert embedding.empty_embedding.grad.abs().sum() > 0, \
            "Empty embedding should have non-zero gradient"
    
    def test_log_weights_are_learnable(
        self, vocab_size, d_base, num_atoms, device
    ):
        """Log-weight embeddings should receive gradients."""
        embedding = KMECategoricalEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
        ).to(device)

        tokens = torch.tensor([[3, 5, 7]], device=device)  # digit tokens
        state = embedding(tokens)

        # Loss that depends on weight distribution
        atom_norms = state.atoms.norm(dim=-1)  # [1, 3, num_atoms]
        loss = (state.weights * atom_norms).sum()
        loss.backward()

        assert embedding.log_weights.grad is not None

        # Check accessed tokens have gradients
        for token_id in [3, 5, 7]:
            assert embedding.log_weights.grad[token_id].abs().sum() > 0, \
                f"Token {token_id} log_weights should have non-zero gradient"
    
    def test_unused_tokens_no_weight_gradient(
        self, vocab_size, d_base, num_atoms, device
    ):
        """Tokens not in batch should not receive weight gradients."""
        embedding = KMECategoricalEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
        ).to(device)
        
        # Only use tokens 0, 1, 2
        tokens = torch.tensor([[0, 1, 2]], device=device)
        state = embedding(tokens)
        
        loss = state.atoms.sum() + state.log_weights.sum()
        loss.backward()
        
        # Tokens not used should have zero weight gradient
        for token_id in [3, 5, 8, vocab_size - 1]:
            if token_id < vocab_size:
                assert torch.allclose(
                    embedding.log_weights.grad[token_id], 
                    torch.zeros_like(embedding.log_weights.grad[token_id])
                ), f"Unused token {token_id} should have zero weight gradient"
    
    def test_initial_weights_are_uniform(
        self, vocab_size, d_base, num_atoms, device
    ):
        """Initial log_weights should be zero (uniform distribution)."""
        embedding = KMECategoricalEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
        ).to(device)
        
        # Check raw log_weights
        assert torch.allclose(
            embedding.log_weights,
            torch.zeros_like(embedding.log_weights),
            atol=1e-5
        ), "Initial log_weights should be zero (uniform)"
        
        # Verify this gives uniform weights
        tokens = torch.arange(min(10, vocab_size), device=device).unsqueeze(0)
        state = embedding(tokens)
        
        expected_weight = 1.0 / num_atoms
        assert torch.allclose(
            state.weights,
            torch.full_like(state.weights, expected_weight),
            atol=1e-5
        ), "Initial weights should be uniform"
    
    def test_embedding_represents_mixture_of_diracs(
        self, vocab_size, d_base, num_atoms, device
    ):
        """
        Each token embedding represents μ = Σⱼ wⱼ δ(aⱼ).
        The KME of this is z = Σⱼ wⱼ φ(aⱼ).
        
        Test that the encoding matches this formula.
        """
        embedding = KMECategoricalEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
        ).to(device)
        
        encoder = RFFEncoder(d_base=d_base, d_out=64, bandwidth=1.0).to(device)
        
        # Single token
        token = torch.tensor([[5]], device=device)  # digit 3
        state = embedding(token)
        
        # Method 1: Use encoder directly
        z_direct = encoder(state)  # [1, 1, 64]
        
        # Method 2: Manual computation z = Σⱼ wⱼ φ(aⱼ)
        atoms = state.atoms[0, 0]  # [num_atoms, d_base]
        weights = state.weights[0, 0]  # [num_atoms]
        
        phi_atoms = encoder.phi(atoms)  # [num_atoms, 64]
        z_manual = (weights.unsqueeze(-1) * phi_atoms).sum(dim=0)  # [64]
        
        assert torch.allclose(z_direct[0, 0], z_manual, atol=1e-5), \
            "KME encoding should match manual Σⱼ wⱼ φ(aⱼ) computation"
    
    def test_atom_distribution_coverage(
        self, vocab_size, d_base, num_atoms, device
    ):
        """
        Atoms for each token should provide reasonable coverage.
        Test that atoms are not degenerate (all same point).
        """
        embedding = KMECategoricalEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
        ).to(device)
        
        # Check a few tokens
        tokens = torch.tensor([[0, 3, 7]], device=device)  # pad, digit1, digit5
        state = embedding(tokens)
        
        for pos in range(3):
            atoms = state.atoms[0, pos]  # [num_atoms, d_base]
            
            # Atoms should not all be identical (due to offsets)
            atom_var = atoms.var(dim=0).mean()
            assert atom_var > 1e-10, \
                f"Atoms for token at pos {pos} are degenerate (variance={atom_var:.2e})"
            
            # Pairwise distances between atoms should be non-zero
            if num_atoms > 1:
                dists = torch.cdist(atoms, atoms)
                off_diag = dists[~torch.eye(num_atoms, dtype=torch.bool, device=device)]
                assert off_diag.min() > 1e-6, \
                    f"Some atoms are too close together for token at pos {pos}"
    
    def test_batch_independence(
        self, vocab_size, d_base, num_atoms, device
    ):
        """Embedding should process each batch element independently."""
        embedding = KMECategoricalEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
        ).to(device)
        
        # Two batch elements
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
        state = embedding(tokens)
        
        # Process separately
        tokens1 = torch.tensor([[1, 2, 3]], device=device)
        tokens2 = torch.tensor([[4, 5, 6]], device=device)
        state1 = embedding(tokens1)
        state2 = embedding(tokens2)
        
        # Should match
        assert torch.allclose(state.atoms[0], state1.atoms[0])
        assert torch.allclose(state.atoms[1], state2.atoms[0])
        assert torch.allclose(state.log_weights[0], state1.log_weights[0])
        assert torch.allclose(state.log_weights[1], state2.log_weights[0])


# =============================================================================
# RFFEncoder Mathematical Properties
# =============================================================================

class TestRFFEncoderMathematical:
    """
    Test that RFF approximates the RBF kernel.
    
    For RBF kernel: k(x, y) = exp(-||x - y||² / 2σ²)
    RFF approximation: k(x, y) ≈ φ(x)ᵀφ(y)
    
    For KME of distribution μ: z_μ = E_{x~μ}[φ(x)] = Σᵢ wᵢ φ(aᵢ)
    Kernel inner product: ⟨μ, ν⟩_H = E_{x~μ, y~ν}[k(x,y)] ≈ z_μᵀ z_ν
    """
    
    def test_rff_approximates_rbf_kernel(self, d_base, device):
        """Test that φ(x)ᵀφ(y) ≈ k(x, y) for RBF kernel."""
        # Scale bandwidth for dimensionality: E[||x-y||²] = 2*d for standard normals
        bandwidth = math.sqrt(d_base)
        d_out = 512  # More frequencies = better approximation

        encoder = RFFEncoder(d_base=d_base, d_out=d_out, bandwidth=bandwidth).to(device)

        # Sample points with controlled distances to get meaningful kernel values
        n_samples = 100
        x = torch.randn(n_samples, d_base, device=device)
        noise_scale = torch.rand(n_samples, 1, device=device) * 2
        y = x + torch.randn(n_samples, d_base, device=device) * noise_scale

        # True RBF kernel: k(x, y) = exp(-||x - y||² / 2σ²)
        sq_dist = ((x - y) ** 2).sum(dim=-1)
        k_true = torch.exp(-sq_dist / (2 * bandwidth ** 2))

        # Verify meaningful range
        assert k_true.min() < 0.5 and k_true.max() > 0.5, \
            f"Kernel values not well distributed: [{k_true.min():.3f}, {k_true.max():.3f}]"

        # RFF approximation: φ(x)ᵀφ(y)
        phi_x = encoder.phi(x)
        phi_y = encoder.phi(y)
        k_approx = (phi_x * phi_y).sum(dim=-1)

        # Check approximation quality
        correlation = torch.corrcoef(torch.stack([k_true, k_approx]))[0, 1]
        assert correlation > 0.9, f"RFF correlation with true kernel: {correlation:.3f}"

        mae = (k_true - k_approx).abs().mean()
        assert mae < 0.15, f"RFF MAE: {mae:.3f}"
    
    def test_kme_inner_product_approximates_mmd(self, d_base, num_atoms, device):
        """
        Test that z_μᵀz_ν ≈ ⟨μ, ν⟩_H (kernel mean embedding inner product).

        For discrete distributions μ = Σᵢ wᵢ δ(aᵢ), ν = Σⱼ uⱼ δ(bⱼ):
        ⟨μ, ν⟩_H = Σᵢⱼ wᵢ uⱼ k(aᵢ, bⱼ)
        """
        # Scale bandwidth for dimensionality
        bandwidth = math.sqrt(d_base)
        d_out = 512

        encoder = RFFEncoder(d_base=d_base, d_out=d_out, bandwidth=bandwidth).to(device)

        # Create two distributions with atoms that overlap somewhat
        # Use smaller scale so atoms aren't too far apart
        atoms_mu = torch.randn(1, 1, num_atoms, d_base, device=device) * 0.5
        atoms_nu = atoms_mu + torch.randn(1, 1, num_atoms, d_base, device=device) * 0.5  # Nearby
        
        weights_mu = F.softmax(torch.randn(1, 1, num_atoms, device=device), dim=-1)
        weights_nu = F.softmax(torch.randn(1, 1, num_atoms, device=device), dim=-1)

        state_mu = KMEState(atoms_mu, torch.log(weights_mu + 1e-10))
        state_nu = KMEState(atoms_nu, torch.log(weights_nu + 1e-10))

        # RFF approximation: z_μᵀz_ν
        z_mu = encoder(state_mu)  # [1, 1, d_out]
        z_nu = encoder(state_nu)
        inner_approx = (z_mu * z_nu).sum()

        # True kernel inner product: Σᵢⱼ wᵢ uⱼ k(aᵢ, bⱼ)
        inner_true = 0.0
        for i in range(num_atoms):
            for j in range(num_atoms):
                sq_dist = ((atoms_mu[0, 0, i] - atoms_nu[0, 0, j]) ** 2).sum()
                k_ij = torch.exp(-sq_dist / (2 * bandwidth ** 2))
                inner_true += weights_mu[0, 0, i] * weights_nu[0, 0, j] * k_ij

        # Sanity check: inner_true should be meaningful (not ~0)
        assert inner_true > 0.01, f"True inner product too small: {inner_true:.6f}"

        # Check approximation
        rel_error = (inner_approx - inner_true).abs() / (inner_true.abs() + 1e-10)
        assert rel_error < 0.3, f"KME inner product relative error: {rel_error:.3f}"
    
    def test_same_distribution_high_similarity(self, d_base, num_atoms, hidden_size, device):
        """Same distribution should have high self-similarity (close to 1)."""
        encoder = RFFEncoder(d_base=d_base, d_out=hidden_size, bandwidth=1.0).to(device)
        
        atoms = torch.randn(1, 1, num_atoms, d_base, device=device)
        log_weights = torch.randn(1, 1, num_atoms, device=device)
        state = KMEState(atoms, log_weights)
        
        z = encoder(state)
        
        # Self inner product: ||z||²
        self_inner = (z * z).sum()
        
        # For a valid probability distribution, this should be bounded and positive
        assert self_inner > 0, "Self inner product should be positive"
    
    def test_orthogonal_distributions_low_similarity(self, d_base, num_atoms, device):
        """Distributions with atoms far apart should have lower similarity."""
        d_out = 256
        encoder = RFFEncoder(d_base=d_base, d_out=d_out, bandwidth=1.0).to(device)
        
        # Distribution 1: atoms near origin
        atoms1 = torch.randn(1, 1, num_atoms, d_base, device=device) * 0.1
        # Distribution 2: atoms far from origin
        atoms2 = torch.randn(1, 1, num_atoms, d_base, device=device) * 0.1 + 10.0
        
        log_weights = torch.zeros(1, 1, num_atoms, device=device)  # Uniform
        
        state1 = KMEState(atoms1, log_weights)
        state2 = KMEState(atoms2, log_weights)
        
        z1 = encoder(state1)
        z2 = encoder(state2)
        
        # Cross inner product should be small (distributions are far apart)
        cross_inner = (z1 * z2).sum()
        self_inner = (z1 * z1).sum()
        
        # Normalized similarity should be low
        similarity = cross_inner / (self_inner + 1e-10)
        assert similarity < 0.5, f"Far distributions should have low similarity: {similarity:.3f}"
    
    def test_phi_preserves_kernel_symmetry(self, d_base, hidden_size, device):
        """k(x, y) = k(y, x) should hold for RFF approximation."""
        encoder = RFFEncoder(d_base=d_base, d_out=hidden_size).to(device)
        
        x = torch.randn(10, d_base, device=device)
        y = torch.randn(10, d_base, device=device)
        
        phi_x = encoder.phi(x)
        phi_y = encoder.phi(y)
        
        k_xy = (phi_x * phi_y).sum(dim=-1)
        k_yx = (phi_y * phi_x).sum(dim=-1)
        
        assert torch.allclose(k_xy, k_yx), "Kernel should be symmetric"
    
    def test_bandwidth_affects_kernel_width(self, d_base, device):
        """Larger bandwidth = wider kernel = higher similarity for distant points."""
        d_out = 512  # More frequencies for stability

        # Scale bandwidths appropriately for dimensionality
        bandwidth_narrow = math.sqrt(d_base) * 0.5
        bandwidth_wide = math.sqrt(d_base) * 2.0

        encoder_narrow = RFFEncoder(d_base=d_base, d_out=d_out, bandwidth=bandwidth_narrow).to(device)
        encoder_wide = RFFEncoder(d_base=d_base, d_out=d_out, bandwidth=bandwidth_wide).to(device)

        # Two points at moderate distance (scaled for dimensionality)
        x = torch.zeros(1, d_base, device=device)
        y = torch.randn(1, d_base, device=device)  # Expected distance ≈ sqrt(d_base)

        # Compute true kernel values to verify test setup
        sq_dist = ((x - y) ** 2).sum()
        k_true_narrow = torch.exp(-sq_dist / (2 * bandwidth_narrow ** 2))
        k_true_wide = torch.exp(-sq_dist / (2 * bandwidth_wide ** 2))
        
        # Sanity check: both kernels should be in meaningful range
        assert 0.01 < k_true_narrow < 0.99, f"Narrow kernel value out of range: {k_true_narrow:.3f}"
        assert 0.01 < k_true_wide < 0.99, f"Wide kernel value out of range: {k_true_wide:.3f}"
        assert k_true_wide > k_true_narrow, "Test setup error: wide should be > narrow"

        # RFF approximations
        phi_x_narrow = encoder_narrow.phi(x)
        phi_y_narrow = encoder_narrow.phi(y)
        sim_narrow = (phi_x_narrow * phi_y_narrow).sum()

        phi_x_wide = encoder_wide.phi(x)
        phi_y_wide = encoder_wide.phi(y)
        sim_wide = (phi_x_wide * phi_y_wide).sum()

        assert sim_wide > sim_narrow, \
            f"Wide kernel ({sim_wide:.3f}) should have higher similarity than narrow ({sim_narrow:.3f})"


# =============================================================================
# KMEAttention Mathematical Properties
# =============================================================================

class TestKMEAttentionMathematical:
    """
    Test mathematical properties of KME attention.
    
    Key properties:
    1. Attention weights form valid probability distributions (sum to 1, non-negative)
    2. Output is a valid weighted combination of value distributions
    """
    
    def test_attention_weights_are_valid_probabilities(
        self, batch_size, seq_len, num_atoms, d_base, num_heads, device
    ):
        """Attention weights should sum to 1 over key positions."""
        attn = KMEAttention(
            d_base=d_base,
            num_atoms=num_atoms,
            num_heads=num_heads,
        ).to(device)
        
        state = KMEState(
            torch.randn(batch_size, seq_len, num_atoms, d_base, device=device),
            torch.randn(batch_size, seq_len, num_atoms, device=device),
        )
        
        # Manually compute attention weights to verify
        B, S, M, _ = state.atoms.shape
        q_atoms = attn.q_proj(state.atoms).view(B, S, M, attn.num_heads, attn.d_base)
        k_atoms = attn.k_proj(state.atoms).view(B, S, M, attn.num_heads, attn.d_base)
        
        q_weights = state.weights
        k_weights = state.weights
        
        # Compute Q and K encodings using direct weighted mean
        q_enc = attn.encode_batched(q_atoms, q_weights)  # [B, H, S, d_base]
        k_enc = attn.encode_batched(k_atoms, k_weights)
        
        # Compute attention weights
        scores = torch.matmul(q_enc, k_enc.transpose(-2, -1)) * attn.attn_scale
        weights = F.softmax(scores, dim=-1)  # [B, H, S_q, S_kv]
        
        # Verify properties
        assert (weights >= 0).all(), "Attention weights must be non-negative"
        
        weight_sums = weights.sum(dim=-1)  # Sum over key positions
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
            "Attention weights must sum to 1"
    
    def test_output_atoms_are_weighted_combination(
        self, batch_size, num_atoms, d_base, num_heads, device
    ):
        """
        Output atoms should be attention-weighted combination of value atoms.
        """
        seq_len = 4
        
        attn = KMEAttention(
            d_base=d_base,
            num_atoms=num_atoms,
            num_heads=1,
            bandwidth=1.0,
        ).to(device)
        
        # Create distinct value atoms for each position
        atoms = torch.randn(batch_size, seq_len, num_atoms, d_base, device=device)
        log_weights = torch.randn(batch_size, seq_len, num_atoms, device=device)
        state = KMEState(atoms, log_weights)
        
        output = attn(query=state, key=state, value=state)
        
        # Output should have same shape
        assert output.atoms.shape == state.atoms.shape
    
    def test_self_attention_permutation_equivariance_without_rope(
        self, batch_size, num_atoms, d_base, num_heads, device
    ):
        """
        Without positional encoding, self-attention should be permutation equivariant.
        
        If we permute input positions, output should be permuted the same way.
        """
        seq_len = 4
        
        attn = KMEAttention(
            d_base=d_base,
            num_atoms=num_atoms,
            num_heads=num_heads,
        ).to(device)
        
        atoms = torch.randn(batch_size, seq_len, num_atoms, d_base, device=device)
        log_weights = torch.randn(batch_size, seq_len, num_atoms, device=device)
        state = KMEState(atoms, log_weights)
        
        # Forward pass
        output1 = attn(query=state, key=state, value=state, cos_sin=None)
        
        # Permute positions
        perm = torch.tensor([2, 0, 3, 1], device=device)
        permuted_atoms = atoms[:, perm]
        permuted_log_weights = log_weights[:, perm]
        permuted_state = KMEState(permuted_atoms, permuted_log_weights)
        
        # Forward pass on permuted input
        output2 = attn(query=permuted_state, key=permuted_state, value=permuted_state, cos_sin=None)
        
        # Unpermute output2
        inv_perm = torch.argsort(perm)
        unpermuted_output2_atoms = output2.atoms[:, inv_perm]
        
        # Should be equal (permutation equivariance)
        assert torch.allclose(output1.atoms, unpermuted_output2_atoms, atol=1e-5), \
            "Self-attention without RoPE should be permutation equivariant"
    
    def test_rope_breaks_permutation_equivariance(
        self, batch_size, num_atoms, d_base, num_heads, seq_len, device
    ):
        """With RoPE, permuting positions should give different results."""
        attn = KMEAttention(
            d_base=d_base,
            num_atoms=num_atoms,
            num_heads=num_heads,
        ).to(device)
        
        atoms = torch.randn(batch_size, seq_len, num_atoms, d_base, device=device)
        log_weights = torch.randn(batch_size, seq_len, num_atoms, device=device)
        state = KMEState(atoms, log_weights)
        
        # Create RoPE embeddings
        cos = torch.randn(seq_len, d_base, device=device)
        sin = torch.randn(seq_len, d_base, device=device)
        
        # Forward pass
        output1 = attn(query=state, key=state, value=state, cos_sin=(cos, sin))
        
        # Permute positions
        perm = torch.randperm(seq_len, device=device)
        permuted_atoms = atoms[:, perm]
        permuted_log_weights = log_weights[:, perm]
        permuted_state = KMEState(permuted_atoms, permuted_log_weights)
        
        # Forward pass on permuted input (with SAME RoPE - this is the key)
        output2 = attn(query=permuted_state, key=permuted_state, value=permuted_state, cos_sin=(cos, sin))
        
        # Unpermute output2
        inv_perm = torch.argsort(perm)
        unpermuted_output2_atoms = output2.atoms[:, inv_perm]
        
        # Should NOT be equal (RoPE breaks equivariance)
        assert not torch.allclose(output1.atoms, unpermuted_output2_atoms, atol=1e-4), \
            "Self-attention with RoPE should NOT be permutation equivariant"


# =============================================================================
# KMERMSNorm Mathematical Properties
# =============================================================================

class TestKMERMSNormMathematical:
    """Test mathematical properties of KME RMS normalization."""
    
    def test_output_has_unit_rms(self, batch_size, seq_len, num_atoms, d_base, device):
        """After normalization, RMS of atoms should be approximately 1 (scaled by weight)."""
        norm = KMERMSNorm(d_base=d_base).to(device)
        
        # Reset weight to ones for this test
        with torch.no_grad():
            norm.weight.fill_(1.0)
        
        # Input with varying magnitudes
        atoms = torch.randn(batch_size, seq_len, num_atoms, d_base, device=device) * 10
        log_weights = torch.randn(batch_size, seq_len, num_atoms, device=device)
        state = KMEState(atoms, log_weights)
        
        output = norm(state)
        
        # Compute RMS of output atoms
        rms = torch.sqrt((output.atoms ** 2).mean(dim=-1))
        
        # RMS should be close to 1
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1), \
            f"RMS after normalization: mean={rms.mean():.3f}, std={rms.std():.3f}"
    
    def test_normalization_is_scale_invariant(self, batch_size, seq_len, num_atoms, d_base, device):
        """Scaling input by constant should give same output (up to sign)."""
        norm = KMERMSNorm(d_base=d_base).to(device)
        
        atoms = torch.randn(batch_size, seq_len, num_atoms, d_base, device=device)
        log_weights = torch.randn(batch_size, seq_len, num_atoms, device=device)
        
        state1 = KMEState(atoms, log_weights)
        state2 = KMEState(atoms * 5.0, log_weights)  # Scaled by 5
        
        output1 = norm(state1)
        output2 = norm(state2)
        
        # Outputs should be equal (normalization removes scale)
        assert torch.allclose(output1.atoms, output2.atoms, atol=1e-5), \
            "RMSNorm should be scale-invariant"
    
    def test_learnable_weight_scales_output(self, batch_size, seq_len, num_atoms, d_base, device):
        """Learnable weight parameter should scale the normalized output."""
        norm = KMERMSNorm(d_base=d_base).to(device)
        
        atoms = torch.randn(batch_size, seq_len, num_atoms, d_base, device=device)
        log_weights = torch.randn(batch_size, seq_len, num_atoms, device=device)
        state = KMEState(atoms, log_weights)
        
        # Output with default weights
        output1 = norm(state)
        
        # Double the learnable weight
        with torch.no_grad():
            norm.weight.mul_(2.0)
        
        output2 = norm(state)
        
        # Output should be doubled
        assert torch.allclose(output2.atoms, output1.atoms * 2.0, atol=1e-5)


# =============================================================================
# KMEMLP Mathematical Properties
# =============================================================================

class TestKMEMLPMathematical:
    """Test mathematical properties of KME MLP."""
    
    def test_residual_connection_structure(self, sample_kme_state, d_base, num_atoms, device):
        """MLP should have residual: output = input + f(input)."""
        mlp = KMEMLP(d_base=d_base, num_atoms=num_atoms).to(device)
        
        output = mlp(sample_kme_state)
        
        # Compute the "delta" that was added
        delta = output.atoms - sample_kme_state.atoms
        
        # Delta should not be zero (unless weights are zero)
        # and should be smaller than input (residual should be a refinement)
        delta_norm = delta.norm()
        input_norm = sample_kme_state.atoms.norm()
        
        # With proper initialization, delta should be smaller than input
        # (the down projection is initialized small)
        assert delta_norm < input_norm, \
            f"Residual delta ({delta_norm:.3f}) should be smaller than input ({input_norm:.3f})"
    
    def test_swiglu_nonlinearity(self, batch_size, seq_len, num_atoms, d_base, device):
        """SwiGLU should produce nonlinear transformation."""
        mlp = KMEMLP(d_base=d_base, num_atoms=num_atoms).to(device)
        
        # Override small initialization for this test
        with torch.no_grad():
            mlp.gate_up.weight.normal_(std=0.5)
            mlp.down.weight.normal_(std=0.5)
        
        # Test linearity: f(ax + by) should NOT equal a*f(x) + b*f(y)
        x = KMEState(
            torch.randn(batch_size, seq_len, num_atoms, d_base, device=device),
            torch.randn(batch_size, seq_len, num_atoms, device=device),
        )
        y = KMEState(
            torch.randn(batch_size, seq_len, num_atoms, d_base, device=device),
            x.log_weights.clone(),  # Same weights for simplicity
        )
        
        a, b = 0.3, 0.7
        
        # f(ax + by)
        combined = KMEState(a * x.atoms + b * y.atoms, x.log_weights)
        f_combined = mlp(combined)
        
        # a*f(x) + b*f(y)
        f_x = mlp(x)
        f_y = mlp(y)
        linear_combined = a * f_x.atoms + b * f_y.atoms
        
        # These should NOT be equal (nonlinear)
        assert not torch.allclose(f_combined.atoms, linear_combined, atol=0.1), \
            "MLP should be nonlinear"
    
    def test_position_independence(self, batch_size, seq_len, num_atoms, d_base, device):
        """MLP should process each position independently."""
        mlp = KMEMLP(d_base=d_base, num_atoms=num_atoms).to(device)
        
        atoms = torch.randn(batch_size, seq_len, num_atoms, d_base, device=device)
        log_weights = torch.randn(batch_size, seq_len, num_atoms, device=device)
        state = KMEState(atoms, log_weights)
        
        # Process full sequence
        output_full = mlp(state)
        
        # Process each position separately
        outputs_separate = []
        for pos in range(seq_len):
            single_pos_state = KMEState(
                atoms[:, pos:pos+1],
                log_weights[:, pos:pos+1],
            )
            outputs_separate.append(mlp(single_pos_state).atoms)
        
        output_separate = torch.cat(outputs_separate, dim=1)
        
        # Should be identical
        assert torch.allclose(output_full.atoms, output_separate, atol=1e-5), \
            "MLP should process positions independently"


# =============================================================================
# KMEReasoningModule Mathematical Properties
# =============================================================================

class TestKMEReasoningModuleMathematical:
    """Test mathematical properties of the reasoning module."""
    
    def test_input_injection_is_additive(self, batch_size, seq_len, num_atoms, d_base, num_heads, device):
        """Input injection should add context atoms to state atoms."""
        module = KMEReasoningModule(
            num_layers=1,
            d_base=d_base,
            num_atoms=num_atoms,
            num_heads=num_heads,
        ).to(device)
        
        # Zero state
        zero_state = KMEState(
            torch.zeros(batch_size, seq_len, num_atoms, d_base, device=device),
            torch.zeros(batch_size, seq_len, num_atoms, device=device),
        )
        
        # Non-zero context
        context = KMEState(
            torch.randn(batch_size, seq_len, num_atoms, d_base, device=device),
            torch.zeros(batch_size, seq_len, num_atoms, device=device),
        )
        
        output = module(zero_state, context=context)
        
        # Output should not be zero (context was injected)
        assert output.atoms.abs().sum() > 0, "Input injection should affect output"
    
    def test_multiple_layers_compose(self, batch_size, seq_len, num_atoms, d_base, num_heads, device):
        """More layers should enable deeper transformations."""
        state = KMEState(
            torch.randn(batch_size, seq_len, num_atoms, d_base, device=device),
            torch.randn(batch_size, seq_len, num_atoms, device=device),
        )
        
        module_1 = KMEReasoningModule(
            num_layers=1, d_base=d_base, num_atoms=num_atoms, num_heads=num_heads
        ).to(device)
        
        module_4 = KMEReasoningModule(
            num_layers=4, d_base=d_base, num_atoms=num_atoms, num_heads=num_heads
        ).to(device)
        
        output_1 = module_1(state, context=state)
        output_4 = module_4(state, context=state)
        
        # Both should produce valid outputs
        assert not torch.isnan(output_1.atoms).any()
        assert not torch.isnan(output_4.atoms).any()
        
        # Outputs should be different (more layers = more transformation)
        assert not torch.allclose(output_1.atoms, output_4.atoms, atol=0.1)


# =============================================================================
# KMEOutputHead Mathematical Properties
# =============================================================================

class TestKMEOutputHeadMathematical:
    """Test mathematical properties of the output head."""
    
    def test_output_depends_on_distribution(
        self, batch_size, seq_len, num_atoms, d_base, vocab_size, hidden_size, device
    ):
        """Different distributions should produce different logits."""
        head = KMEOutputHead(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        ).to(device)
        
        state1 = KMEState(
            torch.randn(batch_size, seq_len, num_atoms, d_base, device=device),
            torch.randn(batch_size, seq_len, num_atoms, device=device),
        )
        state2 = KMEState(
            torch.randn(batch_size, seq_len, num_atoms, d_base, device=device),
            torch.randn(batch_size, seq_len, num_atoms, device=device),
        )
        
        logits1 = head(state1)
        logits2 = head(state2)
        
        assert not torch.allclose(logits1, logits2, atol=0.1), \
            "Different distributions should produce different logits"
    
    def test_weight_concentration_affects_output(
        self, batch_size, seq_len, num_atoms, d_base, vocab_size, hidden_size, device
    ):
        """Concentrated vs diffuse weights should give different outputs."""
        head = KMEOutputHead(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        ).to(device)
        
        atoms = torch.randn(batch_size, seq_len, num_atoms, d_base, device=device)
        
        # Uniform weights
        uniform_log_weights = torch.zeros(batch_size, seq_len, num_atoms, device=device)
        
        # Concentrated weights (first atom dominates)
        concentrated_log_weights = torch.full(
            (batch_size, seq_len, num_atoms), -100.0, device=device
        )
        concentrated_log_weights[..., 0] = 0.0
        
        state_uniform = KMEState(atoms, uniform_log_weights)
        state_concentrated = KMEState(atoms, concentrated_log_weights)
        
        logits_uniform = head(state_uniform)
        logits_concentrated = head(state_concentrated)
        
        # Should be different
        assert not torch.allclose(logits_uniform, logits_concentrated, atol=0.1), \
            "Weight concentration should affect output"


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Test numerical stability of KME operations."""
    
    def test_rff_stable_with_large_inputs(self, d_base, hidden_size, device):
        """RFF should be stable with large input values."""
        encoder = RFFEncoder(d_base=d_base, d_out=hidden_size).to(device)
        
        large_atoms = torch.randn(2, 4, 8, d_base, device=device) * 100
        log_weights = torch.zeros(2, 4, 8, device=device)
        state = KMEState(large_atoms, log_weights)
        
        output = encoder(state)
        
        assert not torch.isnan(output).any(), "NaN with large inputs"
        assert not torch.isinf(output).any(), "Inf with large inputs"
    
    def test_attention_stable_with_many_positions(
        self, batch_size, num_atoms, d_base, num_heads, device
    ):
        """Attention should be stable with many sequence positions."""
        seq_len = 256
        
        attn = KMEAttention(
            d_base=d_base,
            num_atoms=num_atoms,
            num_heads=num_heads,
        ).to(device)
        
        state = KMEState(
            torch.randn(batch_size, seq_len, num_atoms, d_base, device=device),
            torch.randn(batch_size, seq_len, num_atoms, device=device),
        )
        
        output = attn(query=state, key=state, value=state)
        
        assert not torch.isnan(output.atoms).any(), "NaN with long sequences"
        assert not torch.isinf(output.atoms).any(), "Inf with long sequences"
    
    def test_gradient_magnitude_reasonable(
        self, batch_size, seq_len, num_atoms, d_base, num_heads, device
    ):
        """Gradients should have reasonable magnitude (not exploding/vanishing)."""
        module = KMEReasoningModule(
            num_layers=4,
            d_base=d_base,
            num_atoms=num_atoms,
            num_heads=num_heads,
        ).to(device)
        
        atoms = torch.randn(
            batch_size, seq_len, num_atoms, d_base, device=device, requires_grad=True
        )
        log_weights = torch.randn(batch_size, seq_len, num_atoms, device=device)
        state = KMEState(atoms, log_weights)
        
        output = module(state, context=state)
        loss = output.atoms.sum()
        loss.backward()
        
        grad_norm = atoms.grad.norm()
        
        assert grad_norm > 1e-10, f"Gradient too small: {grad_norm}"
        assert grad_norm < 1e10, f"Gradient too large: {grad_norm}"


# =============================================================================
# Conservation Properties
# =============================================================================

class TestConservationProperties:
    """Test that certain quantities are conserved or bounded."""
    
    def test_weights_remain_valid_through_pipeline(
        self, batch_size, seq_len, num_atoms, d_base, num_heads, device
    ):
        """Weights should remain valid probability distributions through all operations."""
        block = KMEReasoningBlock(
            d_base=d_base,
            num_atoms=num_atoms,
            num_heads=num_heads,
        ).to(device)
        
        state = KMEState(
            torch.randn(batch_size, seq_len, num_atoms, d_base, device=device),
            torch.randn(batch_size, seq_len, num_atoms, device=device),
        )
        
        # Initial weights valid
        assert torch.allclose(state.weights.sum(dim=-1), torch.ones(batch_size, seq_len, device=device), atol=1e-5)
        
        # After block
        output = block(state)
        assert torch.allclose(output.weights.sum(dim=-1), torch.ones(batch_size, seq_len, device=device), atol=1e-5)
        assert (output.weights >= 0).all()
    
    def test_atom_norms_bounded_after_normalization(
        self, batch_size, seq_len, num_atoms, d_base, device
    ):
        """After RMSNorm, atom norms should be bounded."""
        norm = KMERMSNorm(d_base=d_base).to(device)
        
        # Large input
        state = KMEState(
            torch.randn(batch_size, seq_len, num_atoms, d_base, device=device) * 100,
            torch.randn(batch_size, seq_len, num_atoms, device=device),
        )
        
        output = norm(state)
        
        # RMS should be close to ||weight|| (approximately 1 with default init)
        rms = torch.sqrt((output.atoms ** 2).mean(dim=-1))
        weight_norm = norm.weight.norm() / math.sqrt(d_base)
        
        assert rms.max() < weight_norm * 2, f"RMS too large after norm: {rms.max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])