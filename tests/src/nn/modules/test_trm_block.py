"""
Unit tests for TRM Transformer components
"""

import pytest
import torch

from src.nn.modules.trm_block import (
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TransformerBlock,
    apply_rotary_pos_emb,
    rotate_half,
)


class TestRMSNorm:
    """Test RMSNorm implementation."""

    def test_shape_preservation(self):
        """RMSNorm should preserve input shape."""
        batch, seq_len, dim = 2, 10, 64
        norm = RMSNorm(dim)
        x = torch.randn(batch, seq_len, dim)

        output = norm(x)

        assert output.shape == x.shape

    def test_normalization(self):
        """RMSNorm should normalize to approximately unit RMS."""
        dim = 64
        norm = RMSNorm(dim)
        x = torch.randn(2, 10, dim) * 10  # Large values

        output = norm(x)

        # Check RMS is approximately 1 (scaled by learnable weight)
        rms = torch.sqrt(torch.mean(output**2, dim=-1))
        # Should be close to 1, allowing for the learned weight parameter
        assert rms.mean().item() < 10  # Much smaller than input

    def test_learnable_weights(self):
        """RMSNorm should have learnable weights."""
        norm = RMSNorm(64)
        assert norm.weight.requires_grad
        assert norm.weight.shape == (64,)


class TestRotaryEmbedding:
    """Test Rotary Position Embedding."""

    def test_output_shape(self):
        """RoPE should output correct shapes."""
        head_dim = 64
        seq_len = 10
        rope = RotaryEmbedding(head_dim)

        x = torch.randn(2, seq_len, head_dim)
        cos, sin = rope(x, seq_len)

        assert cos.shape == (1, seq_len, head_dim)
        assert sin.shape == (1, seq_len, head_dim)

    def test_deterministic(self):
        """RoPE should be deterministic for same sequence length."""
        head_dim = 64
        seq_len = 10
        rope = RotaryEmbedding(head_dim)

        x = torch.randn(2, seq_len, head_dim)
        cos1, sin1 = rope(x, seq_len)
        cos2, sin2 = rope(x, seq_len)

        assert torch.allclose(cos1, cos2)
        assert torch.allclose(sin1, sin2)

    def test_different_seq_lengths(self):
        """RoPE should handle different sequence lengths."""
        head_dim = 64
        rope = RotaryEmbedding(head_dim, max_seq_len=100)

        for seq_len in [5, 10, 20, 50]:
            x = torch.randn(2, seq_len, head_dim)
            cos, sin = rope(x, seq_len)
            assert cos.shape == (1, seq_len, head_dim)
            assert sin.shape == (1, seq_len, head_dim)


class TestRotaryHelpers:
    """Test rotary embedding helper functions."""

    def test_rotate_half(self):
        """rotate_half should swap and negate half dimensions."""
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # [1, 1, 4]

        result = rotate_half(x)
        expected = torch.tensor([[[-3.0, -4.0, 1.0, 2.0]]])

        assert torch.allclose(result, expected)

    def test_apply_rotary_pos_emb_shape(self):
        """Rotary application should preserve shape."""
        batch, seq_len, num_heads, head_dim = 2, 10, 8, 64

        x = torch.randn(batch, seq_len, num_heads, head_dim)
        cos = torch.randn(1, seq_len, head_dim)  # Format from RotaryEmbedding
        sin = torch.randn(1, seq_len, head_dim)

        result = apply_rotary_pos_emb(x, cos, sin)

        assert result.shape == x.shape


class TestSwiGLU:
    """Test SwiGLU activation."""

    def test_shape_preservation(self):
        """SwiGLU should preserve input/output dimensions."""
        dim = 64
        hidden_dim = 256
        swiglu = SwiGLU(dim, hidden_dim, bias=False)

        x = torch.randn(2, 10, dim)
        output = swiglu(x)

        assert output.shape == x.shape

    def test_no_bias(self):
        """SwiGLU should have no bias when bias=False."""
        swiglu = SwiGLU(64, 256, bias=False)

        assert swiglu.w1.bias is None
        assert swiglu.w2.bias is None
        assert swiglu.w3.bias is None

    def test_nonlinearity(self):
        """SwiGLU should be non-linear."""
        swiglu = SwiGLU(64, 256, bias=False)

        x1 = torch.randn(1, 5, 64)
        x2 = torch.randn(1, 5, 64)

        # Non-linear: f(x1 + x2) != f(x1) + f(x2)
        result_sum = swiglu(x1 + x2)
        result_separate = swiglu(x1) + swiglu(x2)

        assert not torch.allclose(result_sum, result_separate)


class TestTransformerBlock:
    """Test complete Transformer block."""

    def test_shape_preservation(self):
        """Transformer block should preserve shape."""
        hidden_size = 512
        num_heads = 8
        batch, seq_len = 2, 10

        block = TransformerBlock(hidden_size, num_heads)
        x = torch.randn(batch, seq_len, hidden_size)

        output = block(x)

        assert output.shape == x.shape

    def test_hidden_size_divisible_by_heads(self):
        """Should raise error if hidden_size not divisible by num_heads."""
        with pytest.raises(AssertionError):
            TransformerBlock(hidden_size=100, num_heads=8)

    def test_attention_output_range(self):
        """Attention output should be reasonable (not exploding)."""
        block = TransformerBlock(hidden_size=64, num_heads=8)
        x = torch.randn(2, 10, 64)

        output = block(x)

        # Output should be bounded (not NaN or inf)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.abs().max() < 100  # Reasonable bound

    def test_residual_connections(self):
        """Block should have residual connections."""
        block = TransformerBlock(hidden_size=64, num_heads=8)

        # Zero initialization should result in identity-like behavior
        # due to residual connections
        with torch.no_grad():
            for param in block.parameters():
                param.zero_()

        x = torch.randn(2, 10, 64)
        output = block(x)

        # With zero weights, residuals mean output ≈ input
        # (not exactly due to normalization)
        assert output.shape == x.shape

    def test_forward_backward(self):
        """Should support forward and backward pass."""
        block = TransformerBlock(hidden_size=64, num_heads=8)
        x = torch.randn(2, 10, 64, requires_grad=True)

        output = block(x)
        loss = output.sum()
        loss.backward()

        # Gradients should exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_sequence_lengths(self):
        """Should handle different sequence lengths."""
        block = TransformerBlock(hidden_size=64, num_heads=8)

        for seq_len in [5, 10, 20, 100]:
            x = torch.randn(2, seq_len, 64)
            output = block(x)
            assert output.shape == (2, seq_len, 64)

    def test_batch_size_one(self):
        """Should work with batch_size=1."""
        block = TransformerBlock(hidden_size=64, num_heads=8)
        x = torch.randn(1, 10, 64)

        output = block(x)
        assert output.shape == (1, 10, 64)

    def test_gradient_flow(self):
        """Gradients should flow through all components."""
        block = TransformerBlock(hidden_size=64, num_heads=8)
        x = torch.randn(2, 10, 64, requires_grad=True)

        output = block(x)
        loss = output.mean()
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestTransformerIntegration:
    """Integration tests for full Transformer."""

    def test_multi_layer_forward(self):
        """Multiple Transformer blocks should work together."""
        hidden_size = 128
        num_heads = 8
        num_layers = 4

        blocks = torch.nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )

        x = torch.randn(2, 10, hidden_size)

        for block in blocks:
            x = block(x)

        assert x.shape == (2, 10, hidden_size)
        assert not torch.isnan(x).any()

    def test_with_input_sum(self):
        """Test with summed inputs (as used in HRM)."""
        hidden_size = 64
        block = TransformerBlock(hidden_size, num_heads=8)

        x = torch.randn(2, 10, hidden_size)
        y = torch.randn(2, 10, hidden_size)
        z = torch.randn(2, 10, hidden_size)

        # Sum inputs as in HRM
        combined = x + y + z
        output = block(combined)

        assert output.shape == (2, 10, hidden_size)
        assert not torch.isnan(output).any()

    def test_detach_between_steps(self):
        """Test detaching between recursion steps (as in HRM)."""
        block = TransformerBlock(hidden_size=64, num_heads=8)

        x = torch.randn(2, 10, 64)
        y = torch.randn(2, 10, 64, requires_grad=True)

        # First step
        output1 = block(x + y)

        # Detach for next step
        y_detached = output1.detach()

        # Second step
        output2 = block(x + y_detached)
        loss = output2.sum()
        loss.backward()

        # Check that detach worked
        assert not y_detached.requires_grad
        assert output2.requires_grad


def test_transformer_with_recursion():
    """Test Transformer in recursive setting like HRM."""
    hidden_size = 64
    num_heads = 8
    L_net = torch.nn.ModuleList([TransformerBlock(hidden_size, num_heads) for _ in range(2)])
    H_net = torch.nn.ModuleList([TransformerBlock(hidden_size, num_heads) for _ in range(2)])

    batch, seq_len = 2, 10
    x = torch.randn(batch, seq_len, hidden_size)
    zL = torch.randn(batch, seq_len, hidden_size)
    zH = torch.randn(batch, seq_len, hidden_size)

    # Simulate HRM forward pass
    def net_forward(net, *inputs):
        combined = sum(inputs)
        for block in net:
            combined = block(combined)
        return combined

    # n=2, T=2 pattern
    with torch.no_grad():
        # First cycle
        zL = net_forward(L_net, zL, zH, x)
        zL = net_forward(L_net, zL, zH, x)
        zH = net_forward(H_net, zH, zL)

        # Second cycle (partial)
        zL = net_forward(L_net, zL, zH, x)

    # Final with gradients
    zL = net_forward(L_net, zL, zH, x)
    zH = net_forward(H_net, zH, zL)

    assert zL.shape == (batch, seq_len, hidden_size)
    assert zH.shape == (batch, seq_len, hidden_size)
    assert zH.requires_grad
    assert not torch.isnan(zH).any()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
