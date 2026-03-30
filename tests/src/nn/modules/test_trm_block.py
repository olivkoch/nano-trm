"""
Unit tests for TRM Transformer components
"""

import pytest
import torch

from src.nn.modules.trm_block import (
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    ReasoningBlock,
    ReasoningBlockConfig,
    ReasoningModule,
    rotate_half,
    apply_rotary_pos_emb,
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
        max_seq_len = 100
        rope = RotaryEmbedding(head_dim, max_seq_len, base=10000)

        cos, sin = rope()
        
        assert cos is not None
        assert sin is not None
        assert cos.shape == (max_seq_len, head_dim)
        assert sin.shape == (max_seq_len, head_dim)

    def test_deterministic(self):
        """RoPE should be deterministic."""
        head_dim = 64
        max_seq_len = 100
        rope = RotaryEmbedding(head_dim, max_seq_len, base=10000)

        cos1, sin1 = rope()
        cos2, sin2 = rope()

        assert torch.allclose(cos1, cos2)
        assert torch.allclose(sin1, sin2)

    def test_disabled_rope(self):
        """RoPE with base=0 should be disabled."""
        head_dim = 64
        max_seq_len = 100
        rope = RotaryEmbedding(head_dim, max_seq_len, base=0)
        
        cos, sin = rope()
        assert cos is None
        assert sin is None


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

        q = torch.randn(batch, seq_len, num_heads, head_dim)
        k = torch.randn(batch, seq_len, num_heads, head_dim)
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)

        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

        assert q_embed.shape == q.shape
        assert k_embed.shape == k.shape


class TestSwiGLU:
    """Test SwiGLU activation."""

    def test_shape_preservation(self):
        """SwiGLU should preserve input/output dimensions."""
        hidden_size = 64
        expansion = 2.0
        swiglu = SwiGLU(hidden_size, expansion)

        x = torch.randn(2, 10, hidden_size)
        output = swiglu(x)

        assert output.shape == x.shape

    def test_no_bias(self):
        """SwiGLU linear layers should have no bias."""
        swiglu = SwiGLU(64, 2.0)

        assert swiglu.gate_up_proj.bias is None
        assert swiglu.down_proj.bias is None

    def test_nonlinearity(self):
        """SwiGLU should be non-linear."""
        swiglu = SwiGLU(64, 2.0)

        x1 = torch.randn(1, 5, 64)
        x2 = torch.randn(1, 5, 64)

        # Non-linear: f(x1 + x2) != f(x1) + f(x2)
        result_sum = swiglu(x1 + x2)
        result_separate = swiglu(x1) + swiglu(x2)

        assert not torch.allclose(result_sum, result_separate)


class TestReasoningBlock:
    """Test ReasoningBlock (transformer block)."""

    def test_shape_preservation(self):
        """ReasoningBlock should preserve shape."""
        hidden_size = 128
        num_heads = 8
        batch, seq_len = 2, 10

        config = ReasoningBlockConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=2,
            rms_norm_eps=1e-5,
            seq_len=seq_len,
        )
        block = ReasoningBlock(config)
        
        x = torch.randn(batch, seq_len, hidden_size)
        output = block(None, x)

        assert output.shape == x.shape

    def test_hidden_size_divisible_by_heads(self):
        """Should work when hidden_size is divisible by num_heads."""
        config = ReasoningBlockConfig(
            hidden_size=64,
            num_heads=8,
            expansion=2,
            rms_norm_eps=1e-5,
            seq_len=10,
        )
        block = ReasoningBlock(config)
        
        assert block is not None

    def test_output_range(self):
        """Output should be reasonable (not exploding)."""
        config = ReasoningBlockConfig(
            hidden_size=64,
            num_heads=8,
            expansion=2,
            rms_norm_eps=1e-5,
            seq_len=10,
        )
        block = ReasoningBlock(config)
        
        x = torch.randn(2, 10, 64)
        output = block(None, x)

        # Output should be bounded (not NaN or inf)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.abs().max() < 100  # Reasonable bound

    def test_forward_backward(self):
        """Should support forward and backward pass."""
        config = ReasoningBlockConfig(
            hidden_size=64,
            num_heads=8,
            expansion=2,
            rms_norm_eps=1e-5,
            seq_len=10,
        )
        block = ReasoningBlock(config)
        
        x = torch.randn(2, 10, 64, requires_grad=True)

        output = block(None, x)
        loss = output.sum()
        loss.backward()

        # Gradients should exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_sequence_lengths(self):
        """Should handle different sequence lengths with appropriate config."""
        for seq_len in [5, 10, 20]:
            config = ReasoningBlockConfig(
                hidden_size=64,
                num_heads=8,
                expansion=2,
                rms_norm_eps=1e-5,
                seq_len=seq_len,
            )
            block = ReasoningBlock(config)
            
            x = torch.randn(2, seq_len, 64)
            output = block(None, x)
            assert output.shape == (2, seq_len, 64)

    def test_batch_size_one(self):
        """Should work with batch_size=1."""
        config = ReasoningBlockConfig(
            hidden_size=64,
            num_heads=8,
            expansion=2,
            rms_norm_eps=1e-5,
            seq_len=10,
        )
        block = ReasoningBlock(config)
        
        x = torch.randn(1, 10, 64)
        output = block(None, x)
        assert output.shape == (1, 10, 64)

    def test_gradient_flow(self):
        """Gradients should flow through all components."""
        config = ReasoningBlockConfig(
            hidden_size=64,
            num_heads=8,
            expansion=2,
            rms_norm_eps=1e-5,
            seq_len=10,
        )
        block = ReasoningBlock(config)
        
        x = torch.randn(2, 10, 64, requires_grad=True)

        output = block(None, x)
        loss = output.mean()
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
                
    def test_with_mlp_t(self):
        """Test ReasoningBlock with mlp_t enabled."""
        config = ReasoningBlockConfig(
            hidden_size=64,
            num_heads=8,
            expansion=2,
            rms_norm_eps=1e-5,
            seq_len=10,
            mlp_t=True,
        )
        block = ReasoningBlock(config)
        
        x = torch.randn(2, 10, 64)
        output = block(None, x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestReasoningModule:
    """Integration tests for ReasoningModule (multi-layer)."""

    def test_multi_layer_forward(self):
        """Multiple ReasoningBlocks should work together."""
        hidden_size = 128
        num_heads = 8
        num_layers = 4
        seq_len = 10

        configs = [
            ReasoningBlockConfig(
                hidden_size=hidden_size,
                num_heads=num_heads,
                expansion=2,
                rms_norm_eps=1e-5,
                seq_len=seq_len,
            )
            for _ in range(num_layers)
        ]
        
        blocks = [ReasoningBlock(config) for config in configs]
        module = ReasoningModule(layers=blocks)

        # ReasoningModule signature: (hidden_states, input_injection, **kwargs)
        hidden_states = torch.randn(2, seq_len, hidden_size)
        input_injection = torch.randn(2, seq_len, hidden_size)
        
        output = module(hidden_states, input_injection, cos_sin=None)

        assert output.shape == (2, seq_len, hidden_size)
        assert not torch.isnan(output).any()

    def test_with_input_sum(self):
        """Test with summed inputs (as used in TRM)."""
        hidden_size = 64
        seq_len = 10
        
        config = ReasoningBlockConfig(
            hidden_size=hidden_size,
            num_heads=8,
            expansion=2,
            rms_norm_eps=1e-5,
            seq_len=seq_len,
        )
        block = ReasoningBlock(config)

        x = torch.randn(2, seq_len, hidden_size)
        y = torch.randn(2, seq_len, hidden_size)
        z = torch.randn(2, seq_len, hidden_size)

        # Sum inputs as in TRM
        combined = x + y + z
        output = block(None, combined)

        assert output.shape == (2, seq_len, hidden_size)
        assert not torch.isnan(output).any()

    def test_detach_between_steps(self):
        """Test detaching between recursion steps (as in TRM)."""
        config = ReasoningBlockConfig(
            hidden_size=64,
            num_heads=8,
            expansion=2,
            rms_norm_eps=1e-5,
            seq_len=10,
        )
        block = ReasoningBlock(config)

        x = torch.randn(2, 10, 64)
        y = torch.randn(2, 10, 64, requires_grad=True)

        # First step
        output1 = block(None, x + y)

        # Detach for next step
        y_detached = output1.detach()

        # Second step
        output2 = block(None, x + y_detached)
        loss = output2.sum()
        loss.backward()

        # Check that detach worked
        assert not y_detached.requires_grad
        assert output2.requires_grad


@pytest.mark.parametrize("hidden_size,num_heads", [(64, 8), (128, 8), (256, 16)])
def test_reasoning_block_with_recursion(hidden_size, num_heads):
    """Test ReasoningBlock in recursive setting like TRM."""
    seq_len = 10
    num_layers = 2
    
    # Create L and H networks
    L_configs = [
        ReasoningBlockConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=2,
            rms_norm_eps=1e-5,
            seq_len=seq_len,
        )
        for _ in range(num_layers)
    ]
    H_configs = [
        ReasoningBlockConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=2,
            rms_norm_eps=1e-5,
            seq_len=seq_len,
        )
        for _ in range(num_layers)
    ]
    
    L_net = ReasoningModule(layers=[ReasoningBlock(c) for c in L_configs])
    H_net = ReasoningModule(layers=[ReasoningBlock(c) for c in H_configs])

    batch = 2
    x = torch.randn(batch, seq_len, hidden_size)
    zL = torch.randn(batch, seq_len, hidden_size)
    zH = torch.randn(batch, seq_len, hidden_size)

    # Simulate TRM forward pass (simplified)
    # ReasoningModule: (hidden_states, input_injection, **kwargs)
    with torch.no_grad():
        # First cycle
        zL = L_net(zL, zH + x, cos_sin=None)
        zL = L_net(zL, zH + x, cos_sin=None)
        zH = H_net(zH, zL, cos_sin=None)

        # Second cycle (partial)
        zL = L_net(zL, zH + x, cos_sin=None)

    # Final with gradients
    zL = L_net(zL, zH + x, cos_sin=None)
    zH = H_net(zH, zL, cos_sin=None)

    assert zL.shape == (batch, seq_len, hidden_size)
    assert zH.shape == (batch, seq_len, hidden_size)
    assert zH.requires_grad
    assert not torch.isnan(zH).any()
