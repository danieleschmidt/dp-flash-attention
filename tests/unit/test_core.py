"""Unit tests for core functionality."""

import pytest
import torch
from unittest.mock import Mock, patch

from dp_flash_attention.core import DPFlashAttention


class TestDPFlashAttention:
    """Test cases for DPFlashAttention module."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        dp_attn = DPFlashAttention(
            embed_dim=768,
            num_heads=12,
            epsilon=1.0,
            delta=1e-5
        )
        assert dp_attn.embed_dim == 768
        assert dp_attn.num_heads == 12
        assert dp_attn.epsilon == 1.0
        assert dp_attn.delta == 1e-5

    def test_init_invalid_epsilon(self):
        """Test initialization with invalid epsilon."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            DPFlashAttention(
                embed_dim=768,
                num_heads=12,
                epsilon=-1.0,
                delta=1e-5
            )

    def test_init_invalid_delta(self):
        """Test initialization with invalid delta."""
        with pytest.raises(ValueError, match="delta must be positive"):
            DPFlashAttention(
                embed_dim=768,
                num_heads=12,
                epsilon=1.0,
                delta=-1e-5
            )

    @pytest.mark.parametrize("embed_dim,num_heads", [
        (768, 12),
        (512, 8),
        (1024, 16)
    ])
    def test_different_configurations(self, embed_dim, num_heads):
        """Test various valid configurations."""
        dp_attn = DPFlashAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            epsilon=1.0,
            delta=1e-5
        )
        assert dp_attn.embed_dim == embed_dim
        assert dp_attn.num_heads == num_heads

    @patch('dp_flash_attention.core.torch.cuda.is_available')
    def test_cuda_unavailable_fallback(self, mock_cuda_available):
        """Test fallback when CUDA is unavailable."""
        mock_cuda_available.return_value = False
        
        dp_attn = DPFlashAttention(
            embed_dim=768,
            num_heads=12,
            epsilon=1.0,
            delta=1e-5
        )
        
        # Should still initialize but with CPU fallback
        assert dp_attn.device_type == "cpu"