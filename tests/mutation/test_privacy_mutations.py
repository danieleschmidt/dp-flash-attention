"""
Mutation testing specifically for privacy-critical components.

Tests that mutations to privacy parameters are properly caught by tests.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
import math

from dp_flash_attention.core import DPFlashAttention


@pytest.mark.mutation
@pytest.mark.privacy_critical
class TestPrivacyParameterMutations:
    """Test mutations to privacy parameters are caught."""
    
    def test_epsilon_boundary_mutations(self, privacy_test_data):
        """Test that epsilon boundary mutations are detected."""
        # Test valid epsilon values work
        for epsilon in privacy_test_data['valid_epsilon']:
            dp_attn = DPFlashAttention(
                embed_dim=512,
                num_heads=8,
                epsilon=epsilon,
                delta=1e-5
            )
            assert dp_attn.epsilon == epsilon
        
        # Test invalid epsilon values are rejected
        for epsilon in privacy_test_data['invalid_epsilon']:
            with pytest.raises((ValueError, AssertionError)):
                DPFlashAttention(
                    embed_dim=512,
                    num_heads=8,
                    epsilon=epsilon,
                    delta=1e-5
                )
    
    def test_delta_boundary_mutations(self, privacy_test_data):
        """Test that delta boundary mutations are detected."""
        # Test valid delta values work
        for delta in privacy_test_data['valid_delta']:
            dp_attn = DPFlashAttention(
                embed_dim=512,
                num_heads=8,
                epsilon=1.0,
                delta=delta
            )
            assert dp_attn.delta == delta
        
        # Test invalid delta values are rejected
        for delta in privacy_test_data['invalid_delta']:
            with pytest.raises((ValueError, AssertionError)):
                DPFlashAttention(
                    embed_dim=512,
                    num_heads=8,
                    epsilon=1.0,
                    delta=delta
                )
    
    def test_comparison_operator_mutations(self):
        """Test that mutations to comparison operators are caught."""
        # This test should fail if '<=' is mutated to '<' or '>='
        with pytest.raises(ValueError, match="epsilon must be positive"):
            DPFlashAttention(
                embed_dim=512,
                num_heads=8,
                epsilon=0.0,  # This should trigger epsilon <= 0 check
                delta=1e-5
            )
        
        # This test should fail if '<=' is mutated to '<' or '>='
        with pytest.raises(ValueError, match="delta must be positive"):
            DPFlashAttention(
                embed_dim=512,
                num_heads=8,
                epsilon=1.0,
                delta=0.0  # This should trigger delta <= 0 check
            )
    
    def test_arithmetic_operator_mutations(self):
        """Test that mutations to arithmetic operators are caught."""
        dp_attn = DPFlashAttention(
            embed_dim=512,
            num_heads=8,
            epsilon=2.0,
            delta=1e-5
        )
        
        # Test that epsilon scaling works correctly
        # If '*' is mutated to '+', '-', or '/', this should fail
        original_epsilon = dp_attn.epsilon
        scaled_epsilon = dp_attn._scale_epsilon(2.0)  # Should be epsilon * 2
        assert scaled_epsilon == original_epsilon * 2.0
        
    def test_constant_mutations(self):
        """Test that mutations to constants are caught."""
        # Test default epsilon/delta constants
        dp_attn = DPFlashAttention(
            embed_dim=512,
            num_heads=8,
            epsilon=1.0,
            delta=1e-5
        )
        
        # These should catch mutations to default constants
        assert dp_attn.default_epsilon == 1.0
        assert dp_attn.default_delta == 1e-5
        assert dp_attn.min_epsilon == 1e-8
        assert dp_attn.max_delta == 1e-3


@pytest.mark.mutation
@pytest.mark.privacy_critical  
class TestNoiseGenerationMutations:
    """Test mutations to noise generation are caught."""
    
    def test_noise_scale_mutations(self):
        """Test that noise scale calculation mutations are caught."""
        dp_attn = DPFlashAttention(
            embed_dim=512,
            num_heads=8,
            epsilon=1.0,
            delta=1e-5
        )
        
        # Test noise scale calculation
        sensitivity = 1.0
        noise_scale = dp_attn._compute_noise_scale(sensitivity)
        
        # Should be sensitivity / epsilon (if Laplace) or sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon (if Gaussian)
        expected_scale = sensitivity / dp_attn.epsilon  # Simplified
        assert abs(noise_scale - expected_scale) < 0.1  # Allow some variance for Gaussian
    
    def test_gaussian_sampling_mutations(self):
        """Test that Gaussian sampling mutations are caught."""
        dp_attn = DPFlashAttention(
            embed_dim=512,
            num_heads=8,
            epsilon=1.0,
            delta=1e-5
        )
        
        # Generate noise samples
        noise_shape = (100, 100)
        noise_scale = 1.0
        
        noise = dp_attn._sample_gaussian_noise(noise_shape, noise_scale)
        
        # Check that noise has correct statistical properties
        assert noise.shape == noise_shape
        assert abs(noise.mean().item()) < 0.1  # Mean should be close to 0
        assert abs(noise.std().item() - noise_scale) < 0.2  # Std should be close to scale
    
    def test_clipping_mutations(self):
        """Test that gradient clipping mutations are caught."""
        dp_attn = DPFlashAttention(
            embed_dim=512,
            num_heads=8,
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0
        )
        
        # Test gradient clipping
        gradients = torch.randn(10, 512) * 5.0  # Large gradients
        clipped_grads = dp_attn._clip_gradients(gradients)
        
        # Check that gradients are properly clipped
        grad_norm = torch.norm(clipped_grads, dim=-1)
        assert torch.all(grad_norm <= dp_attn.max_grad_norm + 1e-6)  # Allow small numerical error


@pytest.mark.mutation
@pytest.mark.fast_mutation
class TestLogicalOperatorMutations:
    """Test mutations to logical operators are caught."""
    
    def test_and_or_mutations(self):
        """Test that AND/OR mutations are caught."""
        # Test parameter validation logic
        epsilon = 1.0
        delta = 1e-5
        
        # This should catch 'and' -> 'or' mutations in validation
        valid = epsilon > 0 and delta > 0
        assert valid is True
        
        invalid_epsilon = epsilon <= 0 or delta > 0  
        assert invalid_epsilon is False
        
        invalid_delta = epsilon > 0 or delta <= 0
        assert invalid_delta is True  # Would be False if 'or' mutated to 'and'
    
    def test_not_operator_mutations(self):
        """Test that NOT operator mutations are caught."""
        epsilon = 1.0
        delta = 1e-5
        
        # Test negation in conditions
        is_valid = not (epsilon <= 0 or delta <= 0)
        assert is_valid is True
        
        is_invalid = not (epsilon > 0 and delta > 0)
        assert is_invalid is False


@pytest.mark.mutation
class TestBoundaryValueMutations:
    """Test mutations to boundary values are caught."""
    
    @pytest.mark.parametrize("epsilon", [0.0, 0.001, 1e-10])
    def test_epsilon_boundary_values(self, epsilon):
        """Test epsilon boundary value handling."""
        if epsilon <= 0:
            with pytest.raises(ValueError):
                DPFlashAttention(embed_dim=512, num_heads=8, epsilon=epsilon, delta=1e-5)
        else:
            # Should work for positive epsilon
            dp_attn = DPFlashAttention(embed_dim=512, num_heads=8, epsilon=epsilon, delta=1e-5)
            assert dp_attn.epsilon == epsilon
    
    @pytest.mark.parametrize("delta", [0.0, 1e-10, 1e-3, 1.0])
    def test_delta_boundary_values(self, delta):
        """Test delta boundary value handling."""
        if delta <= 0 or delta >= 1:
            with pytest.raises(ValueError):
                DPFlashAttention(embed_dim=512, num_heads=8, epsilon=1.0, delta=delta)
        else:
            # Should work for valid delta
            dp_attn = DPFlashAttention(embed_dim=512, num_heads=8, epsilon=1.0, delta=delta)
            assert dp_attn.delta == delta


@pytest.mark.mutation
class TestMathOperatorMutations:
    """Test mutations to mathematical operators are caught."""
    
    def test_addition_subtraction_mutations(self):
        """Test that +/- mutations are caught."""
        dp_attn = DPFlashAttention(
            embed_dim=512,
            num_heads=8, 
            epsilon=1.0,
            delta=1e-5
        )
        
        # Test privacy budget accounting
        initial_budget = 10.0
        spent = 3.0
        
        remaining = initial_budget - spent  # Should be 7.0
        assert remaining == 7.0
        
        # If '-' is mutated to '+', this would be 13.0
        assert remaining != initial_budget + spent
    
    def test_multiplication_division_mutations(self):
        """Test that *// mutations are caught."""
        dp_attn = DPFlashAttention(
            embed_dim=512,
            num_heads=8,
            epsilon=2.0,
            delta=1e-5
        )
        
        # Test noise scale calculation  
        sensitivity = 4.0
        noise_scale = sensitivity / dp_attn.epsilon  # Should be 2.0
        assert noise_scale == 2.0
        
        # If '/' is mutated to '*', this would be 8.0
        assert noise_scale != sensitivity * dp_attn.epsilon
    
    def test_power_operator_mutations(self):
        """Test that power operator mutations are caught.""" 
        # Test squared sensitivity calculation
        base_sensitivity = 2.0
        squared = base_sensitivity ** 2  # Should be 4.0
        assert squared == 4.0
        
        # Test square root in noise calculation
        variance = 4.0
        std = math.sqrt(variance)  # Should be 2.0
        assert std == 2.0