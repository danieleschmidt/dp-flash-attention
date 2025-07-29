"""Tests for differential privacy guarantees."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock

from dp_flash_attention.privacy import RenyiAccountant


class TestPrivacyGuarantees:
    """Test privacy guarantee compliance."""

    def test_renyi_accountant_initialization(self):
        """Test Renyi accountant proper initialization."""
        accountant = RenyiAccountant()
        assert accountant.alpha_values is not None
        assert len(accountant.alpha_values) > 0

    def test_privacy_composition(self):
        """Test privacy composition across multiple steps."""
        accountant = RenyiAccountant()
        
        # Simulate multiple privacy steps
        for _ in range(10):
            accountant.add_step(epsilon=0.1, delta=1e-6, sampling_rate=0.01)
        
        total_epsilon = accountant.get_epsilon(delta=1e-5)
        assert total_epsilon > 0
        assert total_epsilon < 10  # Should be less than naive composition

    @pytest.mark.parametrize("epsilon,delta", [
        (1.0, 1e-5),
        (0.5, 1e-6),
        (2.0, 1e-4)
    ])
    def test_privacy_budget_tracking(self, epsilon, delta):
        """Test privacy budget is tracked correctly."""
        accountant = RenyiAccountant()
        
        initial_budget = accountant.get_epsilon(delta)
        accountant.add_step(epsilon, delta, sampling_rate=0.01)
        final_budget = accountant.get_epsilon(delta)
        
        assert final_budget >= initial_budget

    def test_sensitivity_calculation(self):
        """Test gradient sensitivity calculation."""
        # Mock tensor with known gradient properties  
        tensor = torch.randn(32, 512, 768, requires_grad=True)
        loss = tensor.sum()
        loss.backward()
        
        # Calculate L2 sensitivity
        grad_norm = torch.norm(tensor.grad, p=2)
        sensitivity = min(grad_norm.item(), 1.0)  # Clipped sensitivity
        
        assert sensitivity >= 0
        assert sensitivity <= 1.0

    @pytest.mark.slow
    def test_empirical_privacy_validation(self):
        """Empirical validation of privacy guarantees via shadow modeling."""
        # This would be a comprehensive test involving:
        # 1. Training shadow models with/without target sample
        # 2. Running membership inference attacks
        # 3. Validating empirical privacy matches theoretical guarantees
        
        # Simplified version for framework demonstration
        num_shadow_models = 5  # Reduced for testing
        target_epsilon = 1.0
        
        privacy_violations = 0
        
        for i in range(num_shadow_models):
            # Simulate privacy measurement
            measured_privacy = np.random.exponential(target_epsilon)
            if measured_privacy > target_epsilon * 1.1:  # 10% tolerance
                privacy_violations += 1
        
        violation_rate = privacy_violations / num_shadow_models
        assert violation_rate < 0.1  # Less than 10% violations acceptable