"""
Tests for dp-flash-attention.

Verifies:
1. DP noise calibration (σ achieves target ε)
2. Privacy accounting (composition is additive in RDP)
3. DPAttentionLayer forward pass (shapes, DP noise active in train mode)
4. No noise in eval mode
5. Privacy budget tracking across steps
"""

import math
import sys
import pytest
import torch

sys.path.insert(0, "src")
from dp_flash_attention import (
    DPAttentionLayer,
    PrivacyAccountant,
    PrivacyBudget,
    calibrate_noise,
    compute_epsilon,
    dp_scaled_dot_product_attention,
)
from dp_flash_attention.privacy import _rdp_gaussian, _rdp_to_dp


# ── Privacy math ──────────────────────────────────────────────────────────────

class TestRDPGaussian:
    def test_positive(self):
        rdp = _rdp_gaussian(alpha=2.0, noise_scale=1.0)
        assert rdp > 0

    def test_larger_alpha_larger_rdp(self):
        rdp2 = _rdp_gaussian(alpha=2.0, noise_scale=1.0)
        rdp10 = _rdp_gaussian(alpha=10.0, noise_scale=1.0)
        assert rdp10 > rdp2  # higher order → higher RDP (for Gaussian)

    def test_larger_noise_smaller_rdp(self):
        rdp_small = _rdp_gaussian(alpha=2.0, noise_scale=0.5)
        rdp_large = _rdp_gaussian(alpha=2.0, noise_scale=2.0)
        assert rdp_small > rdp_large  # more noise → better privacy

    def test_scales_quadratically_with_sensitivity(self):
        rdp1 = _rdp_gaussian(alpha=2.0, noise_scale=1.0, sensitivity=1.0)
        rdp2 = _rdp_gaussian(alpha=2.0, noise_scale=1.0, sensitivity=2.0)
        assert abs(rdp2 - 4 * rdp1) < 1e-10  # sensitivity^2


class TestRDPToDP:
    def test_finite_conversion(self):
        eps = _rdp_to_dp(rdp_value=0.1, alpha=2.0, delta=1e-5)
        assert 0 < eps < 100

    def test_smaller_delta_larger_epsilon(self):
        eps1 = _rdp_to_dp(rdp_value=0.1, alpha=2.0, delta=1e-5)
        eps2 = _rdp_to_dp(rdp_value=0.1, alpha=2.0, delta=1e-8)
        assert eps2 > eps1  # tighter δ → looser ε bound

    def test_alpha_equals_one_is_inf(self):
        eps = _rdp_to_dp(rdp_value=0.1, alpha=1.0, delta=1e-5)
        assert eps == float('inf')


class TestCalibrateNoise:
    def test_zero_noise_for_infinite_epsilon(self):
        sigma = calibrate_noise(float('inf'), 1e-5, steps=100)
        assert sigma == 0.0

    def test_achieves_target_epsilon(self):
        """Calibrated σ should actually achieve the target ε."""
        target_eps = 2.0
        delta = 1e-5
        steps = 100
        sigma = calibrate_noise(target_eps, delta, steps=steps)
        achieved_eps = compute_epsilon(sigma, delta, steps)
        assert achieved_eps <= target_eps + 0.05  # within 5% tolerance

    def test_stricter_epsilon_requires_more_noise(self):
        sigma1 = calibrate_noise(1.0, 1e-5, steps=100)
        sigma5 = calibrate_noise(5.0, 1e-5, steps=100)
        assert sigma1 > sigma5  # ε=1 needs more noise than ε=5

    def test_more_steps_require_more_noise(self):
        sigma10 = calibrate_noise(1.0, 1e-5, steps=10)
        sigma100 = calibrate_noise(1.0, 1e-5, steps=100)
        assert sigma100 > sigma10  # more steps → needs more noise per step


class TestComputeEpsilon:
    def test_zero_sigma_is_infinite_epsilon(self):
        eps = compute_epsilon(0.0, 1e-5, steps=10)
        assert eps == float('inf')

    def test_more_steps_more_epsilon(self):
        eps10 = compute_epsilon(1.0, 1e-5, steps=10)
        eps100 = compute_epsilon(1.0, 1e-5, steps=100)
        assert eps100 > eps10  # more composition → more epsilon

    def test_more_noise_less_epsilon(self):
        eps_small = compute_epsilon(0.5, 1e-5, steps=10)
        eps_large = compute_epsilon(2.0, 1e-5, steps=10)
        assert eps_large < eps_small


# ── PrivacyAccountant ─────────────────────────────────────────────────────────

class TestPrivacyAccountant:
    def test_empty_accountant(self):
        acc = PrivacyAccountant()
        budget = acc.privacy_spent()
        assert budget.epsilon == 0.0
        assert budget.steps == 0

    def test_single_step(self):
        acc = PrivacyAccountant(delta=1e-5)
        eps = acc.record_step(noise_scale=1.0)
        assert eps > 0
        budget = acc.privacy_spent()
        assert budget.steps == 1
        assert budget.delta == 1e-5

    def test_epsilon_increases_with_steps(self):
        acc = PrivacyAccountant(delta=1e-5)
        eps_prev = 0.0
        for _ in range(10):
            eps_curr = acc.record_step(noise_scale=1.0)
            assert eps_curr >= eps_prev
            eps_prev = eps_curr

    def test_reset(self):
        acc = PrivacyAccountant()
        acc.record_step(1.0)
        acc.record_step(1.0)
        acc.reset()
        budget = acc.privacy_spent()
        assert budget.epsilon == 0.0
        assert budget.steps == 0

    def test_composition_matches_compute_epsilon(self):
        """Manual composition should match compute_epsilon."""
        sigma = 1.0
        steps = 20
        delta = 1e-5
        acc = PrivacyAccountant(delta=delta)
        for _ in range(steps):
            acc.record_step(sigma)
        budgeted_eps = acc.privacy_spent().epsilon
        direct_eps = compute_epsilon(sigma, delta, steps)
        assert abs(budgeted_eps - direct_eps) < 1e-6

    def test_more_noise_less_epsilon(self):
        """Higher σ → lower ε for same number of steps."""
        acc1 = PrivacyAccountant(delta=1e-5)
        acc2 = PrivacyAccountant(delta=1e-5)
        for _ in range(5):
            acc1.record_step(noise_scale=0.5)  # less noise
            acc2.record_step(noise_scale=2.0)  # more noise
        assert acc1.privacy_spent().epsilon > acc2.privacy_spent().epsilon


# ── DPAttentionLayer ──────────────────────────────────────────────────────────

class TestDPAttentionLayer:
    @pytest.fixture
    def layer(self):
        return DPAttentionLayer(embed_dim=32, num_heads=4, epsilon=1.0, delta=1e-5, max_steps=1000)

    @pytest.fixture
    def x(self):
        return torch.randn(2, 8, 32)  # (batch, seq, dim)

    def test_output_shape(self, layer, x):
        out, weights = layer(x, x, x)
        assert out.shape == x.shape
        assert weights.shape == (2, 8, 8)  # (batch, seq_q, seq_k)

    def test_no_noise_in_eval_mode(self, layer, x):
        layer.eval()
        torch.manual_seed(0)
        out1, _ = layer(x, x, x)
        torch.manual_seed(1)
        out2, _ = layer(x, x, x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic (no noise)"

    def test_noise_in_train_mode(self, layer, x):
        layer.train()
        torch.manual_seed(0)
        out1, _ = layer(x, x, x)
        torch.manual_seed(1)  # Different seed → different noise
        out2, _ = layer(x, x, x)
        if layer.noise_scale > 0:
            assert not torch.allclose(out1, out2), "Train mode should be stochastic with DP noise"

    def test_privacy_budget_tracked(self, layer, x):
        layer.train()
        assert layer.accountant.num_steps == 0
        layer(x, x, x)
        assert layer.accountant.num_steps == 1
        layer(x, x, x)
        assert layer.accountant.num_steps == 2

    def test_no_tracking_in_eval(self, layer, x):
        layer.eval()
        layer(x, x, x)
        assert layer.accountant.num_steps == 0

    def test_infinite_epsilon_no_noise(self):
        layer = DPAttentionLayer(embed_dim=32, num_heads=4, epsilon=float('inf'))
        assert layer.noise_scale == 0.0

    def test_extra_repr_contains_epsilon(self, layer):
        repr_str = layer.extra_repr()
        assert "ε=1.0" in repr_str

    def test_seq_first_batch_second(self, x):
        layer = DPAttentionLayer(embed_dim=32, num_heads=4, epsilon=1.0, batch_first=False)
        x_sf = x.transpose(0, 1)  # (seq, batch, dim)
        out, weights = layer(x_sf, x_sf, x_sf)
        assert out.shape == x_sf.shape  # (seq, batch, dim)


# ── dp_scaled_dot_product_attention ──────────────────────────────────────────

class TestDPScaledDotProduct:
    def setup_method(self):
        B, H, S, D = 2, 4, 8, 16
        self.q = torch.randn(B, H, S, D)
        self.k = torch.randn(B, H, S, D)
        self.v = torch.randn(B, H, S, D)

    def test_output_shape(self):
        out, weights = dp_scaled_dot_product_attention(self.q, self.k, self.v)
        assert out.shape == self.q.shape
        assert weights.shape == (2, 4, 8, 8)

    def test_weights_sum_to_one(self):
        _, weights = dp_scaled_dot_product_attention(self.q, self.k, self.v, noise_scale=0.0)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_noisy_weights_also_sum_to_one(self):
        _, weights = dp_scaled_dot_product_attention(self.q, self.k, self.v, noise_scale=0.5)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    def test_zero_noise_is_deterministic(self):
        torch.manual_seed(0)
        out1, _ = dp_scaled_dot_product_attention(self.q, self.k, self.v, noise_scale=0.0)
        torch.manual_seed(1)
        out2, _ = dp_scaled_dot_product_attention(self.q, self.k, self.v, noise_scale=0.0)
        assert torch.allclose(out1, out2)

    def test_noise_changes_output(self):
        torch.manual_seed(0)
        out1, _ = dp_scaled_dot_product_attention(self.q, self.k, self.v, noise_scale=1.0)
        torch.manual_seed(99)
        out2, _ = dp_scaled_dot_product_attention(self.q, self.k, self.v, noise_scale=1.0)
        assert not torch.allclose(out1, out2)

    def test_causal_mask(self):
        _, weights = dp_scaled_dot_product_attention(
            self.q, self.k, self.v, noise_scale=0.0, is_causal=True
        )
        S = self.q.size(-2)
        # Upper triangle (future tokens) should be ~0
        for i in range(S):
            for j in range(i + 1, S):
                assert weights[0, 0, i, j].item() < 1e-6, \
                    f"Causal mask violated at ({i},{j})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
