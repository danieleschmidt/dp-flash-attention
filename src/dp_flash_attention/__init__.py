"""
dp-flash-attention: Differentially private attention for PyTorch.

Implements Flash-Attention style scaled dot-product attention with
integrated Rényi DP noise injection and privacy budget tracking.

Core API:
    DPAttentionLayer  — drop-in nn.MultiheadAttention replacement
    PrivacyAccountant — tracks cumulative (ε, δ) privacy budget
    calibrate_noise   — compute σ for a target (ε, δ, steps) budget
    compute_epsilon   — compute ε for a given σ and step count

Example::

    from dp_flash_attention import DPAttentionLayer, PrivacyAccountant

    layer = DPAttentionLayer(embed_dim=128, num_heads=4, epsilon=1.0, delta=1e-5)
    x = torch.randn(8, 32, 128)  # (batch, seq, dim)
    out, _ = layer(x, x, x)

    print(layer.privacy_budget)  # ε spent so far
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"

from .attention import DPAttentionLayer, dp_scaled_dot_product_attention
from .privacy import PrivacyAccountant, PrivacyBudget, calibrate_noise, compute_epsilon

__all__ = [
    "DPAttentionLayer",
    "dp_scaled_dot_product_attention",
    "PrivacyAccountant",
    "PrivacyBudget",
    "calibrate_noise",
    "compute_epsilon",
]
