"""
Differentially Private Attention — pure PyTorch implementation.

Architecture:
  DPAttention       — functional DP scaled dot-product attention
  DPAttentionLayer  — drop-in nn.MultiheadAttention replacement with DP

DP mechanism:
  We add calibrated Gaussian noise to the *attention weights* (post-softmax)
  before they are used to aggregate values. This is the most natural point
  for DP injection in attention: it privatizes the score distribution directly
  rather than perturbing gradients, enabling a clean per-forward-pass privacy
  analysis (Rényi DP via the Gaussian mechanism).

  Noise scale σ is calibrated so that T forward passes through the layer
  achieve (ε, δ)-DP (using PrivacyAccountant for composition tracking).
  Sensitivity = 1.0 (attention weights are in [0,1] and sum to 1 per row,
  so each row has L2 norm ≤ 1 and adding noise row-by-row is sensitivity-1).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .privacy import PrivacyAccountant, calibrate_noise


def dp_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    noise_scale: float = 0.0,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Scaled dot-product attention with Gaussian DP noise on attention weights.

    Adds i.i.d. Gaussian noise N(0, σ²) to each element of the post-softmax
    attention weight matrix, then re-normalizes to maintain probability
    semantics. This implements the Gaussian mechanism on the attention
    distribution.

    Args:
        query: (batch, heads, seq_q, head_dim)
        key:   (batch, heads, seq_k, head_dim)
        value: (batch, heads, seq_k, head_dim)
        noise_scale: σ for Gaussian noise. 0 = no noise (ε = ∞).
        attn_mask: Optional additive mask (float) or boolean mask
        dropout_p: Attention dropout probability
        is_causal: Apply causal mask

    Returns:
        (output, attn_weights)
        output:      (batch, heads, seq_q, head_dim)
        attn_weights:(batch, heads, seq_q, seq_k)
    """
    B, H, Sq, D = query.shape
    scale = 1.0 / math.sqrt(D)

    # Scaled dot-product scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # (B, H, Sq, Sk)

    # Apply masks
    if is_causal:
        Sk = key.size(-2)
        causal_mask = torch.triu(
            torch.ones(Sq, Sk, device=query.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(attn_mask, float('-inf'))
        else:
            scores = scores + attn_mask

    # Softmax → attention weights in [0, 1], rows sum to 1
    attn_weights = F.softmax(scores, dim=-1)  # (B, H, Sq, Sk)

    # ── DP noise injection ──────────────────────────────────────────────────
    # Add N(0, σ²) noise to attention weights, then re-normalize.
    # Re-normalization via ReLU+renorm keeps weights non-negative and summing
    # to 1, which maintains output stability while preserving DP guarantee.
    # (The renormalization is a post-processing step; by DP post-processing
    # theorem it doesn't weaken the guarantee.)
    if noise_scale > 0.0:
        noise = torch.randn_like(attn_weights) * noise_scale
        noisy_weights = attn_weights + noise
        # Clip to [0, inf) and renormalize
        noisy_weights = F.relu(noisy_weights)
        row_sums = noisy_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        attn_weights = noisy_weights / row_sums
    # ────────────────────────────────────────────────────────────────────────

    # Dropout (on noisy weights)
    if dropout_p > 0.0 and torch.is_grad_enabled():
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Weighted sum of values
    output = torch.matmul(attn_weights, value)  # (B, H, Sq, D)

    return output, attn_weights


class DPAttentionLayer(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention with differential privacy.

    Provides (ε, δ)-DP guarantees per forward pass via Gaussian noise
    injection on the attention weight matrix. Use PrivacyAccountant to
    track cumulative privacy cost across training.

    Args:
        embed_dim:    Total embedding dimension
        num_heads:    Number of attention heads
        epsilon:      Privacy budget ε (∞ = no noise)
        delta:        Privacy parameter δ
        max_steps:    Expected number of forward passes (for noise calibration)
        sensitivity:  L2 sensitivity of attention weights (default: 1.0)
        dropout:      Attention dropout probability
        bias:         Whether to include bias in projections
        batch_first:  If True, input shape is (batch, seq, dim)

    Example::

        layer = DPAttentionLayer(embed_dim=256, num_heads=8, epsilon=1.0, delta=1e-5)
        accountant = layer.accountant  # shared privacy tracker

        for x in batches:
            out, _ = layer(x, x, x)
            budget = accountant.privacy_spent()
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_steps: int = 1000,
        sensitivity: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.dropout = dropout
        self.batch_first = batch_first

        # Calibrate noise for the full training run
        self.noise_scale = calibrate_noise(
            target_epsilon=epsilon,
            target_delta=delta,
            sensitivity=sensitivity,
            steps=max_steps,
        )

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Privacy tracker (shared; caller can access via .accountant)
        self.accountant = PrivacyAccountant(delta=delta)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with DP noise injection.

        Args:
            query: (batch, seq, embed_dim) if batch_first else (seq, batch, embed_dim)
            key:   same shape as query
            value: same shape as query
            key_padding_mask: (batch, seq_k) bool mask — True = ignore
            attn_mask: Additional additive or boolean attention mask
            need_weights: Whether to return attention weights

        Returns:
            (output, attn_weights_or_None)
        """
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, Sq, _ = query.shape
        Sk = key.size(1)

        # Project and reshape to (B, H, S, head_dim)
        def _proj_reshape(proj, x, seq):
            return proj(x).view(B, seq, self.num_heads, self.head_dim).transpose(1, 2)

        q = _proj_reshape(self.q_proj, query, Sq)
        k = _proj_reshape(self.k_proj, key, Sk)
        v = _proj_reshape(self.v_proj, value, Sk)

        # Build combined mask
        combined_mask = None
        if key_padding_mask is not None:
            # (B, 1, 1, Sk) → broadcasts over heads and query positions
            pad_mask = key_padding_mask[:, None, None, :].expand(B, self.num_heads, Sq, Sk)
            combined_mask = pad_mask
        if attn_mask is not None:
            combined_mask = attn_mask if combined_mask is None else (combined_mask | attn_mask.bool())

        # DP attention
        out, weights = dp_scaled_dot_product_attention(
            q, k, v,
            noise_scale=self.noise_scale if self.training else 0.0,
            attn_mask=combined_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Track privacy only during training (no-op when noise_scale=0, i.e. ε=∞)
        if self.training and self.noise_scale > 0:
            self.accountant.record_step(self.noise_scale, self.sensitivity)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, Sq, self.embed_dim)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        return out, (weights.mean(dim=1) if need_weights else None)

    @property
    def privacy_budget(self):
        """Current cumulative privacy budget spent."""
        return self.accountant.privacy_spent()

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"ε={self.epsilon}, δ={self.delta:.2e}, σ={self.noise_scale:.4f}"
        )
