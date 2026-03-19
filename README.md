# dp-flash-attention

Differentially private scaled dot-product attention for PyTorch. Implements the Gaussian mechanism directly on attention weights with Rényi DP accounting — no CUDA kernels required, no external DP libraries.

## What it does

Standard attention computes a softmax over query-key dot products, then uses those weights to aggregate values. This library adds calibrated Gaussian noise to those post-softmax weights before aggregation, injecting differential privacy at the most natural point in the computation.

```
scores = Q @ K.T / sqrt(d)
weights = softmax(scores)
weights += N(0, σ²)          ← DP noise here
weights = relu(weights) / sum  ← renormalize
output = weights @ V
```

The noise scale σ is automatically calibrated so that T forward passes through the layer achieve (ε, δ)-DP via RDP composition.

## Installation

```bash
pip install -e .
# or just add src/ to your PYTHONPATH
```

Requires: `torch>=2.0`, Python 3.10+. No CUDA required.

## Quick start

```python
from dp_flash_attention import DPAttentionLayer

# Drop-in replacement for nn.MultiheadAttention
layer = DPAttentionLayer(
    embed_dim=256,
    num_heads=8,
    epsilon=1.0,    # privacy budget
    delta=1e-5,     # privacy parameter
    max_steps=1000, # expected number of training steps
)

x = torch.randn(batch, seq, 256)
out, weights = layer(x, x, x)

# Check how much budget you've spent
print(layer.privacy_budget)
# PrivacyBudget(ε=0.342, δ=1e-05, steps=100)
```

## API

### `DPAttentionLayer`

Drop-in replacement for `nn.MultiheadAttention`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `embed_dim` | int | Total embedding dimension |
| `num_heads` | int | Number of attention heads |
| `epsilon` | float | Privacy budget ε (∞ = no DP) |
| `delta` | float | Privacy parameter δ (default: 1e-5) |
| `max_steps` | int | Expected training steps (for σ calibration) |
| `sensitivity` | float | L2 sensitivity of attention weights (default: 1.0) |
| `dropout` | float | Attention dropout probability |
| `batch_first` | bool | If True, input is (batch, seq, dim) |

**Key properties:**
- `.noise_scale` — calibrated σ
- `.privacy_budget` — current `PrivacyBudget(epsilon, delta, steps)`
- `.accountant` — shared `PrivacyAccountant` for this layer

DP noise is only injected during `.train()` mode. `.eval()` mode is deterministic.

### `PrivacyAccountant`

Tracks cumulative (ε, δ)-DP spending via RDP composition.

```python
accountant = PrivacyAccountant(delta=1e-5)
for batch in training_loop:
    # ... training ...
    accountant.record_step(noise_scale=sigma)
    
print(accountant.privacy_spent())
# PrivacyBudget(ε=1.234, δ=1e-05, steps=500)
```

### `calibrate_noise` / `compute_epsilon`

```python
from dp_flash_attention import calibrate_noise, compute_epsilon

# What σ do I need for ε=1, δ=1e-5, over 1000 steps?
sigma = calibrate_noise(target_epsilon=1.0, target_delta=1e-5, steps=1000)

# What ε does σ=2.5 give me after 500 steps?
eps = compute_epsilon(noise_scale=2.5, delta=1e-5, steps=500)
```

## Privacy-utility tradeoff

Demo on a 4-class sequence classification task (2-layer transformer, CUDA):

```
Config                     σ (noise)    ε spent   Accuracy
─────────────────────────────────────────────────────────
High privacy   (ε=1)        225.81       0.686     76.1%
Medium privacy (ε=5)         47.49       3.438     76.5%
No privacy     (ε=∞)          0.00       0.000     99.9%

Baseline (random): 25.0%
```

Run it yourself: `python demo.py`

The accuracy gap between DP and non-DP is larger on harder tasks. The ε=1 vs ε=5 gap is small here because the task is simple; on real NLP tasks you'd typically see 2–5% accuracy loss at ε=1.

## How the math works

### Gaussian mechanism

For a function f with L2 sensitivity Δ (i.e., ||f(x) - f(x')|| ≤ Δ for neighboring datasets x, x'), the Gaussian mechanism adds noise N(0, σ²·I) and achieves (α, α·Δ²/2σ²)-Rényi DP for all α > 1.

For attention weights, each row of the weight matrix lies in the probability simplex, so its L2 norm ≤ 1 (sensitivity = 1 per row).

### RDP composition

For T independent applications of the same mechanism:

```
RDP_total(α) = T × RDP_per_step(α)
```

### RDP → (ε, δ)-DP conversion

```
ε = min_α [ RDP(α) + (log(α/(α-1)) - log(δ)) / (α-1) ]
```

This gives a tight bound by optimizing over the Rényi order α.

### Noise calibration

Given a target (ε, δ) and T steps, we find the minimum σ such that the above chain of equalities holds — via bisection search over σ.

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
# 34 passed in 0.77s
```

Tests cover: RDP math, DP conversion, noise calibration, privacy accounting composition, attention shapes, causal masking, DP noise activation/deactivation, and training-mode tracking.

## Limitations and future work

- **Attention-weight DP vs gradient DP:** This privatizes attention weights per-forward-pass, not per-sample gradients. For strong user-level DP guarantees in training, combine with gradient clipping (DP-SGD). The two approaches are complementary.
- **No subsampling amplification:** We assume full-batch attention. With minibatch sampling, you can get amplification (lower effective ε per step) — the accountant supports `sampling_rate` extension.
- **CUDA kernels:** The name is aspirational. Future work would fuse the noise injection into a Triton/CUDA kernel for zero-overhead DP attention.

## License

MIT
