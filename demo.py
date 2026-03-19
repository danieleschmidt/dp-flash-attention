#!/usr/bin/env python3
"""
DP-Flash-Attention Demo: Privacy-Utility Tradeoff

Trains a small transformer on a synthetic sequence classification task
at three privacy budgets (ε=1, ε=5, ε=∞) and shows accuracy vs ε.

Task: Classify 4-class sequences of length 16 by majority token.
Architecture: 2-layer transformer encoder with DPAttentionLayer.
"""

import math
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Add src to path if running directly
sys.path.insert(0, "src")
from dp_flash_attention import DPAttentionLayer, PrivacyAccountant, calibrate_noise, compute_epsilon


# ── Synthetic dataset ──────────────────────────────────────────────────────────

NUM_CLASSES = 4
SEQ_LEN = 16
VOCAB_SIZE = NUM_CLASSES + 1  # tokens 1..NUM_CLASSES + padding


def make_batch(batch_size: int, device: torch.device):
    """
    Create a batch of (input_ids, labels).
    Label = most frequent token in the sequence.
    """
    tokens = torch.randint(1, NUM_CLASSES + 1, (batch_size, SEQ_LEN), device=device)
    # Label = most common token (argmax of bincount per row)
    labels = torch.stack([
        tokens[i].bincount(minlength=NUM_CLASSES + 1)[1:].argmax()
        for i in range(batch_size)
    ])
    return tokens, labels


# ── Model ─────────────────────────────────────────────────────────────────────

class DPTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, epsilon: float, delta: float, max_steps: int):
        super().__init__()
        self.attn = DPAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            epsilon=epsilon,
            delta=delta,
            max_steps=max_steps,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class DPTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        seq_len: int,
        epsilon: float,
        delta: float,
        max_steps: int,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            DPTransformerBlock(embed_dim, num_heads, epsilon, delta, max_steps)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.seq_len = seq_len

    def forward(self, x):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            h = block(h)
        # Global average pooling
        h = h.mean(dim=1)
        return self.classifier(h)

    @property
    def privacy_budget(self):
        """Cumulative privacy budget across all layers (worst case = max)."""
        budgets = [block.attn.privacy_budget for block in self.blocks]
        # Conservative: take the layer with highest ε
        return max(budgets, key=lambda b: b.epsilon)


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_eval(
    epsilon: float,
    delta: float = 1e-5,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    batch_size: int = 64,
    epochs: int = 20,
    steps_per_epoch: int = 50,
    eval_batches: int = 20,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> dict:
    """Train a DP transformer and return accuracy + privacy stats."""
    torch.manual_seed(seed)
    random.seed(seed)

    total_steps = epochs * steps_per_epoch * num_layers  # per-layer step count

    model = DPTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=NUM_CLASSES,
        seq_len=SEQ_LEN,
        epsilon=epsilon,
        delta=delta,
        max_steps=total_steps,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=3e-4)

    # Report calibrated noise
    noise_scale = model.blocks[0].attn.noise_scale
    print(f"\n  ε={epsilon:4.1f}  σ={noise_scale:.4f}  (calibrated for {total_steps} steps)")

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            x, y = make_batch(batch_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / steps_per_epoch
            budget = model.privacy_budget
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  ε_spent={budget.epsilon:.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = make_batch(batch_size, device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    budget = model.privacy_budget

    return {
        "epsilon_target": epsilon,
        "epsilon_spent": budget.epsilon,
        "delta": delta,
        "noise_scale": noise_scale,
        "accuracy": accuracy,
        "steps": budget.steps,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task: {NUM_CLASSES}-class sequence classification (majority token, seq_len={SEQ_LEN})")
    print("=" * 60)

    configs = [
        {"epsilon": 1.0,          "label": "High privacy   (ε=1)"},
        {"epsilon": 5.0,          "label": "Medium privacy (ε=5)"},
        {"epsilon": float('inf'), "label": "No privacy     (ε=∞)"},
    ]

    results = []
    for cfg in configs:
        print(f"\n{'─'*60}")
        print(f"Training: {cfg['label']}")
        result = train_and_eval(
            epsilon=cfg["epsilon"],
            device=device,
        )
        results.append({**result, "label": cfg["label"]})

    # Summary table
    print(f"\n{'='*60}")
    print("PRIVACY-UTILITY TRADEOFF SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'σ (noise)':>10} {'ε spent':>10} {'Accuracy':>10}")
    print(f"{'─'*25} {'─'*10} {'─'*10} {'─'*10}")
    for r in results:
        eps_str = f"{r['epsilon_spent']:.3f}" if r['epsilon_spent'] < 999 else "∞"
        print(f"{r['label']:<25} {r['noise_scale']:>10.4f} {eps_str:>10} {r['accuracy']:>9.1%}")

    print(f"\nBaseline (random): {1/NUM_CLASSES:.1%}")
    print("\nConclusion:")
    print("  Higher ε → less noise → better accuracy")
    print("  ε=1 provides strong DP guarantees at some accuracy cost")
    print("  ε=∞ (no noise) is the non-private upper bound")


if __name__ == "__main__":
    main()
