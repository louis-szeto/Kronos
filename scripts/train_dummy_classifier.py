#!/usr/bin/env python3
"""
Train a dummy Kronos classification model with random weights.
Outputs a single sigmoid score [0, 1] per the ML contract (GAP-02).
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification.kronos_classification_base import KronosClassificationONNXWrapper

SEQ_LEN = 64
HIDDEN_SIZE = 256
NUM_SAMPLES = 100
EPOCHS = 5
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "dummy_classifier_v1")


def generate_synthetic_ohlcv(n: int, seq_len: int, label: int) -> torch.Tensor:
    """Generate synthetic OHLCV sequences.

    label=1: uptrend (close increases), label=0: flat/down.
    """
    data = torch.zeros(n, seq_len, 5)
    for i in range(n):
        if label == 1:
            # Uptrend: close drifts up
            close = 100 + torch.cumsum(torch.randn(seq_len) * 0.5 + 0.3, dim=0)
        else:
            # Flat/slightly down
            close = 100 + torch.cumsum(torch.randn(seq_len) * 0.5 - 0.1, dim=0)

        close = close.clamp(min=1)
        high = close + torch.abs(torch.randn(seq_len)) * 0.5
        low = close - torch.abs(torch.randn(seq_len)) * 0.5
        low = low.clamp(min=0.1)
        open_ = close + torch.randn(seq_len) * 0.2
        open_ = open_.clamp(min=0.1)
        volume = torch.rand(seq_len) * 1000 + 100

        data[i] = torch.stack([open_, high, low, close, volume], dim=1)
    return data


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Creating dummy classification model...")
    model = KronosClassificationONNXWrapper(d_in=5, hidden_size=HIDDEN_SIZE, seq_len=SEQ_LEN)

    # Generate synthetic data: 50 uptrend (label=1), 50 flat (label=0)
    print("Generating 100 synthetic OHLCV sequences...")
    x_pos = generate_synthetic_ohlcv(50, SEQ_LEN, label=1)
    x_neg = generate_synthetic_ohlcv(50, SEQ_LEN, label=0)
    x_all = torch.cat([x_pos, x_neg], dim=0)
    y_all = torch.cat([torch.ones(50), torch.zeros(50)])

    # Shuffle
    perm = torch.randperm(NUM_SAMPLES)
    x_all = x_all[perm]
    y_all = y_all[perm]

    # Normalize OHLCV per-sample (simple z-score on price columns)
    # Only normalize first 4 columns (OHLC), keep volume as-is
    price_mean = x_all[:, :, :4].mean(dim=1, keepdim=True)
    price_std = x_all[:, :, :4].std(dim=1, keepdim=True).clamp(min=1e-8)
    x_all[:, :, :4] = (x_all[:, :, :4] - price_mean) / price_std

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    print(f"Training for {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        scores = model(x_all).squeeze(-1)  # [100]
        loss = loss_fn(scores, y_all)
        loss.backward()
        optimizer.step()

        acc = ((scores > 0.5).float() == y_all).float().mean()
        print(f"  Epoch {epoch+1}/{EPOCHS}: loss={loss.item():.4f}, acc={acc.item():.4f}")

    # Save
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"d_in": 5, "hidden_size": HIDDEN_SIZE, "seq_len": SEQ_LEN},
    }, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
