#!/usr/bin/env python3
"""
Export dummy classification model to ONNX and verify outputs are in [0, 1].
GAP-02 contract: single sigmoid score [0.0, 1.0] per input.
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification.kronos_classification_base import KronosClassificationONNXWrapper

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "dummy_classifier_v1")
ONNX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "dummy_classifier_v1.onnx")
SEQ_LEN = 64


def main():
    # Load model
    print(f"Loading model from {MODEL_DIR}...")
    ckpt = torch.load(os.path.join(MODEL_DIR, "model.pt"), map_location="cpu", weights_only=True)
    config = ckpt["config"]
    model = KronosClassificationONNXWrapper(**config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Export to ONNX
    print(f"Exporting to {ONNX_PATH}...")
    model.export_to_onnx(ONNX_PATH, opset_version=17)

    # Additional verification: 10 random inputs, all in [0, 1]
    import onnxruntime as ort

    sess = ort.InferenceSession(ONNX_PATH)
    test_inputs = np.random.randn(10, SEQ_LEN, 5).astype(np.float32)
    outputs = sess.run(None, {"ohlcv": test_inputs})
    scores = outputs[0]

    print(f"\nVerification: 10 random inputs")
    print(f"  Output shape: {scores.shape}")
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    assert scores.shape == (10, 1), f"Shape mismatch: {scores.shape}"
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0), "Scores out of [0, 1]!"
    print("  ✅ All scores in [0.0, 1.0] — ML contract satisfied")
    print(f"\nONNX file: {ONNX_PATH}")


if __name__ == "__main__":
    main()
