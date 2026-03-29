"""
Shared test fixtures and configuration for Kronos test suite.

All external dependencies (HuggingFace Hub downloads, GPU) are mocked so tests
run fully offline on CPU without network access.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "model"))
sys.path.insert(0, str(PROJECT_ROOT / "classification"))
sys.path.insert(0, str(PROJECT_ROOT / "finetune_csv"))

# Force CPU for every test
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# ---------------------------------------------------------------------------
# Small mock Kronos backbone / tokenizer used throughout tests
# ---------------------------------------------------------------------------

class _MockConfig:
    """Minimal config object that mimics Kronos backbone config."""
    def __init__(self, n_embd=64):
        self.n_embd = n_embd


class MockKronosBackbone(nn.Module):
    """Tiny stand-in for the Kronos model that produces hidden_states."""

    def __init__(self, hidden_size=64, vocab_size=1024):
        super().__init__()
        self.config = _MockConfig(hidden_size)
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, output_hidden_states=False, **kwargs):
        x = self.embedding(input_ids)
        x = self.linear(x)
        # KronosClassificationModel expects outputs.hidden_states[-1]
        Output = type("Output", (), {"hidden_states": (x,)})
        return Output()

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()


class MockKronosTokenizer:
    """Minimal tokenizer stand-in."""

    def encode(self, data, timestamps=None):
        # Return a fixed-length token sequence based on data length
        if hasattr(data, 'shape'):
            length = data.shape[0]
        else:
            length = len(data)
        return list(range(length))

    def decode(self, tokens, **kwargs):
        return tokens

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()

    def save_pretrained(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump({"mock": True}, f)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _force_cpu():
    """Patch torch.cuda.is_available to always return False."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture()
def mock_backbone():
    return MockKronosBackbone(hidden_size=64)


@pytest.fixture()
def mock_tokenizer():
    return MockKronosTokenizer()


@pytest.fixture()
def sample_ohlcv_df():
    """DataFrame with standard OHLCV+amount columns (50 rows)."""
    np.random.seed(42)
    n = 50
    base = 100.0
    closes = base + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open": closes + np.random.randn(n) * 0.1,
        "high": closes + np.abs(np.random.randn(n)) * 0.3,
        "low": closes - np.abs(np.random.randn(n)) * 0.3,
        "close": closes,
        "volume": np.random.uniform(1000, 10000, n),
        "amount": np.zeros(n),
    })
    df["amount"] = df["close"] * df["volume"]
    return df


@pytest.fixture()
def sample_ohlcv_no_volume_df():
    """DataFrame with only OHLC columns."""
    np.random.seed(42)
    n = 50
    base = 100.0
    closes = base + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": closes + np.random.randn(n) * 0.1,
        "high": closes + np.abs(np.random.randn(n)) * 0.3,
        "low": closes - np.abs(np.random.randn(n)) * 0.3,
        "close": closes,
    })


@pytest.fixture()
def sample_timeseries_df():
    """DataFrame with OHLCV + timestamps (200 rows), suitable for finetune_csv tests."""
    np.random.seed(123)
    n = 200
    base = 100.0
    closes = base + np.cumsum(np.random.randn(n) * 0.5)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "timestamps": timestamps,
        "open": closes + np.random.randn(n) * 0.1,
        "high": closes + np.abs(np.random.randn(n)) * 0.3,
        "low": closes - np.abs(np.random.randn(n)) * 0.3,
        "close": closes,
        "volume": np.random.uniform(1000, 10000, n),
        "amount": np.zeros(n),
    })
    df["amount"] = df["close"] * df["volume"]
    return df


@pytest.fixture()
def tmp_dir(tmp_path):
    """Return a temporary directory path."""
    return tmp_path


@pytest.fixture()
def sample_classification_data(tmp_path):
    """Create a JSON file mimicking the classification training data format."""
    data = {"results": []}
    for i in range(20):
        n_pts = 60
        chart_data = {
            "opens": list(np.random.randn(n_pts) * 2 + 100),
            "highs": list(np.random.randn(n_pts) * 2 + 102),
            "lows": list(np.random.randn(n_pts) * 2 + 98),
            "closes": list(np.random.randn(n_pts) * 2 + 100),
            "volumes": list(np.abs(np.random.randn(n_pts)) * 1000),
            "dates": list(range(1700000000000, 1700000000000 + n_pts * 3600000, 3600000)),
        }
        data["results"].append({
            "assigned_label": i % 2,
            "chart_data": chart_data,
        })
    fp = tmp_path / "train_data.json"
    with open(fp, "w") as f:
        json.dump(data, f)
    return str(fp)


@pytest.fixture()
def sample_yaml_config(tmp_path):
    """Create a minimal YAML config for finetune_csv."""
    config = {
        "data": {
            "data_path": str(tmp_path / "data.csv"),
            "lookback_window": 30,
            "predict_window": 5,
            "clip": 5.0,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "training": {
            "tokenizer_epochs": 1,
            "basemodel_epochs": 1,
            "batch_size": 4,
            "log_interval": 1,
            "num_workers": 0,
            "seed": 42,
            "tokenizer_learning_rate": 1e-4,
            "predictor_learning_rate": 1e-4,
        },
        "model_paths": {
            "exp_name": "test_exp",
            "base_path": str(tmp_path / "output"),
            "pretrained_tokenizer": str(tmp_path / "dummy_tokenizer"),
            "pretrained_predictor": str(tmp_path / "dummy_predictor"),
            "tokenizer_save_name": "tokenizer",
            "basemodel_save_name": "basemodel",
            "finetuned_tokenizer": "",
        },
        "experiment": {
            "name": "test",
            "description": "test run",
            "train_tokenizer": True,
            "train_basemodel": True,
        },
        "device": {"use_cuda": False, "device_id": 0},
        "distributed": {"use_ddp": False},
    }
    fp = tmp_path / "config.yaml"
    import yaml
    with open(fp, "w") as f:
        yaml.dump(config, f)
    return str(fp)
