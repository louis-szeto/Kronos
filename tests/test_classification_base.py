"""
Tests for classification/kronos_classification_base.py —
KronosClassificationModel, KronosClassificationConfig.

The HuggingFace model downloads are mocked so tests run offline.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from conftest import MockKronosBackbone, MockKronosTokenizer


def _mock_init(cls, kronos_model_path="mock", tokenizer_path="mock", **kwargs):
    """Replacement __init__ that skips HF downloads."""
    nn.Module.__init__(cls)
    cls.tokenizer = MockKronosTokenizer()
    cls.backbone = MockKronosBackbone(hidden_size=kwargs.get("hidden_size", 64))

    cls.max_context = kwargs.get("max_context", 512)
    cls.min_context = kwargs.get("min_context", 20)
    cls.use_volume = kwargs.get("use_volume", True)
    cls.num_exogenous = kwargs.get("num_exogenous", 0)
    cls.pooling_strategy = kwargs.get("pooling_strategy", "mean")
    cls.padding_strategy = kwargs.get("padding_strategy", "right")
    cls.loss_type = kwargs.get("loss_type") or "cross_entropy"
    cls.label_smoothing = kwargs.get("label_smoothing", 0.1)
    cls.num_classes = kwargs.get("num_classes", 2)
    cls.d_in = 4 + (1 if cls.use_volume else 0) + (1 if cls.num_exogenous > 0 else 0)
    cls.hidden_size = cls.backbone.config.n_embd

    if cls.pooling_strategy == "attention":
        cls.attention_weights = nn.Linear(cls.hidden_size, 1)

    hidden_size = cls.hidden_size
    classifier_hidden_size = kwargs.get("classifier_hidden_size", None) or hidden_size
    cls.classification_head = nn.Sequential(
        nn.Dropout(kwargs.get("hidden_dropout_prob", 0.1)),
        nn.Linear(hidden_size, classifier_hidden_size),
        nn.GELU(),
        nn.Dropout(kwargs.get("hidden_dropout_prob", 0.1)),
        nn.Linear(classifier_hidden_size, cls.num_classes),
    )

    if kwargs.get("freeze_backbone", False):
        for param in cls.backbone.parameters():
            param.requires_grad = False


@pytest.fixture()
def make_model():
    """Factory fixture that creates KronosClassificationModel with mocked backbone."""
    import classification.kronos_classification_base as mod

    def _make(**kwargs):
        orig_init = mod.KronosClassificationModel.__init__
        mod.KronosClassificationModel.__init__ = _mock_init
        try:
            m = mod.KronosClassificationModel(**kwargs)
            return m
        finally:
            mod.KronosClassificationModel.__init__ = orig_init

    return _make


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

class TestClassificationModelInit:

    def test_default_config(self, make_model):
        m = make_model()
        assert m.num_classes == 2
        assert m.use_volume is True
        assert m.pooling_strategy == "mean"
        assert m.max_context == 512
        assert m.min_context == 20

    def test_hidden_size_set(self, make_model):
        m = make_model(hidden_size=64)
        assert m.hidden_size == 64

    def test_classification_head_structure(self, make_model):
        m = make_model()
        head = m.classification_head
        assert isinstance(head, nn.Sequential)
        last_linear = [layer for layer in head if isinstance(layer, nn.Linear)][-1]
        assert last_linear.out_features == 2

    def test_multiclass(self, make_model):
        m = make_model(num_classes=5)
        last = [layer for layer in m.classification_head if isinstance(layer, nn.Linear)][-1]
        assert last.out_features == 5

    def test_freeze_backbone(self, make_model):
        m = make_model(freeze_backbone=True)
        for p in m.backbone.parameters():
            assert not p.requires_grad


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestClassificationForward:

    def test_forward_returns_dict(self, make_model):
        m = make_model(max_context=64)
        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)

        out = m(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        assert "logits" in out
        assert "hidden_states" in out
        assert out["logits"].shape == (2, 2)
        assert out["loss"] is None  # No labels

    def test_forward_with_labels(self, make_model):
        m = make_model(max_context=64)
        input_ids = torch.randint(0, 100, (4, 16))
        attention_mask = torch.ones(4, 16, dtype=torch.long)
        labels = torch.tensor([0, 1, 0, 1])

        out = m(input_ids=input_ids, attention_mask=attention_mask,
                labels=labels, return_dict=True)
        assert out["loss"] is not None
        assert out["loss"].shape == ()
        assert out["logits"].shape == (4, 2)

    def test_forward_without_mask(self, make_model):
        m = make_model(max_context=64)
        input_ids = torch.randint(0, 100, (2, 16))
        out = m(input_ids=input_ids)
        assert out["logits"].shape[0] == 2

    def test_gradient_flow(self, make_model):
        m = make_model(max_context=64)
        input_ids = torch.randint(0, 100, (2, 16))
        labels = torch.tensor([0, 1])
        out = m(input_ids=input_ids, labels=labels, return_dict=True)
        out["loss"].backward()

        for name, param in m.classification_head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# Pooling strategies
# ---------------------------------------------------------------------------

class TestPoolingStrategies:

    def test_mean_pooling(self, make_model):
        m = make_model(pooling_strategy="mean", max_context=64)
        out = m(input_ids=torch.randint(0, 100, (2, 16)))
        assert out["logits"].shape == (2, 2)

    def test_last_pooling(self, make_model):
        m = make_model(pooling_strategy="last", max_context=64)
        out = m(input_ids=torch.randint(0, 100, (2, 16)))
        assert out["logits"].shape == (2, 2)

    def test_max_pooling(self, make_model):
        m = make_model(pooling_strategy="max", max_context=64)
        out = m(input_ids=torch.randint(0, 100, (2, 16)),
                attention_mask=torch.ones(2, 16, dtype=torch.long))
        assert out["logits"].shape == (2, 2)

    def test_attention_pooling(self, make_model):
        m = make_model(pooling_strategy="attention", max_context=64)
        out = m(input_ids=torch.randint(0, 100, (2, 16)),
                attention_mask=torch.ones(2, 16, dtype=torch.long))
        assert out["logits"].shape == (2, 2)

    def test_unknown_pooling_raises(self, make_model):
        m = make_model(pooling_strategy="mean")
        m.pooling_strategy = "nonexistent_strategy"
        with pytest.raises(ValueError, match="Unknown pooling"):
            m._pool_sequence(torch.randn(2, 8, 64), torch.ones(2, 8, dtype=torch.long))


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class TestLossFunctions:

    def test_cross_entropy_loss(self, make_model):
        m = make_model(loss_type="cross_entropy")
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        loss = m._compute_loss(logits, labels)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_focal_loss(self, make_model):
        m = make_model(loss_type="focal")
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        loss = m._compute_loss(logits, labels)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_label_smoothing_loss(self, make_model):
        m = make_model(loss_type="label_smoothing", label_smoothing=0.1)
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        loss = m._compute_loss(logits, labels)
        assert loss.item() > 0

    def test_binary_cross_entropy_loss(self, make_model):
        m = make_model(loss_type="binary_cross_entropy", num_classes=2)
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        loss = m._compute_loss(logits, labels)
        assert loss.item() > 0

    def test_loss_with_class_weights(self, make_model):
        m = make_model(loss_type="cross_entropy")
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        weights = torch.tensor([1.0, 2.0])
        loss = m._compute_loss(logits, labels, class_weights=weights)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# Tokenize timeseries
# ---------------------------------------------------------------------------

class TestTokenizeTimeseries:

    def test_basic_tokenize(self, make_model):
        m = make_model(max_context=64, min_context=5)
        df = pd.DataFrame({
            "open": [100, 101, 102], "high": [103, 104, 105],
            "low": [99, 100, 101], "close": [101, 102, 103],
            "volume": [1000, 1100, 1200], "amount": [101000, 112200, 123600],
        })
        result = m.tokenize_timeseries(df)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].dtype == torch.long

    def test_rejects_nan(self, make_model):
        m = make_model(max_context=64, min_context=5)
        df = pd.DataFrame({
            "open": [100, float("nan")], "high": [103, 104],
            "low": [99, 100], "close": [101, 102],
            "volume": [1000, 1100], "amount": [101000, 112200],
        })
        with pytest.raises(ValueError, match="NaN"):
            m.tokenize_timeseries(df)

    def test_rejects_inf(self, make_model):
        m = make_model(max_context=64, min_context=5)
        df = pd.DataFrame({
            "open": [100, float("inf")], "high": [103, 104],
            "low": [99, 100], "close": [101, 102],
            "volume": [1000, 1100], "amount": [101000, 112200],
        })
        with pytest.raises(ValueError, match="infinite"):
            m.tokenize_timeseries(df)

    def test_rejects_negative_prices(self, make_model):
        m = make_model(max_context=64, min_context=5)
        df = pd.DataFrame({
            "open": [-1, 101], "high": [103, 104],
            "low": [99, 100], "close": [101, 102],
            "volume": [1000, 1100], "amount": [101000, 112200],
        })
        with pytest.raises(ValueError, match="negative"):
            m.tokenize_timeseries(df)

    def test_missing_columns(self, make_model):
        m = make_model(max_context=64, min_context=5)
        df = pd.DataFrame({"open": [100], "high": [103]})  # missing low, close
        with pytest.raises(ValueError, match="Missing"):
            m.tokenize_timeseries(df)

    def test_padding_short_sequence(self, make_model):
        m = make_model(max_context=64, min_context=5)
        df = pd.DataFrame({
            "open": [100], "high": [103], "low": [99], "close": [101],
            "volume": [1000], "amount": [101000],
        })
        result = m.tokenize_timeseries(df)
        assert result["input_ids"].size(0) >= m.min_context


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_and_load_safetensors(self, make_model, tmp_path):
        m = make_model(max_context=64)
        save_dir = str(tmp_path / "model_out")
        m.save_pretrained(save_dir, save_format="both")

        assert os.path.exists(os.path.join(save_dir, "model.safetensors"))
        assert os.path.exists(os.path.join(save_dir, "config.json"))

    def test_save_pytorch_format(self, make_model, tmp_path):
        m = make_model(max_context=64)
        save_dir = str(tmp_path / "model_pt")
        m.save_pretrained(save_dir, save_format="pytorch")
        assert os.path.exists(os.path.join(save_dir, "pytorch_model.bin"))


# ---------------------------------------------------------------------------
# KronosClassificationConfig
# ---------------------------------------------------------------------------

class TestClassificationConfig:

    def test_to_dict(self):
        import classification.kronos_classification_base as mod
        cfg = mod.KronosClassificationConfig(num_classes=3, max_context=256)
        d = cfg.to_dict()
        assert d["num_classes"] == 3
        assert d["max_context"] == 256

    def test_from_dict(self):
        import classification.kronos_classification_base as mod
        d = {"num_classes": 5, "max_context": 128, "pooling_strategy": "max"}
        cfg = mod.KronosClassificationConfig.from_dict(d)
        assert cfg.num_classes == 5
        assert cfg.max_context == 128
        assert cfg.pooling_strategy == "max"


# ---------------------------------------------------------------------------
# Checkpoint validation
# ---------------------------------------------------------------------------

class TestCheckpointValidation:

    def test_validate_nonexistent_file(self):
        import classification.kronos_classification_base as mod
        with pytest.raises(FileNotFoundError):
            mod._validate_checkpoint("/nonexistent/path.bin")

    def test_validate_empty_file(self, tmp_path):
        import classification.kronos_classification_base as mod
        fp = tmp_path / "empty.bin"
        fp.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            mod._validate_checkpoint(str(fp))

    def test_validate_valid_file(self, tmp_path):
        import classification.kronos_classification_base as mod
        fp = tmp_path / "valid.bin"
        fp.write_bytes(b"\x00\x01\x02")
        assert mod._validate_checkpoint(str(fp)) is True
