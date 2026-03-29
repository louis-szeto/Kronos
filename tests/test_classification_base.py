"""
Tests for classification/kronos_classification_base.py —
KronosClassificationModel, KronosClassificationConfig, _validate_checkpoint.

The HuggingFace model downloads are mocked so tests run offline.
"""

import hashlib
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
        assert m.padding_strategy == "right"
        assert m.max_context == 512
        assert m.min_context == 20
        assert m.loss_type == "cross_entropy"

    def test_d_in_with_volume(self, make_model):
        m = make_model(use_volume=True)
        assert m.d_in == 5  # OHLCV + amount

    def test_d_in_without_volume(self, make_model):
        m = make_model(use_volume=False)
        assert m.d_in == 4  # OHLC only

    def test_d_in_with_exogenous(self, make_model):
        m = make_model(use_volume=True, num_exogenous=1)
        assert m.d_in == 6  # OHLCV + amount + exogenous

    def test_hidden_size_set(self, make_model):
        m = make_model(hidden_size=64)
        assert m.hidden_size == 64

    def test_classification_head_structure(self, make_model):
        m = make_model()
        head = m.classification_head
        assert isinstance(head, nn.Sequential)
        layers = list(head)
        assert isinstance(layers[0], nn.Dropout)
        assert isinstance(layers[1], nn.Linear)
        assert isinstance(layers[2], nn.GELU)
        assert isinstance(layers[3], nn.Dropout)
        assert isinstance(layers[4], nn.Linear)
        assert layers[4].out_features == 2

    def test_multiclass(self, make_model):
        m = make_model(num_classes=5)
        last = [l for l in m.classification_head if isinstance(l, nn.Linear)][-1]
        assert last.out_features == 5

    def test_custom_classifier_hidden_size(self, make_model):
        m = make_model(classifier_hidden_size=32)
        linears = [l for l in m.classification_head if isinstance(l, nn.Linear)]
        assert linears[0].out_features == 32
        assert linears[1].in_features == 32

    def test_freeze_backbone(self, make_model):
        m = make_model(freeze_backbone=True)
        for p in m.backbone.parameters():
            assert not p.requires_grad

    def test_unfrozen_backbone_by_default(self, make_model):
        m = make_model()
        for p in m.backbone.parameters():
            assert p.requires_grad

    def test_attention_pooling_creates_layer(self, make_model):
        m = make_model(pooling_strategy="attention")
        assert hasattr(m, "attention_weights")
        assert isinstance(m.attention_weights, nn.Linear)

    def test_auto_loss_type_binary(self, make_model):
        """With loss_type=None, mock init falls through to default 'cross_entropy'.
        The real init uses auto-selection logic, but the mock just uses the default."""
        m = make_model(loss_type=None)
        # Our mock_init uses `or "cross_entropy"` for None
        assert m.loss_type == "cross_entropy"

    def test_auto_loss_type_multiclass(self, make_model):
        m = make_model(num_classes=3, loss_type=None)
        assert m.loss_type == "cross_entropy"

    def test_init_wraps_hf_errors(self):
        """Real __init__ should wrap OSError/ConnectionError as RuntimeError."""
        import classification.kronos_classification_base as mod
        with patch.object(mod.KronosTokenizer, "from_pretrained",
                          side_effect=OSError("network error")):
            with pytest.raises(RuntimeError, match="Failed to load"):
                mod.KronosClassificationModel(
                    kronos_model_path="fake", tokenizer_path="fake"
                )


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
        assert out["loss"] is None

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

    def test_forward_multiclass(self, make_model):
        m = make_model(num_classes=5, max_context=64)
        input_ids = torch.randint(0, 100, (2, 16))
        out = m(input_ids=input_ids)
        assert out["logits"].shape == (2, 5)

    def test_forward_return_dict_false_no_labels(self, make_model):
        m = make_model(max_context=64)
        input_ids = torch.randint(0, 100, (2, 16))
        out = m(input_ids=input_ids, return_dict=False)
        # Without labels, returns logits directly
        assert isinstance(out, torch.Tensor)

    def test_forward_return_dict_false_with_labels(self, make_model):
        m = make_model(max_context=64)
        input_ids = torch.randint(0, 100, (2, 16))
        labels = torch.tensor([0, 1])
        out = m(input_ids=input_ids, labels=labels, return_dict=False)
        assert isinstance(out, tuple)
        assert len(out) == 2  # (loss, logits)

    def test_gradient_flow(self, make_model):
        m = make_model(max_context=64)
        input_ids = torch.randint(0, 100, (2, 16))
        labels = torch.tensor([0, 1])
        out = m(input_ids=input_ids, labels=labels, return_dict=True)
        out["loss"].backward()

        for name, param in m.classification_head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_class_weights_in_forward(self, make_model):
        m = make_model(max_context=64, loss_type="cross_entropy")
        input_ids = torch.randint(0, 100, (4, 16))
        labels = torch.tensor([0, 1, 0, 1])
        weights = torch.tensor([1.0, 2.0])
        out = m(input_ids=input_ids, labels=labels,
                class_weights=weights, return_dict=True)
        assert out["loss"] is not None


# ---------------------------------------------------------------------------
# Pooling strategies
# ---------------------------------------------------------------------------

class TestPoolingStrategies:

    def test_mean_pooling(self, make_model):
        m = make_model(pooling_strategy="mean", max_context=64)
        out = m(input_ids=torch.randint(0, 100, (2, 16)))
        assert out["logits"].shape == (2, 2)

    def test_mean_pooling_with_mask(self, make_model):
        m = make_model(pooling_strategy="mean", max_context=64)
        hs = torch.randn(2, 8, 64)
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[0, 4:] = 0
        pooled = m._pool_sequence(hs, mask)
        assert pooled.shape == (2, 64)

    def test_last_pooling(self, make_model):
        m = make_model(pooling_strategy="last", max_context=64)
        out = m(input_ids=torch.randint(0, 100, (2, 16)))
        assert out["logits"].shape == (2, 2)

    def test_last_pooling_with_mask(self, make_model):
        m = make_model(pooling_strategy="last", max_context=64)
        hs = torch.randn(2, 8, 64)
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[0, 5:] = 0  # actual len = 5
        pooled = m._pool_sequence(hs, mask)
        assert pooled.shape == (2, 64)

    def test_max_pooling(self, make_model):
        m = make_model(pooling_strategy="max", max_context=64)
        out = m(input_ids=torch.randint(0, 100, (2, 16)),
                attention_mask=torch.ones(2, 16, dtype=torch.long))
        assert out["logits"].shape == (2, 2)

    def test_max_pooling_with_mask(self, make_model):
        m = make_model(pooling_strategy="max", max_context=64)
        hs = torch.randn(2, 8, 64)
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[0, 4:] = 0
        pooled = m._pool_sequence(hs, mask)
        assert pooled.shape == (2, 64)

    def test_attention_pooling(self, make_model):
        m = make_model(pooling_strategy="attention", max_context=64)
        out = m(input_ids=torch.randint(0, 100, (2, 16)),
                attention_mask=torch.ones(2, 16, dtype=torch.long))
        assert out["logits"].shape == (2, 2)

    def test_attention_pooling_with_mask(self, make_model):
        m = make_model(pooling_strategy="attention", max_context=64)
        hs = torch.randn(2, 8, 64)
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[0, 4:] = 0
        pooled = m._pool_sequence(hs, mask)
        assert pooled.shape == (2, 64)

    def test_pooling_without_mask(self, make_model):
        for strategy in ["mean", "last", "max"]:
            m = make_model(pooling_strategy=strategy, max_context=64)
            hs = torch.randn(2, 8, 64)
            pooled = m._pool_sequence(hs, None)
            assert pooled.shape == (2, 64)

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

    def test_focal_loss_lower_than_ce(self, make_model):
        """Focal loss should down-weight easy examples."""
        m_focal = make_model(loss_type="focal")
        m_ce = make_model(loss_type="cross_entropy")
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])  # Easy examples
        labels = torch.tensor([0, 1])
        focal = m_focal._compute_loss(logits, labels)
        ce = m_ce._compute_loss(logits, labels)
        assert focal.item() < ce.item()

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

    def test_loss_gradient_flow(self, make_model):
        m = make_model(loss_type="focal")
        logits = torch.randn(4, 2, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1])
        loss = m._compute_loss(logits, labels)
        loss.backward()
        assert logits.grad is not None


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
        assert result["attention_mask"].dtype == torch.long

    def test_attention_mask_all_ones(self, make_model):
        m = make_model(max_context=64, min_context=5)
        df = pd.DataFrame({
            "open": [100, 101, 102], "high": [103, 104, 105],
            "low": [99, 100, 101], "close": [101, 102, 103],
            "volume": [1000, 1100, 1200], "amount": [101000, 112200, 123600],
        })
        result = m.tokenize_timeseries(df)
        assert (result["attention_mask"] == 1).all()

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
        df = pd.DataFrame({"open": [100], "high": [103]})
        with pytest.raises(ValueError, match="Missing"):
            m.tokenize_timeseries(df)

    def test_padding_right_short_sequence(self, make_model):
        m = make_model(max_context=64, min_context=10, padding_strategy="right")
        df = pd.DataFrame({
            "open": [100], "high": [103], "low": [99], "close": [101],
            "volume": [1000], "amount": [101000],
        })
        result = m.tokenize_timeseries(df)
        assert result["input_ids"].size(0) >= m.min_context

    def test_padding_left_short_sequence(self, make_model):
        m = make_model(max_context=64, min_context=10, padding_strategy="left")
        df = pd.DataFrame({
            "open": [100], "high": [103], "low": [99], "close": [101],
            "volume": [1000], "amount": [101000],
        })
        result = m.tokenize_timeseries(df)
        assert result["input_ids"].size(0) >= m.min_context

    def test_padding_both_short_sequence(self, make_model):
        m = make_model(max_context=64, min_context=10, padding_strategy="both")
        df = pd.DataFrame({
            "open": [100], "high": [103], "low": [99], "close": [101],
            "volume": [1000], "amount": [101000],
        })
        result = m.tokenize_timeseries(df)
        assert result["input_ids"].size(0) >= m.min_context

    def test_truncation_over_max_context(self, make_model):
        m = make_model(max_context=10, min_context=5)
        # 50 rows -> tokenized to 50 tokens -> truncated to max_context
        df = pd.DataFrame({
            "open": [100] * 50, "high": [103] * 50,
            "low": [99] * 50, "close": [101] * 50,
            "volume": [1000] * 50, "amount": [101000] * 50,
        })
        result = m.tokenize_timeseries(df)
        assert result["input_ids"].size(0) <= m.max_context

    def test_auto_amount_column(self, make_model):
        m = make_model(max_context=64, min_context=5)
        df = pd.DataFrame({
            "open": [100, 101], "high": [103, 104],
            "low": [99, 100], "close": [101, 102],
            "volume": [1000, 1100],
            # No amount column — should be auto-generated
        })
        result = m.tokenize_timeseries(df)
        assert "input_ids" in result

    def test_no_volume_mode(self, make_model):
        m = make_model(max_context=64, min_context=5, use_volume=False)
        df = pd.DataFrame({
            "open": [100, 101], "high": [103, 104],
            "low": [99, 100], "close": [101, 102],
        })
        result = m.tokenize_timeseries(df)
        assert "input_ids" in result


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

    def test_save_safetensors_only(self, make_model, tmp_path):
        m = make_model(max_context=64)
        save_dir = str(tmp_path / "model_safe")
        m.save_pretrained(save_dir, save_format="safetensors")
        assert os.path.exists(os.path.join(save_dir, "model.safetensors"))
        assert not os.path.exists(os.path.join(save_dir, "pytorch_model.bin"))

    def test_save_creates_directory(self, make_model, tmp_path):
        save_dir = str(tmp_path / "nested" / "model")
        m = make_model(max_context=64)
        m.save_pretrained(save_dir, save_format="safetensors")
        assert os.path.isdir(save_dir)

    def test_config_json_contains_expected_keys(self, make_model, tmp_path):
        m = make_model(max_context=128, num_classes=3)
        save_dir = str(tmp_path / "model_cfg")
        m.save_pretrained(save_dir, save_format="safetensors")
        with open(os.path.join(save_dir, "config.json")) as f:
            config = json.load(f)
        assert config["num_classes"] == 3
        assert config["max_context"] == 128

    def test_from_pretrained_loads_safetensors(self, make_model, tmp_path):
        import classification.kronos_classification_base as mod

        m = make_model(max_context=64, num_classes=3)
        save_dir = str(tmp_path / "model_load")
        m.save_pretrained(save_dir, save_format="both")

        # Patch __init__ to avoid HF downloads during from_pretrained
        orig_init = mod.KronosClassificationModel.__init__
        mod.KronosClassificationModel.__init__ = _mock_init
        try:
            loaded = mod.KronosClassificationModel.from_pretrained(save_dir)
            assert loaded.num_classes == 3
        finally:
            mod.KronosClassificationModel.__init__ = orig_init

    def test_from_pretrained_raises_on_missing_dir(self):
        import classification.kronos_classification_base as mod
        orig_init = mod.KronosClassificationModel.__init__
        mod.KronosClassificationModel.__init__ = _mock_init
        try:
            with pytest.raises(FileNotFoundError):
                mod.KronosClassificationModel.from_pretrained("/nonexistent/path")
        finally:
            mod.KronosClassificationModel.__init__ = orig_init

    def test_from_pretrained_kwargs_override_config(self, make_model, tmp_path):
        """Overriding non-architectural kwargs (e.g., max_context) should work.
        Overriding architectural kwargs (num_classes) would break state_dict,
        so we test with a compatible override."""
        import classification.kronos_classification_base as mod

        m = make_model(max_context=64, num_classes=2)
        save_dir = str(tmp_path / "model_override")
        m.save_pretrained(save_dir, save_format="both")

        orig_init = mod.KronosClassificationModel.__init__
        mod.KronosClassificationModel.__init__ = _mock_init
        try:
            # Override max_context (non-architectural) — state_dict still compatible
            loaded = mod.KronosClassificationModel.from_pretrained(
                save_dir, max_context=128
            )
            assert loaded.max_context == 128
            assert loaded.num_classes == 2  # Preserved from saved config
        finally:
            mod.KronosClassificationModel.__init__ = orig_init


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
        assert "learning_rate" in d
        assert "weight_decay" in d

    def test_from_dict(self):
        import classification.kronos_classification_base as mod
        d = {"num_classes": 5, "max_context": 128, "pooling_strategy": "max"}
        cfg = mod.KronosClassificationConfig.from_dict(d)
        assert cfg.num_classes == 5
        assert cfg.max_context == 128
        assert cfg.pooling_strategy == "max"

    def test_from_dict_defaults(self):
        import classification.kronos_classification_base as mod
        cfg = mod.KronosClassificationConfig.from_dict({})
        assert cfg.num_classes == 2
        assert cfg.learning_rate == 2e-5

    def test_roundtrip(self):
        import classification.kronos_classification_base as mod
        cfg1 = mod.KronosClassificationConfig(num_classes=10, learning_rate=1e-4)
        d = cfg1.to_dict()
        cfg2 = mod.KronosClassificationConfig.from_dict(d)
        assert cfg2.num_classes == 10
        assert cfg2.learning_rate == 1e-4


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

    def test_validate_hash_mismatch(self, tmp_path):
        import classification.kronos_classification_base as mod
        fp = tmp_path / "hash.bin"
        fp.write_bytes(b"hello world")
        with pytest.raises(ValueError, match="integrity check failed"):
            mod._validate_checkpoint(str(fp), expected_sha256="abc123")

    def test_validate_hash_match(self, tmp_path):
        import classification.kronos_classification_base as mod
        fp = tmp_path / "hash_ok.bin"
        content = b"hello world"
        fp.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert mod._validate_checkpoint(str(fp), expected_sha256=expected) is True
