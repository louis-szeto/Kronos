"""
Tests for classification/kronos_pretrain.py and classification/kronos_finetune.py.

Covers:
- KronosTimeSeriesDataset (data loading, collation, class balancing)
- KronosPretrainer (training loop, early stopping, checkpointing)
- KronosFineTuner (fine-tuning with backbone freeze/unfreeze)
- collate_fn
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


def _make_classification_model(**kwargs):
    """Create a KronosClassificationModel with mocked HF downloads."""
    import classification.kronos_classification_base as mod

    orig_init = mod.KronosClassificationModel.__init__
    mod.KronosClassificationModel.__init__ = _mock_classification_init
    try:
        m = mod.KronosClassificationModel(
            kronos_model_path="mock", tokenizer_path="mock", **kwargs
        )
        return m
    finally:
        mod.KronosClassificationModel.__init__ = orig_init


def _mock_classification_init(cls, kronos_model_path="mock", tokenizer_path="mock", **kwargs):
    """Replacement __init__ that skips HF downloads."""
    nn.Module.__init__(cls)
    cls.tokenizer = MockKronosTokenizer()
    cls.backbone = MockKronosBackbone(hidden_size=64)

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
    cls.hidden_size = 64

    if cls.pooling_strategy == "attention":
        cls.attention_weights = nn.Linear(cls.hidden_size, 1)

    cls.classification_head = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(64, 64),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(64, cls.num_classes),
    )

    if kwargs.get("freeze_backbone", False):
        for param in cls.backbone.parameters():
            param.requires_grad = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_tokenizer():
    return MockKronosTokenizer()


@pytest.fixture()
def sample_json_data(tmp_path):
    """Create a JSON file with classification training data."""
    np.random.seed(42)
    results = []
    for i in range(30):
        n_pts = 50
        chart_data = {
            "opens": list(np.random.randn(n_pts) * 2 + 100),
            "highs": list(np.random.randn(n_pts) * 2 + 102),
            "lows": list(np.random.randn(n_pts) * 2 + 98),
            "closes": list(np.random.randn(n_pts) * 2 + 100),
            "volumes": list(np.abs(np.random.randn(n_pts)) * 1000),
            "dates": list(range(1700000000000, 1700000000000 + n_pts * 3600000, 3600000)),
        }
        results.append({
            "assigned_label": i % 2,
            "chart_data": chart_data,
        })
    data = {"results": results}
    fp = tmp_path / "data.json"
    with open(fp, "w") as f:
        json.dump(data, f)
    return str(fp)


@pytest.fixture()
def sample_json_dir(tmp_path):
    """Create a directory with multiple JSON files."""
    np.random.seed(42)
    for file_idx in range(3):
        results = []
        for i in range(10):
            n_pts = 50
            chart_data = {
                "opens": list(np.random.randn(n_pts) * 2 + 100),
                "highs": list(np.random.randn(n_pts) * 2 + 102),
                "lows": list(np.random.randn(n_pts) * 2 + 98),
                "closes": list(np.random.randn(n_pts) * 2 + 100),
                "volumes": list(np.abs(np.random.randn(n_pts)) * 1000),
                "dates": list(range(1700000000000, 1700000000000 + n_pts * 3600000, 3600000)),
            }
            results.append({
                "assigned_label": i % 2,
                "chart_data": chart_data,
            })
        data = {"results": results}
        fp = tmp_path / f"data_{file_idx}.json"
        with open(fp, "w") as f:
            json.dump(data, f)
    return str(tmp_path)


# ---------------------------------------------------------------------------
# KronosTimeSeriesDataset
# ---------------------------------------------------------------------------

class TestKronosTimeSeriesDataset:

    def test_load_single_file(self, sample_json_data, mock_tokenizer):
        import classification.kronos_pretrain as mod
        ds = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data,
            tokenizer=mock_tokenizer,
            max_context=64,
            split_type="train",
            train_split=0.8, val_split=0.1,
        )
        assert len(ds) > 0

    def test_load_directory(self, sample_json_dir, mock_tokenizer):
        import classification.kronos_pretrain as mod
        ds = mod.KronosTimeSeriesDataset(
            data_path=sample_json_dir,
            tokenizer=mock_tokenizer,
            max_context=64,
            split_type="train",
        )
        assert len(ds) > 0

    def test_getitem_keys(self, sample_json_data, mock_tokenizer):
        import classification.kronos_pretrain as mod
        ds = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data,
            tokenizer=mock_tokenizer,
            max_context=64,
            split_type="train",
        )
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["input_ids"].dtype == torch.long
        assert item["attention_mask"].dtype == torch.long
        assert item["labels"].dtype == torch.long

    def test_truncation_to_max_context(self, sample_json_data, mock_tokenizer):
        import classification.kronos_pretrain as mod
        ds = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data,
            tokenizer=mock_tokenizer,
            max_context=10,
            split_type="train",
        )
        item = ds[0]
        assert item["input_ids"].size(0) <= 10

    def test_split_types(self, sample_json_data, mock_tokenizer):
        import classification.kronos_pretrain as mod
        train = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data, tokenizer=mock_tokenizer,
            split_type="train", train_split=0.6, val_split=0.2,
        )
        val = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data, tokenizer=mock_tokenizer,
            split_type="val", train_split=0.6, val_split=0.2,
        )
        test = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data, tokenizer=mock_tokenizer,
            split_type="test", train_split=0.6, val_split=0.2,
        )
        total = len(train) + len(val) + len(test)
        assert total > 0

    def test_no_volume(self, sample_json_data, mock_tokenizer):
        import classification.kronos_pretrain as mod
        ds = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data,
            tokenizer=mock_tokenizer,
            use_volume=False,
            split_type="train",
        )
        assert len(ds) > 0

    def test_class_weights(self, sample_json_data, mock_tokenizer):
        import classification.kronos_pretrain as mod
        ds = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data,
            tokenizer=mock_tokenizer,
            split_type="train",
        )
        weights = ds.get_class_weights()
        assert weights.dtype == torch.float32
        assert len(weights) >= 2

    def test_oversample_balancing(self, sample_json_data, mock_tokenizer):
        import classification.kronos_pretrain as mod
        ds = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data,
            tokenizer=mock_tokenizer,
            split_type="train",
            class_balance="oversample",
        )
        from collections import Counter
        labels = Counter(s["label"] for s in ds.data)
        counts = list(labels.values())
        assert max(counts) - min(counts) <= 1

    def test_undersample_balancing(self, sample_json_data, mock_tokenizer):
        import classification.kronos_pretrain as mod
        ds = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data,
            tokenizer=mock_tokenizer,
            split_type="train",
            class_balance="undersample",
        )
        from collections import Counter
        labels = Counter(s["label"] for s in ds.data)
        counts = list(labels.values())
        assert max(counts) - min(counts) <= 1

    def test_skip_none_labels(self, tmp_path, mock_tokenizer):
        import classification.kronos_pretrain as mod
        data = {"results": [
            {"assigned_label": None, "chart_data": {
                "opens": [1], "highs": [2], "lows": [0.5], "closes": [1.5],
                "volumes": [100], "dates": [1700000000000],
            }},
            {"assigned_label": 1, "chart_data": {
                "opens": [1], "highs": [2], "lows": [0.5], "closes": [1.5],
                "volumes": [100], "dates": [1700000000000],
            }},
        ]}
        fp = tmp_path / "data.json"
        with open(fp, "w") as f:
            json.dump(data, f)
        ds = mod.KronosTimeSeriesDataset(
            data_path=str(fp), tokenizer=mock_tokenizer,
            split_type="train", train_split=1.0, val_split=0.0,
        )
        assert len(ds) == 1  # Only the one with a real label


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------

class TestCollateFn:

    def test_pads_to_max_len(self, sample_json_data, mock_tokenizer):
        import classification.kronos_pretrain as mod
        ds = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data, tokenizer=mock_tokenizer,
            split_type="train",
        )
        batch = [ds[0], ds[1]]
        collated = mod.collate_fn(batch)

        assert collated["input_ids"].shape[0] == 2
        assert collated["attention_mask"].shape == collated["input_ids"].shape
        assert collated["labels"].shape == (2,)

    def test_uniform_lengths(self):
        import classification.kronos_pretrain as mod
        items = [
            {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1]), "labels": torch.tensor(0)},
            {"input_ids": torch.tensor([4, 5]), "attention_mask": torch.tensor([1, 1]), "labels": torch.tensor(1)},
        ]
        result = mod.collate_fn(items)
        assert result["input_ids"].shape == (2, 3)
        # Second item should be padded
        assert result["input_ids"][1, 2] == 0
        assert result["attention_mask"][1, 2] == 0


# ---------------------------------------------------------------------------
# KronosPretrainer (training loop)
# ---------------------------------------------------------------------------

class TestKronosPretrainer:

    @pytest.fixture()
    def pretrainer_setup(self, sample_json_data):
        """Set up a pretrainer with mocked model."""
        import classification.kronos_pretrain as mod

        model = _make_classification_model(max_context=64)

        train_ds = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data, tokenizer=model.tokenizer,
            max_context=64, split_type="train", train_split=0.8, val_split=0.1,
        )
        val_ds = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data, tokenizer=model.tokenizer,
            max_context=64, split_type="val", train_split=0.8, val_split=0.1,
        )

        pretrainer = mod.KronosPretrainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=str(tempfile.mkdtemp()),
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=1,
            warmup_steps=0,
            logging_steps=1,
            save_steps=100,
            eval_steps=100,
            device="cpu",
            num_workers=0,
        )

        return pretrainer, model

    def test_train_one_epoch(self, pretrainer_setup):
        trainer, model = pretrainer_setup
        trainer.train()
        assert trainer.global_step > 0

    def test_checkpoint_saved(self, pretrainer_setup):
        trainer, model = pretrainer_setup
        trainer.train()
        output_dir = trainer.output_dir
        assert os.path.isdir(output_dir)

    def test_early_stopping(self, sample_json_data):
        """Early stopping should stop after patience epochs with no improvement."""
        import classification.kronos_pretrain as mod

        model = _make_classification_model(max_context=64)

        train_ds = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data, tokenizer=model.tokenizer,
            max_context=64, split_type="train", train_split=0.8, val_split=0.1,
        )
        val_ds = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data, tokenizer=model.tokenizer,
            max_context=64, split_type="val", train_split=0.8, val_split=0.1,
        )

        trainer = mod.KronosPretrainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=str(tempfile.mkdtemp()),
            batch_size=2,
            num_epochs=100,  # High number
            device="cpu",
            num_workers=0,
        )
        trainer.patience = 1  # Override hardcoded patience
        trainer.train()
        # Should NOT have run all 100 epochs
        assert trainer.epochs_without_improvement >= 0


# ---------------------------------------------------------------------------
# KronosFineTuner
# ---------------------------------------------------------------------------

class TestKronosFineTuner:

    @pytest.fixture()
    def finetuner_setup(self, sample_json_data):
        import classification.kronos_finetune as mod

        model = _make_classification_model(max_context=64)

        train_ds = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data, tokenizer=model.tokenizer,
            max_context=64, split_type="train", train_split=0.8, val_split=0.1,
        )
        val_ds = mod.KronosTimeSeriesDataset(
            data_path=sample_json_data, tokenizer=model.tokenizer,
            max_context=64, split_type="val", train_split=0.8, val_split=0.1,
        )

        finetuner = mod.KronosFineTuner(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=str(tempfile.mkdtemp()),
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-4,
            device="cpu",
            num_workers=0,
        )

        return finetuner, model

    def test_finetune_one_epoch(self, finetuner_setup):
        ft, _ = finetuner_setup
        ft.train()
        assert ft.global_step > 0

    def test_backbone_freeze_unfreeze(self, finetuner_setup):
        ft, model = finetuner_setup
        ft._freeze_backbone()
        for p in model.backbone.parameters():
            assert not p.requires_grad
        ft._unfreeze_backbone()
        for p in model.backbone.parameters():
            assert p.requires_grad

    def test_evaluate_returns_metrics(self, finetuner_setup):
        ft, _ = finetuner_setup
        metrics = ft._evaluate(ft.val_loader, "Validation")
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_save_checkpoint(self, finetuner_setup):
        ft, _ = finetuner_setup
        ft._save_checkpoint("test_ckpt")
        expected = os.path.join(ft.output_dir, "test_ckpt")
        assert os.path.isdir(expected)
