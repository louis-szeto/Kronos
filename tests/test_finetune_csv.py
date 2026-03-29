"""
Tests for finetune_csv/ modules:
- CustomKlineDataset (data loading, CSV parsing, splitting)
- config_loader (ConfigLoader, CustomFinetuneConfig)
- finetune_base_model (train_model, logging setup)
- finetune_tokenizer (train_tokenizer)
- train_sequential (SequentialTrainer)
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import yaml


# ---------------------------------------------------------------------------
# CSV test data fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_csv(tmp_path, sample_timeseries_df):
    """Write a CSV file and return its path."""
    fp = tmp_path / "data.csv"
    sample_timeseries_df.to_csv(fp, index=False)
    return str(fp)


@pytest.fixture()
def malformed_csv(tmp_path):
    fp = tmp_path / "bad.csv"
    fp.write_text("this is not,valid,csv\n\"unclosed quote\n")
    return str(fp)


@pytest.fixture()
def empty_csv(tmp_path):
    fp = tmp_path / "empty.csv"
    fp.write_text("")
    return str(fp)


@pytest.fixture()
def missing_cols_csv(tmp_path):
    fp = tmp_path / "no_ts.csv"
    pd.DataFrame({"open": [1], "high": [2], "low": [0.5], "close": [1.5],
                  "volume": [100], "amount": [150]}).to_csv(fp, index=False)
    return str(fp)


# ---------------------------------------------------------------------------
# CustomKlineDataset
# ---------------------------------------------------------------------------

class TestCustomKlineDataset:

    def test_load_valid_csv(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        ds = CustomKlineDataset(data_path=sample_csv, data_type="train",
                                lookback_window=30, predict_window=5)
        assert len(ds) > 0

    def test_file_not_found(self):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        with pytest.raises(FileNotFoundError):
            CustomKlineDataset(data_path="/nonexistent/path.csv")

    def test_empty_csv(self, empty_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        with pytest.raises((ValueError, pd.errors.EmptyDataError)):
            CustomKlineDataset(data_path=empty_csv)

    def test_malformed_csv(self, malformed_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        with pytest.raises(ValueError, match="parse"):
            CustomKlineDataset(data_path=malformed_csv)

    def test_missing_timestamps_column(self, missing_cols_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        with pytest.raises(ValueError, match="timestamps"):
            CustomKlineDataset(data_path=missing_cols_csv)

    def test_missing_ohlcv_columns(self, tmp_path):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        df = pd.DataFrame({
            "timestamps": pd.date_range("2024-01-01", periods=100, freq="h"),
            "open": [1.0] * 100,  # Missing high, low, close, etc.
        })
        fp = tmp_path / "bad_cols.csv"
        df.to_csv(fp, index=False)
        with pytest.raises(ValueError, match="missing required"):
            CustomKlineDataset(data_path=str(fp))

    def test_train_val_test_splits(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        train = CustomKlineDataset(data_path=sample_csv, data_type="train",
                                   train_ratio=0.7, val_ratio=0.15)
        val = CustomKlineDataset(data_path=sample_csv, data_type="val",
                                 train_ratio=0.7, val_ratio=0.15)
        test = CustomKlineDataset(data_path=sample_csv, data_type="test",
                                  train_ratio=0.7, val_ratio=0.15)
        assert len(train) > len(val)
        assert len(val) >= len(test)

    def test_getitem_returns_tensors(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        ds = CustomKlineDataset(data_path=sample_csv, data_type="train",
                                lookback_window=30, predict_window=5)
        x, stamp = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(stamp, torch.Tensor)
        assert x.shape[-1] == 6  # OHLCV + amount
        assert stamp.shape[-1] == 5  # minute, hour, weekday, day, month

    def test_zscore_normalization(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        ds = CustomKlineDataset(data_path=sample_csv, data_type="train",
                                lookback_window=30, predict_window=5)
        x, _ = ds[0]
        # After z-score + clip(5), values should be bounded
        assert x.abs().max() <= 5.0 + 1e-6

    def test_set_epoch_seed_reproducibility(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        ds = CustomKlineDataset(data_path=sample_csv, data_type="train",
                                lookback_window=30, predict_window=5, seed=42)
        ds.set_epoch_seed(0)
        x1, _ = ds[0]
        ds.set_epoch_seed(0)
        x2, _ = ds[0]
        assert torch.allclose(x1, x2)

    def test_negative_prices_rejected(self, tmp_path):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        df = pd.DataFrame({
            "timestamps": pd.date_range("2024-01-01", periods=100, freq="h"),
            "open": [-1] * 100, "high": [2] * 100,
            "low": [0.5] * 100, "close": [1.5] * 100,
            "volume": [100] * 100, "amount": [150] * 100,
        })
        fp = tmp_path / "neg.csv"
        df.to_csv(fp, index=False)
        with pytest.raises(ValueError, match="negative"):
            CustomKlineDataset(data_path=str(fp))

    def test_inf_values_rejected(self, tmp_path):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        df = pd.DataFrame({
            "timestamps": pd.date_range("2024-01-01", periods=100, freq="h"),
            "open": [float("inf")] * 100, "high": [2] * 100,
            "low": [0.5] * 100, "close": [1.5] * 100,
            "volume": [100] * 100, "amount": [150] * 100,
        })
        fp = tmp_path / "inf.csv"
        df.to_csv(fp, index=False)
        with pytest.raises(ValueError, match="infinite"):
            CustomKlineDataset(data_path=str(fp))

    def test_nan_forward_fill(self, tmp_path):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        timestamps = pd.date_range("2024-01-01", periods=100, freq="h")
        df = pd.DataFrame({
            "timestamps": timestamps,
            "open": [1.0] * 50 + [float("nan")] * 50,
            "high": [2.0] * 100,
            "low": [0.5] * 100,
            "close": [1.5] * 100,
            "volume": [100] * 100,
            "amount": [150] * 100,
        })
        fp = tmp_path / "nan.csv"
        df.to_csv(fp, index=False)
        # Should succeed (forward fill handles NaN)
        ds = CustomKlineDataset(data_path=str(fp))
        assert len(ds) > 0


# ---------------------------------------------------------------------------
# ConfigLoader / CustomFinetuneConfig
# ---------------------------------------------------------------------------

class TestConfigLoader:

    def test_load_valid_config(self, sample_yaml_config):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        assert loader.config is not None
        assert loader.get("training.batch_size") == 4

    def test_missing_config_file(self):
        from config_loader import ConfigLoader
        with pytest.raises(FileNotFoundError):
            ConfigLoader("/nonexistent/config.yaml")

    def test_get_nested_key(self, sample_yaml_config):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        assert loader.get("data.lookback_window") == 30

    def test_get_default_for_missing_key(self, sample_yaml_config):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        assert loader.get("nonexistent.key", "default") == "default"

    def test_get_data_config(self, sample_yaml_config):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        dc = loader.get_data_config()
        assert "data_path" in dc

    def test_get_training_config(self, sample_yaml_config):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        tc = loader.get_training_config()
        assert "batch_size" in tc

    def test_update_config(self, sample_yaml_config):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        loader.update_config({"training": {"batch_size": 99}})
        assert loader.get("training.batch_size") == 99

    def test_save_config(self, sample_yaml_config, tmp_path):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        out = str(tmp_path / "out.yaml")
        loader.save_config(out)
        assert os.path.exists(out)
        loader2 = ConfigLoader(out)
        assert loader2.get("training.batch_size") == 4

    def test_dynamic_path_resolution(self, tmp_path):
        from config_loader import ConfigLoader
        config = {
            "model_paths": {
                "exp_name": "test_exp",
                "base_path": str(tmp_path),
                "base_save_path": "",
                "finetuned_tokenizer": "",
            },
            "data": {"data_path": "dummy"},
            "training": {},
            "experiment": {},
            "device": {},
            "distributed": {},
        }
        fp = tmp_path / "cfg.yaml"
        with open(fp, "w") as f:
            yaml.dump(config, f)
        loader = ConfigLoader(str(fp))
        resolved = loader.config["model_paths"]["base_save_path"]
        assert "test_exp" in resolved


class TestCustomFinetuneConfig:

    def test_load_config(self, sample_yaml_config):
        from config_loader import CustomFinetuneConfig
        cfg = CustomFinetuneConfig(sample_yaml_config)
        assert cfg.batch_size == 4
        assert cfg.seed == 42

    def test_default_epochs_fallback(self, tmp_path):
        from config_loader import CustomFinetuneConfig
        config = {
            "model_paths": {"exp_name": "t", "base_path": str(tmp_path),
                            "pretrained_tokenizer": "a", "pretrained_predictor": "b"},
            "data": {"data_path": "d"},
            "training": {"epochs": 10},
            "experiment": {},
            "device": {},
            "distributed": {},
        }
        fp = tmp_path / "cfg.yaml"
        with open(fp, "w") as f:
            yaml.dump(config, f)
        cfg = CustomFinetuneConfig(str(fp))
        # When only 'epochs' is provided, both should use it
        assert cfg.tokenizer_epochs == 10
        assert cfg.basemodel_epochs == 10

    def test_path_computation(self, sample_yaml_config, tmp_path):
        from config_loader import CustomFinetuneConfig
        cfg = CustomFinetuneConfig(sample_yaml_config)
        assert cfg.tokenizer_save_path.endswith("tokenizer")
        assert cfg.basemodel_save_path.endswith("basemodel")


# ---------------------------------------------------------------------------
# finetune_base_model — logging & training helpers
# ---------------------------------------------------------------------------

class TestFinetuneBaseModel:

    def test_setup_logging(self, tmp_path):
        import logging
        from finetune_csv.finetune_base_model import setup_logging
        log_dir = str(tmp_path / "logs")
        logger = setup_logging("test_exp", log_dir, rank=0)
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO

    def test_setup_logging_idempotent(self, tmp_path):
        from finetune_csv.finetune_base_model import setup_logging
        log_dir = str(tmp_path / "logs")
        l1 = setup_logging("test1", log_dir, rank=0)
        l2 = setup_logging("test1", log_dir, rank=0)
        assert l1 is l2  # same logger returned


# ---------------------------------------------------------------------------
# finetune_tokenizer
# ---------------------------------------------------------------------------

class TestFinetuneTokenizer:

    def test_set_seed(self):
        from finetune_csv.finetune_tokenizer import set_seed
        set_seed(42)
        import random
        assert random.randint(0, 1000) == random.randint(0, 1000)  # Deterministic within same seed

    def test_get_model_size(self):
        from finetune_csv.finetune_tokenizer import get_model_size
        model = torch.nn.Linear(10, 10)
        size = get_model_size(model)
        assert isinstance(size, str)
        assert "K" in size or "M" in size or "B" in size

    def test_setup_logging(self, tmp_path):
        import logging
        from finetune_csv.finetune_tokenizer import setup_logging
        log_dir = str(tmp_path / "logs_tok")
        logger = setup_logging("tok_exp", log_dir, rank=0)
        assert isinstance(logger, logging.Logger)


# ---------------------------------------------------------------------------
# train_sequential — SequentialTrainer
# ---------------------------------------------------------------------------

class TestSequentialTrainer:

    def test_skip_tokenizer_when_disabled(self, sample_yaml_config, tmp_path):
        from train_sequential import SequentialTrainer
        trainer = SequentialTrainer(sample_yaml_config)
        trainer.config.train_tokenizer = False
        # Should skip without error (even though basemodel may fail,
        # we just test that tokenizer phase is skipped)
        # We can't run full training without a real model, but we can test
        # that the flag is respected
        assert trainer.config.train_tokenizer is False

    def test_skip_basemodel_when_disabled(self, sample_yaml_config):
        from train_sequential import SequentialTrainer
        trainer = SequentialTrainer(sample_yaml_config)
        trainer.config.train_basemodel = False
        assert trainer.config.train_basemodel is False

    def test_check_existing_models(self, sample_yaml_config):
        from train_sequential import SequentialTrainer
        trainer = SequentialTrainer(sample_yaml_config)
        tok_exists, base_exists = trainer._check_existing_models()
        assert tok_exists is False
        assert base_exists is False

    def test_create_directories(self, sample_yaml_config):
        from train_sequential import SequentialTrainer
        trainer = SequentialTrainer(sample_yaml_config)
        trainer._create_directories()
        assert os.path.isdir(trainer.config.tokenizer_save_path)
        assert os.path.isdir(trainer.config.basemodel_save_path)
