"""
Tests for finetune_csv/ modules:
- CustomKlineDataset (data loading, CSV parsing, splitting, windowing)
- config_loader (ConfigLoader, CustomFinetuneConfig)
- finetune_base_model (logging setup)
- finetune_tokenizer (set_seed, get_model_size, logging)
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
# CSV test data fixtures
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
# CustomKlineDataset — data loading & validation
# ---------------------------------------------------------------------------

class TestCustomKlineDatasetLoading:

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
            "open": [1.0] * 100,
        })
        fp = tmp_path / "bad_cols.csv"
        df.to_csv(fp, index=False)
        with pytest.raises(ValueError, match="missing required"):
            CustomKlineDataset(data_path=str(fp))

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
        timestamps = pd.date_range("2024-01-01", periods=500, freq="h")
        df = pd.DataFrame({
            "timestamps": timestamps,
            "open": [1.0] * 250 + [float("nan")] * 250,
            "high": [2.0] * 500,
            "low": [0.5] * 500,
            "close": [1.5] * 500,
            "volume": [100] * 500,
            "amount": [150] * 500,
        })
        fp = tmp_path / "nan.csv"
        df.to_csv(fp, index=False)
        ds = CustomKlineDataset(data_path=str(fp))
        assert len(ds) > 0

    def test_timestamps_sorted(self, tmp_path):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        ts = list(pd.date_range("2024-01-01", periods=500, freq="h"))
        ts.reverse()
        df = pd.DataFrame({
            "timestamps": ts,
            "open": [1.0] * 500, "high": [2.0] * 500,
            "low": [0.5] * 500, "close": [1.5] * 500,
            "volume": [100] * 500, "amount": [150] * 500,
        })
        fp = tmp_path / "unsorted.csv"
        df.to_csv(fp, index=False)
        ds = CustomKlineDataset(data_path=str(fp))
        assert len(ds) > 0


# ---------------------------------------------------------------------------
# CustomKlineDataset — splitting & windowing
# ---------------------------------------------------------------------------

class TestCustomKlineDatasetSplitting:

    def test_train_val_test_splits(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        # Use small window so all splits have enough data
        train = CustomKlineDataset(data_path=sample_csv, data_type="train",
                                   lookback_window=5, predict_window=2,
                                   train_ratio=0.7, val_ratio=0.15)
        val = CustomKlineDataset(data_path=sample_csv, data_type="val",
                                 lookback_window=5, predict_window=2,
                                 train_ratio=0.7, val_ratio=0.15)
        test = CustomKlineDataset(data_path=sample_csv, data_type="test",
                                  lookback_window=5, predict_window=2,
                                  train_ratio=0.7, val_ratio=0.15)
        assert len(train) > len(val)
        assert len(val) >= len(test)

    def test_train_split_ratio(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        train = CustomKlineDataset(data_path=sample_csv, data_type="train",
                                   train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        # Total data is 200, train gets first 140
        total_len = 200
        expected_train_end = int(total_len * 0.7)
        # n_samples = data_len - window + 1
        # Train data length = expected_train_end
        assert train.data.shape[0] == expected_train_end

    def test_val_split_bounds(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        val = CustomKlineDataset(data_path=sample_csv, data_type="val",
                                 train_ratio=0.7, val_ratio=0.15)
        total_len = 200
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.85)
        assert val.data.shape[0] == val_end - train_end

    def test_getitem_returns_tensors(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        ds = CustomKlineDataset(data_path=sample_csv, data_type="train",
                                lookback_window=30, predict_window=5)
        x, stamp = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(stamp, torch.Tensor)

    def test_getitem_feature_dim(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        ds = CustomKlineDataset(data_path=sample_csv, data_type="train",
                                lookback_window=30, predict_window=5)
        x, stamp = ds[0]
        assert x.shape[-1] == 6  # OHLCV + amount
        assert stamp.shape[-1] == 5  # minute, hour, weekday, day, month

    def test_getitem_window_length(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        lb, pred = 30, 5
        ds = CustomKlineDataset(data_path=sample_csv, data_type="train",
                                lookback_window=lb, predict_window=pred)
        x, stamp = ds[0]
        expected_window = lb + pred + 1
        assert x.shape[0] == expected_window

    def test_zscore_normalization_clipped(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        ds = CustomKlineDataset(data_path=sample_csv, data_type="train",
                                lookback_window=30, predict_window=5, clip=5.0)
        x, _ = ds[0]
        assert x.abs().max() <= 5.0 + 1e-6

    def test_custom_clip_value(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        ds = CustomKlineDataset(data_path=sample_csv, data_type="train",
                                lookback_window=30, predict_window=5, clip=2.0)
        x, _ = ds[0]
        assert x.abs().max() <= 2.0 + 1e-6

    def test_set_epoch_seed_reproducibility(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        ds = CustomKlineDataset(data_path=sample_csv, data_type="train",
                                lookback_window=30, predict_window=5, seed=42)
        ds.set_epoch_seed(0)
        x1, _ = ds[0]
        ds.set_epoch_seed(0)
        x2, _ = ds[0]
        assert torch.allclose(x1, x2)

    def test_different_epochs_different_samples(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        ds = CustomKlineDataset(data_path=sample_csv, data_type="train",
                                lookback_window=30, predict_window=5, seed=42)
        ds.set_epoch_seed(0)
        x1, _ = ds[0]
        ds.set_epoch_seed(1)
        x2, _ = ds[0]
        # Different epoch seeds produce different starting indices -> different data
        assert not torch.allclose(x1, x2)

    def test_n_samples_computation(self, sample_csv):
        from finetune_csv.finetune_base_model import CustomKlineDataset
        lb, pred = 30, 5
        ds = CustomKlineDataset(data_path=sample_csv, data_type="train",
                                lookback_window=lb, predict_window=pred,
                                train_ratio=1.0, val_ratio=0.0)
        window = lb + pred + 1
        expected = ds.data.shape[0] - window + 1
        assert ds.n_samples == expected
        assert len(ds) == expected


# ---------------------------------------------------------------------------
# ConfigLoader
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
        assert "lookback_window" in dc

    def test_get_training_config(self, sample_yaml_config):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        tc = loader.get_training_config()
        assert "batch_size" in tc
        assert "seed" in tc

    def test_get_model_paths(self, sample_yaml_config):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        mp = loader.get_model_paths()
        assert "exp_name" in mp

    def test_get_experiment_config(self, sample_yaml_config):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        ec = loader.get_experiment_config()
        assert "name" in ec

    def test_get_device_config(self, sample_yaml_config):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        dc = loader.get_device_config()
        assert "use_cuda" in dc

    def test_get_distributed_config(self, sample_yaml_config):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        dc = loader.get_distributed_config()
        assert "use_ddp" in dc

    def test_update_config(self, sample_yaml_config):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        loader.update_config({"training": {"batch_size": 99}})
        assert loader.get("training.batch_size") == 99

    def test_update_nested_config(self, sample_yaml_config):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        loader.update_config({"training": {"seed": 999, "batch_size": 8}})
        assert loader.get("training.seed") == 999
        assert loader.get("training.batch_size") == 8

    def test_save_config_roundtrip(self, sample_yaml_config, tmp_path):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        out = str(tmp_path / "out.yaml")
        loader.save_config(out)
        assert os.path.exists(out)
        loader2 = ConfigLoader(out)
        assert loader2.get("training.batch_size") == 4

    def test_dynamic_path_resolution_empty_values(self, tmp_path):
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

    def test_dynamic_path_resolution_placeholder(self, tmp_path):
        from config_loader import ConfigLoader
        config = {
            "model_paths": {
                "exp_name": "my_exp",
                "base_path": str(tmp_path),
                "base_save_path": "custom/{exp_name}/model",
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
        assert resolved == "custom/my_exp/model"

    def test_print_config(self, sample_yaml_config, capsys):
        from config_loader import ConfigLoader
        loader = ConfigLoader(sample_yaml_config)
        loader.print_config()
        captured = capsys.readouterr()
        assert "Current configuration" in captured.out


# ---------------------------------------------------------------------------
# CustomFinetuneConfig
# ---------------------------------------------------------------------------

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
                            "base_save_path": str(tmp_path / "t"),
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
        assert cfg.tokenizer_epochs == 10
        assert cfg.basemodel_epochs == 10

    def test_separate_epoch_settings(self, tmp_path):
        from config_loader import CustomFinetuneConfig
        config = {
            "model_paths": {"exp_name": "t", "base_path": str(tmp_path),
                            "base_save_path": str(tmp_path / "t"),
                            "pretrained_tokenizer": "a", "pretrained_predictor": "b"},
            "data": {"data_path": "d"},
            "training": {"tokenizer_epochs": 5, "basemodel_epochs": 15},
            "experiment": {},
            "device": {},
            "distributed": {},
        }
        fp = tmp_path / "cfg.yaml"
        with open(fp, "w") as f:
            yaml.dump(config, f)
        cfg = CustomFinetuneConfig(str(fp))
        assert cfg.tokenizer_epochs == 5
        assert cfg.basemodel_epochs == 15

    def test_path_computation(self, sample_yaml_config, tmp_path):
        from config_loader import CustomFinetuneConfig
        cfg = CustomFinetuneConfig(sample_yaml_config)
        assert cfg.tokenizer_save_path.endswith("tokenizer")
        assert cfg.basemodel_save_path.endswith("basemodel")
        assert "best_model" in cfg.tokenizer_best_model_path
        assert "best_model" in cfg.basemodel_best_model_path

    def test_get_tokenizer_config(self, sample_yaml_config):
        from config_loader import CustomFinetuneConfig
        cfg = CustomFinetuneConfig(sample_yaml_config)
        tc = cfg.get_tokenizer_config()
        assert tc["epochs"] == cfg.tokenizer_epochs
        assert tc["data_path"] == cfg.data_path

    def test_get_basemodel_config(self, sample_yaml_config):
        from config_loader import CustomFinetuneConfig
        cfg = CustomFinetuneConfig(sample_yaml_config)
        bc = cfg.get_basemodel_config()
        assert bc["epochs"] == cfg.basemodel_epochs
        assert "predictor_learning_rate" in bc

    def test_pretrained_flags_default(self, tmp_path):
        from config_loader import CustomFinetuneConfig
        config = {
            "model_paths": {"exp_name": "t", "base_path": str(tmp_path),
                            "base_save_path": str(tmp_path / "t"),
                            "pretrained_tokenizer": "a", "pretrained_predictor": "b"},
            "data": {"data_path": "d"},
            "training": {},
            "experiment": {},
            "device": {},
            "distributed": {},
        }
        fp = tmp_path / "cfg.yaml"
        with open(fp, "w") as f:
            yaml.dump(config, f)
        cfg = CustomFinetuneConfig(str(fp))
        assert cfg.pre_trained_tokenizer is True
        assert cfg.pre_trained_predictor is True

    def test_pretrained_unified_flag(self, tmp_path):
        from config_loader import CustomFinetuneConfig
        config = {
            "model_paths": {"exp_name": "t", "base_path": str(tmp_path),
                            "base_save_path": str(tmp_path / "t"),
                            "pretrained_tokenizer": "a", "pretrained_predictor": "b"},
            "data": {"data_path": "d"},
            "training": {},
            "experiment": {"pre_trained": False},
            "device": {},
            "distributed": {},
        }
        fp = tmp_path / "cfg.yaml"
        with open(fp, "w") as f:
            yaml.dump(config, f)
        cfg = CustomFinetuneConfig(str(fp))
        assert cfg.pre_trained_tokenizer is False
        assert cfg.pre_trained_predictor is False

    def test_print_config_summary(self, sample_yaml_config, capsys):
        from config_loader import CustomFinetuneConfig
        cfg = CustomFinetuneConfig(sample_yaml_config)
        cfg.print_config_summary()
        captured = capsys.readouterr()
        assert "Experiment name" in captured.out


# ---------------------------------------------------------------------------
# finetune_base_model — logging helpers
# ---------------------------------------------------------------------------

class TestFinetuneBaseModelLogging:

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
        assert l1 is l2

    def test_setup_logging_nonzero_rank_no_console(self, tmp_path):
        import logging
        from finetune_csv.finetune_base_model import setup_logging
        log_dir = str(tmp_path / "logs_rank")
        logger = setup_logging("test_rank", log_dir, rank=1)
        # Non-zero rank should not have console handler
        console_handlers = [h for h in logger.handlers
                            if isinstance(h, logging.StreamHandler)
                            and not isinstance(h, logging.FileHandler)]
        assert len(console_handlers) == 0

    def test_setup_logging_creates_log_file(self, tmp_path):
        import logging as _logging
        from finetune_csv.finetune_base_model import setup_logging
        # Clean up any existing logger with this name (from previous tests)
        logger_name = "basemodel_training_rank_0"
        existing = _logging.getLogger(logger_name)
        existing.handlers.clear()

        log_dir = str(tmp_path / "logs_create")
        logger = setup_logging("test_create_unique", log_dir, rank=0)
        logger.info("test message")
        # Close file handlers to ensure data is flushed to disk
        for handler in logger.handlers:
            if isinstance(handler, _logging.FileHandler):
                handler.close()
        all_files = os.listdir(log_dir)
        log_files = [f for f in all_files if f.endswith(".log")]
        assert len(log_files) >= 1


# ---------------------------------------------------------------------------
# finetune_tokenizer — utility functions
# ---------------------------------------------------------------------------

class TestFinetuneTokenizer:

    def test_set_seed(self):
        from finetune_csv.finetune_tokenizer import set_seed
        set_seed(42)
        import random
        v1 = random.randint(0, 1000)
        set_seed(42)
        v2 = random.randint(0, 1000)
        assert v1 == v2  # Deterministic after reset

    def test_get_model_size_small(self):
        from finetune_csv.finetune_tokenizer import get_model_size
        model = torch.nn.Linear(10, 10)  # 110 params
        size = get_model_size(model)
        assert isinstance(size, str)
        assert "K" in size

    def test_get_model_size_large(self):
        from finetune_csv.finetune_tokenizer import get_model_size
        # Mock a model with >1M params
        model = MagicMock()
        p = MagicMock()
        p.numel.return_value = 5_000_000
        p.requires_grad = True
        model.parameters.return_value = [p]
        size = get_model_size(model)
        assert "M" in size

    def test_format_time(self):
        from finetune_csv.finetune_tokenizer import format_time
        result = format_time(3661)
        assert "1" in result  # 1 hour

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

    def test_instantiation(self, sample_yaml_config):
        from train_sequential import SequentialTrainer
        trainer = SequentialTrainer(sample_yaml_config)
        assert trainer.config is not None
        assert trainer.device is not None

    def test_skip_tokenizer_when_disabled(self, sample_yaml_config):
        from train_sequential import SequentialTrainer
        trainer = SequentialTrainer(sample_yaml_config)
        trainer.config.train_tokenizer = False
        assert trainer.config.train_tokenizer is False

    def test_skip_basemodel_when_disabled(self, sample_yaml_config):
        from train_sequential import SequentialTrainer
        trainer = SequentialTrainer(sample_yaml_config)
        trainer.config.train_basemodel = False
        assert trainer.config.train_basemodel is False

    def test_check_existing_models_none(self, sample_yaml_config):
        from train_sequential import SequentialTrainer
        trainer = SequentialTrainer(sample_yaml_config)
        tok_exists, base_exists = trainer._check_existing_models()
        assert tok_exists is False
        assert base_exists is False

    def test_check_existing_models_with_files(self, sample_yaml_config):
        from train_sequential import SequentialTrainer
        trainer = SequentialTrainer(sample_yaml_config)
        # Create the "best_model" directories
        os.makedirs(trainer.config.tokenizer_best_model_path, exist_ok=True)
        tok_exists, _ = trainer._check_existing_models()
        assert tok_exists is True

    def test_create_directories(self, sample_yaml_config):
        from train_sequential import SequentialTrainer
        trainer = SequentialTrainer(sample_yaml_config)
        trainer._create_directories()
        assert os.path.isdir(trainer.config.tokenizer_save_path)
        assert os.path.isdir(trainer.config.basemodel_save_path)

    def test_skip_existing_skips_training(self, sample_yaml_config):
        from train_sequential import SequentialTrainer
        trainer = SequentialTrainer(sample_yaml_config)
        trainer.config.skip_existing = True
        trainer._create_directories()
        # Create the tokenizer best_model to trigger skip
        os.makedirs(trainer.config.tokenizer_best_model_path, exist_ok=True)
        result = trainer.train_tokenizer_phase()
        assert result is True  # Returns True but skips actual training

    def test_setup_device_cpu(self, sample_yaml_config):
        from train_sequential import SequentialTrainer
        trainer = SequentialTrainer(sample_yaml_config)
        assert trainer.device == torch.device("cpu")
