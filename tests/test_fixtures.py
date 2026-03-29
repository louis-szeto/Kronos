"""
Shared fixture definitions that need to be available across test modules.
This is imported by conftest.py-style fixtures.
"""

import os
import pytest
import yaml


@pytest.fixture()
def sample_yaml_config(tmp_path):
    """Create a minimal YAML config file for finetune_csv tests."""
    config = {
        "data": {
            "data_path": str(tmp_path / "data.csv"),
            "lookback_window": 30,
            "predict_window": 5,
            "max_context": 64,
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
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_weight_decay": 0.1,
            "accumulation_steps": 1,
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
            "skip_existing": False,
        },
        "device": {"use_cuda": False, "device_id": 0},
        "distributed": {"use_ddp": False},
    }
    fp = tmp_path / "config.yaml"
    with open(fp, "w") as f:
        yaml.dump(config, f)
    return str(fp)
