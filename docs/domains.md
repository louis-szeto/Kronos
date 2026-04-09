# Domains

Module dependency rules and boundaries.

## Domain Map

```
┌─────────────────────────────────────────────────────┐
│                     webui/                           │
│  Flask app, REST API, Plotly charts                  │
│  Depends on: model/                                  │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│                 classification/                       │
│  Classification head, pretrain/finetune/RL/inference │
│  Depends on: model/                                  │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│               finetune_csv/                          │
│  CSV-based sequential finetuning pipeline            │
│  Depends on: model/                                  │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│                  finetune/                            │
│  Qlib-based finetuning (tokenizer + predictor)       │
│  Depends on: model/                                  │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│                   model/                             │
│  Core: KronosTokenizer, Kronos, KronosPredictor      │
│  Depends on: torch, einops, huggingface_hub          │
└─────────────────────────────────────────────────────┘
```

## Dependency Rules

| From | May Import | May NOT Import |
|------|-----------|----------------|
| `model/` | `torch`, `einops`, `pandas`, `numpy`, `huggingface_hub` | `classification/`, `finetune/`, `finetune_csv/`, `webui/` |
| `classification/` | `model/` (via `from model import ...`) | `finetune/`, `finetune_csv/`, `webui/` |
| `finetune/` | `model/`, own `config.py`, `dataset.py`, `utils/` | `classification/`, `finetune_csv/`, `webui/` |
| `finetune_csv/` | `model/`, own modules | `classification/`, `finetune/`, `webui/` |
| `webui/` | `model/` | `classification/`, `finetune/`, `finetune_csv/` |
| `tests/` | All modules (via `conftest.py` path setup) | Production code may not import `tests/` |

## Cross-Domain Patterns

### Shared Backbone

`model/` is the foundation. All other domains import from it:

```
from model import Kronos, KronosTokenizer, KronosPredictor    # common pattern
from model.kronos import Kronos, KronosTokenizer               # direct import
```

Classification additionally uses `sys.path.append` for relative imports within its directory.

### Duplicated Patterns

These patterns appear in both `classification/` and `finetune_csv/` independently:

| Pattern | `classification/` | `finetune_csv/` |
|---------|-------------------|-----------------|
| Dataset class | `KronosTimeSeriesDataset` | `CustomKlineDataset` |
| Collate function | `collate_fn` | N/A (DataLoader default) |
| Trainer class | `KronosPretrainer` / `KronosFineTuner` | `train_model()` function |
| Config | `KronosClassificationConfig` | `CustomFinetuneConfig` (YAML) |
| Class balancing | oversample/undersample/weights | N/A |

These are intentionally independent - each domain is self-contained.

### Import Resolution

All scripts use `sys.path.append('../')` or `sys.path.append(os.path.dirname(...))` to resolve imports. The project must be run from its root directory or have the root in `PYTHONPATH`.

## Configuration Isolation

| Domain | Config Method | Config Location |
|--------|--------------|-----------------|
| `model/` | HuggingFace Hub config.json | Auto-downloaded |
| `classification/` | Python `KronosClassificationConfig` class | Inline / argparse defaults |
| `finetune/` | Python `Config` class | `finetune/config.py` |
| `finetune_csv/` | YAML files | `finetune_csv/configs/*.yaml` |
| `webui/` | Hardcoded + runtime API params | `webui/app.py` |

## Key Files Per Domain

### model/
- `module.py` - All neural network building blocks
- `kronos.py` - KronosTokenizer, Kronos, KronosPredictor, inference functions
- `__init__.py` - Public API exports + `get_model_class()` registry

### classification/
- `kronos_classification_base.py` - Model + config + serialization
- `kronos_pretrain.py` - Dataset + pretrainer + CLI
- `kronos_finetune.py` - Dataset + finetuner + CLI (metrics-based)
- `kronos_rl_finetune.py` - REINFORCE finetuner + CLI
- `kronos_inference.py` - Pipeline + data conversion + checkpoint analysis

### finetune/
- `config.py` - Global configuration
- `dataset.py` - QlibDataset (pickle-based)
- `train_tokenizer.py` - Tokenizer DDP training
- `train_predictor.py` - Predictor DDP training
- `utils/training_utils.py` - Shared DDP/logging utilities

### finetune_csv/
- `config_loader.py` - YAML parsing + CustomFinetuneConfig
- `finetune_tokenizer.py` - Tokenizer training on CSV data
- `finetune_base_model.py` - Predictor training + CustomKlineDataset
- `train_sequential.py` - SequentialTrainer orchestrator
- `configs/` - YAML config templates
