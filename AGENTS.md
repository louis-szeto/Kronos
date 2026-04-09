# Kronos - Harness Engineering Map

Quick-start reference for AI agents working in this repo.

## Project Summary

Kronos is a time-series foundation model for financial markets. It tokenizes OHLCV data via Binary Spherical Quantization (BSQ), then uses a dual-head Transformer for autoregressive prediction and classification.

## Directory Map

| Directory | Purpose | Entry Points |
|-----------|---------|-------------|
| `model/` | Core neural nets (KronosTokenizer, Kronos, KronosPredictor) | `model/__init__.py` |
| `classification/` | Classification head, pretrain/finetune/RL scripts | `classification/kronos_classification_base.py` |
| `finetune/` | Qlib-based finetuning (tokenizer + predictor) | `finetune/train_tokenizer.py`, `finetune/train_predictor.py` |
| `finetune_csv/` | CSV-based sequential finetuning pipeline | `finetune_csv/train_sequential.py` |
| `webui/` | Flask web interface for interactive prediction | `webui/app.py` |
| `examples/` | Prediction demo scripts | `examples/prediction_example.py` |
| `tests/` | Test suite (mocked HF/torch, runs offline on CPU) | `tests/conftest.py` |

## Key Classes

| Class | Location | Role |
|-------|----------|------|
| `KronosTokenizer` | `model/kronos.py` | Encoder-decoder with BSQ quantizer |
| `Kronos` | `model/kronos.py` | Dual-head Transformer predictor |
| `KronosPredictor` | `model/kronos.py` | High-level predict/predict_batch API |
| `KronosClassificationModel` | `classification/kronos_classification_base.py` | Wraps Kronos backbone + classification head |
| `KronosPretrainer` | `classification/kronos_pretrain.py` | Multi-GPU pretraining loop |
| `KronosFineTuner` | `classification/kronos_finetune.py` | Supervised fine-tuning loop |
| `PolicyGradientFinetuner` | `classification/kronos_rl_finetune.py` | REINFORCE RL fine-tuning |
| `KronosClassificationPipeline` | `classification/kronos_inference.py` | Batch inference on DataFrames |
| `SequentialTrainer` | `finetune_csv/train_sequential.py` | Orchestrates tokenizer then predictor training |
| `CustomFinetuneConfig` | `finetune_csv/config_loader.py` | YAML-driven config for CSV finetuning |

## Model Architecture (2-sentence version)

Input OHLCV is encoded by `KronosTokenizer` into s1/s2 dual-bit tokens via BSQ. `Kronos` model uses a hierarchical embedding + dependency-aware Transformer to autoregressively predict next tokens, decoded back to OHLCV space.

## Data Flow

```
CSV/JSON → DataFrame → z-score normalize → KronosTokenizer.encode → s1,s2 tokens
→ Kronos.forward → s1_logits,s2_logits → decode → denormalize → predictions
```

## Running Tests

```bash
pytest tests/ -v           # All tests, offline, CPU-only
pytest tests/test_kronos_model.py  # Model unit tests
```

## Key Constraints

- All pretrained model paths use HuggingFace Hub (`NeoQuasar/Kronos-*`)
- Classification scripts use `from model import ...` (must run from project root or have it in PYTHONPATH)
- `finetune/` requires Qlib data; `finetune_csv/` works with plain CSV
- Web UI runs on port 7070, loads models on-demand via `/api/load-model`

## Further Reading

- `docs/architecture.md` - Full ML pipeline and component interactions
- `docs/domains.md` - Module dependency rules
- `docs/golden-principles.md` - Conventions and patterns
- `docs/quality.md` - Quality grades per domain
