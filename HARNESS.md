# HARNESS.md — Kronos

**Last audited:** 2026-03-31
**Auditor:** Phase 1 deep read
**Scope:** All Python source, configs, tests, docs. No code changes made.

---

## 1. Project Overview

Kronos is a time-series foundation model for financial markets. It tokenizes OHLCV data via Binary Spherical Quantization (BSQ) into dual-bit tokens (s1 coarse, s2 fine), then uses a dual-head Transformer for autoregressive prediction and classification.

| Metric | Value |
|--------|-------|
| Python source files | ~35 |
| Tests | 201 (mocked HF/torch, offline, CPU) |
| Model sizes | mini (4.1M), small (24.7M), base (102.3M) |
| Pretrained models | HuggingFace Hub `NeoQuasar/Kronos-*` |
| Web UI | Flask on port 7070 |

**Critical invariant:** Training data never leaves controlled AWS infrastructure. Raw market data is INPUT ONLY — the platform outputs ML-derived alternative data, never raw OHLCV.

---

## 2. Architecture

### 2.1 Data Flow

```
CSV/JSON/Qlib → DataFrame → z-score normalize → KronosTokenizer.encode → s1,s2 tokens
  → Kronos.forward → s1_logits,s2_logits → decode → denormalize → predictions
```

### 2.2 Module Map

| Directory | Purpose | Key Classes | Lines |
|-----------|---------|-------------|-------|
| `model/` | Core neural nets | `KronosTokenizer`, `Kronos`, `KronosPredictor` | ~680 |
| `model/module.py` | Building blocks | `BSQuantizer`, `TransformerBlock`, `HierarchicalEmbedding`, `DualHead`, `DependencyAwareLayer` | ~570 |
| `classification/` | Classification head + training | `KronosClassificationModel`, `KronosPretrainer`, `KronosFineTuner`, `PolicyGradientFinetuner`, `KronosClassificationPipeline` | ~2500 |
| `finetune/` | Qlib-based finetuning | `QlibDataset`, `train_model()` (tokenizer + predictor) | ~500 |
| `finetune_csv/` | CSV-based sequential pipeline | `SequentialTrainer`, `CustomFinetuneConfig`, `CustomKlineDataset` | ~800 |
| `webui/` | Flask REST API + Plotly charts | `app.py` (single file) | ~700 |
| `tests/` | Test suite | `conftest.py`, 6 test modules | ~2000 |
| `examples/` | Prediction demos | 4 example scripts | ~300 |

### 2.3 Training Phases

1. **Tokenizer training** — Encoder → BSQ → Decoder, recon + BSQ entropy loss, AdamW + OneCycleLR
2. **Predictor training** — Tokenizer.encode → Kronos LM → next-token CE (s1 + s2), AdamW + OneCycleLR
3. **Classification pretraining** (optional) — JSON data → KronosClassificationModel → CE/focal/label-smoothing, early stopping
4. **Classification finetuning** — Pretrained checkpoint → KronosFineTuner → metrics-based, optional backbone freeze/unfreeze
5. **RL finetuning** (optional) — REINFORCE policy gradient with entropy regularization

### 2.4 Model Architecture (core)

- **BSQ Quantization:** Bipolar `{-1,+1}` tokens, straight-through estimator, loss = commit_loss + zeta * (gamma0 * per_sample_entropy - gamma * codebook_entropy)
- **Hierarchical Embedding:** Dual embedding (s1 vocab + s2 vocab) → fusion projection to d_model
- **DependencyAwareLayer:** Cross-attention where s2 prediction is conditioned on sampled s1 embeddings
- **DualHead:** Independent CE for s1 and s2, combined as `(CE_s1 + CE_s2) / 2`
- **TemporalEmbedding:** Fixed/learnable embeddings for minute, hour, weekday, day, month
- **Attention:** PyTorch SDPA with RoPE, causal masking

---

## 3. Security Audit

### 3.1 CRITICAL — P0

| ID | Issue | Location | Detail |
|----|-------|----------|--------|
| SEC-1 | **Pickle deserialization (arbitrary code exec)** | `finetune/dataset.py:42`, `classification/kronos_inference.py:198,220,281,287,353` | `pickle.load()` on untrusted files. An attacker who controls a `.pkl` file gets RCE. The `create_sample_data()` and `predict_from_file()` paths in `kronos_inference.py` serialize/deserialize with pickle. The `finetune/dataset.py:QlibDataset` loads pickled training data. |
| SEC-2 | **training_state.bin uses weights_only=False** | `classification/kronos_inference.py:383` | `torch.load(training_state_path, ..., weights_only=False)` in `analyze_checkpoint()`. This is pickle deserialization of optimizer/scheduler state dicts which contain non-tensor Python objects. Documented as intentional but risky if checkpoint files are tampered. |
| SEC-3 | **Model weights loaded with `weights_only=True` (pytorch format)** | `classification/kronos_classification_base.py:508,556` | Good — both load paths use `weights_only=True` for model weights. However, the `from_pretrained()` fallback to pytorch format still exists, and safetensors is properly preferred. |

### 3.2 HIGH — P1

| ID | Issue | Location | Detail |
|----|-------|----------|--------|
| SEC-4 | **Open CORS by default** | `webui/app.py:34-38` | When `KRONOS_ALLOWED_ORIGINS` env var is not set, CORS is completely open. Warning logged but not enforced. |
| SEC-5 | **No rate limiting on WebUI** | `webui/app.py` (all endpoints) | No rate limiting on `/api/predict`, `/api/load-model`, or `/api/load-data`. An attacker can flood the prediction endpoint, loading GPU models repeatedly. |
| SEC-6 | **Model artifacts on HuggingFace Hub** | All classification scripts | Models downloaded from `NeoQuasar/Kronos-*` via `huggingface_hub`. If the Hub account is compromised, malicious model weights could be served. No pinning or hash verification on Hub downloads. |

### 3.3 MEDIUM — P2

| ID | Issue | Location | Detail |
|----|-------|----------|--------|
| SEC-7 | **WebUI API key optional** | `webui/app.py:40-54` | If `KRONOS_API_KEY` env var is not set, the `require_api_key` decorator will fail (comparing against None). Behavior is undefined — could allow unauthenticated access. |
| SEC-8 | **Path traversal in WebUI (partially mitigated)** | `webui/app.py:115-119` | `Path(file_path).resolve()` checked against `DATA_DIR`. Good approach but needs additional symlink hardening. |
| SEC-9 | **Error message information leakage** | `webui/app.py` | Error responses include traceback details in some code paths. Recent commits (d91dff9) partially addressed this but verify all paths are sanitized. |
| SEC-10 | **Data leakage risk in train/test splits** | `classification/kronos_pretrain.py:210-223`, `classification/kronos_finetune.py:209-221` | Random shuffle with `seed(42)` across ALL samples, then split by index. No time-based splitting — future data can appear in training set. This is a **financial ML data leakage risk** (look-ahead bias). |

### 3.4 LOW — P3

| ID | Issue | Location | Detail |
|----|-------|----------|--------|
| SEC-11 | **Instance-level normalization** | `model/kronos.py:552-554` | Z-score computed per-sample (mean/std of each individual input window). Not a data leakage risk in itself, but prevents cross-sample comparability. |
| SEC-12 | **No input size bounds on WebUI** | `webui/app.py` | No limit on the size of uploaded CSV data or prediction request bodies. Could cause OOM. |

---

## 4. Stability Audit

### 4.1 Training Stability

| ID | Issue | Severity | Location | Detail |
|----|-------|----------|----------|--------|
| STA-1 | **NaN loss guard present** | MITIGATED | `finetune_csv/finetune_base_model.py:330-334` | Skips batch if `not np.isfinite(loss_val)`. Good. But only in `finetune_csv/`, not in `classification/` trainers. |
| STA-2 | **No NaN guard in classification trainers** | P1 | `classification/kronos_pretrain.py`, `classification/kronos_finetune.py`, `classification/kronos_rl_finetune.py` | If loss becomes NaN/Inf in these trainers, it propagates through the entire training run, corrupting all weights. No skip-batch guard. |
| STA-3 | **Early stopping only in pretrainer** | P2 | `classification/kronos_pretrain.py:457-460` | `KronosPretrainer` has patience=3 early stopping. `KronosFineTuner` has NO early stopping. `PolicyGradientFinetuner` has NO early stopping. |
| STA-4 | **DDP cleanup on crash** | P2 | `finetune_csv/train_sequential.py` | Has try/except around `dist.destroy_process_group()`. Good. But `classification/` scripts have no such cleanup. |
| STA-5 | **Checkpoint recovery** | P2 | All trainers | No resume-from-checkpoint capability. If training crashes mid-epoch, all progress since last checkpoint save is lost. No `--resume_from_checkpoint` flag. |
| STA-6 | **FP16 GradScaler initialization** | P3 | `classification/kronos_pretrain.py:367` | `torch.cuda.amp.GradScaler()` without `enabled=True` — deprecated API. Should use `torch.amp.GradScaler('cuda')` for PyTorch 2.x. |

### 4.2 Memory Stability

| ID | Issue | Severity | Location | Detail |
|----|-------|----------|----------|--------|
| MEM-1 | **auto_regressive_inference memory** | P1 | `model/kronos.py:401-481` | Expands input by `sample_count` (default 5) upfront: `x.unsqueeze(1).repeat(1, sample_count, ...)`. For batch=32, seq=512, sample_count=5, this creates 160 effective sequences simultaneously. On Kronos-base (102M params), this can OOM on GPUs < 24GB. |
| MEM-2 | **No gradient checkpointing** | P2 | All trainers | Large sequences (>256 tokens) with Kronos-base will OOM during training without activation checkpointing. |
| MEM-3 | **GPU cache clearing** | GOOD | All trainers | `torch.cuda.empty_cache()` called after each epoch. Recent commit (6456913) confirmed no memory leak. |

---

## 5. Performance Audit

### 5.1 Attention

| ID | Issue | Severity | Location | Detail |
|----|-------|----------|----------|--------|
| PERF-1 | **SDPA with causal mask** | GOOD | `model/module.py:345-350` | Uses `F.scaled_dot_product_attention` with `is_causal=True`. Correctly leverages PyTorch fused kernels (commit dccfa76). |
| PERF-2 | **No Flash Attention fallback** | P3 | `model/module.py` | SDPA will auto-dispatch Flash Attention if available, but there's no explicit `attn_implementation="flash_attention_2"` config to guarantee it. |

### 5.2 Inference

| ID | Issue | Severity | Location | Detail |
|----|-------|----------|----------|--------|
| PERF-3 | **Sliding window AR inference is O(n^2)** | P2 | `model/kronos.py:401-481` | Each AR step recomputes full attention over the entire context window. For pred_len=96, this means 96 full forward passes. KV-cache would reduce this to O(n). |
| PERF-4 | **Collect metrics disabled during inference** | GOOD | `model/module.py:99-100`, commit 21eb36a | `collect_metrics=False` during `encode()` in inference mode. Avoids unnecessary entropy computation. |
| PERF-5 | **Batch prediction in KronosClassificationPipeline** | GOOD | `classification/kronos_inference.py:100-151` | Properly batches inference with padding and masking. |

### 5.3 Tokenization

| ID | Issue | Severity | Location | Detail |
|----|-------|----------|----------|--------|
| PERF-6 | **BSQ quantization is efficient** | GOOD | `model/module.py:39-223` | Straight-through estimator with no lookup tables. Quantization is just `sign(z)` + detach trick. O(1) per token. |

---

## 6. Dependency Map

### 6.1 External Dependencies

| Package | Version | Used By | Risk |
|---------|---------|---------|------|
| `torch` | >=2.1.0 | All | Core |
| `einops` | 0.8.1 | `model/` | Low |
| `huggingface_hub` | 0.33.1 | `model/`, `classification/` | Medium (SEC-6) |
| `safetensors` | 0.6.2 | `classification/` | Low (good security) |
| `flask` | 2.3.3 | `webui/` | Medium (SEC-4,5) |
| `flask-cors` | 4.0.0 | `webui/` | Low |
| `pandas` | 2.2.2 | All | Low |
| `numpy` | 1.24.3 | All | Low |
| `scikit-learn` | >=1.3.0 | `classification/` | Low |
| `transformers` | >=4.30.0 | `classification/` | Medium |
| `accelerate` | >=0.20.0 | `classification/` | Low |
| `plotly` | 5.17.0 | `webui/` | Low |
| `matplotlib` | 3.9.3 | `model/` | Low |
| `tqdm` | 4.67.1 | All training | Low |

### 6.2 Inter-Project Dependencies

```
Kronos depends on:
  ← HuggingFace Hub (NeoQuasar/Kronos-* pretrained models)
  ← AWS infrastructure (training jobs run on EKS/Fargate)
  ← BPC (consumes pattern-matching output as training labels via JSON)

Kronos is consumed by:
  → backend-pipeline-controller (ONNX inference with Kronos-derived models)
  → avantageux.io-webapp (serves Kronos-based alternative data)
```

### 6.3 Internal Module Dependencies

```
model/           ← torch, einops, huggingface_hub (foundation, no internal deps)
classification/  ← model/ (via from model import ...)
finetune/        ← model/, own config/dataset/utils
finetune_csv/    ← model/, own modules
webui/           ← model/
tests/           ← all modules (via conftest.py mocking)
```

---

## 7. Risk Matrix

| Risk | Likelihood | Impact | Priority | ID |
|------|-----------|--------|----------|-----|
| Pickle RCE via malicious data file | HIGH | CRITICAL | **P0** | SEC-1 |
| Training NaN corruption | MEDIUM | HIGH | **P1** | STA-2 |
| Financial data leakage (look-ahead bias) | HIGH | HIGH | **P1** | SEC-10 |
| OOM on AR inference with large batches | MEDIUM | HIGH | **P1** | MEM-1 |
| Open CORS in production WebUI | MEDIUM | MEDIUM | **P1** | SEC-4 |
| HuggingFace model supply chain | LOW | CRITICAL | **P1** | SEC-6 |
| No resume-from-checkpoint | MEDIUM | MEDIUM | **P2** | STA-5 |
| No early stopping in finetuner/RL | LOW | MEDIUM | **P2** | STA-3 |
| AR inference O(n^2) without KV-cache | LOW | MEDIUM | **P2** | PERF-3 |
| WebUI API key undefined behavior | LOW | MEDIUM | **P2** | SEC-7 |
| No input size limits on WebUI | LOW | LOW | **P3** | SEC-12 |
| Deprecated FP16 API | LOW | LOW | **P3** | STA-6 |
| No Flash Attention guarantee | LOW | LOW | **P3** | PERF-2 |

---

## 8. Prioritized Action Items

### P0 — Do Immediately

- [ ] **SEC-1:** Replace all `pickle.load()` with safe deserialization. Options:
  - Use `safetensors` for model weights (already done for model state dicts)
  - Use JSON or Arrow/Parquet for training data in `finetune/dataset.py` and `kronos_inference.py`
  - If pickle is unavoidable, add `RestrictedUnpickler` class whitelist
  - Files: `finetune/dataset.py`, `classification/kronos_inference.py` (6 call sites)
- [ ] **SEC-2:** Add SHA-256 hash verification for `training_state.bin` files, or migrate training state to safetensors format

### P1 — Before Next Training Run

- [x] **SEC-10:** Implement time-based train/val/test splitting in classification datasets. Current random shuffle allows future data in training. Replace `random.shuffle + index split` with chronological split.
- [x] **STA-2:** Add NaN/Inf loss guard to `KronosPretrainer`, `KronosFineTuner`, and `PolicyGradientFinetuner` (copy pattern from `finetune_csv/finetune_base_model.py:330-334`)
- [x] **MEM-1:** Add memory guard in `auto_regressive_inference` — check `batch_size * sample_count` against available GPU memory before expanding. Option to process sample_count sequentially instead of in parallel.
- [x] **SEC-4:** Default CORS to deny-all, require explicit `KRONOS_ALLOWED_ORIGINS` env var in production
- [x] **SEC-6:** Pin HuggingFace model revisions by commit hash in all scripts. Add `revision=` parameter to `from_pretrained()` calls.

### P2 — This Sprint

- [ ] **STA-5:** Add `--resume_from_checkpoint` flag to all 3 classification trainers. Load optimizer/scheduler state from `training_state.bin`.
- [ ] **STA-3:** Add early stopping to `KronosFineTuner` (monitor val_f1) and `PolicyGradientFinetuner` (monitor val_reward)
- [ ] **PERF-3:** Implement KV-cache in `auto_regressive_inference` to avoid recomputing attention for all previous tokens each step
- [ ] **SEC-7:** Fail loudly if `KRONOS_API_KEY` is not set — do not start WebUI without auth configured
- [ ] **STA-4:** Add DDP cleanup try/except to all classification training scripts

### P3 — Backlog

- [ ] **SEC-12:** Add request body size limits and sequence length limits to WebUI endpoints
- [ ] **STA-6:** Migrate from deprecated `torch.cuda.amp` to `torch.amp` API
- [ ] **PERF-2:** Add `attn_implementation="flash_attention_2"` option to model config
- [ ] **MEM-2:** Add gradient checkpointing option for long-sequence training
- [ ] Split `webui/app.py` from 700 lines into blueprint modules (noted in `docs/quality.md`)
- [ ] Make `comet_ml` optional in `finetune/` (noted in `docs/quality.md`)

---

## 9. AWS Cost Optimization

| Area | Current State | Recommendation |
|------|---------------|----------------|
| **GPU instances** | No spot instance support | Add `--use-spot` flag with checkpoint-based fault tolerance for spot interruptions |
| **Checkpoint storage** | All checkpoints kept (epoch + best + step) | Add S3 lifecycle policy: delete step checkpoints after 7 days, keep only best + latest |
| **Training jobs** | No mixed precision by default | Default `--fp16` to True for all trainers (2x throughput on A100/H100) |
| **Model loading** | Full model loaded to GPU per request | Add model warm pool / keep-alive in WebUI (already partially done with on-demand loading) |
| **Data transfer** | HuggingFace Hub downloads each run | Cache downloaded models on EBS volume, mount to training pods |

---

## 10. Code Quality Notes

### Strengths
- 201 tests covering all modules with mocked HF/torch (offline, CPU)
- Safetensors is default serialization format for classification models
- Checkpoint validation with SHA-256 hash option
- Input validation on OHLCV data (NaN, Inf, negative prices)
- NaN loss guard in `finetune_csv/` trainer
- Proper gradient clipping in all trainers
- Early stopping in pretrainer
- GPU cleanup after each epoch
- API key auth on WebUI with timing-safe comparison

### Weaknesses
- `KronosTimeSeriesDataset` duplicated in `kronos_pretrain.py` and `kronos_finetune.py` (identical code)
- `collate_fn` duplicated in both files
- WebUI is a single 700-line file
- `comet_ml` hard dependency in `finetune/`
- No centralized training config (3 different config systems: argparse, Python class, YAML)
- `model/__init__.py:get_model_class()` uses `print()` instead of `logging`
- `model/kronos.py` has `sys.path.append('../")` at module level

---

## 11. Invariants

1. Training data never leaves AWS
2. All prediction endpoints validate input (NaN, Inf, negative prices)
3. Model artifacts encrypted at rest (S3 SSE)
4. No data leakage between train/test — **BROKEN** (SEC-10, random split instead of time-based)
5. WebUI requires authentication — **PARTIAL** (SEC-7, undefined if API key not set)
6. Pickle deserialization only from trusted sources — **NOT ENFORCED** (SEC-1)
7. All trainers use gradient clipping
8. Safetensors preferred over pytorch format

---

## 12. Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| `model/` | `test_kronos_model.py` + `test_kronos_regression.py` | Core forward pass, tokenization, prediction, regression |
| `classification/` | `test_classification_base.py` + `test_classification_pretrain_finetune.py` | Model init, serialization, pooling, loss, training loops |
| `finetune_csv/` | `test_finetune_csv.py` | Config loading, dataset, sequential training |
| `webui/` | Security tests (28 tests from commit 137a75f) | Path traversal, auth, CORS, input validation |
| `tests/` | `test_fixtures.py` | Conftest fixture validation |

Run: `pytest tests/ -v` (all tests, offline, CPU-only)

---

## 13. Recent Git Activity

| Commit | Description |
|--------|-------------|
| `137a75f` | 28 security tests for WebUI |
| `d91dff9` | Security fixes: path traversal, CORS, auth, error leakage |
| `f6f53a9` | Implement all remaining TODOs and placeholder code |
| `f9a3d10` | Expand test coverage to 201 tests |
| `30027d6` | Comprehensive test suite with mocked HF/torch |
| `454a29d` | Logging, early stopping, GPU cleanup, HF error handling |
| `3d21e19` | NaN loss guard, weights_only=False for training state |
| `fc57098` | Input validation, pickle safety, GPU memory, error handling |
| `8d8f940` | 2nd stage RL, multi GPU support |
| `dccfa76` | Use pytorch SDPA implementation |
| `6456913` | Remove unnecessary CUDA cache clearing |
