# Golden Principles

Conventions and patterns extracted from the Kronos codebase.

## Model Patterns

### Dual-Token Architecture
- All tokenization produces **two indices**: `s1` (coarse) and `s2` (fine), each `s1_bits`/`s2_bits` wide
- Vocabulary sizes: `vocab_s1 = 2^s1_bits`, `vocab_s2 = 2^s2_bits`
- Always use `half=True` for encoding during prediction (returns tuple of s1/s2 indices)
- The `DependencyAwareLayer` conditions s2 prediction on s1 via cross-attention

### Quantization Convention
- BSQ outputs bipolar values `{-1, +1}`, scaled by `1/sqrt(codebook_dim)`
- `quantize()` uses straight-through estimator: `z + (zhat - z).detach()`
- Loss = `commit_loss + zeta * (gamma0 * per_sample_entropy - gamma * codebook_entropy)`

### Hierarchical Embedding
- Composite token ID = `(s1_id << s2_bits) | s2_id`
- `split_token()` extracts: `s2_ids = token_ids & mask`, `s1_ids = token_ids >> s2_bits`
- Fusion: `Linear(concat(emb_s1, emb_s2)) → d_model`

### DualHead Loss
- Cross-entropy computed independently for s1 and s2
- Combined: `(CE_s1 + CE_s2) / 2`
- Padding mask zeroes out invalid positions before loss computation

## Data Validation

### Input Requirements
- **Required columns**: `open`, `high`, `low`, `close`
- **Optional**: `volume`, `amount` (auto-derived if missing: `amount = close * volume`)
- **No NaN** in OHLCV columns → raises `ValueError`
- **No Inf** → raises `ValueError`
- **No negative prices** → raises `ValueError`
- Volume defaults to `0.0` if column absent

### Normalization
- Instance-level z-score: `(x - mean) / (std + 1e-5)` per sample
- Clip to `[-clip, clip]` (default clip=5.0)
- `1e-5` epsilon prevents division by zero
- Denormalize after prediction: `pred * (std + 1e-5) + mean`

### Timestamps
- Decomposed into 5 temporal features: `minute, hour, weekday, day, month`
- Either fixed (`FixedEmbedding`) or learnable (`nn.Embedding`)
- `TemporalEmbedding` sums all five components

## Training Configs

### Optimizer Convention
- **Always AdamW** with decoupled weight decay
- Bias and norm params excluded from weight decay
- Two param groups: `[with_decay, without_decay]`

### Learning Rate Schedules
| Domain | Scheduler | Warmup |
|--------|-----------|--------|
| `finetune/` | `OneCycleLR` (pct_start=0.03, div_factor=10) | Implicit |
| `classification/` | `get_linear_schedule_with_warmup` (transformers) | Explicit steps/ratio |
| Default predictor LR | `4e-5` | |
| Default tokenizer LR | `2e-4` | |

### Gradient Handling
- Clip: `max_norm=1.0` (classification), `max_norm=2.0` (tokenizer), `max_norm=3.0` (predictor)
- NaN loss guard: skip batch if `not np.isfinite(loss_val)`
- Accumulation: divide loss by `gradient_accumulation_steps` before backward

### Distributed Training
- `torchrun` for multi-GPU via DDP
- `DistributedSampler` with `set_epoch()` each epoch
- Barrier after validation for synchronization
- Only rank 0 saves checkpoints and prints summaries

## Serialization

### Model Saving
- Prefer **safetensors** format (default in classification)
- Also support PyTorch `.bin` format for compatibility
- `save_pretrained(save_dir, save_format="both")` writes both
- Config saved as `config.json` alongside weights

### Model Loading
- `from_pretrained(load_dir)` auto-detects safetensors vs PyTorch
- Checkpoint validation: file existence, non-empty, optional SHA-256 hash
- `weights_only=True` for trusted PyTorch loads, `weights_only=False` only for training state

### Checkpoint Contents
| File | Content |
|------|---------|
| `model.safetensors` | Model weights |
| `pytorch_model.bin` | Model weights (legacy) |
| `config.json` | Model configuration dict |
| `training_state.bin` | Optimizer + scheduler + global_step + best_metric |

## Code Conventions

### Import Style
- Top-level scripts use `sys.path.append('../')` + `from model import ...`
- Classification scripts use `from kronos_classification_base import ...` (intra-directory)
- `__init__.py` in `model/` provides clean public API

### Error Handling
- Explicit `try/except` around HuggingFace downloads with user-friendly messages
- Input validation at API boundaries (predict, forward, data loading)
- Guard for `NaN` and `Inf` in both data and loss values

### Logging
- `classification/` uses Python `logging` module with `logger = logging.getLogger(__name__)`
- `finetune_csv/` uses `logging.Logger` with `RotatingFileHandler`
- `finetune/` uses Comet ML for experiment tracking (optional)
- Progress bars via `tqdm` (disabled on non-rank-0 in DDP)

### GPU Management
- Auto-detect GPU with most free memory: `torch.cuda.mem_get_info(i)[0]`
- `torch.cuda.empty_cache()` after each epoch and at training end
- FP16 via `torch.cuda.amp.autocast` + `GradScaler` (opt-in with `--fp16`)
