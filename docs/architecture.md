# Architecture

End-to-end ML pipeline map for Kronos.

## Pipeline Stages

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐
│  Data Input  │───>│ Preprocessing│───>│  Tokenization   │───>│ Model Forward │
│ CSV/JSON/Qlib│    │ z-score+clip │    │ BSQ Quantizer   │    │ Transformer   │
└─────────────┘    └──────────────┘    └─────────────────┘    └───────┬───────┘
                                                                        │
                        ┌───────────────────────────────────────────────┘
                        │
              ┌─────────▼──────────┐    ┌─────────────┐
              │  Task Head          │───>│  Inference   │───> Predictions/Labels
              │  Prediction/Classify│    │  Denormalize │
              └────────────────────┘    └─────────────┘
```

## Component Inventory

### 1. Model Core (`model/`)

| File | Component | Description |
|------|-----------|-------------|
| `module.py` | `BinarySphericalQuantizer` | BSQ: bipolar quantization with entropy loss |
| `module.py` | `BSQuantizer` | Wrapper: L2 normalize → BSQ → dual-index (s1, s2) |
| `module.py` | `TransformerBlock` | Pre-norm self-attention + SwiGLU FFN |
| `module.py` | `MultiHeadAttentionWithRoPE` | Causal self-attention with rotary positional encoding |
| `module.py` | `MultiHeadCrossAttentionWithRoPE` | Cross-attention for dependency-aware layer |
| `module.py` | `HierarchicalEmbedding` | Dual embedding (s1 vocab + s2 vocab) → fusion projection |
| `module.py` | `DependencyAwareLayer` | Cross-attn: s2 conditioned on sampled s1 embeddings |
| `module.py` | `DualHead` | Parallel s1/s2 prediction heads with combined CE loss |
| `module.py` | `TemporalEmbedding` | Fixed/learnable time features (minute, hour, weekday, day, month) |
| `module.py` | `RMSNorm`, `FeedForward` | Standard sub-layers |
| `kronos.py` | `KronosTokenizer` | Encoder → BSQ quantize → Decoder; `encode()` / `decode()` |
| `kronos.py` | `Kronos` | Hierarchical LM: embed → transformer → s1 head → dep-layer → s2 head |
| `kronos.py` | `KronosPredictor` | End-to-end: normalize → tokenize → AR generate → decode → denormalize |
| `kronos.py` | `auto_regressive_inference` | Sliding-window AR with top-k/top-p sampling |

### 2. Classification Pipeline (`classification/`)

| File | Component | Description |
|------|-----------|-------------|
| `kronos_classification_base.py` | `KronosClassificationModel` | Backbone + pooling + classification head |
| `kronos_classification_base.py` | `KronosClassificationConfig` | Hyperparameter container |
| `kronos_pretrain.py` | `KronosPretrainer` | Multi-GPU pretraining with early stopping |
| `kronos_pretrain.py` | `KronosTimeSeriesDataset` | JSON → OHLCV DataFrame → tokenize → tensor |
| `kronos_finetune.py` | `KronosFineTuner` | Supervised FT with optional backbone freeze |
| `kronos_rl_finetune.py` | `PolicyGradientFinetuner` | REINFORCE with entropy regularization |
| `kronos_inference.py` | `KronosClassificationPipeline` | Batch prediction, auto GPU selection |

### 3. Finetune - Qlib (`finetune/`)

| File | Component | Description |
|------|-----------|-------------|
| `config.py` | `Config` | Central config: data paths, training HPs, model paths |
| `dataset.py` | `QlibDataset` | Pickled Qlib data → sliding window → z-score normalize |
| `train_tokenizer.py` | `train_model()` | DDP tokenizer training: recon + BSQ loss |
| `train_predictor.py` | `train_model()` | DDP predictor training: next-token CE on s1/s2 |
| `utils/training_utils.py` | Utilities | DDP setup/cleanup, seed, model size, time formatting |

### 4. Finetune - CSV (`finetune_csv/`)

| File | Component | Description |
|------|-----------|-------------|
| `config_loader.py` | `ConfigLoader` | YAML config with path template resolution |
| `config_loader.py` | `CustomFinetuneConfig` | Flattened config object from YAML sections |
| `finetune_tokenizer.py` | `train_tokenizer()` | Single-GPU tokenizer finetuning on CSV |
| `finetune_base_model.py` | `train_model()` | Predictor finetuning with DDP support |
| `finetune_base_model.py` | `CustomKlineDataset` | CSV → time-based split → sliding window |
| `train_sequential.py` | `SequentialTrainer` | Orchestrates tokenizer → predictor in sequence |

### 5. Web UI (`webui/`)

| File | Component | Description |
|------|-----------|-------------|
| `app.py` | Flask app | REST API: load data, load model, predict, save results |
| `templates/index.html` | Frontend | Plotly candlestick charts with prediction overlay |
| `run.py` / `start.sh` | Launcher | Convenience entry points |

## Training Phases

```
Phase 1: Tokenizer Training
  Data → normalize → Encoder → BSQ → Decoder → recon_loss + bsq_loss
  Optimizer: AdamW + OneCycleLR

Phase 2: Predictor Training (uses frozen/tokenized outputs)
  Data → Tokenizer.encode → (s1_ids, s2_ids) → Kronos LM → CE loss (s1 + s2)
  Optimizer: AdamW + OneCycleLR

Phase 3: Classification Pretraining (optional)
  JSON data → Tokenizer → KronosClassificationModel → CE/focal/label-smoothing loss
  Optimizer: AdamW + linear warmup schedule

Phase 4: Classification Finetuning
  Pretrained checkpoint → KronosFineTuner → metrics-based early stopping
  Optional: backbone freeze for N epochs, then unfreeze

Phase 5: RL Finetuning (optional)
  Supervised checkpoint → REINFORCE policy gradient → reward = correct-class prob
  Entropy regularization for exploration
```

## Inference Flow

```
KronosPredictor.predict(df, timestamps):
  1. Validate input (OHLCV, no NaN/inf/neg prices)
  2. z-score normalize (instance-level)
  3. Tokenizer.encode(normalized_data, half=True) → (s1_ids, s2_ids)
  4. auto_regressive_inference(tokenizer, model, ...) → sliding window generation
  5. Tokenizer.decode(generated_tokens, half=True) → reconstructed data
  6. Average over sample_count runs
  7. Denormalize (un-z-score)
  8. Return DataFrame with OHLCV predictions

KronosClassificationPipeline.predict(df, timestamps):
  1. Tokenize each DataFrame
  2. Pad/collate batch
  3. KronosClassificationModel.forward → pool → classify
  4. Return predictions + optional probabilities
```

## Model Sizes (from HuggingFace Hub)

| Model | Params | Context | Tokenizer |
|-------|--------|---------|-----------|
| Kronos-mini | 4.1M | 2048 | Kronos-Tokenizer-2k |
| Kronos-small | 24.7M | 512 | Kronos-Tokenizer-base |
| Kronos-base | 102.3M | 512 | Kronos-Tokenizer-base |
