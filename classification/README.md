# Kronos Time Series Classification Model

A classification model built on top of the Kronos time series foundation model. This project removes the original prediction head from Kronos and replaces it with a custom classification head for time series classification tasks.

## Overview

Kronos is a time series foundation model that uses a specialized tokenizer to quantize continuous K-line data (OHLCV) into hierarchical discrete tokens. This project adapts Kronos for classification by:

1. **Removing the original prediction head** from the pretrained Kronos model
2. **Adding a custom classification head** for multi-class time series classification
3. **Providing pretraining and fine-tuning pipelines** with multi-GPU support
4. **Supporting the original Kronos tokenizer** for OHLCV time series data

## Key Features

- **Flexible Input Dimensions**: Supports N×4 (OHLC), N×5 (OHLCV), or N×6 (with exogenous features)
- **Variable Length Sequences**: Handles sequences from 20 to 200 timesteps with smart padding
- **Adaptive Loss Functions**: Auto-detects binary vs multi-class, supports focal loss and label smoothing
- **Class Imbalance Handling**: Oversampling, undersampling, and class-weighted loss
- **RL Fine-tuning**: REINFORCE policy gradient for further improvement
- **Multiple Checkpoint Formats**: SafeTensors and PyTorch (.pth) support
- **Comprehensive Metrics**: Accuracy, F1, precision, recall on train/val/test sets
- **GPU Optimization**: Auto GPU selection, FP16 mixed precision, optimized data loading
- **Multi-GPU Support**: Distributed training with torchrun (3.5-4x speedup on 4 GPUs)

## Prerequisites

First, clone and install the Kronos repository:

```bash
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos
pip install -r requirements.txt
pip install scikit-learn transformers accelerate safetensors
```

## Input Data Format

The model expects JSON files with the following structure:

```json
{
  "info": {
    "ticker": "stocks/A",
    "total_patterns": 1283,
    "label_statistics": {
      "1": 233,
      "0": 1000
    }
  },
  "results": [
    {
      "ticker": "stocks/A",
      "timeframe": "4h",
      "assigned_label": 1,
      "chart_data": {
        "dates": [1262793600000, 1262808000000, ...],
        "opens": [9.74, 9.615, ...],
        "highs": [9.75, 9.75, ...],
        "lows": [9.555, 9.54, ...],
        "closes": [9.619, 9.62, ...],
        "volumes": [8390295.0, 4702441.0, ...]
      }
    }
  ]
}
```

**Important**: Only samples with `assigned_label` set (not `null`) will be used for training.

The model will automatically:
- Load all JSON files from the input directory
- Filter out samples without labels
- Calculate `amount` from `close × volume`
- Split data into train/val/test sets (default 80/10/10)

## Quick Start Guide

### 1. Pre-training (Supervised Learning)

Train the model from scratch on your labeled data:

```bash
# Auto-detect fastest GPU (recommended)
python classification/kronos_pretrain.py \
    --data_dir /path/to/labeled_data \
    --kronos_model NeoQuasar/Kronos-base \
    --tokenizer_path NeoQuasar/Kronos-Tokenizer-base \
    --num_classes 2 \
    --output_dir ./pretrain_checkpoints \
    --batch_size 24 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --pooling_strategy mean \
    --loss_type focal \
    --class_balance oversample \
    --save_format safetensors \
    --fp16

# Use specific GPU
python classification/kronos_pretrain.py \
    --data_dir /path/to/labeled_data \
    --kronos_model NeoQuasar/Kronos-base \
    --num_classes 2 \
    --device cuda:2 \
    --fp16

# Multi-GPU training (all 4 GPUs, 3.5-4x faster)
torchrun --standalone --nproc_per_node=4 classification/kronos_pretrain.py \
    --data_dir /path/to/labeled_data \
    --kronos_model NeoQuasar/Kronos-base \
    --num_classes 2 \
    --batch_size 24 \
    --fp16
```

### 2. Fine-tuning

Fine-tune a pretrained model on new data:

```bash
# Auto-detect fastest GPU
python classification/kronos_finetune.py \
    --data_dir /path/to/new_data \
    --pretrained_checkpoint ./pretrain_checkpoints/best_model \
    --num_classes 2 \
    --output_dir ./finetuned_checkpoints \
    --batch_size 48 \
    --learning_rate 5e-5 \
    --num_epochs 5 \
    --freeze_backbone_epochs 1 \
    --class_balance class_weights \
    --fp16

# Multi-GPU
torchrun --standalone --nproc_per_node=4 classification/kronos_finetune.py \
    --data_dir /path/to/new_data \
    --pretrained_checkpoint ./pretrain_checkpoints/best_model \
    --num_classes 2 \
    --batch_size 24 \
    --fp16
```

**Key Parameters:**
- `--freeze_backbone_epochs`: Number of epochs to freeze backbone (train only head)
- Use lower learning rate for fine-tuning (5e-5 vs 2e-5 for pretraining)

### 3. RL Fine-tuning (Optional)

Further improve accuracy using REINFORCE policy gradient after supervised training:

```bash
# Single GPU
python classification/kronos_rl_finetune.py \
    --model_path ./finetuned_checkpoints/best_model \
    --data_dir /path/to/labeled_data \
    --output_dir ./rl_checkpoints \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --num_epochs 3 \
    --fp16

# Multi-GPU
torchrun --standalone --nproc_per_node=4 classification/kronos_rl_finetune.py \
    --model_path ./finetuned_checkpoints/best_model \
    --batch_size 16 \
    --fp16
```

**Key Parameters:**
- `--gamma`: Discount factor for rewards (default: 0.99)
- `--entropy_coef`: Entropy regularization (default: 0.01)
- `--reward_scale`: Scale factor for rewards (default: 1.0)

### 4. Inference

Run inference on new data:

```python
from classification.kronos_inference import KronosClassificationPipeline
import pandas as pd
import json

# Initialize pipeline (auto-detects fastest GPU)
pipeline = KronosClassificationPipeline(
    model_path="./finetuned_checkpoints/best_model",
    batch_size=64
)

# Or specify device
# pipeline = KronosClassificationPipeline(
#     model_path="./finetuned_checkpoints/best_model",
#     device="cuda:2",
#     batch_size=64
# )

# Load a sample from your JSON file
with open('/path/to/data.json', 'r') as f:
    data = json.load(f)

sample = data['results'][0]
chart_data = sample['chart_data']

# Create DataFrame
df = pd.DataFrame({
    'open': chart_data['opens'],
    'high': chart_data['highs'],
    'low': chart_data['lows'],
    'close': chart_data['closes'],
    'volume': chart_data['volumes']
})
df['amount'] = df['close'] * df['volume']

timestamps = pd.to_datetime(chart_data['dates'], unit='ms')

# Single prediction
prediction = pipeline.predict(df, timestamps)
print(f"Predicted class: {prediction}")

# With probabilities
results = pipeline.predict(df, timestamps, return_probs=True)
print(f"Prediction: {results['predictions']}")
print(f"Probabilities: {results['probabilities']}")

# Top-k predictions
results = pipeline.predict(df, timestamps, return_probs=True, return_top_k=3)
print(f"Top-3 classes: {results['top_k_predictions']}")
print(f"Top-3 probabilities: {results['top_k_probabilities']}")
```

## Model Architecture

```
Input: OHLCV Time Series Data [N x 4/5/6]
    ↓
Smart Padding (if N < min_context) or Truncation (if N > max_context)
    ↓
Kronos Tokenizer (quantizes to discrete tokens)
    ↓
Tokenized Sequence [batch_size, seq_len]
    ↓
Kronos Transformer Backbone (pretrained)
    ↓
Hidden States [batch_size, seq_len, hidden_size]
    ↓
Pooling Layer (mean/last/max/attention)
    ↓
Pooled Representation [batch_size, hidden_size]
    ↓
Classification Head:
  - Dropout
  - Linear(hidden_size → classifier_hidden_size)
  - GELU
  - Dropout
  - Linear(classifier_hidden_size → num_classes)
    ↓
Logits [batch_size, num_classes]
```

## Input Dimension Support

The model supports flexible input dimensions:

| Dimension | Columns | Use Case |
|-----------|---------|----------|
| N × 4 | `open, high, low, close` | Price-only data |
| N × 5 | `open, high, low, close, volume` | Standard OHLCV |
| N × 6 | `open, high, low, close, volume, indicator` | With exogenous features |

Set `--no_volume` flag for N×4 input. The `amount` column is automatically computed as `close × volume`.

For N×6 input with exogenous features, use `--num_exogenous 1` and ensure your data has a column named `indicator` or `exogenous_0`.

## Class Imbalance Handling

The model provides multiple strategies for handling class imbalance:

### 1. Oversampling (Recommended for Small Datasets)
```bash
--class_balance oversample --oversample_ratio 1.0
```
- Resamples minority class to match majority class
- `oversample_ratio=1.0` means equal samples per class

### 2. Undersampling
```bash
--class_balance undersample
```
- Reduces majority class to match minority class
- Loses data but faster training

### 3. Class Weights
```bash
--class_balance class_weights
```
- Uses inverse frequency weights in loss function
- No data duplication, preserves original distribution

### 4. Focal Loss
```bash
--loss_type focal
```
- Automatically focuses on hard-to-classify examples
- Works well with severe class imbalance

## Loss Functions

The model supports adaptive loss functions:

| Loss Type | Description | Best For |
|-----------|-------------|----------|
| `auto` (None) | CrossEntropy for multi-class, BCE for binary | General purpose |
| `cross_entropy` | Standard cross-entropy | Multi-class, balanced data |
| `focal` | Focal loss with gamma=2 | Imbalanced datasets |
| `label_smoothing` | Label smoothed cross-entropy | Preventing overconfidence |

## Advanced Usage

### Variable Length Sequences

The model handles sequences from 20-200 timesteps:

```bash
python classification/kronos_pretrain.py \
    --min_context 20 \
    --max_context 200 \
    --padding_strategy right \
    ...
```

**Padding Strategies:**
- `right`: Pad at the end (default, preserves most recent data)
- `left`: Pad at the beginning (preserves latest data)
- `both`: Pad on both sides (centered)

### Analyze Model Checkpoint

```bash
python classification/kronos_inference.py analyze ./checkpoints/best_model
```

Output:
```
Model Statistics:
Total parameters: 102,345,678
Trainable parameters: 102,345,678
Backbone parameters: 101,234,567 (98.9%)
Classification head parameters: 1,111,111 (1.1%)

Training State:
Global step: 5000
Best validation metric: 0.8756

Model Configuration:
Number of classes: 2
Input dimensions: 5 (OHLCV)
Max context: 512
Min context: 20
Use volume: True
Pooling strategy: mean
Padding strategy: right
Loss type: focal
```

### Batch Prediction from File

```bash
python classification/kronos_inference.py predict \
    ./checkpoints/best_model \
    input.pkl \
    output.pkl
```

## Performance Optimization

### GPU Selection

The model supports automatic GPU selection and manual specification:

```bash
# Auto-detect fastest GPU (recommended)
python classification/kronos_pretrain.py --data_dir ... --fp16

# Manually specify GPU
python classification/kronos_pretrain.py --data_dir ... --device cuda:2 --fp16

# Use all 4 GPUs (3.5-4x speedup)
torchrun --standalone --nproc_per_node=4 classification/kronos_pretrain.py \
    --data_dir ... --fp16

# Use specific GPUs only
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nproc_per_node=3 \
    classification/kronos_pretrain.py --data_dir ... --fp16
```

**For detailed GPU training guide, see [GPU_TRAINING_GUIDE.md](GPU_TRAINING_GUIDE.md)**

### Memory Optimization

1. **Use FP16 training** (~2x speedup, ~50% memory reduction):
```bash
--fp16
```

2. **Reduce batch size with gradient accumulation**:
```bash
--batch_size 8 --gradient_accumulation_steps 4
```

3. **Reduce max_context** for shorter sequences:
```bash
--max_context 256
```

4. **Adjust data loading workers**:
```bash
--num_workers 8 --prefetch_factor 4
```

### Multi-GPU Training

Use `torchrun` for distributed training:

```bash
torchrun --standalone --nproc_per_node=4 classification/kronos_pretrain.py \
    --data_dir /path/to/data \
    --num_classes 2 \
    --batch_size 24 \
    --output_dir ./checkpoints \
    --fp16
```

**Performance benchmarks:**
- Single GPU (FP16): ~60-70 samples/second
- 4× GPU (FP16): ~220 samples/second (3.5-4x speedup)

### Training Speed Tips

1. **Always use FP16** for 2x speedup
2. Increase batch size if GPU memory allows
3. Increase `--num_workers` for faster data loading (4-8 recommended)
4. Use `--prefetch_factor 2-4` for overlapping CPU/GPU work
5. Pin memory is enabled by default

## Troubleshooting

### Import Error: "model" module not found

Make sure you're running from the Kronos repository directory:

```bash
cd /path/to/Kronos
python classification/kronos_pretrain.py ...
```

Or set PYTHONPATH:

```bash
export PYTHONPATH=/path/to/Kronos:$PYTHONPATH
```

### CUDA Out of Memory

```bash
# Reduce batch size
--batch_size 4

# Increase gradient accumulation
--gradient_accumulation_steps 8

# Use FP16
--fp16

# Reduce max_context
--max_context 256
```

### Class Imbalance Warnings

If you see severe class imbalance warnings, try:

```bash
--class_balance oversample --loss_type focal
```

This combines data-level and algorithm-level techniques.

### Poor Performance on Minority Class

Combine multiple techniques:

```bash
--class_balance oversample --loss_type focal --class_weights
```

Or use RL fine-tuning after supervised pretraining:

```bash
# First: supervised pretraining
python classification/kronos_pretrain.py ... --output_dir ./supervised

# Then: RL fine-tuning
python classification/kronos_rl_finetune.py --model_path ./supervised/best_model ...
```

## Checkpoint Formats

The model supports multiple checkpoint formats:

### SafeTensors (Recommended)
- **Format**: `.safetensors` + `config.json`
- **Advantages**: Safe, fast, zero-copy loading
- **Use**: Production deployment

### PyTorch
- **Format**: `pytorch_model.bin`
- **Advantages**: Compatible with older code
- **Use**: Legacy support

### Both
- **Format**: Saves both formats
- **Use**: Maximum compatibility

## Model Performance Metrics

During training, the following metrics are tracked:

- **Loss**: Cross-entropy or focal loss
- **Accuracy**: Percentage of correct predictions
- **F1 Score**: Weighted F1 score
- **Precision**: Weighted precision
- **Recall**: Weighted recall

Metrics are computed on both validation and test sets.

## Citation

If you use this project, please cite the original Kronos paper:

```bibtex
@article{shi2025kronos,
  title={Kronos: A Foundation Model for the Language of Financial Markets},
  author={Shi, Yu and Fu, Zongliang and Chen, Shuo and Zhao, Bohan and Xu, Wei and Zhang, Changshui and Li, Jian},
  journal={arXiv preprint arXiv:2508.02739},
  year={2025}
}
```

## License

This project follows the MIT License from the original Kronos repository.

## Resources

- [Kronos GitHub Repository](https://github.com/shiyu-coder/Kronos)
- [Kronos Model on Hugging Face](https://huggingface.co/NeoQuasar/Kronos-base)
- [Kronos Paper](https://arxiv.org/abs/2508.02739)
- [Live Demo](https://shiyu-coder.github.io/Kronos-demo/)
