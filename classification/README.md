# Kronos Time Series Classification Model

A classification model built on top of the Kronos time series foundation model. This project removes the original prediction head from Kronos and replaces it with a custom classification head for time series classification tasks.

## Overview

Kronos is a time series foundation model that uses a specialized tokenizer to quantize continuous K-line data (OHLCV) into hierarchical discrete tokens. This project adapts Kronos for classification by:

1. **Removing the original prediction head** from the pretrained Kronos model
2. **Adding a custom classification head** for multi-class time series classification
3. **Providing pretraining and fine-tuning pipelines** with multi-GPU support
4. **Supporting the original Kronos tokenizer** for OHLCV time series data

## Key Features

- **Time Series Input**: Processes OHLCV (Open, High, Low, Close, Volume) data
- **Flexible Architecture**: Multiple pooling strategies (mean, last, max, attention)
- **Multi-GPU Training**: Full distributed training support with DDP
- **Mixed Precision**: FP16 support for faster training
- **Incremental Training**: Pretrain on large datasets, then fine-tune on specific tasks

## Prerequisites

First, clone and install the Kronos repository:

```bash
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos
pip install -r requirements.txt
```

Then install additional dependencies:

```bash
pip install scikit-learn transformers accelerate
```

## Project Structure

```
kronos-classification/
├── kronos_classification_base.py    # Base model architecture
├── kronos_pretrain.py               # Pretraining script
├── kronos_finetune.py               # Fine-tuning script
├── kronos_inference.py              # Inference and utilities
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Data Format

Your data should be in JSON format with the following structure:

```json
{
  "info": {
    "ticker": "stocks/AMD",
    "total_patterns": 1283,
    "label_statistics": {
      "1": 131,
      "0": 1138
    }
  },
  "results": [
    {
      "ticker": "stocks/AMD",
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

**Important**: Only samples with `assigned_label` set (not `null`) will be used for training. The dataset automatically:
- Loads all JSON files from a directory
- Filters out samples without labels
- Calculates `amount` from price × volume
- Splits data into train/val/test sets (default 80/10/10)

## Quick Start

### 1. Prepare Your Data

Place all your labeled JSON files in a directory (e.g., `./data/`). The scripts will:
- Load all `*.json` files in the directory
- Use only samples where `assigned_label` is not `null`
- Automatically split into train/val/test sets (80/10/10 by default)

### 2. Pretrain the Model

#### Single GPU:

```bash
python kronos_pretrain.py \
    --data_dir ./data \
    --kronos_model NeoQuasar/Kronos-base \
    --tokenizer_path NeoQuasar/Kronos-Tokenizer-base \
    --num_classes 2 \
    --output_dir ./pretrain_checkpoints \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --train_split 0.8 \
    --val_split 0.1 \
    --pooling_strategy mean \
    --fp16
```

#### Multiple GPUs (4 GPUs):

```bash
torchrun --standalone --nproc_per_node=4 kronos_pretrain.py \
    --data_dir ./data \
    --kronos_model NeoQuasar/Kronos-base \
    --tokenizer_path NeoQuasar/Kronos-Tokenizer-base \
    --num_classes 2 \
    --output_dir ./pretrain_checkpoints \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --train_split 0.8 \
    --val_split 0.1 \
    --fp16
```

### 3. Fine-tune on Specific Task

```bash
python kronos_finetune.py \
    --data_dir ./data \
    --pretrained_checkpoint ./pretrain_checkpoints/best_model \
    --num_classes 2 \
    --output_dir ./finetune_checkpoints \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_epochs 5 \
    --train_split 0.8 \
    --val_split 0.1 \
    --freeze_backbone_epochs 1 \
    --fp16
```

### 4. Run Inference

```python
from kronos_inference import KronosClassificationPipeline
import pandas as pd
import json

# Initialize pipeline
pipeline = KronosClassificationPipeline(
    model_path="./finetune_checkpoints/best_model",
    device="cuda",
    batch_size=32
)

# Load a sample from your JSON file
with open('./data/your_data.json', 'r') as f:
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
```

## Model Architecture

```
Input: OHLCV Time Series Data
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

## Configuration Options

### Pooling Strategies

- `mean`: Mean pooling over sequence (default)
- `last`: Use last time step representation
- `max`: Max pooling over sequence
- `attention`: Learned attention-based pooling

Example:
```bash
python kronos_pretrain.py --pooling_strategy attention ...
```

### Freeze Backbone

Train only the classification head for first N epochs:

```bash
python kronos_finetune.py --freeze_backbone_epochs 2 ...
```

### Without Volume Data

If your data doesn't include volume/amount:

```bash
python kronos_pretrain.py --no_volume ...
```

## Advanced Usage

### Analyze Checkpoint

```bash
python kronos_inference.py analyze ./pretrain_checkpoints/best_model
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
Hidden size: 768
Max context: 512
Use volume: True
Pooling strategy: mean
```

### Batch Prediction from File

```bash
python kronos_inference.py predict ./finetune_checkpoints/best_model input.pkl output.pkl
```

### Custom Data Preprocessing

```python
import pickle
import pandas as pd

# Load your raw data
df = pd.read_csv('market_data.csv')

# Create windowed samples
samples = []
window_size = 200

for i in range(window_size, len(df)):
    window_df = df.iloc[i - window_size:i]
    
    # Define label (example: predict next day direction)
    label = 1 if df.iloc[i]['close'] > df.iloc[i-1]['close'] else 0
    
    samples.append({
        'data': window_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
        'label': label,
        'timestamps': pd.to_datetime(window_df['timestamp'])
    })

# Save
with open('my_dataset.pkl', 'wb') as f:
    pickle.dump(samples, f)
```

## Performance Optimization

### Memory Optimization

1. Use FP16 training:
```bash
--fp16
```

2. Reduce batch size and increase gradient accumulation:
```bash
--batch_size 8 --gradient_accumulation_steps 4
```

3. Reduce max_context if possible:
```bash
--max_context 256
```

### Training Speed

1. Use multiple GPUs with `torchrun`
2. Increase batch size if GPU memory allows
3. Enable pin_memory (already enabled by default)
4. Use FP16 for ~2x speedup

## Troubleshooting

### Import Error: "model" module not found

Make sure you're running from the Kronos repository directory or have added it to your Python path:

```bash
export PYTHONPATH=/path/to/Kronos:$PYTHONPATH
```

Or run from Kronos directory:

```bash
cd /path/to/Kronos
python /path/to/kronos_pretrain.py ...
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

### Different Number of Features

If your data has different columns:

1. **Without volume**: Use `--no_volume`
2. **Custom features**: Modify the `kronos_classification_base.py` to handle your specific feature set

## Model Checkpoints

Each checkpoint directory contains:
- `pytorch_model.bin`: Model weights and configuration
- `training_state.bin`: Optimizer and scheduler state (for resuming training)
- Tokenizer files from the original Kronos tokenizer

To load a checkpoint:

```python
from kronos_classification_base import KronosClassificationModel

model = KronosClassificationModel.from_pretrained('./checkpoints/best_model')
```

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