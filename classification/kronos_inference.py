"""
Kronos Inference and Utility Scripts
Includes inference pipeline and helper utilities for time series classification.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
import json
from tqdm import tqdm
import os
import sys
import pickle


class KronosClassificationPipeline:
    """Inference pipeline for Kronos classification model."""

    def __init__(
        self,
        model_path: str,
        device: str = None,  # Auto-detect if None
        batch_size: int = 32,
        max_context: int = 512,
    ):
        """
        Initialize inference pipeline.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on (None for auto-detect)
            batch_size: Batch size for inference
            max_context: Maximum context length
        """
        from kronos_classification_base import KronosClassificationModel

        # Auto-detect fastest GPU if not specified
        if device is None:
            if torch.cuda.is_available():
                # Find GPU with most free memory
                max_free_memory = 0
                best_device = 0
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    free_memory = torch.cuda.mem_get_info(i)[0] / (1024**3)  # GB
                    if free_memory > max_free_memory:
                        max_free_memory = free_memory
                        best_device = i
                device = f"cuda:{best_device}"
                print(f"Auto-selected GPU {best_device} with {max_free_memory:.1f}GB free memory")
            else:
                device = "cpu"

        print(f"Loading model from {model_path}...")
        self.model = KronosClassificationModel.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.batch_size = batch_size
        self.max_context = max_context
        self.tokenizer = self.model.tokenizer
        self.use_volume = self.model.use_volume

        print(f"Model loaded on {device}")
    
    def predict(
        self,
        data: Union[pd.DataFrame, List[pd.DataFrame]],
        timestamps: Optional[Union[pd.Series, List[pd.Series]]] = None,
        return_probs: bool = False,
        return_top_k: Optional[int] = None,
    ) -> Union[int, List[int], Dict]:
        """
        Predict class labels for time series data.
        
        Args:
            data: Single DataFrame or list of DataFrames with OHLCV columns
            timestamps: Timestamps for the data (optional)
            return_probs: Whether to return class probabilities
            return_top_k: Return top-k predictions
            
        Returns:
            Predictions (labels, probabilities, or both)
        """
        # Handle single DataFrame
        if isinstance(data, pd.DataFrame):
            data = [data]
            if timestamps is not None and not isinstance(timestamps, list):
                timestamps = [timestamps]
            single_input = True
        else:
            single_input = False
            if timestamps is None:
                timestamps = [None] * len(data)
        
        all_logits = []
        
        # Process in batches
        with torch.no_grad():
            for i in range(0, len(data), self.batch_size):
                batch_data = data[i:i + self.batch_size]
                batch_timestamps = timestamps[i:i + self.batch_size] if timestamps else [None] * len(batch_data)
                
                # Tokenize batch
                batch_inputs = []
                max_len = 0
                
                for df, ts in zip(batch_data, batch_timestamps):
                    # Prepare columns
                    required_cols = ['open', 'high', 'low', 'close']
                    if self.use_volume and 'volume' in df.columns:
                        required_cols += ['volume', 'amount']
                    
                    ts_data = df[required_cols].values
                    tokens = self.tokenizer.encode(ts_data, ts)
                    input_ids = torch.tensor(tokens, dtype=torch.long)
                    
                    if input_ids.size(0) > self.max_context:
                        input_ids = input_ids[-self.max_context:]
                    
                    batch_inputs.append(input_ids)
                    max_len = max(max_len, input_ids.size(0))
                
                # Pad sequences
                padded_inputs = []
                attention_masks = []
                
                for input_ids in batch_inputs:
                    seq_len = input_ids.size(0)
                    
                    padded_ids = torch.zeros(max_len, dtype=torch.long)
                    padded_ids[:seq_len] = input_ids
                    padded_inputs.append(padded_ids)
                    
                    mask = torch.zeros(max_len, dtype=torch.long)
                    mask[:seq_len] = 1
                    attention_masks.append(mask)
                
                input_ids_batch = torch.stack(padded_inputs).to(self.device)
                attention_mask_batch = torch.stack(attention_masks).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    return_dict=True
                )
                
                all_logits.append(outputs['logits'].cpu())
        
        # Concatenate all batches
        logits = torch.cat(all_logits, dim=0)
        
        # Get predictions
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = np.argmax(probs, axis=-1)
        
        # Format output
        if return_probs:
            result = {
                'predictions': preds.tolist(),
                'probabilities': probs.tolist(),
            }
            
            if return_top_k:
                top_k_probs, top_k_indices = torch.topk(
                    torch.from_numpy(probs),
                    k=min(return_top_k, probs.shape[1]),
                    dim=-1
                )
                result['top_k_predictions'] = top_k_indices.numpy().tolist()
                result['top_k_probabilities'] = top_k_probs.numpy().tolist()
            
            if single_input:
                result = {k: v[0] if isinstance(v, list) else v for k, v in result.items()}
            
            return result
        else:
            return preds[0] if single_input else preds.tolist()
    
    def predict_from_file(
        self,
        input_file: str,
        output_file: str,
        return_probs: bool = True,
    ):
        """
        Predict on data from file and save results.

        Args:
            input_file: Path to input JSON file
            output_file: Path to save predictions (JSON)
            return_probs: Whether to save probabilities
        """
        # Load data from JSON file
        with open(input_file, 'r') as f:
            raw_data = json.load(f)

        # Convert JSON records back to DataFrames
        data = []
        for item in raw_data:
            df = pd.DataFrame(item['data'])
            data.append({
                'data': df,
                'timestamps': pd.to_datetime(item['timestamps']) if item.get('timestamps') else None,
                'label': item.get('label'),
            })

        print(f"Loaded {len(data)} samples from {input_file}")
        
        # Extract DataFrames and timestamps
        dataframes = [item['data'] for item in data]
        timestamps = [item.get('timestamps', None) for item in data]
        
        # Predict
        print("Running inference...")
        results = self.predict(dataframes, timestamps, return_probs=return_probs)
        
        # Add predictions to data
        if return_probs:
            for i, item in enumerate(data):
                item['predicted_label'] = int(results['predictions'][i])
                item['predicted_probabilities'] = results['probabilities'][i]
        else:
            for i, item in enumerate(data):
                item['predicted_label'] = int(results[i])
        
        # Save results as JSON
        output_data = []
        for item in data:
            record = {
                'predicted_label': item.get('predicted_label'),
            }
            if 'predicted_probabilities' in item:
                record['predicted_probabilities'] = item['predicted_probabilities']
            if item.get('data') is not None:
                record['data'] = item['data'].to_dict(orient='list') if isinstance(item['data'], pd.DataFrame) else item['data']
            output_data.append(record)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"Predictions saved to {output_file}")
        return data


def create_sample_data():
    """
    Create sample time series data for testing.
    Demonstrates the expected data format.
    """
    import random
    
    def generate_random_ohlcv(n_points=100):
        """Generate random OHLCV data."""
        base_price = 100
        data = []
        
        for _ in range(n_points):
            open_price = base_price + random.uniform(-2, 2)
            close_price = open_price + random.uniform(-3, 3)
            high_price = max(open_price, close_price) + random.uniform(0, 2)
            low_price = min(open_price, close_price) - random.uniform(0, 2)
            volume = random.uniform(1000, 10000)
            amount = volume * (high_price + low_price) / 2
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'amount': amount,
            })
            
            base_price = close_price
        
        return pd.DataFrame(data)
    
    # Create sample datasets
    train_data = []
    for _ in range(100):
        df = generate_random_ohlcv(200)
        label = random.randint(0, 1)
        train_data.append({'data': df, 'label': label})
    
    val_data = []
    for _ in range(20):
        df = generate_random_ohlcv(200)
        label = random.randint(0, 1)
        val_data.append({'data': df, 'label': label})
    
    test_data = []
    for _ in range(20):
        df = generate_random_ohlcv(200)
        label = random.randint(0, 1)
        test_data.append({'data': df, 'label': label})
    
    # Save files as JSON
    import json as _json

    def _serialize_samples(samples):
        """Convert sample DataFrames to JSON-serializable format."""
        serialized = []
        for s in samples:
            serialized.append({
                'data': s['data'].to_dict(orient='list'),
                'label': s['label'],
            })
        return serialized

    with open('train_sample.json', 'w') as f:
        _json.dump(_serialize_samples(train_data), f, indent=2)

    with open('val_sample.json', 'w') as f:
        _json.dump(_serialize_samples(val_data), f, indent=2)

    with open('test_sample.json', 'w') as f:
        _json.dump(_serialize_samples(test_data), f, indent=2)

    print("Sample data files created:")
    print("- train_sample.json (100 samples)")
    print("- val_sample.json (20 samples)")
    print("- test_sample.json (20 samples)")
    print("\nEach sample contains:")
    print("  - 'data': DataFrame with columns [open, high, low, close, volume, amount]")
    print("  - 'label': Classification label (0 or 1)")
    print("  - 'timestamps': Optional timestamps for each row")


def convert_csv_to_classification_data(
    csv_file: str,
    output_file: str,
    window_size: int = 200,
    label_column: Optional[str] = None,
    label_func: Optional[callable] = None,
):
    """
    Convert CSV file to classification dataset format.

    Args:
        csv_file: Path to CSV file with OHLCV data
        output_file: Output JSON file path (pickle extension auto-converted to .json)
        window_size: Size of sliding window
        label_column: Column name for labels (if available)
        label_func: Function to generate labels from data (if label_column is None)
    """
    df = pd.read_csv(csv_file)
    
    # Check for required columns
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    samples = []
    
    # Create sliding windows
    for i in range(window_size, len(df)):
        window_df = df.iloc[i - window_size:i][required_cols + 
                            (['volume', 'amount'] if 'volume' in df.columns else [])]
        
        # Determine label
        if label_column:
            label = int(df.iloc[i][label_column])
        elif label_func:
            label = label_func(df.iloc[i - window_size:i + 1])
        else:
            # Default: predict if price goes up or down
            label = 1 if df.iloc[i]['close'] > df.iloc[i - 1]['close'] else 0
        
        # Timestamps if available
        timestamps = None
        if 'timestamp' in df.columns or 'timestamps' in df.columns:
            ts_col = 'timestamp' if 'timestamp' in df.columns else 'timestamps'
            timestamps = pd.to_datetime(df.iloc[i - window_size:i][ts_col])
        
        samples.append({
            'data': window_df.reset_index(drop=True),
            'label': label,
            'timestamps': timestamps
        })
    
    # Save as JSON
    serialized = []
    for s in samples:
        record = {
            'data': s['data'].to_dict(orient='list'),
            'label': s['label'],
        }
        if s.get('timestamps') is not None:
            record['timestamps'] = [str(ts) for ts in s['timestamps']]
        serialized.append(record)

    with open(output_file.replace('.pkl', '.json'), 'w') as f:
        json.dump(serialized, f, indent=2)
    
    print(f"Created {len(samples)} samples from {csv_file}")
    print(f"Saved to {output_file}")


def analyze_checkpoint(checkpoint_path: str):
    """Analyze a saved checkpoint and print statistics."""
    from kronos_classification_base import KronosClassificationModel
    
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    model = KronosClassificationModel.from_pretrained(checkpoint_path)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.classification_head.parameters())
    
    print("\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Backbone parameters: {backbone_params:,} ({backbone_params/total_params*100:.1f}%)")
    print(f"Classification head parameters: {head_params:,} ({head_params/total_params*100:.1f}%)")
    
    training_state_path = os.path.join(checkpoint_path, 'training_state.bin')
    if os.path.exists(training_state_path):
        # Verify SHA-256 hash before loading (SEC-2)
        hash_path = training_state_path + '.sha256'
        if os.path.exists(hash_path):
            import hashlib
            sha256 = hashlib.sha256()
            with open(training_state_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            actual_hash = sha256.hexdigest()
            with open(hash_path, 'r') as f:
                expected_hash = f.read().strip()
            if actual_hash != expected_hash:
                print(f"\nWARNING: training_state.bin integrity check FAILED!")
                print(f"  Expected: {expected_hash}")
                print(f"  Actual:   {actual_hash}")
                print(f"  File may be tampered. Skipping training state load.")
            else:
                # SECURITY: Use weights_only=True to prevent arbitrary code execution.
                # Only load training state files produced by our own training code
                # and verified via SHA-256 above.
                try:
                    training_state = torch.load(training_state_path, map_location='cpu', weights_only=True)
                except Exception:
                    import logging
                    logging.getLogger(__name__).warning(
                        "weights_only=True failed for training_state.bin; "
                        "file may contain non-tensor objects. Retrying with safe_globals.")
                    import io
                    class SafeUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            allowed = {'collections.OrderedDict', 'dict', 'list', 'tuple', 'set', 'frozenset', 'int', 'float', 'str', 'bool', 'bytes'}
                            fqn = f"{module}.{name}"
                            if fqn in allowed or name in allowed:
                                return super().find_class(module, name)
                            raise pickle.UnpicklingError(f"Blocked: {fqn}")
                    with open(training_state_path, 'rb') as _f:
                        training_state = SafeUnpickler(_f).load()
                print("\nTraining State:")
                print(f"Global step: {training_state.get('global_step', 'N/A')}")
                print(f"Best validation metric: {training_state.get('best_val_metric', training_state.get('best_val_loss', 'N/A'))}")
                print(f"Integrity: SHA-256 verified")
        else:
            print("\nWARNING: No .sha256 sidecar file for training_state.bin. "
                  "Cannot verify integrity. Skipping load.")
    
    print("\nModel Configuration:")
    print(f"Number of classes: {model.num_classes}")
    print(f"Hidden size: {model.hidden_size}")
    print(f"Max context: {model.max_context}")
    print(f"Use volume: {model.use_volume}")
    print(f"Pooling strategy: {model.pooling_strategy}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python kronos_inference.py create_sample")
        print("  python kronos_inference.py convert_csv <csv_file> <output_file> [window_size]")
        print("  python kronos_inference.py analyze <checkpoint_path>")
        print("  python kronos_inference.py predict <model_path> <input_file> <output_file>")
        print("  (Data files use JSON format)")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create_sample":
        create_sample_data()
    
    elif command == "convert_csv":
        if len(sys.argv) < 4:
            print("Usage: python kronos_inference.py convert_csv <csv_file> <output_file> [window_size]")
            sys.exit(1)
        window_size = int(sys.argv[4]) if len(sys.argv) > 4 else 200
        convert_csv_to_classification_data(sys.argv[2], sys.argv[3], window_size)
    
    elif command == "analyze":
        if len(sys.argv) < 3:
            print("Usage: python kronos_inference.py analyze <checkpoint_path>")
            sys.exit(1)
        analyze_checkpoint(sys.argv[2])
    
    elif command == "predict":
        if len(sys.argv) < 5:
            print("Usage: python kronos_inference.py predict <model_path> <input_file> <output_file>")
            sys.exit(1)
        pipeline = KronosClassificationPipeline(sys.argv[2])
        pipeline.predict_from_file(sys.argv[3], sys.argv[4])
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)