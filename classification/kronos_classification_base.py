"""
Kronos Classification Model - Base Architecture
Removes the original prediction head and adds a classification head
for time series classification tasks.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import sys
import os
import hashlib
import json
import logging
from safetensors.torch import save_file as safe_save_file
from safetensors.torch import load_file as safe_load_file

logger = logging.getLogger(__name__)

# Import Kronos components
try:
    from model import Kronos, KronosTokenizer
except ImportError:
    print("ERROR: Kronos model not found. Please clone the Kronos repository:")
    print("git clone https://github.com/shiyu-coder/Kronos.git")
    print("Then add the repository to your Python path or run from the Kronos directory")
    sys.exit(1)


def _validate_checkpoint(path: str, expected_sha256: str = None) -> bool:
    """Validate checkpoint file integrity.

    Args:
        path: Path to checkpoint file
        expected_sha256: Optional expected SHA-256 hash. If None, only checks file is readable.

    Returns:
        True if valid

    Raises:
        ValueError: If file is corrupted or hash mismatch
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if os.path.getsize(path) == 0:
        raise ValueError(f"Checkpoint file is empty: {path}")

    if expected_sha256 is not None:
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest()
        if actual_hash != expected_sha256:
            raise ValueError(
                f"Checkpoint integrity check failed for {path}. "
                f"Expected {expected_sha256}, got {actual_hash}"
            )

    return True


class KronosClassificationModel(nn.Module):
    """
    Kronos model with custom classification head.
    Removes the original prediction head and adds a new classification layer.

    Input: Time series data with OHLCV columns (open, high, low, close, volume)
    Output: Classification logits for num_classes

    Supports:
    - Variable length sequences (20-200) with smart padding
    - N x 4 (without volume), N x 5 (OHLCV), N x 6 (with exogenous features)
    - Binary and multi-class classification
    - Class imbalance handling via weighted loss
    """

    def __init__(
        self,
        kronos_model_path: str = "NeoQuasar/Kronos-base",
        tokenizer_path: str = "NeoQuasar/Kronos-Tokenizer-base",
        num_classes: int = 2,
        hidden_dropout_prob: float = 0.1,
        classifier_hidden_size: Optional[int] = None,
        freeze_backbone: bool = False,
        max_context: int = 512,
        min_context: int = 20,
        use_volume: bool = True,
        num_exogenous: int = 0,
        pooling_strategy: str = "mean",  # "mean", "last", "max", "attention"
        padding_strategy: str = "right",  # "right", "left", "both"
        loss_type: Optional[str] = None,  # None=auto, "cross_entropy", "focal", "label_smoothing"
        label_smoothing: float = 0.1,
    ):
        """
        Initialize Kronos Classification Model.

        Args:
            kronos_model_path: Path to pretrained Kronos model
            tokenizer_path: Path to pretrained Kronos tokenizer
            num_classes: Number of classification classes
            hidden_dropout_prob: Dropout probability for classifier
            classifier_hidden_size: Hidden size for classifier (if None, uses model's hidden size)
            freeze_backbone: Whether to freeze the backbone model during training
            max_context: Maximum context length for time series
            min_context: Minimum context length (sequences shorter will be padded)
            use_volume: Whether to use volume and amount data
            num_exogenous: Number of additional exogenous features (0-1)
            pooling_strategy: How to pool sequence representations ("mean", "last", "max", "attention")
            padding_strategy: How to pad sequences ("right", "left", "both")
            loss_type: Loss function type (None=auto based on num_classes)
            label_smoothing: Label smoothing factor for cross-entropy
        """
        super().__init__()

        logger.info(f"Loading Kronos tokenizer from {tokenizer_path}...")
        try:
            # SEC-6: pin revision to guard against supply-chain attacks on HF Hub.
            # Env var KRONOS_REVISION overrides the pinned default.
            _tok_revision = os.environ.get('KRONOS_TOKENIZER_REVISION')
            self.tokenizer = KronosTokenizer.from_pretrained(
                tokenizer_path,
                **({'revision': _tok_revision} if _tok_revision else {}),
            )
        except (OSError, ConnectionError) as e:
            raise RuntimeError(
                f"Failed to load Kronos tokenizer from '{tokenizer_path}'. "
                f"Check network connectivity and model path. Error: {e}"
            ) from e

        logger.info(f"Loading Kronos backbone from {kronos_model_path}...")
        try:
            _model_revision = os.environ.get('KRONOS_MODEL_REVISION')
            self.backbone = Kronos.from_pretrained(
                kronos_model_path,
                **({'revision': _model_revision} if _model_revision else {}),
            )
        except (OSError, ConnectionError) as e:
            raise RuntimeError(
                f"Failed to load Kronos backbone from '{kronos_model_path}'. "
                f"Check network connectivity and model path. Error: {e}"
            ) from e

        # Store configuration
        self.max_context = max_context
        self.min_context = min_context
        self.use_volume = use_volume
        self.num_exogenous = num_exogenous
        self.pooling_strategy = pooling_strategy
        self.padding_strategy = padding_strategy
        self.loss_type = loss_type if loss_type else ("cross_entropy" if num_classes > 2 else "binary_cross_entropy")
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

        # Determine input dimensions
        # Base: OHLC (4), + Volume (1) if use_volume, + Exogenous (num_exogenous)
        self.d_in = 4 + (1 if use_volume else 0) + (1 if num_exogenous > 0 else 0)

        # Freeze backbone if requested
        if freeze_backbone:
            logger.info("Freezing backbone parameters...")
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get hidden size from backbone config
        self.hidden_size = self.backbone.config.n_embd

        # Build classification head
        if classifier_hidden_size is None:
            classifier_hidden_size = self.hidden_size

        # Attention pooling layer (if using attention pooling)
        if pooling_strategy == "attention":
            self.attention_weights = nn.Linear(self.hidden_size, 1)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(self.hidden_size, classifier_hidden_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(classifier_hidden_size, num_classes)
        )

        logger.info(f"Classification head initialized with {num_classes} classes")
        logger.info(f"Input dimensions: {self.d_in} (OHLC{'V' if use_volume else ''}{' + ' + str(num_exogenous) if num_exogenous > 0 else ''})")
        logger.info(f"Pooling strategy: {pooling_strategy}, Padding: {padding_strategy}")
    
    def _pool_sequence(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence representations based on pooling strategy.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            pooled_output: [batch_size, hidden_size]
        """
        if self.pooling_strategy == "last":
            # Use last token representation
            if attention_mask is not None:
                # Get actual sequence lengths
                seq_lengths = attention_mask.sum(dim=1) - 1
                pooled_output = hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]
            else:
                pooled_output = hidden_states[:, -1, :]
        
        elif self.pooling_strategy == "mean":
            # Mean pooling over sequence
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled_output = sum_hidden / sum_mask
            else:
                pooled_output = torch.mean(hidden_states, dim=1)
        
        elif self.pooling_strategy == "max":
            # Max pooling over sequence
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                hidden_states = hidden_states.masked_fill(mask_expanded == 0, -1e9)
            pooled_output = torch.max(hidden_states, dim=1)[0]
        
        elif self.pooling_strategy == "attention":
            # Attention-based pooling
            # Compute attention weights
            attention_scores = self.attention_weights(hidden_states).squeeze(-1)  # [batch_size, seq_len]
            
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
            
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]
            pooled_output = torch.sum(hidden_states * attention_weights, dim=1)  # [batch_size, hidden_size]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return pooled_output
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss based on configuration.

        Args:
            logits: Model logits [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
            class_weights: Optional class weights

        Returns:
            Loss tensor
        """
        if self.loss_type == "focal":
            # Focal loss for handling class imbalance
            from torch.nn.functional import softmax

            ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')(logits, labels)
            p_t = torch.exp(-ce_loss)
            focal_loss = ((1 - p_t) ** 2) * ce_loss  # gamma=2
            return focal_loss.mean()

        elif self.loss_type == "label_smoothing" or self.label_smoothing > 0:
            # Label smoothed cross entropy
            return nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=self.label_smoothing
            )(logits, labels)

        else:  # cross_entropy or binary_cross_entropy
            if self.num_classes == 2 and self.loss_type == "binary_cross_entropy":
                # Binary classification with BCE
                probs = torch.softmax(logits, dim=-1)[:, 1]  # Probability of class 1
                loss_fct = nn.BCELoss(weight=class_weights[1] if class_weights is not None else None)
                return loss_fct(probs, labels.float())
            else:
                # Standard cross entropy
                return nn.CrossEntropyLoss(weight=class_weights)(logits, labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size]
            return_dict: Whether to return dictionary
            class_weights: Optional class weights for loss calculation

        Returns:
            Dictionary containing loss (if labels provided) and logits
        """
        # Get backbone outputs
        # The Kronos model returns a tuple: (logits, hidden_states)
        outputs = self.backbone(
            input_ids,
            output_hidden_states=True
        )

        # Extract hidden states from the last layer
        # outputs.hidden_states is a tuple of hidden states from each layer
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

        # Pool sequence representations
        pooled_output = self._pool_sequence(hidden_states, attention_mask)

        # Pass through classification head
        logits = self.classification_head(pooled_output)  # [batch_size, num_classes]

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            if class_weights is not None:
                class_weights = class_weights.to(logits.device)
            loss = self._compute_loss(logits, labels, class_weights)

        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": pooled_output,
            }

        return (loss, logits) if loss is not None else logits
    
    def tokenize_timeseries(
        self,
        df: pd.DataFrame,
        timestamps: Optional[pd.Series] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize time series data using Kronos tokenizer with smart padding.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'amount']
                or ['open', 'high', 'low', 'close'] if use_volume=False
            timestamps: Timestamps for the data (optional)

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        if self.use_volume and 'volume' in df.columns:
            required_cols += ['volume']
            # Ensure amount column exists
            if 'amount' not in df.columns:
                df = df.copy()
                df['amount'] = df['close'] * df['volume']
            required_cols += ['amount']

        # Handle exogenous features
        if self.num_exogenous > 0:
            # Look for exogenous column (e.g., 'exogenous_0', 'indicator', etc.)
            exog_cols = [col for col in df.columns if col.startswith('exogenous_') or col == 'indicator']
            if exog_cols:
                required_cols.extend(exog_cols[:self.num_exogenous])

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Got: {df.columns.tolist()}")

        # Validate OHLCV data
        ohlcv_cols = [c for c in required_cols if c in ('open', 'high', 'low', 'close', 'volume', 'amount')]
        ohlcv_data = df[ohlcv_cols]
        if ohlcv_data.isnull().any().any():
            raise ValueError(f"Input data contains NaN values in columns: {ohlcv_cols}")
        if np.isinf(ohlcv_data.values).any():
            raise ValueError(f"Input data contains infinite values in columns: {ohlcv_cols}")
        price_cols = [c for c in ('open', 'high', 'low', 'close') if c in ohlcv_cols]
        if (df[price_cols] < 0).any().any():
            raise ValueError(f"Price columns contain negative values: {price_cols}")

        # Convert to numpy array
        data = df[required_cols].values

        # Handle padding for short sequences
        seq_len = len(data)
        if seq_len < self.min_context:
            # Pad to min_context
            pad_len = self.min_context - seq_len

            if self.padding_strategy == "right":
                # Pad at the end with zeros
                padding = np.zeros((pad_len, data.shape[1]))
                data = np.vstack([data, padding])
            elif self.padding_strategy == "left":
                # Pad at the beginning with zeros
                padding = np.zeros((pad_len, data.shape[1]))
                data = np.vstack([padding, data])
            else:  # "both"
                # Pad on both sides
                pad_left = pad_len // 2
                pad_right = pad_len - pad_left
                left_padding = np.zeros((pad_left, data.shape[1]))
                right_padding = np.zeros((pad_right, data.shape[1]))
                data = np.vstack([left_padding, data, right_padding])

        # Tokenize using Kronos tokenizer
        tokens = self.tokenizer.encode(data, timestamps)

        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)

        # Truncate if exceeds max_context
        if input_ids.size(0) > self.max_context:
            input_ids = input_ids[-self.max_context:]

        # Create attention mask (all ones for valid tokens)
        attention_mask = torch.ones(input_ids.size(0), dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def save_pretrained(
        self,
        save_directory: str,
        save_format: str = "both"  # "safetensors", "pytorch", or "both"
    ):
        """
        Save model and tokenizer to directory.

        Args:
            save_directory: Directory to save to
            save_format: Format to save ("safetensors", "pytorch", or "both")
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

        # Save config
        config = {
            'num_classes': self.num_classes,
            'hidden_size': self.hidden_size,
            'max_context': self.max_context,
            'min_context': self.min_context,
            'use_volume': self.use_volume,
            'num_exogenous': self.num_exogenous,
            'pooling_strategy': self.pooling_strategy,
            'padding_strategy': self.padding_strategy,
            'loss_type': self.loss_type,
            'label_smoothing': self.label_smoothing,
            'd_in': self.d_in,
        }

        # Save in requested format(s)
        if save_format in ["pytorch", "both"]:
            torch.save({
                'model_state_dict': self.state_dict(),
                'config': config,
            }, os.path.join(save_directory, 'pytorch_model.bin'))
            logger.info(f"Model saved in PyTorch format to {save_directory}")

        if save_format in ["safetensors", "both"]:
            # Save model weights in safetensors format
            state_dict = self.state_dict()
            # Add config to the safetensors file as metadata
            safe_save_file(
                state_dict,
                os.path.join(save_directory, 'model.safetensors'),
                metadata={"config": str(config)}
            )

            # Save config separately as JSON for easy loading
            import json
            with open(os.path.join(save_directory, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Model saved in SafeTensors format to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        **kwargs
    ):
        """
        Load model from directory.

        Auto-detects format (safetensors or pytorch) and loads accordingly.
        """
        import json

        # Load config
        config_path = os.path.join(load_directory, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Try loading from pytorch_model.bin (legacy)
            pytorch_path = os.path.join(load_directory, 'pytorch_model.bin')
            if os.path.exists(pytorch_path):
                # SECURITY NOTE: weights_only=False uses pickle deserialization,
                # which can execute arbitrary code. Only load checkpoints from trusted sources.
                # Prefer the safetensors format (auto-detected below) for untrusted files.
                checkpoint = torch.load(pytorch_path, map_location='cpu', weights_only=True)
                config = checkpoint.get('config', {})
                # Legacy format support
                if 'num_classes' in checkpoint:
                    config['num_classes'] = checkpoint['num_classes']
                if 'max_context' in checkpoint:
                    config['max_context'] = checkpoint['max_context']
                if 'use_volume' in checkpoint:
                    config['use_volume'] = checkpoint['use_volume']
                if 'pooling_strategy' in checkpoint:
                    config['pooling_strategy'] = checkpoint['pooling_strategy']
            else:
                raise FileNotFoundError(f"No config found in {load_directory}")

        # Merge with kwargs (kwargs take precedence)
        for key, value in kwargs.items():
            if key in config:
                config[key] = value

        # Create model instance with loaded config
        model = cls(
            tokenizer_path=load_directory,
            num_classes=config.get('num_classes', 2),
            max_context=config.get('max_context', 512),
            min_context=config.get('min_context', 20),
            use_volume=config.get('use_volume', True),
            num_exogenous=config.get('num_exogenous', 0),
            pooling_strategy=config.get('pooling_strategy', 'mean'),
            padding_strategy=config.get('padding_strategy', 'right'),
            loss_type=config.get('loss_type', None),
            label_smoothing=config.get('label_smoothing', 0.1),
        )

        # Load weights (prefer safetensors for security)
        safetensors_path = os.path.join(load_directory, 'model.safetensors')
        pytorch_path = os.path.join(load_directory, 'pytorch_model.bin')

        if os.path.exists(safetensors_path):
            # Validate file integrity before loading
            _validate_checkpoint(safetensors_path)
            state_dict = safe_load_file(safetensors_path)
            model.load_state_dict(state_dict)
            logger.info(f"Model loaded from SafeTensors format: {load_directory}")
        elif os.path.exists(pytorch_path):
            # Validate file integrity before loading
            _validate_checkpoint(pytorch_path)
            # SECURITY NOTE: weights_only=False uses pickle deserialization.
            # Prefer safetensors format for untrusted checkpoints.
            checkpoint = torch.load(pytorch_path, map_location='cpu', weights_only=True)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise ValueError(f"Invalid checkpoint format in {pytorch_path}")
            logger.info(f"Model loaded from PyTorch format: {load_directory}")
        else:
            raise FileNotFoundError(f"No model weights found in {load_directory}")

        return model


class KronosClassificationConfig:
    """Configuration for Kronos Classification Model."""
    
    def __init__(
        self,
        kronos_model_path: str = "NeoQuasar/Kronos-base",
        tokenizer_path: str = "NeoQuasar/Kronos-Tokenizer-base",
        num_classes: int = 2,
        hidden_dropout_prob: float = 0.1,
        classifier_hidden_size: Optional[int] = None,
        freeze_backbone: bool = False,
        max_context: int = 512,
        use_volume: bool = True,
        pooling_strategy: str = "mean",
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        self.kronos_model_path = kronos_model_path
        self.tokenizer_path = tokenizer_path
        self.num_classes = num_classes
        self.hidden_dropout_prob = hidden_dropout_prob
        self.classifier_hidden_size = classifier_hidden_size
        self.freeze_backbone = freeze_backbone
        self.max_context = max_context
        self.use_volume = use_volume
        self.pooling_strategy = pooling_strategy
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        return cls(**config_dict)