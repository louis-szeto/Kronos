"""
Kronos Classification Model - Base Architecture
Removes the original prediction head and adds a classification head
for time series classification tasks.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import sys
import os

# Import Kronos components
try:
    from model import Kronos, KronosTokenizer
except ImportError:
    print("ERROR: Kronos model not found. Please clone the Kronos repository:")
    print("git clone https://github.com/shiyu-coder/Kronos.git")
    print("Then add the repository to your Python path or run from the Kronos directory")
    sys.exit(1)


class KronosClassificationModel(nn.Module):
    """
    Kronos model with custom classification head.
    Removes the original prediction head and adds a new classification layer.
    
    Input: Time series data with OHLCV columns (open, high, low, close, volume)
    Output: Classification logits for num_classes
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
        use_volume: bool = True,
        pooling_strategy: str = "mean",  # "mean", "last", "max", "attention"
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
            use_volume: Whether to use volume and amount data
            pooling_strategy: How to pool sequence representations ("mean", "last", "max", "attention")
        """
        super().__init__()
        
        print(f"Loading Kronos tokenizer from {tokenizer_path}...")
        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        
        print(f"Loading Kronos backbone from {kronos_model_path}...")
        self.backbone = Kronos.from_pretrained(kronos_model_path)
        
        # Store configuration
        self.max_context = max_context
        self.use_volume = use_volume
        self.pooling_strategy = pooling_strategy
        
        # Freeze backbone if requested
        if freeze_backbone:
            print("Freezing backbone parameters...")
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
        
        self.num_classes = num_classes
        print(f"Classification head initialized with {num_classes} classes")
        print(f"Pooling strategy: {pooling_strategy}")
    
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
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size]
            return_dict: Whether to return dictionary
            
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
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        
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
        Tokenize time series data using Kronos tokenizer.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'amount']
                or ['open', 'high', 'low', 'close'] if use_volume=False
            timestamps: Timestamps for the data (optional)
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        if self.use_volume:
            required_cols += ['volume', 'amount']
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to numpy array
        data = df[required_cols].values
        
        # Tokenize using Kronos tokenizer
        # The tokenizer expects data in shape [sequence_length, num_features]
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
    
    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer to directory."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        # Save model weights and config
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'hidden_size': self.hidden_size,
            'max_context': self.max_context,
            'use_volume': self.use_volume,
            'pooling_strategy': self.pooling_strategy,
        }, os.path.join(save_directory, 'pytorch_model.bin'))
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load model from directory."""
        # Load checkpoint
        checkpoint = torch.load(
            os.path.join(load_directory, 'pytorch_model.bin'),
            map_location='cpu'
        )
        
        # Create model instance
        model = cls(
            tokenizer_path=load_directory,
            num_classes=checkpoint['num_classes'],
            max_context=checkpoint.get('max_context', 512),
            use_volume=checkpoint.get('use_volume', True),
            pooling_strategy=checkpoint.get('pooling_strategy', 'mean'),
            **kwargs
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {load_directory}")
        
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