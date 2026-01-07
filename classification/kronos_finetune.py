"""
Kronos Fine-tuning Script
Fine-tune pretrained classification model on specific time series classification tasks.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import argparse
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


class KronosTimeSeriesDataset(Dataset):
    """Dataset for fine-tuning Kronos classification model."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_context: int = 512,
        use_volume: bool = True,
        train_split: float = 0.8,
        val_split: float = 0.1,
        split_type: str = 'train',  # 'train', 'val', or 'test'
        class_balance: str = 'none',  # 'none', 'oversample', 'undersample', 'class_weights'
        oversample_ratio: float = 1.0,
    ):
        """
        Initialize time series dataset.
        
        Args:
            data_path: Path to JSON file or directory containing JSON files
            tokenizer: Kronos tokenizer
            max_context: Maximum context length
            use_volume: Whether to use volume and amount
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            split_type: Which split to use ('train', 'val', 'test')
            class_balance: Method to handle class imbalance
            oversample_ratio: Target ratio for oversampling (1.0 = equal samples)
        """
        self.tokenizer = tokenizer
        self.max_context = max_context
        self.use_volume = use_volume
        self.class_balance = class_balance
        
        print(f"Loading data from {data_path}...")
        self.data = self._load_json_data(data_path, train_split, val_split, split_type)
        
        # Apply class balancing only for training split
        if split_type == 'train' and class_balance != 'none':
            self.data = self._apply_class_balancing(self.data, class_balance, oversample_ratio)
        
        print(f"Loaded {len(self.data)} samples for {split_type} split")
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print class distribution statistics."""
        from collections import Counter
        label_counts = Counter([sample['label'] for sample in self.data])
        print(f"Class distribution: {dict(label_counts)}")
        if len(label_counts) > 1:
            minority_class = min(label_counts, key=label_counts.get)
            majority_class = max(label_counts, key=label_counts.get)
            ratio = label_counts[majority_class] / label_counts[minority_class]
            print(f"Class imbalance ratio: {ratio:.2f}:1 (majority:minority)")
    
    def _apply_class_balancing(self, data, method, oversample_ratio):
        """Apply class balancing technique."""
        from collections import Counter
        import random
        
        # Get class distribution
        labels = [sample['label'] for sample in data]
        label_counts = Counter(labels)
        
        print(f"\nOriginal distribution: {dict(label_counts)}")
        
        if method == 'oversample':
            # Oversample minority classes
            max_count = max(label_counts.values())
            target_count = int(max_count * oversample_ratio)
            
            balanced_data = []
            for label in label_counts.keys():
                # Get all samples for this class
                class_samples = [s for s in data if s['label'] == label]
                
                if len(class_samples) < target_count:
                    # Oversample with replacement
                    oversampled = random.choices(class_samples, k=target_count)
                    balanced_data.extend(oversampled)
                    print(f"Class {label}: {len(class_samples)} -> {target_count} (oversampled)")
                else:
                    balanced_data.extend(class_samples)
                    print(f"Class {label}: {len(class_samples)} (kept as is)")
            
            random.shuffle(balanced_data)
            return balanced_data
        
        elif method == 'undersample':
            # Undersample majority classes
            min_count = min(label_counts.values())
            
            balanced_data = []
            for label in label_counts.keys():
                class_samples = [s for s in data if s['label'] == label]
                
                if len(class_samples) > min_count:
                    # Undersample randomly
                    undersampled = random.sample(class_samples, min_count)
                    balanced_data.extend(undersampled)
                    print(f"Class {label}: {len(class_samples)} -> {min_count} (undersampled)")
                else:
                    balanced_data.extend(class_samples)
                    print(f"Class {label}: {len(class_samples)} (kept as is)")
            
            random.shuffle(balanced_data)
            return balanced_data
        
        return data
    
    def get_class_weights(self):
        """Calculate class weights for weighted loss (inverse frequency)."""
        from collections import Counter
        import numpy as np
        
        labels = [sample['label'] for sample in self.data]
        label_counts = Counter(labels)
        
        # Calculate inverse frequency weights
        total = len(labels)
        num_classes = len(label_counts)
        
        weights = {}
        for label in range(num_classes):
            if label in label_counts:
                weights[label] = total / (num_classes * label_counts[label])
            else:
                weights[label] = 1.0
        
        # Convert to tensor
        weight_list = [weights[i] for i in range(num_classes)]
        return torch.tensor(weight_list, dtype=torch.float32)
    
    def _load_json_data(self, data_path: str, train_split: float, val_split: float, split_type: str):
        """Load and split data from JSON files."""
        import glob
        import json
        import random
        
        all_samples = []
        
        # Handle directory or single file
        if os.path.isdir(data_path):
            json_files = glob.glob(os.path.join(data_path, "*.json"))
        else:
            json_files = [data_path]
        
        print(f"Found {len(json_files)} JSON files")
        
        # Load all samples from all files
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract samples from 'results' field
            for result in data.get('results', []):
                # Skip samples without assigned labels
                if result.get('assigned_label') is None:
                    continue
                
                chart_data = result['chart_data']
                
                # Create DataFrame from chart_data
                df = pd.DataFrame({
                    'open': chart_data['opens'],
                    'high': chart_data['highs'],
                    'low': chart_data['lows'],
                    'close': chart_data['closes'],
                    'volume': chart_data['volumes']
                })
                
                # Calculate amount (price * volume) if not present
                df['amount'] = df['close'] * df['volume']
                
                # Convert dates to timestamps
                timestamps = pd.to_datetime(chart_data['dates'], unit='ms')
                
                all_samples.append({
                    'data': df,
                    'label': result['assigned_label'],
                    'timestamps': timestamps
                })
        
        print(f"Total samples loaded: {len(all_samples)}")
        
        # Shuffle and split data
        random.seed(42)
        random.shuffle(all_samples)
        
        n_total = len(all_samples)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        if split_type == 'train':
            return all_samples[:n_train]
        elif split_type == 'val':
            return all_samples[n_train:n_train + n_val]
        else:  # test
            return all_samples[n_train + n_val:]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.data[idx]
        
        df = sample['data']
        label = sample['label']
        timestamps = sample.get('timestamps', None)
        
        # Prepare columns
        required_cols = ['open', 'high', 'low', 'close']
        if self.use_volume and 'volume' in df.columns:
            required_cols += ['volume', 'amount']
        
        data = df[required_cols].values
        
        # Tokenize
        tokens = self.tokenizer.encode(data, timestamps)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        if input_ids.size(0) > self.max_context:
            input_ids = input_ids[-self.max_context:]
        
        attention_mask = torch.ones(input_ids.size(0), dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch):
    """Collate function to pad sequences."""
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    input_ids = []
    attention_masks = []
    labels = []
    
    for item in batch:
        seq_len = item['input_ids'].size(0)
        
        padded_ids = torch.zeros(max_len, dtype=torch.long)
        padded_ids[:seq_len] = item['input_ids']
        input_ids.append(padded_ids)
        
        padded_mask = torch.zeros(max_len, dtype=torch.long)
        padded_mask[:seq_len] = item['attention_mask']
        attention_masks.append(padded_mask)
        
        labels.append(item['labels'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': torch.stack(labels)
    }


class KronosFineTuner:
    """Fine-tuner for Kronos classification model."""
    
    def __init__(
        self,
        model,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        output_dir: str = "./finetuned_checkpoints",
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        num_epochs: int = 5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        logging_steps: int = 50,
        save_steps: int = 500,
        eval_steps: int = 500,
        local_rank: int = -1,
        fp16: bool = False,
        freeze_backbone_epochs: int = 0,
        class_weights: Optional[torch.Tensor] = None,
        device: str = None,
        num_workers: int = 4,
        prefetch_factor: int = 2,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.local_rank = local_rank
        self.fp16 = fp16
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.class_weights = class_weights
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        os.makedirs(output_dir, exist_ok=True)

        self.is_distributed = local_rank != -1
        if self.is_distributed:
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=False)
        else:
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            self.model = self.model.to(self.device)

        print(f"Training on device: {self.device}")
        
        self._setup_dataloaders()
        self._setup_optimizer()
        
        self.scaler = torch.cuda.amp.GradScaler() if fp16 else None
        self.global_step = 0
        self.best_val_metric = 0.0
    
    def _setup_dataloaders(self):
        """Setup data loaders."""
        if self.is_distributed:
            train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            val_sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.val_dataset else None
            test_sampler = DistributedSampler(self.test_dataset, shuffle=False) if self.test_dataset else None
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=collate_fn,
            num_workers=self.num_workers if not self.is_distributed else 4,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers if not self.is_distributed else 4,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        ) if self.val_dataset else None

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=test_sampler,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers if not self.is_distributed else 4,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        ) if self.test_dataset else None
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate
        )
        
        total_steps = len(self.train_loader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        model = self.model.module if self.is_distributed else self.model
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")
    
    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        model = self.model.module if self.is_distributed else self.model
        for param in model.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")
    
    def train(self):
        """Main training loop."""
        print(f"Starting fine-tuning for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            if epoch < self.freeze_backbone_epochs:
                if epoch == 0:
                    self._freeze_backbone()
            elif epoch == self.freeze_backbone_epochs and self.freeze_backbone_epochs > 0:
                self._unfreeze_backbone()
                self._setup_optimizer()
            
            self._train_epoch(epoch)
            
            if self.val_loader and (self.local_rank <= 0):
                metrics = self._evaluate(self.val_loader, "Validation")
                print(f"Validation Metrics: {metrics}")
                
                if metrics['f1'] > self.best_val_metric:
                    self.best_val_metric = metrics['f1']
                    self._save_checkpoint("best_model")
            
            if self.local_rank <= 0:
                self._save_checkpoint(f"epoch_{epoch + 1}")
        
        if self.test_loader and (self.local_rank <= 0):
            print("\n" + "="*50)
            print("Final Test Evaluation")
            print("="*50)
            test_metrics = self._evaluate(self.test_loader, "Test")
            print(f"Test Metrics: {test_metrics}")
    
    def _train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
        
        epoch_loss = 0
        progress_bar = tqdm(
            self.train_loader,
            desc="Training",
            disable=(self.local_rank > 0)
        )
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            if self.fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch, class_weights=self.class_weights)
                    loss = outputs['loss'] / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(**batch, class_weights=self.class_weights)
                loss = outputs['loss'] / self.gradient_accumulation_steps
                loss.backward()
            
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                if self.global_step % self.logging_steps == 0 and self.local_rank <= 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})
                
                if self.global_step % self.save_steps == 0 and self.local_rank <= 0:
                    self._save_checkpoint(f"checkpoint-{self.global_step}")
                
                if (self.val_loader and self.global_step % self.eval_steps == 0 and self.local_rank <= 0):
                    metrics = self._evaluate(self.val_loader, "Validation")
                    print(f"\nStep {self.global_step} - Validation: {metrics}")
                    self.model.train()
    
    def _evaluate(self, dataloader: DataLoader, split_name: str = "Validation") -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}", disable=(self.local_rank > 0)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                total_loss += outputs['loss'].item()
                
                preds = torch.argmax(outputs['logits'], dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        avg_loss = total_loss / len(dataloader)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        if split_name == "Test" and self.local_rank <= 0:
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds))
        
        return metrics
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save checkpoint."""
        save_path = os.path.join(self.output_dir, checkpoint_name)
        
        model_to_save = self.model.module if self.is_distributed else self.model
        model_to_save.save_pretrained(save_path)
        
        torch.save({
            'global_step': self.global_step,
            'best_val_metric': self.best_val_metric,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, os.path.join(save_path, 'training_state.bin'))
        
        print(f"Checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Kronos Classification Model")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing JSON files or single JSON file")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="./finetuned_checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_context", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Fraction of data for training")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Fraction of data for validation")
    parser.add_argument("--class_balance", type=str, default="none",
                       choices=["none", "oversample", "undersample", "class_weights"],
                       help="Method to handle class imbalance")
    parser.add_argument("--oversample_ratio", type=float, default=1.0,
                       help="Target ratio for minority class (1.0 = equal samples)")
    parser.add_argument("--no_volume", action="store_true", help="Don't use volume/amount")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training (FP16)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda:0, cuda:1, etc. or 'auto' for fastest available)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers (0 for single-threaded)")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                       help="Number of batches to prefetch per worker")

    args = parser.parse_args()

    # Auto-detect fastest GPU if not in distributed mode
    if args.local_rank == -1 and args.device is None:
        if torch.cuda.is_available():
            max_free_memory = 0
            best_device = 0
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_memory = torch.cuda.mem_get_info(i)[0] / (1024**3)  # GB
                print(f"GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f}GB total, {free_memory:.1f}GB free)")
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_device = i
            args.device = f"cuda:{best_device}"
            print(f"Auto-selected GPU {best_device} with {max_free_memory:.1f}GB free memory")
        else:
            args.device = "cpu"

    # Set device for single-GPU training
    if args.local_rank == -1:
        torch.cuda.set_device(int(args.device.split(":")[1]) if ":" in args.device else 0)

    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')

    from kronos_classification_base import KronosClassificationModel
    
    print(f"Loading pretrained model from {args.pretrained_checkpoint}...")
    model = KronosClassificationModel.from_pretrained(
        args.pretrained_checkpoint,
        num_classes=args.num_classes
    )
    
    train_dataset = KronosTimeSeriesDataset(
        args.data_dir,
        model.tokenizer,
        max_context=args.max_context,
        use_volume=not args.no_volume,
        train_split=args.train_split,
        val_split=args.val_split,
        split_type='train',
        class_balance=args.class_balance,
        oversample_ratio=args.oversample_ratio
    )
    
    val_dataset = KronosTimeSeriesDataset(
        args.data_dir,
        model.tokenizer,
        max_context=args.max_context,
        use_volume=not args.no_volume,
        train_split=args.train_split,
        val_split=args.val_split,
        split_type='val',
        class_balance='none',  # Don't balance validation set
        oversample_ratio=1.0
    )
    
    test_dataset = KronosTimeSeriesDataset(
        args.data_dir,
        model.tokenizer,
        max_context=args.max_context,
        use_volume=not args.no_volume,
        train_split=args.train_split,
        val_split=args.val_split,
        split_type='test',
        class_balance='none',  # Don't balance test set
        oversample_ratio=1.0
    )
    
    # Get class weights if using weighted loss
    class_weights = None
    if args.class_balance == 'class_weights':
        class_weights = train_dataset.get_class_weights()
        print(f"Using class weights: {class_weights}")
    
    finetuner = KronosFineTuner(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        local_rank=args.local_rank,
        fp16=args.fp16,
        class_weights=class_weights,
        device=args.device,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    
    finetuner.train()
    print("Fine-tuning completed!")


if __name__ == "__main__":
    main()