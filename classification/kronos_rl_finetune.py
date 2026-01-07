"""
Kronos RL Fine-tuning Script
Fine-tune classification model using Reinforcement Learning (Policy Gradient).
This is useful for further improving accuracy after supervised training.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import json
from collections import Counter
import glob
import random


# Import the dataset class from pretrain script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kronos_pretrain import KronosTimeSeriesDataset, collate_fn


class PolicyGradientFinetuner:
    """
    Fine-tune using REINFORCE (Policy Gradient) algorithm.

    This treats classification as a policy decision and optimizes
    based on reward signals (correctness with potential weighting).
    """

    def __init__(
        self,
        model,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        output_dir: str = "./rl_checkpoints",
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        num_epochs: int = 3,
        gamma: float = 0.99,  # Discount factor for future rewards
        entropy_coef: float = 0.01,  # Entropy regularization coefficient
        reward_scale: float = 1.0,  # Scale for rewards
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        logging_steps: int = 50,
        eval_steps: int = 500,
        local_rank: int = -1,
        fp16: bool = False,
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
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.reward_scale = reward_scale
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.local_rank = local_rank
        self.fp16 = fp16
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        os.makedirs(output_dir, exist_ok=True)

        # Setup distributed training
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

        print(f"RL training on device: {self.device}")

        self._setup_dataloaders()
        self._setup_optimizer()

        self.scaler = torch.cuda.amp.GradScaler() if fp16 else None
        self.global_step = 0
        self.best_val_reward = float('-inf')

    def _setup_dataloaders(self):
        """Setup training and validation data loaders."""
        if self.is_distributed:
            train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False
            ) if self.val_dataset else None
            test_sampler = DistributedSampler(
                self.test_dataset, shuffle=False
            ) if self.test_dataset else None
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
        """Setup optimizer and learning rate scheduler."""
        # Only optimize classification head parameters initially
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight', 'backbone']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.01
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
        warmup_steps = int(total_steps * 0.1)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    def compute_rewards(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for each action (prediction).

        Args:
            logits: Model logits [batch_size, num_classes]
            labels: Ground truth labels [batch_size]

        Returns:
            rewards: Reward for each sample [batch_size]
        """
        probs = F.softmax(logits, dim=-1)
        predicted_probs = probs[torch.arange(len(labels)), labels]

        # Reward is the probability assigned to the correct class
        # Scale by reward_scale
        rewards = predicted_probs * self.reward_scale

        # Additional bonus for correct predictions
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).float()
        rewards = rewards + correct * self.reward_scale

        return rewards

    def compute_policy_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute policy gradient loss (REINFORCE).

        Args:
            logits: Model logits [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
            rewards: Reward for each sample [batch_size]

        Returns:
            loss: Policy loss
        """
        # Compute log probabilities for the taken actions (correct labels)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs[torch.arange(len(labels)), labels]

        # Policy gradient: -log_prob * reward
        # We want to maximize reward, so we minimize negative reward
        policy_loss = -(action_log_probs * rewards.detach()).mean()

        # Add entropy regularization to encourage exploration
        entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy

        total_loss = policy_loss + entropy_loss
        return total_loss

    def train(self):
        """Main training loop."""
        print(f"Starting RL fine-tuning for {self.num_epochs} epochs...")
        print(f"Algorithm: REINFORCE (Policy Gradient)")

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            self._train_epoch(epoch)

            if self.val_loader and (self.local_rank <= 0):
                avg_reward = self._evaluate(self.val_loader, "Validation")
                print(f"Validation Average Reward: {avg_reward:.4f}")

                if avg_reward > self.best_val_reward:
                    self.best_val_reward = avg_reward
                    self._save_checkpoint("best_model")

            if self.local_rank <= 0:
                self._save_checkpoint(f"epoch_{epoch + 1}")

        if self.test_loader and (self.local_rank <= 0):
            print("\n" + "=" * 50)
            print("Final Test Evaluation")
            print("=" * 50)
            test_reward = self._evaluate(self.test_loader, "Test", detailed=True)
            print(f"Test Average Reward: {test_reward:.4f}")

    def _train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()

        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0
        epoch_reward = 0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc="Training",
            disable=(self.local_rank > 0)
        )

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch, return_dict=True)
                    logits = outputs['logits']
                    labels = batch['labels']

                    # Compute rewards
                    rewards = self.compute_rewards(logits, labels)

                    # Compute policy loss
                    loss = self.compute_policy_loss(logits, labels, rewards) / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(**batch, return_dict=True)
                logits = outputs['logits']
                labels = batch['labels']

                # Compute rewards
                rewards = self.compute_rewards(logits, labels)

                # Compute policy loss
                loss = self.compute_policy_loss(logits, labels, rewards) / self.gradient_accumulation_steps
                loss.backward()

            epoch_loss += loss.item() * self.gradient_accumulation_steps
            epoch_reward += rewards.mean().item()
            num_batches += 1

            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.logging_steps == 0 and self.local_rank <= 0:
                    avg_loss = epoch_loss / (step + 1)
                    avg_reward = epoch_reward / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'reward': f'{avg_reward:.4f}',
                        'lr': f'{lr:.2e}'
                    })

                if (self.val_loader and
                    self.global_step % self.eval_steps == 0 and
                    self.local_rank <= 0):
                    val_reward = self._evaluate(self.val_loader, "Validation")
                    print(f"\nStep {self.global_step} - Validation Reward: {val_reward:.4f}")
                    self.model.train()

    def _evaluate(
        self,
        dataloader: DataLoader,
        split_name: str = "Validation",
        detailed: bool = False
    ) -> float:
        """Evaluate model."""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_rewards = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}", disable=(self.local_rank > 0)):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                logits = outputs['logits']
                labels = batch['labels']

                # Compute rewards
                rewards = self.compute_rewards(logits, labels)
                all_rewards.extend(rewards.cpu().numpy())

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_reward = np.mean(all_rewards)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

        if detailed and self.local_rank <= 0:
            from sklearn.metrics import classification_report, f1_score
            print(f"\n{split_name} Accuracy: {accuracy:.4f}")
            print(f"{split_name} Average Reward: {avg_reward:.4f}")
            print(f"{split_name} F1 Score: {f1_score(all_labels, all_preds, average='weighted'):.4f}")
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds))

        return avg_reward

    def _save_checkpoint(self, checkpoint_name: str):
        """Save checkpoint."""
        from kronos_classification_base import KronosClassificationModel

        save_path = os.path.join(self.output_dir, checkpoint_name)

        model_to_save = self.model.module if self.is_distributed else self.model
        model_to_save.save_pretrained(save_path, save_format="both")

        torch.save({
            'global_step': self.global_step,
            'best_val_reward': self.best_val_reward,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, os.path.join(save_path, 'rl_training_state.bin'))

        print(f"RL checkpoint saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="RL Fine-tune Kronos Classification Model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to pretrained/supervised-trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing JSON files or single JSON file")
    parser.add_argument("--output_dir", type=str, default="./rl_finetuned_checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_context", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor for rewards")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                       help="Entropy regularization coefficient")
    parser.add_argument("--reward_scale", type=float, default=1.0,
                       help="Scale factor for rewards")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Fraction of data for training")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Fraction of data for validation")
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

    print(f"Loading pretrained model from {args.model_path}...")
    model = KronosClassificationModel.from_pretrained(args.model_path)

    train_dataset = KronosTimeSeriesDataset(
        args.data_dir,
        model.tokenizer,
        max_context=args.max_context,
        use_volume=not args.no_volume,
        train_split=args.train_split,
        val_split=args.val_split,
        split_type='train',
        class_balance='none',  # No class balancing in RL phase
        oversample_ratio=1.0
    )

    val_dataset = KronosTimeSeriesDataset(
        args.data_dir,
        model.tokenizer,
        max_context=args.max_context,
        use_volume=not args.no_volume,
        train_split=args.train_split,
        val_split=args.val_split,
        split_type='val',
        class_balance='none',
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
        class_balance='none',
        oversample_ratio=1.0
    )

    rl_finetuner = PolicyGradientFinetuner(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        reward_scale=args.reward_scale,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        local_rank=args.local_rank,
        fp16=args.fp16,
        device=args.device,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    rl_finetuner.train()
    print("RL fine-tuning completed!")


if __name__ == "__main__":
    main()
