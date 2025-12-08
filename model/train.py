"""
Multi-Task Training Script for Hierarchical Go Coder Model

Trains the HRM/TRM model with:
1. Planning loss (predict plan tokens)
2. Generation loss (predict code tokens)
3. Refinement loss (learn when to refine/continue/stop)
"""

import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.hrm_model import HierarchicalGoCoderModel
from model.config import HRMConfig, get_small_config, get_medium_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Data
    train_data_path: str = "data/tokenized/train.jsonl"
    val_data_path: str = "data/tokenized/val.jsonl"

    # Training
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Loss weights
    plan_loss_weight: float = 0.4
    code_loss_weight: float = 0.4
    refinement_loss_weight: float = 0.2

    # Validation
    eval_every: int = 500
    save_every: int = 1000

    # Checkpoint
    output_dir: str = "checkpoints"
    resume_from: Optional[str] = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4


class HierarchicalDataset(Dataset):
    """Dataset for hierarchical training data."""

    def __init__(self, data_path: str, max_length: int = 1024):
        """
        Initialize dataset.

        Args:
            data_path: Path to tokenized JSONL file
            max_length: Maximum sequence length
        """
        self.data_path = data_path
        self.max_length = max_length
        self.samples = []

        # Load data
        logger.info(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                self.samples.append(record)

        logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns dict with:
            - input_ids: Full sequence
            - attention_mask: Mask for padding
            - For now, we'll split into problem/plan/code during training
        """
        sample = self.samples[idx]

        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']

        # Pad or truncate
        if len(input_ids) < self.max_length:
            padding = self.max_length - len(input_ids)
            input_ids = input_ids + [0] * padding  # 0 is pad token
            attention_mask = attention_mask + [0] * padding
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }


class HRMTrainer:
    """Trainer for Hierarchical Recursive Model."""

    def __init__(
        self,
        model: HierarchicalGoCoderModel,
        train_config: TrainingConfig,
        model_config: HRMConfig,
    ):
        """
        Initialize trainer.

        Args:
            model: HRM model instance
            train_config: Training configuration
            model_config: Model configuration
        """
        self.model = model
        self.train_config = train_config
        self.model_config = model_config

        # Move model to device
        self.device = torch.device(train_config.device)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

        # Loss functions
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

        # Metrics
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Create output directory
        Path(train_config.output_dir).mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_plan_loss = 0
        total_code_loss = 0
        total_refinement_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # For now, use simple splitting strategy
            # TODO: Implement proper problem/plan/code splitting based on special tokens
            seq_len = input_ids.size(1)

            # Split roughly: first 1/3 problem, middle 1/3 plan, last 1/3 code
            problem_len = seq_len // 3
            plan_len = seq_len // 3

            problem_ids = input_ids[:, :problem_len]
            target_plan = input_ids[:, problem_len:problem_len+plan_len]
            target_code = input_ids[:, problem_len+plan_len:]

            # Forward pass
            output = self.model(
                problem_ids=problem_ids,
                target_plan=target_plan,
                target_code=target_code,
            )

            # Compute losses
            # Plan loss
            plan_logits = output['plan_logits']
            plan_loss = self.criterion(
                plan_logits.reshape(-1, self.model_config.vocab_size),
                target_plan.reshape(-1)
            )

            # Code loss
            code_logits = output['code_logits']
            code_loss = self.criterion(
                code_logits.reshape(-1, self.model_config.vocab_size),
                target_code.reshape(-1)
            )

            # Refinement loss
            # TODO: Integrate GoSyntaxValidator to provide real validation feedback
            # For now, use a simple heuristic: assume DONE (target=2) for complete sequences
            refinement_logits = output['refinement_logits']

            # Improved heuristic: check if sequence looks complete
            # If target_code contains EOS token, assume DONE (2), otherwise CONTINUE (0)
            eos_token_id = self.model_config.eos_token_id
            has_eos = (target_code == eos_token_id).any(dim=1)

            # Decision: DONE (2) if has EOS, otherwise CONTINUE (0)
            # (REFINE=1 would require more sophisticated logic with validation)
            refinement_target = torch.where(
                has_eos,
                torch.tensor(2, dtype=torch.long, device=self.device),
                torch.tensor(0, dtype=torch.long, device=self.device)
            )

            refinement_loss = self.criterion(
                refinement_logits,
                refinement_target
            )

            # Combined loss
            loss = (
                self.train_config.plan_loss_weight * plan_loss +
                self.train_config.code_loss_weight * code_loss +
                self.train_config.refinement_loss_weight * refinement_loss
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_config.max_grad_norm
            )

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_plan_loss += plan_loss.item()
            total_code_loss += code_loss.item()
            total_refinement_loss += refinement_loss.item()

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'plan': f"{plan_loss.item():.4f}",
                'code': f"{code_loss.item():.4f}",
                'refine': f"{refinement_loss.item():.4f}",
            })

            # Save checkpoint
            if self.global_step % self.train_config.save_every == 0:
                self.save_checkpoint(f"step_{self.global_step}")

        # Return average losses
        num_batches = len(dataloader)
        return {
            'loss': total_loss / num_batches,
            'plan_loss': total_plan_loss / num_batches,
            'code_loss': total_code_loss / num_batches,
            'refinement_loss': total_refinement_loss / num_batches,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0

        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)

            # Same splitting as training
            seq_len = input_ids.size(1)
            problem_len = seq_len // 3
            plan_len = seq_len // 3

            problem_ids = input_ids[:, :problem_len]
            target_plan = input_ids[:, problem_len:problem_len+plan_len]
            target_code = input_ids[:, problem_len+plan_len:]

            # Forward pass
            output = self.model(
                problem_ids=problem_ids,
                target_plan=target_plan,
                target_code=target_code,
            )

            # Compute losses
            plan_loss = self.criterion(
                output['plan_logits'].reshape(-1, self.model_config.vocab_size),
                target_plan.reshape(-1)
            )
            code_loss = self.criterion(
                output['code_logits'].reshape(-1, self.model_config.vocab_size),
                target_code.reshape(-1)
            )

            refinement_target = torch.zeros(
                output['refinement_logits'].size(0),
                dtype=torch.long,
                device=self.device
            )
            refinement_loss = self.criterion(
                output['refinement_logits'],
                refinement_target
            )

            loss = (
                self.train_config.plan_loss_weight * plan_loss +
                self.train_config.code_loss_weight * code_loss +
                self.train_config.refinement_loss_weight * refinement_loss
            )

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return {'val_loss': avg_loss}

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = Path(self.train_config.output_dir) / f"{name}.pt"

        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': asdict(self.model_config),
            'train_config': asdict(self.train_config),
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']

        logger.info(f"Resumed from step {self.global_step}")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Hierarchical Go Coder")
    parser.add_argument('--config', type=str, default='small', choices=['tiny', 'small', 'medium'])
    parser.add_argument('--train-data', type=str, default='data/tokenized/train.jsonl')
    parser.add_argument('--val-data', type=str, default='data/tokenized/val.jsonl')
    parser.add_argument('--output-dir', type=str, default='checkpoints')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')

    args = parser.parse_args()

    # Model config
    if args.config == 'tiny':
        from model.config import get_tiny_config
        model_config = get_tiny_config()
    elif args.config == 'small':
        model_config = get_small_config()
    else:
        model_config = get_medium_config()

    logger.info(f"Model config: {model_config}")

    # Training config
    train_config = TrainingConfig(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        resume_from=args.resume,
    )

    # Create model
    logger.info("Creating model...")
    model = HierarchicalGoCoderModel(model_config)

    param_counts = model.get_num_params()
    logger.info(f"Model parameters:")
    for name, count in param_counts.items():
        logger.info(f"  {name}: {count/1e6:.2f}M")

    # Create datasets
    logger.info("Loading datasets...")
    # Use the smaller of planner and generator n_positions
    max_seq_length = min(
        model_config.planner_config.n_positions,
        model_config.generator_config.n_positions
    )
    logger.info(f"Using max sequence length: {max_seq_length}")

    train_dataset = HierarchicalDataset(
        train_config.train_data_path,
        max_length=max_seq_length
    )

    if Path(train_config.val_data_path).exists():
        val_dataset = HierarchicalDataset(
            train_config.val_data_path,
            max_length=max_seq_length
        )
    else:
        logger.warning("No validation data found, skipping validation")
        val_dataset = None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    ) if val_dataset else None

    # Create trainer
    trainer = HRMTrainer(model, train_config, model_config)

    # Resume if checkpoint provided
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(train_config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{train_config.num_epochs}")

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch + 1)
        logger.info(f"Train metrics: {train_metrics}")

        # Validate
        if val_loader:
            val_metrics = trainer.validate(val_loader)
            logger.info(f"Validation metrics: {val_metrics}")

            # Save best model
            if val_metrics['val_loss'] < trainer.best_val_loss:
                trainer.best_val_loss = val_metrics['val_loss']
                trainer.save_checkpoint('best')

        # Save epoch checkpoint
        trainer.save_checkpoint(f'epoch_{epoch + 1}')

    logger.info("\nTraining complete!")
    logger.info(f"Checkpoints saved to {train_config.output_dir}")


if __name__ == "__main__":
    main()
