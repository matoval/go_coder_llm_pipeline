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
from torch.optim.lr_scheduler import LambdaLR
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
    batch_size: int = 4  # Reduced from 16 to fit in 12GB GPU
    gradient_accumulation_steps: int = 4  # Accumulate to effective batch size of 16
    num_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True  # Enable to save memory during backward pass

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

    def __init__(self, data_path: str, max_length: int = 1024, pad_token_id: int = 3):
        """
        Initialize dataset.

        Args:
            data_path: Path to tokenized JSONL file
            max_length: Maximum sequence length
            pad_token_id: Token ID to use for padding (default 3)
        """
        self.data_path = data_path
        self.max_length = max_length
        self.pad_token_id = pad_token_id
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
            input_ids = input_ids + [self.pad_token_id] * padding
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

        # Enable gradient checkpointing if requested (saves memory during backward pass)
        if train_config.gradient_checkpointing:
            if hasattr(self.model.planner, 'gradient_checkpointing_enable'):
                self.model.planner.gradient_checkpointing_enable()
            if hasattr(self.model.generator, 'gradient_checkpointing_enable'):
                self.model.generator.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

        # Learning rate scheduler with warmup
        def lr_lambda(current_step: int):
            if current_step < train_config.warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, train_config.warmup_steps))
            # Cosine decay after warmup
            progress = float(current_step - train_config.warmup_steps) / float(
                max(1, train_config.num_epochs * 100 - train_config.warmup_steps)
            )
            import math
            return max(0.0, 0.5 * (1.0 + math.cos(progress * math.pi)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # Loss functions
        # FIXED: Use model_config.pad_token_id (3) instead of hardcoded 0
        self.criterion = nn.CrossEntropyLoss(ignore_index=model_config.pad_token_id)  # Ignore padding for plan/code
        self.refinement_criterion = nn.CrossEntropyLoss()  # No ignore_index for refinement

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

            # Split based on special tokens: <PLAN> and <CODE>
            problem_ids, target_plan, target_code = self.split_sequence(input_ids)

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
            # Handle NaN (occurs when all tokens in batch are padding)
            if torch.isnan(plan_loss):
                plan_loss = torch.tensor(0.0, device=self.device)

            # Code loss
            code_logits = output['code_logits']
            code_loss = self.criterion(
                code_logits.reshape(-1, self.model_config.vocab_size),
                target_code.reshape(-1)
            )
            # Handle NaN (occurs when all tokens in batch are padding)
            if torch.isnan(code_loss):
                code_loss = torch.tensor(0.0, device=self.device)

            # Refinement loss - use validation feedback from data
            refinement_logits = output['refinement_logits']

            # Extract validation signals from input sequence
            # Format: <SYNTAX_OK> true/false <TEST_PASS> true/false
            syntax_ok_token_id = self.model_config.special_tokens.get('<SYNTAX_OK>', 28)
            test_pass_token_id = self.model_config.special_tokens.get('<TEST_PASS>', 30)

            # Find validation tokens in the batch
            refinement_target = self._extract_refinement_targets(
                batch['input_ids'],
                syntax_ok_token_id,
                test_pass_token_id
            )

            refinement_loss = self.refinement_criterion(
                refinement_logits,
                refinement_target
            )

            # Combined loss
            loss = (
                self.train_config.plan_loss_weight * plan_loss +
                self.train_config.code_loss_weight * code_loss +
                self.train_config.refinement_loss_weight * refinement_loss
            )

            # Track metrics (before scaling for gradient accumulation)
            total_loss += loss.item()
            total_plan_loss += plan_loss.item()
            total_code_loss += code_loss.item()
            total_refinement_loss += refinement_loss.item()

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.train_config.gradient_accumulation_steps

            # Backward pass
            scaled_loss.backward()

            # Only update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.train_config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'plan': f"{plan_loss.item():.4f}",
                'code': f"{code_loss.item():.4f}",
                'refine': f"{refinement_loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
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

    def split_sequence(self, input_ids: torch.Tensor):
        """Split input sequence into problem, plan, and code based on special tokens."""
        plan_token_id = self.model_config.special_tokens['<PLAN>']
        code_token_id = self.model_config.special_tokens['<CODE>']

        batch_size = input_ids.size(0)
        problem_ids_list = []
        target_plan_list = []
        target_code_list = []

        for i in range(batch_size):
            seq = input_ids[i]

            # Find where <PLAN> and <CODE> tokens appear
            plan_positions = (seq == plan_token_id).nonzero(as_tuple=True)[0]
            code_positions = (seq == code_token_id).nonzero(as_tuple=True)[0]

            # Fallback to 1/3 splitting if tokens not found
            if len(plan_positions) == 0 or len(code_positions) == 0:
                seq_len = seq.size(0)
                problem_len = seq_len // 3
                plan_len = seq_len // 3

                problem_ids_list.append(seq[:problem_len])
                target_plan_list.append(seq[problem_len:problem_len+plan_len])
                target_code_list.append(seq[problem_len+plan_len:])
            else:
                # Use first occurrence of each token
                plan_start = plan_positions[0].item()
                code_start = code_positions[0].item()

                # Split: problem is before <PLAN>, plan is <PLAN> to <CODE>, code is from <CODE> onward
                problem_ids_list.append(seq[:plan_start])
                target_plan_list.append(seq[plan_start:code_start])
                target_code_list.append(seq[code_start:])

        # Pad sequences to same length within batch
        def pad_sequences(seqs):
            max_len = max(s.size(0) for s in seqs)
            # FIXED: Use pad_token_id (3) instead of 0
            padded = torch.full((batch_size, max_len), self.model_config.pad_token_id, dtype=torch.long, device=self.device)
            for i, s in enumerate(seqs):
                padded[i, :s.size(0)] = s
            return padded

        problem_ids = pad_sequences(problem_ids_list)
        target_plan = pad_sequences(target_plan_list)
        target_code = pad_sequences(target_code_list)

        return problem_ids, target_plan, target_code

    def _extract_refinement_targets(
        self,
        input_ids: torch.Tensor,
        syntax_ok_token_id: int,
        test_pass_token_id: int
    ) -> torch.Tensor:
        """
        Extract refinement targets from validation tokens in the sequence.

        Returns:
            Tensor of shape (batch_size,) with values:
            - 0 (CONTINUE): No validation info found
            - 1 (REFINE): Validation failed (would need <SYNTAX_ERR> or <TEST_FAIL> tokens)
            - 2 (DONE): Validation passed (both <SYNTAX_OK> and <TEST_PASS> found)

        Note: This is a simplified version. Full implementation would parse
        the actual true/false values after the validation tokens.
        """
        batch_size = input_ids.size(0)
        refinement_targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i in range(batch_size):
            seq = input_ids[i]

            # Check if validation tokens exist
            has_syntax_ok = (seq == syntax_ok_token_id).any()
            has_test_pass = (seq == test_pass_token_id).any()

            # Simple heuristic:
            # - If both validation tokens present, assume tests passed → DONE (2)
            # - Otherwise, no validation info → CONTINUE (0)
            # TODO: Parse actual true/false values for more accuracy
            if has_syntax_ok and has_test_pass:
                refinement_targets[i] = 2  # DONE
            else:
                refinement_targets[i] = 0  # CONTINUE

        return refinement_targets

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0

        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)

            # Split based on special tokens
            problem_ids, target_plan, target_code = self.split_sequence(input_ids)

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
            # Handle NaN (occurs when all tokens in batch are padding)
            if torch.isnan(plan_loss):
                plan_loss = torch.tensor(0.0, device=self.device)

            code_loss = self.criterion(
                output['code_logits'].reshape(-1, self.model_config.vocab_size),
                target_code.reshape(-1)
            )
            # Handle NaN (occurs when all tokens in batch are padding)
            if torch.isnan(code_loss):
                code_loss = torch.tensor(0.0, device=self.device)

            refinement_target = torch.zeros(
                output['refinement_logits'].size(0),
                dtype=torch.long,
                device=self.device
            )
            refinement_loss = self.refinement_criterion(
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
            'scheduler_state_dict': self.scheduler.state_dict(),
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

        # Load scheduler state if available (for backwards compatibility)
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.global_step = checkpoint['global_step']

        logger.info(f"Resumed from step {self.global_step}")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Hierarchical Go Coder")
    parser.add_argument('--config', type=str, default='small', choices=['tiny', 'small', 'medium'])
    parser.add_argument('--train-data', type=str, default='data/tokenized/train.jsonl')
    parser.add_argument('--val-data', type=str, default='data/tokenized/val.jsonl')
    parser.add_argument('--output-dir', type=str, default='checkpoints')
    parser.add_argument('--batch-size', type=int, default=4)
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
        max_length=max_seq_length,
        pad_token_id=model_config.pad_token_id
    )

    if Path(train_config.val_data_path).exists():
        val_dataset = HierarchicalDataset(
            train_config.val_data_path,
            max_length=max_seq_length,
            pad_token_id=model_config.pad_token_id
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
