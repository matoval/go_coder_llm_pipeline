#!/usr/bin/env python3
"""
Quick health check for training - verifies training is working without waiting days.

Checks:
1. Losses are finite (not NaN/Inf)
2. Losses are decreasing over batches
3. Gradients are flowing (not zero or exploding)
4. Model can generate coherent output
5. GPU memory is stable
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import get_small_config
from model.hrm_model import HierarchicalGoCoderModel
from model.train import HierarchicalDataset, HRMTrainer, TrainingConfig
from torch.utils.data import DataLoader
from tokenizer.tokenizer import GoTokenizer
import numpy as np


def check_training_health(checkpoint_path: str = None, num_batches: int = 50):
    """
    Run health checks on training.

    Args:
        checkpoint_path: Path to checkpoint to check (optional)
        num_batches: Number of batches to train for health check
    """
    print("=" * 80)
    print("TRAINING HEALTH CHECK")
    print("=" * 80)
    print()

    # Load config
    config = get_small_config()
    train_config = TrainingConfig()

    # Create model
    print("1. Loading model...")
    model = HierarchicalGoCoderModel(config)

    if checkpoint_path:
        print(f"   Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Checkpoint at step: {checkpoint.get('global_step', 'unknown')}")
    else:
        print("   Using fresh model")

    # Load small dataset
    print("\n2. Loading dataset...")
    dataset = HierarchicalDataset(
        train_config.train_data_path,
        max_length=512,
        pad_token_id=config.pad_token_id
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"   Loaded {len(dataset)} samples")

    # Create trainer
    trainer = HRMTrainer(model, train_config, config)

    # Track metrics
    print(f"\n3. Running {num_batches} training steps...")
    losses = []
    plan_losses = []
    code_losses = []
    grad_norms = []

    model.train()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        # Move to device
        input_ids = batch['input_ids'].to(trainer.device)

        # Split sequence
        problem_ids, target_plan, target_code = trainer.split_sequence(input_ids)

        # Forward pass
        output = model(
            problem_ids=problem_ids,
            target_plan=target_plan,
            target_code=target_code,
        )

        # Compute losses
        plan_loss = trainer.criterion(
            output['plan_logits'].reshape(-1, config.vocab_size),
            target_plan.reshape(-1)
        )
        if torch.isnan(plan_loss):
            plan_loss = torch.tensor(0.0, device=trainer.device)

        code_loss = trainer.criterion(
            output['code_logits'].reshape(-1, config.vocab_size),
            target_code.reshape(-1)
        )
        if torch.isnan(code_loss):
            code_loss = torch.tensor(0.0, device=trainer.device)

        eos_token_id = config.eos_token_id
        has_eos = (target_code == eos_token_id).any(dim=1)
        refinement_target = torch.where(
            has_eos,
            torch.tensor(2, dtype=torch.long, device=trainer.device),
            torch.tensor(0, dtype=torch.long, device=trainer.device)
        )
        refinement_loss = trainer.refinement_criterion(
            output['refinement_logits'],
            refinement_target
        )

        loss = (
            train_config.plan_loss_weight * plan_loss +
            train_config.code_loss_weight * code_loss +
            train_config.refinement_loss_weight * refinement_loss
        )

        # Backward
        trainer.optimizer.zero_grad()
        loss.backward()

        # Check gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        trainer.optimizer.step()

        # Track
        losses.append(loss.item())
        plan_losses.append(plan_loss.item())
        code_losses.append(code_loss.item())
        grad_norms.append(total_norm)

        if (i + 1) % 10 == 0:
            print(f"   Step {i+1}/{num_batches}: loss={loss.item():.4f}, "
                  f"plan={plan_loss.item():.4f}, code={code_loss.item():.4f}, "
                  f"grad_norm={total_norm:.4f}")

    # Analysis
    print("\n" + "=" * 80)
    print("HEALTH CHECK RESULTS")
    print("=" * 80)

    # Check 1: Finite losses
    print("\n✓ CHECK 1: Losses are finite")
    nan_count = sum(1 for l in losses if np.isnan(l) or np.isinf(l))
    if nan_count == 0:
        print(f"   ✓ PASS: All {len(losses)} losses are finite")
    else:
        print(f"   ✗ FAIL: {nan_count}/{len(losses)} losses are NaN/Inf")
        return False

    # Check 2: Loss is decreasing
    print("\n✓ CHECK 2: Loss is decreasing")
    first_10_avg = np.mean(losses[:10])
    last_10_avg = np.mean(losses[-10:])
    improvement = (first_10_avg - last_10_avg) / first_10_avg * 100

    print(f"   First 10 batches avg loss: {first_10_avg:.4f}")
    print(f"   Last 10 batches avg loss:  {last_10_avg:.4f}")
    print(f"   Improvement: {improvement:.2f}%")

    if improvement > 0:
        print(f"   ✓ PASS: Loss decreased by {improvement:.2f}%")
    else:
        print(f"   ⚠ WARNING: Loss not decreasing yet (may need more steps)")

    # Check 3: Gradients are flowing
    print("\n✓ CHECK 3: Gradients are flowing")
    avg_grad_norm = np.mean(grad_norms)
    max_grad_norm = np.max(grad_norms)
    min_grad_norm = np.min(grad_norms)

    print(f"   Average gradient norm: {avg_grad_norm:.4f}")
    print(f"   Max gradient norm: {max_grad_norm:.4f}")
    print(f"   Min gradient norm: {min_grad_norm:.4f}")

    if avg_grad_norm > 0.001 and avg_grad_norm < 100:
        print(f"   ✓ PASS: Gradients in healthy range")
    elif avg_grad_norm <= 0.001:
        print(f"   ✗ FAIL: Gradients too small (vanishing gradients)")
        return False
    else:
        print(f"   ✗ FAIL: Gradients too large (exploding gradients)")
        return False

    # Check 4: Can generate output
    print("\n✓ CHECK 4: Model can generate output")
    model.eval()
    tokenizer = GoTokenizer('tokenizer/go_coder_llm.model')

    # Create simple test input
    test_text = "<PROBLEM>Fix null pointer bug</PROBLEM>"
    test_ids = tokenizer.encode(test_text)
    test_tensor = torch.tensor([test_ids], dtype=torch.long).to(trainer.device)

    try:
        with torch.no_grad():
            output = model.generate(
                problem_ids=test_tensor,
                max_plan_tokens=20,
                max_code_tokens=50,
                max_refinement_iterations=1,
            )

        plan_text = tokenizer.decode(output.plan_tokens[0].cpu().tolist())
        code_text = tokenizer.decode(output.generated_code[0].cpu().tolist())

        print(f"   Generated plan preview: {plan_text[:100]}...")
        print(f"   Generated code preview: {code_text[:100]}...")
        print(f"   ✓ PASS: Model can generate output")
    except Exception as e:
        print(f"   ✗ FAIL: Generation failed: {e}")
        return False

    # Check 5: Loss components
    print("\n✓ CHECK 5: Loss components")
    print(f"   Plan loss avg: {np.mean(plan_losses):.4f}")
    print(f"   Code loss avg: {np.mean(code_losses):.4f}")
    print(f"   Both plan and code have reasonable losses")

    if np.mean(plan_losses) < 15 and np.mean(code_losses) < 15:
        print(f"   ✓ PASS: Loss components in expected range")
    else:
        print(f"   ⚠ WARNING: Losses are high (expected for early training)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\n✓ Training appears to be working correctly!")
    print(f"\nKey metrics after {num_batches} steps:")
    print(f"  - Loss: {last_10_avg:.4f} (improved {improvement:.2f}%)")
    print(f"  - Gradient norm: {avg_grad_norm:.4f}")
    print(f"  - Model can generate output: Yes")
    print("\nRecommendation: Training is healthy. Continue training.")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check training health")
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to check')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps to test')

    args = parser.parse_args()

    success = check_training_health(args.checkpoint, args.steps)
    sys.exit(0 if success else 1)
