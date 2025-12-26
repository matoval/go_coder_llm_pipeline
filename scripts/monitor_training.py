#!/usr/bin/env python3
"""
Monitor live training progress - check latest checkpoint and show metrics.
"""

import sys
import torch
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import get_small_config
from tokenizer.tokenizer import GoTokenizer


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints"):
    """Find the most recent checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    checkpoints = list(checkpoint_path.glob("*.pt"))
    if not checkpoints:
        return None

    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def check_checkpoint(checkpoint_path: str):
    """Load and display checkpoint info."""
    print(f"\n{'='*80}")
    print(f"CHECKPOINT: {Path(checkpoint_path).name}")
    print(f"{'='*80}\n")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Basic info
        print("Training Progress:")
        print(f"  Global step: {checkpoint.get('global_step', 'unknown')}")

        # Model config
        if 'model_config' in checkpoint:
            cfg = checkpoint['model_config']
            print(f"\nModel Config:")
            print(f"  Vocab size: {cfg.get('vocab_size', 'unknown')}")
            print(f"  Planner layers: {cfg.get('planner_config', {}).get('n_layer', 'unknown')}")
            print(f"  Generator layers: {cfg.get('generator_config', {}).get('n_layer', 'unknown')}")

        # Try to test generation
        print(f"\nTesting model generation...")
        config = get_small_config()
        from model.hrm_model import HierarchicalGoCoderModel
        model = HierarchicalGoCoderModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        tokenizer = GoTokenizer('tokenizer/go_coder_llm.model')

        # Simple test
        test_text = "<PROBLEM>Fix the authentication bug in the login handler</PROBLEM>"
        test_ids = tokenizer.encode(test_text)
        test_tensor = torch.tensor([test_ids], dtype=torch.long)

        with torch.no_grad():
            output = model.generate(
                problem_ids=test_tensor,
                max_plan_tokens=30,
                max_code_tokens=100,
                max_refinement_iterations=1,
                temperature=0.8,
            )

        plan_text = tokenizer.decode(output.plan_tokens[0].tolist())
        code_text = tokenizer.decode(output.generated_code[0].tolist())

        print(f"\nSample Generation:")
        print(f"  Input: {test_text}")
        print(f"\n  Generated Plan:")
        print(f"    {plan_text[:200]}")
        print(f"\n  Generated Code:")
        print(f"    {code_text[:200]}")

        print(f"\n✓ Model can generate output successfully!")

    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()


def monitor_training(checkpoint_dir: str = "checkpoints", interval: int = 60):
    """Monitor training progress continuously."""
    print(f"Monitoring training in: {checkpoint_dir}")
    print(f"Checking every {interval} seconds...")
    print(f"Press Ctrl+C to stop\n")

    last_checkpoint = None

    try:
        while True:
            latest = find_latest_checkpoint(checkpoint_dir)

            if latest and latest != last_checkpoint:
                check_checkpoint(latest)
                last_checkpoint = latest
            elif not latest:
                print(f"No checkpoints found yet in {checkpoint_dir}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory containing checkpoints')
    parser.add_argument('--checkpoint', type=str,
                        help='Specific checkpoint to check')
    parser.add_argument('--interval', type=int, default=60,
                        help='Monitoring interval in seconds')
    parser.add_argument('--watch', action='store_true',
                        help='Continuously monitor for new checkpoints')

    args = parser.parse_args()

    if args.checkpoint:
        # Check specific checkpoint
        check_checkpoint(args.checkpoint)
    elif args.watch:
        # Monitor continuously
        monitor_training(args.checkpoint_dir, args.interval)
    else:
        # Check latest checkpoint once
        latest = find_latest_checkpoint(args.checkpoint_dir)
        if latest:
            check_checkpoint(latest)
        else:
            print(f"No checkpoints found in {args.checkpoint_dir}")
