#!/usr/bin/env python3
"""
Inference script for HRM Go Coder model.
Load a trained checkpoint and generate code for test problems.
"""

import sys
import json
import torch
from pathlib import Path
from typing import List, Dict
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.hrm_model import HierarchicalGoCoderModel
from model.config import HRMConfig, ModuleConfig
from tokenizer.tokenizer import GoTokenizer


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Recreate model config
    config_dict = checkpoint['model_config']

    # Reconstruct ModuleConfig objects from dicts
    planner_config = ModuleConfig(**config_dict['planner_config'])
    generator_config = ModuleConfig(**config_dict['generator_config'])

    # Create HRMConfig with proper module configs
    model_config = HRMConfig()
    model_config.vocab_size = config_dict['vocab_size']
    model_config.bos_token_id = config_dict['bos_token_id']
    model_config.eos_token_id = config_dict['eos_token_id']
    model_config.pad_token_id = config_dict['pad_token_id']
    model_config.planner_config = planner_config
    model_config.generator_config = generator_config
    model_config.special_tokens = config_dict['special_tokens']

    # Create model
    model = HierarchicalGoCoderModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Global step: {checkpoint['global_step']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model, model_config


def prepare_problem(problem: Dict, tokenizer: GoTokenizer, max_length: int = 256) -> torch.Tensor:
    """
    Convert problem to tokenized format expected by model.

    Args:
        problem: Dict with 'description' and optionally 'context'
        tokenizer: Tokenizer instance
        max_length: Max input length

    Returns:
        problem_ids: Tensor of shape [1, T]
    """
    # Format problem
    lines = []
    lines.append("<PROBLEM>")
    lines.append(problem['description'])
    lines.append("</PROBLEM>")

    if problem.get('context'):
        lines.append("")
        lines.append("<CONTEXT>")
        lines.append(problem['context'])
        lines.append("</CONTEXT>")

    text = "\n".join(lines)

    # Tokenize
    token_ids = tokenizer.encode(text)

    # Truncate if needed
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]

    # Convert to tensor
    problem_ids = torch.tensor([token_ids], dtype=torch.long)

    return problem_ids


def generate_solution(
    model: HierarchicalGoCoderModel,
    tokenizer: GoTokenizer,
    problem: Dict,
    device: str = 'cpu',
    max_plan_tokens: int = 50,
    max_code_tokens: int = 300,
    temperature: float = 0.8,
) -> Dict:
    """
    Generate code solution for a problem.

    Returns:
        Dict with 'plan', 'code', 'iterations', and 'raw_tokens'
    """
    # Prepare input
    problem_ids = prepare_problem(problem, tokenizer).to(device)

    # Generate
    with torch.no_grad():
        output = model.generate(
            problem_ids=problem_ids,
            max_plan_tokens=max_plan_tokens,
            max_code_tokens=max_code_tokens,
            temperature=temperature,
        )

    # Decode plan
    plan_tokens = output.plan_tokens[0].cpu().tolist()
    plan_text = tokenizer.decode(plan_tokens)

    # Decode final code
    code_tokens = output.generated_code[0].cpu().tolist()
    code_text = tokenizer.decode(code_tokens)

    return {
        'plan': plan_text,
        'code': code_text,
        'iterations': output.num_iterations,
        'refinement_decisions': output.refinement_decisions,
        'plan_tokens': plan_tokens,
        'code_tokens': code_tokens,
    }


def run_inference(
    checkpoint_path: str,
    test_problems: List[Dict],
    tokenizer_path: str,
    device: str = 'cpu',
    output_file: str = None,
    temperature: float = 0.8,
):
    """
    Run inference on test problems.

    Args:
        checkpoint_path: Path to model checkpoint
        test_problems: List of problem dicts
        tokenizer_path: Path to tokenizer model
        device: Device to use
        output_file: Optional file to save results
        temperature: Sampling temperature
    """
    # Load model and tokenizer
    model, config = load_model(checkpoint_path, device)
    tokenizer = GoTokenizer(tokenizer_path)

    print(f"\nRunning inference on {len(test_problems)} problems...")
    print(f"Temperature: {temperature}")
    print()

    results = []

    for i, problem in enumerate(test_problems):
        print(f"=" * 80)
        print(f"Problem {i+1}/{len(test_problems)}: {problem['title']}")
        print(f"=" * 80)
        print(f"\nDescription:")
        print(problem['description'])
        print()

        # Generate solution
        solution = generate_solution(
            model, tokenizer, problem, device, temperature=temperature
        )

        print(f"Plan ({solution['iterations']} iterations):")
        print(solution['plan'])
        print()

        print(f"Generated Code:")
        print(solution['code'])
        print()

        # Store result
        result = {
            'problem': problem,
            'solution': solution,
        }
        results.append(result)

    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference with HRM model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--tokenizer', type=str, default='tokenizer/go_coder_llm.model', help='Tokenizer path')
    parser.add_argument('--problems', type=str, help='JSONL file with test problems')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')

    args = parser.parse_args()

    # Load test problems
    if args.problems:
        test_problems = []
        with open(args.problems, 'r') as f:
            for line in f:
                test_problems.append(json.loads(line))
    else:
        # Use default test problems
        test_problems = [
            {
                'title': 'Simple HTTP Handler',
                'description': 'Write a Go HTTP handler that returns "Hello, World!" for GET requests to /hello',
            },
            {
                'title': 'JSON Parser',
                'description': 'Write a function that parses a JSON string into a map[string]interface{}',
            },
            {
                'title': 'Concurrent Counter',
                'description': 'Implement a thread-safe counter using sync.Mutex with Increment, Decrement, and Get methods',
            },
        ]

    # Run inference
    run_inference(
        args.checkpoint,
        test_problems,
        args.tokenizer,
        args.device,
        args.output,
        args.temperature,
    )


if __name__ == "__main__":
    main()
