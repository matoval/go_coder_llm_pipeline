#!/usr/bin/env python3
"""
Prepare training data by tokenizing hierarchical files and creating train/val split.
"""

import json
import sys
import random
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenizer.tokenizer import GoTokenizer


def serialize_record(record: Dict) -> str:
    """
    Serialize hierarchical record to text format for tokenization.

    Args:
        record: Hierarchical PR record

    Returns:
        Formatted text string
    """
    lines = []

    # Problem section
    lines.append("<PROBLEM>")
    lines.append(record["problem"]["description"])
    if record["problem"].get("details"):
        lines.append(record["problem"]["details"])
    lines.append("</PROBLEM>")
    lines.append("")

    # Context section
    if record.get("context"):
        lines.append("<CONTEXT>")
        lines.append(f'<FILE path="{record["context"]["file"]}">')
        if record["context"].get("before"):
            lines.append(record["context"]["before"])
        lines.append("</FILE>")
        lines.append("</CONTEXT>")
        lines.append("")

    # Plan section
    if record.get("plan"):
        plan = record["plan"]
        lines.append("<PLAN>")

        for step in plan["steps"]:
            parts = ["<STEP>"]

            if "intent" in step:
                parts.append(f"<INTENT:{step['intent']}>")

            if "target" in step:
                # Parse target like "func:handle"
                if ":" in step["target"]:
                    target_type, target_name = step["target"].split(":", 1)
                    parts.append(f"<TARGET:{target_type}> {target_name}")
                else:
                    parts.append(step["target"])

            if "description" in step:
                parts.append(step["description"])

            lines.append(" ".join(parts))

        lines.append("</PLAN>")
        lines.append("")

    # Code solution
    if record.get("solution"):
        lines.append("<CODE>")
        if record["solution"].get("code"):
            lines.append(record["solution"]["code"])
        lines.append("</CODE>")
        lines.append("")

        # Validation (if available)
        if record["solution"].get("validation"):
            val = record["solution"]["validation"]
            lines.append("<VALIDATE>")
            lines.append(f"<SYNTAX_OK> {str(val.get('syntax', True)).lower()}")
            lines.append(f"<TEST_PASS> {str(val.get('tests', True)).lower()}")
            lines.append("</VALIDATE>")

    return "\n".join(lines)


def tokenize_and_split(
    input_dir: str,
    output_dir: str,
    tokenizer_path: str,
    max_length: int = 512,
    val_split: float = 0.1,
    seed: int = 42
):
    """
    Tokenize hierarchical files and split into train/val sets.

    Args:
        input_dir: Directory with hierarchical JSONL files
        output_dir: Output directory for tokenized data
        tokenizer_path: Path to trained tokenizer model
        max_length: Maximum sequence length
        val_split: Fraction of data for validation
        seed: Random seed for reproducibility
    """
    print(f"Preparing training data...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Tokenizer: {tokenizer_path}")
    print(f"  Max length: {max_length}")
    print(f"  Val split: {val_split:.1%}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GoTokenizer(tokenizer_path)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print()

    # Find all hierarchical files
    input_path = Path(input_dir)
    hierarchical_files = sorted(input_path.glob("*_hierarchical.jsonl"))

    if not hierarchical_files:
        print(f"Error: No hierarchical files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(hierarchical_files)} hierarchical files")
    print()

    # Collect all records
    print("Loading records...")
    all_records = []
    total_loaded = 0
    total_errors = 0

    for file in hierarchical_files:
        with open(file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    all_records.append(record)
                    total_loaded += 1

                    if total_loaded % 10000 == 0:
                        print(f"  Loaded {total_loaded} records...")

                except Exception as e:
                    total_errors += 1

    print(f"  Total records: {total_loaded}")
    print(f"  Errors: {total_errors}")
    print()

    # Shuffle and split
    print("Splitting into train/val sets...")
    random.seed(seed)
    random.shuffle(all_records)

    val_size = int(len(all_records) * val_split)
    train_records = all_records[val_size:]
    val_records = all_records[:val_size]

    print(f"  Train: {len(train_records)} records")
    print(f"  Val: {len(val_records)} records")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Tokenize and save train set
    print("Tokenizing train set...")
    train_file = output_path / "train.jsonl"
    tokenize_dataset(train_records, train_file, tokenizer, max_length)

    # Tokenize and save val set
    print("Tokenizing validation set...")
    val_file = output_path / "val.jsonl"
    tokenize_dataset(val_records, val_file, tokenizer, max_length)

    print()
    print("✓ Training data preparation complete!")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")


def tokenize_dataset(
    records: List[Dict],
    output_file: Path,
    tokenizer: GoTokenizer,
    max_length: int
):
    """
    Tokenize records and save to file.

    Args:
        records: List of hierarchical records
        output_file: Output file path
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
    """
    count = 0

    with open(output_file, 'w') as fout:
        for record in records:
            try:
                # Serialize to text
                text = serialize_record(record)

                # Tokenize
                token_ids = tokenizer.encode(text)

                # Truncate if needed
                if len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]

                # Create training record
                training_record = {
                    'input_ids': token_ids,
                    'attention_mask': [1] * len(token_ids),
                    'metadata': {
                        'repo': record.get('repo', ''),
                        'pr_number': record.get('pr_number', 0)
                    }
                }

                fout.write(json.dumps(training_record) + '\n')
                count += 1

                if count % 1000 == 0:
                    print(f"  Tokenized {count} records...")

            except Exception as e:
                print(f"  Error tokenizing record: {e}")
                continue

    print(f"  ✓ Saved {count} records to {output_file.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument(
        "--input",
        default="data/processed",
        help="Input directory with hierarchical JSONL files"
    )
    parser.add_argument(
        "--output",
        default="data/tokenized",
        help="Output directory for tokenized data"
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer/go_coder_llm.model",
        help="Path to tokenizer model"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (0.0-1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    tokenize_and_split(
        args.input,
        args.output,
        args.tokenizer,
        args.max_length,
        args.val_split,
        args.seed
    )
