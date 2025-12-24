#!/usr/bin/env python3
"""
Train SentencePiece tokenizer with HRM/TRM special tokens
"""

import sentencepiece as spm
import sys
from pathlib import Path

def train_tokenizer(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 50000
):
    """
    Train SentencePiece tokenizer.

    Args:
        input_file: Path to training corpus
        model_prefix: Output model prefix (without extension)
        vocab_size: Vocabulary size
    """
    # HRM/TRM special tokens
    user_defined_symbols = [
        "<REPO>",
        "<PR_TITLE>",
        "<PR_BODY>",
        "<FILE>",
        "<BEFORE>",
        "<AFTER>",
        "<COMMENTS>",
        "<CODE>",
        "<STRUCTURE>",
        "<PLAN>",
        "</PLAN>",
        "<STEP>",
        "<INTENT:FIX>",
        "<INTENT:ADD>",
        "<INTENT:REFACTOR>",
        "<INTENT:OPTIMIZE>",
        "<INTENT:UPDATE>",
        "<INTENT:REMOVE>",
        "<TARGET:func>",
        "<TARGET:type>",
        "<TARGET:interface>",
        "<TARGET:method>",
        "<TARGET:var>",
        "<VALIDATE>",
        "<SYNTAX_OK>",
        "<SYNTAX_ERR>",
        "<TEST_PASS>",
        "<TEST_FAIL>",
        "<REFINE>",
        "<CONTEXT>",
        "<PROBLEM>",
    ]

    print(f"Training SentencePiece tokenizer...")
    print(f"  Input: {input_file}")
    print(f"  Model: {model_prefix}.model")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Special tokens: {len(user_defined_symbols)}")
    print()

    # Check input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Training parameters
    train_args = {
        'input': input_file,
        'model_prefix': model_prefix,
        'vocab_size': vocab_size,
        'character_coverage': 1.0,
        'model_type': 'bpe',
        'unk_id': 0,
        'bos_id': 1,
        'eos_id': 2,
        'pad_id': 3,
        'user_defined_symbols': user_defined_symbols,
        'input_sentence_size': 10000000,
        'shuffle_input_sentence': True,
        'num_threads': 4,
    }

    # Train the model
    print("Training in progress...")
    spm.SentencePieceTrainer.train(**train_args)

    print()
    print("✓ Tokenizer training complete!")
    print(f"Output files:")
    print(f"  - {model_prefix}.model")
    print(f"  - {model_prefix}.vocab")

    # Verify outputs
    model_file = Path(f"{model_prefix}.model")
    vocab_file = Path(f"{model_prefix}.vocab")

    if model_file.exists() and vocab_file.exists():
        model_size = model_file.stat().st_size
        with open(vocab_file, 'r') as f:
            actual_vocab_size = sum(1 for _ in f)

        print()
        print(f"Model size: {model_size / 1024 / 1024:.2f} MB")
        print(f"Vocabulary entries: {actual_vocab_size}")
    else:
        print()
        print("Error: Output files were not created!")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument(
        "--input",
        default="../data/corpus/training_corpus.txt",
        help="Input corpus file"
    )
    parser.add_argument(
        "--model-prefix",
        default="go_coder_llm",
        help="Output model prefix"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50000,
        help="Vocabulary size"
    )

    args = parser.parse_args()

    train_tokenizer(
        args.input,
        args.model_prefix,
        args.vocab_size
    )
