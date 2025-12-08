"""
Create Synthetic Test Data for Quick Training Verification

Generates minimal fake data to test the training pipeline without
needing to collect real GitHub data.
"""

import json
import argparse
from pathlib import Path
import random


# Sample problem descriptions
PROBLEMS = [
    "Fix nil pointer dereference in handler",
    "Add input validation to user registration",
    "Refactor authentication middleware",
    "Optimize database query performance",
    "Update error handling in API endpoints",
    "Add logging to critical functions",
    "Remove deprecated function calls",
    "Fix race condition in concurrent handler",
    "Add unit tests for validator",
    "Refactor config loading logic",
]

# Sample Go code before fixes
CODE_BEFORE = [
    "func handle(r *Request) { process(r) }",
    "func register(user User) error { db.Save(user); return nil }",
    "func authenticate(token string) bool { return true }",
    "func getUsers() []User { return db.Query(\"SELECT * FROM users\") }",
    "func handleError(err error) { fmt.Println(err) }",
    "func process(data Data) { compute(data) }",
    "func oldFunction() { deprecated() }",
    "func concurrent() { go process(); process() }",
    "func validate(input string) bool { return true }",
    "func loadConfig() Config { return Config{} }",
]

# Sample Go code after fixes
CODE_AFTER = [
    "func handle(r *Request) { if r == nil { return }; process(r) }",
    "func register(user User) error { if user.Email == \"\" { return errors.New(\"invalid email\") }; db.Save(user); return nil }",
    "func authenticate(token string) bool { if token == \"\" { return false }; return validateToken(token) }",
    "func getUsers() []User { return db.Where(\"active = ?\", true).Find(&users) }",
    "func handleError(err error) { if err != nil { log.Error(err) } }",
    "func process(data Data) { log.Info(\"processing\"); compute(data) }",
    "func newFunction() { modernImplementation() }",
    "func concurrent() { var wg sync.WaitGroup; wg.Add(1); go func() { defer wg.Done(); process() }(); wg.Wait() }",
    "func validate(input string) bool { if len(input) == 0 { return false }; return regex.Match(input) }",
    "func loadConfig() Config { cfg, _ := os.ReadFile(\"config.json\"); return parseConfig(cfg) }",
]

# Sample intents
INTENTS = ["FIX", "ADD", "REFACTOR", "OPTIMIZE", "UPDATE"]

# Sample targets
TARGETS = [
    "func:handle",
    "func:register",
    "func:authenticate",
    "func:getUsers",
    "func:handleError",
    "type:Config",
    "method:Process",
]


def generate_sample(idx: int) -> dict:
    """Generate a single synthetic training sample."""

    # Pick random components
    problem = PROBLEMS[idx % len(PROBLEMS)]
    before = CODE_BEFORE[idx % len(CODE_BEFORE)]
    after = CODE_AFTER[idx % len(CODE_AFTER)]
    intent = INTENTS[idx % len(INTENTS)]
    target = TARGETS[idx % len(TARGETS)]

    # Create structured format
    return {
        "repo": f"test/repo-{idx % 5}",
        "pr_number": 100 + idx,
        "problem": {
            "description": problem,
            "details": f"This PR addresses the issue in the codebase. Sample {idx}."
        },
        "context": {
            "file": f"internal/handler_{idx % 3}.go",
            "before": before
        },
        "plan": {
            "steps": [
                {
                    "intent": intent,
                    "target": target,
                    "description": f"{intent} the issue in {target}"
                }
            ],
            "intent": intent,
            "targets": [target]
        },
        "solution": {
            "code": after,
            "validation": {
                "syntax": True,
                "tests": True
            }
        }
    }


def serialize_to_training_format(record: dict) -> str:
    """Convert record to text format for tokenization."""
    lines = []

    # Problem
    lines.append("<PROBLEM>")
    lines.append(record["problem"]["description"])
    if record["problem"].get("details"):
        lines.append(record["problem"]["details"])
    lines.append("</PROBLEM>")
    lines.append("")

    # Context
    lines.append("<CONTEXT>")
    lines.append(f'<FILE path="{record["context"]["file"]}">')
    lines.append(record["context"]["before"])
    lines.append("</FILE>")
    lines.append("</CONTEXT>")
    lines.append("")

    # Plan
    lines.append("<PLAN>")
    for step in record["plan"]["steps"]:
        parts = ["<STEP>"]
        if "intent" in step:
            parts.append(f"<INTENT:{step['intent']}>")
        if "target" in step:
            parts.append(f"<TARGET:{step['target'].split(':')[0]}> {step['target'].split(':')[1]}")
        if "description" in step:
            parts.append(step["description"])
        lines.append(" ".join(parts))
    lines.append("</PLAN>")
    lines.append("")

    # Code
    lines.append("<CODE>")
    lines.append(record["solution"]["code"])
    lines.append("</CODE>")
    lines.append("")

    # Validation
    lines.append("<VALIDATE>")
    lines.append(f"<SYNTAX_OK> {str(record['solution']['validation']['syntax']).lower()}")
    lines.append(f"<TEST_PASS> {str(record['solution']['validation']['tests']).lower()}")
    lines.append("</VALIDATE>")

    return "\n".join(lines)


def create_tokenized_record(record: dict, max_length: int = 256) -> dict:
    """Create a mock tokenized record."""

    # Serialize to text
    text = serialize_to_training_format(record)

    # Mock tokenization: just convert characters to pseudo-token-ids
    # In reality, this would use SentencePiece
    token_ids = [hash(char) % 50000 for char in text][:max_length]

    # Pad to max_length if needed
    if len(token_ids) < max_length:
        token_ids.extend([0] * (max_length - len(token_ids)))  # Pad with 0

    attention_mask = [1 if i < len(text) else 0 for i in range(max_length)]

    return {
        'input_ids': token_ids,
        'attention_mask': attention_mask,
        'metadata': {
            'repo': record['repo'],
            'pr_number': record['pr_number']
        }
    }


def main():
    """Generate synthetic test data."""
    parser = argparse.ArgumentParser(description="Create synthetic test data")
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--output-dir', type=str, default='data/test',
                       help='Output directory')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                       help='Train/val split ratio')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_samples} synthetic samples...")

    # Generate samples
    samples = [generate_sample(i) for i in range(args.num_samples)]

    # Split train/val
    split_idx = int(args.num_samples * args.split_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")

    # Save hierarchical format
    with open(output_dir / "hierarchical.jsonl", 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"✓ Saved hierarchical format: {output_dir / 'hierarchical.jsonl'}")

    # Save tokenized format (train)
    with open(output_dir / "train.jsonl", 'w') as f:
        for sample in train_samples:
            tokenized = create_tokenized_record(sample)
            f.write(json.dumps(tokenized) + '\n')

    print(f"✓ Saved tokenized train: {output_dir / 'train.jsonl'}")

    # Save tokenized format (val)
    with open(output_dir / "val.jsonl", 'w') as f:
        for sample in val_samples:
            tokenized = create_tokenized_record(sample)
            f.write(json.dumps(tokenized) + '\n')

    print(f"✓ Saved tokenized val: {output_dir / 'val.jsonl'}")

    # Save text corpus (for tokenizer training)
    with open(output_dir / "corpus.txt", 'w') as f:
        for sample in samples:
            text = serialize_to_training_format(sample)
            f.write(text + '\n\n')

    print(f"✓ Saved text corpus: {output_dir / 'corpus.txt'}")

    print(f"\n✅ Synthetic test data created!")
    print(f"\nQuick start:")
    print(f"  python model/train.py \\")
    print(f"    --config tiny \\")
    print(f"    --train-data {output_dir / 'train.jsonl'} \\")
    print(f"    --val-data {output_dir / 'val.jsonl'} \\")
    print(f"    --batch-size 4 \\")
    print(f"    --epochs 1 \\")
    print(f"    --output-dir checkpoints/test")


if __name__ == "__main__":
    main()
