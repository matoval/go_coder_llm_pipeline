"""
Data Processing Pipeline for Hierarchical Training Data

Processes raw GitHub PR data into hierarchical format suitable
for HRM/TRM training.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.plan_extractor import PlanExtractor
from tokenizer.tokenizer import GoTokenizer


class DataPipeline:
    """Process raw PR data into hierarchical training format."""

    def __init__(self, tokenizer_path: str = None):
        """
        Initialize pipeline.

        Args:
            tokenizer_path: Path to tokenizer model (optional, for tokenization)
        """
        self.plan_extractor = PlanExtractor()
        self.tokenizer = None

        if tokenizer_path and Path(tokenizer_path).exists():
            self.tokenizer = GoTokenizer(tokenizer_path)

    def serialize_record(self, record: Dict) -> str:
        """
        Serialize hierarchical record to text format for training.

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

    def process_to_hierarchical(
        self,
        input_path: str,
        output_path: str
    ) -> int:
        """
        Convert raw PR data to hierarchical format.

        Args:
            input_path: Path to raw JSONL file
            output_path: Path for hierarchical JSONL output

        Returns:
            Number of records processed
        """
        count = 0

        with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
            for line in fin:
                try:
                    record = json.loads(line)
                    hierarchical = self.plan_extractor.process_pr_record(record)
                    fout.write(json.dumps(hierarchical) + '\n')
                    count += 1

                    if count % 100 == 0:
                        print(f"Processed {count} records...")

                except Exception as e:
                    print(f"Error processing record: {e}")
                    continue

        return count

    def create_training_corpus(
        self,
        input_path: str,
        output_path: str
    ) -> int:
        """
        Create text corpus for tokenizer training.

        Args:
            input_path: Path to hierarchical JSONL file
            output_path: Path for text corpus output

        Returns:
            Number of records processed
        """
        count = 0

        with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
            for line in fin:
                try:
                    record = json.loads(line)
                    text = self.serialize_record(record)
                    fout.write(text + '\n\n')
                    count += 1

                    if count % 100 == 0:
                        print(f"Created corpus from {count} records...")

                except Exception as e:
                    print(f"Error creating corpus: {e}")
                    continue

        return count

    def tokenize_dataset(
        self,
        input_path: str,
        output_path: str,
        max_length: int = 1024
    ) -> int:
        """
        Tokenize hierarchical records for training.

        Args:
            input_path: Path to hierarchical JSONL file
            output_path: Path for tokenized JSONL output
            max_length: Maximum sequence length

        Returns:
            Number of records processed
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialized")

        count = 0

        with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
            for line in fin:
                try:
                    record = json.loads(line)
                    text = self.serialize_record(record)

                    # Tokenize
                    token_ids = self.tokenizer.encode(text)

                    # Truncate if needed
                    if len(token_ids) > max_length:
                        token_ids = token_ids[:max_length]

                    # Create training record
                    training_record = {
                        'input_ids': token_ids,
                        'attention_mask': [1] * len(token_ids),
                        'metadata': {
                            'repo': record['repo'],
                            'pr_number': record['pr_number']
                        }
                    }

                    fout.write(json.dumps(training_record) + '\n')
                    count += 1

                    if count % 100 == 0:
                        print(f"Tokenized {count} records...")

                except Exception as e:
                    print(f"Error tokenizing record: {e}")
                    continue

        return count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process PR data for hierarchical training"
    )
    parser.add_argument(
        "command",
        choices=["hierarchical", "corpus", "tokenize"],
        help="Processing command to run"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file or directory"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file or directory"
    )
    parser.add_argument(
        "--tokenizer",
        help="Path to tokenizer model (required for 'tokenize' command)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = DataPipeline(tokenizer_path=args.tokenizer)

    # Process based on command
    if args.command == "hierarchical":
        print(f"Converting to hierarchical format...")
        print(f"  Input: {args.input}")
        print(f"  Output: {args.output}")

        count = pipeline.process_to_hierarchical(args.input, args.output)
        print(f"\nProcessed {count} records")

    elif args.command == "corpus":
        print(f"Creating training corpus...")
        print(f"  Input: {args.input}")
        print(f"  Output: {args.output}")

        count = pipeline.create_training_corpus(args.input, args.output)
        print(f"\nCreated corpus from {count} records")

    elif args.command == "tokenize":
        if not args.tokenizer:
            print("Error: --tokenizer required for tokenize command")
            sys.exit(1)

        if not Path(args.tokenizer).exists():
            print(f"Error: Tokenizer not found at {args.tokenizer}")
            sys.exit(1)

        print(f"Tokenizing dataset...")
        print(f"  Input: {args.input}")
        print(f"  Output: {args.output}")
        print(f"  Tokenizer: {args.tokenizer}")
        print(f"  Max length: {args.max_length}")

        count = pipeline.tokenize_dataset(
            args.input,
            args.output,
            args.max_length
        )
        print(f"\nTokenized {count} records")


if __name__ == "__main__":
    main()
