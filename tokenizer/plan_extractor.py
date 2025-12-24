"""
Plan Extractor for Hierarchical Training Data

Extracts high-level plans from PR information to create
hierarchical training data for the HRM/TRM model.
"""

import json
import re
from typing import List, Dict, Optional
from pathlib import Path
import argparse


class PlanExtractor:
    """Extract structured plans from PR data."""

    # Intent classification patterns
    INTENT_PATTERNS = {
        "FIX": [
            r"\bfix(es|ed|ing)?\b",
            r"\bbug\b",
            r"\berror\b",
            r"\bissue\b",
            r"\bpanic\b",
            r"\bcrash\b",
            r"\bnil pointer\b",
        ],
        "ADD": [
            r"\badd(s|ed|ing)?\b",
            r"\bnew\b",
            r"\bimplement(s|ed|ing)?\b",
            r"\bcreate(s|d)?\b",
            r"\bintroduce(s|d)?\b",
        ],
        "REFACTOR": [
            r"\brefactor(s|ed|ing)?\b",
            r"\brestructure(s|d)?\b",
            r"\breorganize(s|d)?\b",
            r"\bsimplify\b",
            r"\bclean\s*up\b",
        ],
        "OPTIMIZE": [
            r"\boptimiz(e|es|ed|ing)\b",
            r"\bperformance\b",
            r"\bfaster\b",
            r"\bimprove(s|d)?\b",
            r"\befficiency\b",
        ],
        "UPDATE": [
            r"\bupdate(s|d)?\b",
            r"\bmodify\b",
            r"\bchange(s|d)?\b",
            r"\badjust(s|ed)?\b",
        ],
        "REMOVE": [
            r"\bremove(s|d)?\b",
            r"\bdelete(s|d)?\b",
            r"\bdrop(s|ped)?\b",
            r"\beliminate(s|d)?\b",
        ],
    }

    # Target identification patterns
    TARGET_PATTERNS = {
        "func": r"\bfunc(?:tion)?\s+(\w+)\s*\(",
        "type": r"\btype\s+(\w+)\s+struct",
        "interface": r"\btype\s+(\w+)\s+interface",
        "method": r"\bfunc\s+\([^)]+\)\s+(\w+)\s*\(",
        "var": r"\bvar\s+(\w+)\s",
    }

    # Action verb extraction
    ACTION_PATTERNS = [
        r"(add|implement|create)\s+([^.\n]+)",
        r"(fix|resolve)\s+([^.\n]+)",
        r"(refactor|restructure)\s+([^.\n]+)",
        r"(optimize|improve)\s+([^.\n]+)",
        r"(update|modify|change)\s+([^.\n]+)",
        r"(remove|delete)\s+([^.\n]+)",
    ]

    def classify_intent(self, title: str, body: str) -> str:
        """
        Classify the primary intent of a PR.

        Args:
            title: PR title
            body: PR description

        Returns:
            Intent classification (FIX, ADD, REFACTOR, etc.)
        """
        text = (title + " " + (body or "")).lower()

        # Count matches for each intent
        scores = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = sum(
                len(re.findall(pattern, text, re.IGNORECASE))
                for pattern in patterns
            )
            scores[intent] = score

        # Return intent with highest score, default to UPDATE
        if max(scores.values()) == 0:
            return "UPDATE"

        return max(scores, key=scores.get)

    def identify_targets(self, file_changes: List[Dict]) -> List[tuple]:
        """
        Identify target entities (functions, types, etc.) from code changes.

        Args:
            file_changes: List of file change dictionaries

        Returns:
            List of (target_type, target_name) tuples
        """
        targets = []

        for file_change in file_changes:
            # Look in both before and after code
            code = file_change.get("patch", "") or ""
            code += file_change.get("after", "") or ""

            # Find all targets
            for target_type, pattern in self.TARGET_PATTERNS.items():
                matches = re.findall(pattern, code)
                for match in matches:
                    targets.append((target_type, match))

        # Remove duplicates while preserving order
        seen = set()
        unique_targets = []
        for target in targets:
            if target not in seen:
                seen.add(target)
                unique_targets.append(target)

        return unique_targets

    def extract_actions(self, body: str) -> List[str]:
        """
        Extract action descriptions from PR body.

        Args:
            body: PR description text

        Returns:
            List of action descriptions
        """
        if not body:
            return []

        actions = []

        for pattern in self.ACTION_PATTERNS:
            matches = re.findall(pattern, body, re.IGNORECASE)
            for verb, description in matches:
                # Clean up description
                description = description.strip()
                if len(description) > 5:  # Skip very short descriptions
                    actions.append(f"{verb} {description}")

        return actions[:5]  # Limit to top 5 actions

    def extract_plan(
        self,
        title: str,
        body: str,
        file_changes: List[Dict]
    ) -> Dict:
        """
        Extract high-level plan from PR information.

        Args:
            title: PR title
            body: PR description
            file_changes: List of file changes

        Returns:
            Structured plan dictionary
        """
        # Classify overall intent
        intent = self.classify_intent(title, body)

        # Identify target entities
        targets = self.identify_targets(file_changes)

        # Extract action steps
        actions = self.extract_actions(body or "")

        # Construct plan steps
        steps = []

        # If we have specific targets and actions, combine them
        if targets and actions:
            for i, action in enumerate(actions):
                step = {
                    "intent": intent,
                    "description": action
                }

                # Associate first action with first target, etc.
                if i < len(targets):
                    target_type, target_name = targets[i]
                    step["target"] = f"{target_type}:{target_name}"

                steps.append(step)

        # If we have targets but no actions
        elif targets:
            for target_type, target_name in targets:
                steps.append({
                    "intent": intent,
                    "target": f"{target_type}:{target_name}",
                    "description": f"{intent.lower()} {target_name}"
                })

        # If we have actions but no targets
        elif actions:
            for action in actions:
                steps.append({
                    "intent": intent,
                    "description": action
                })

        # Fallback: use title as single step
        else:
            steps.append({
                "intent": intent,
                "description": title
            })

        return {
            "steps": steps,
            "intent": intent,
            "targets": [f"{t[0]}:{t[1]}" for t in targets]
        }

    def process_pr_record(self, record: Dict) -> Dict:
        """
        Process a PR record into hierarchical format.

        Args:
            record: Raw PR record

        Returns:
            Hierarchical annotated record
        """
        # Extract plan
        plan = self.extract_plan(
            record["pull_request"]["title"],
            record["pull_request"].get("body", ""),
            record.get("files") or []
        )

        # Get first file for context (simplified)
        context = None
        solution = None

        if record.get("files"):
            first_file = record["files"][0]
            context = {
                "file": first_file["filename"],
                "before": first_file.get("before", "")
            }
            solution = {
                "code": first_file.get("after", ""),
                "validation": {
                    "syntax": True,  # Assume valid (will be checked during training)
                    "tests": True    # Assume valid
                }
            }

        return {
            "repo": record["repo"]["full_name"],
            "pr_number": record["pull_request"]["number"],
            "problem": {
                "description": record["pull_request"]["title"],
                "details": record["pull_request"].get("body", "")
            },
            "context": context,
            "plan": plan,
            "solution": solution
        }

    def process_file(self, input_path: str, output_path: str):
        """
        Process a JSONL file of PR records.

        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file
        """
        processed_count = 0
        error_count = 0

        with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
            for line_num, line in enumerate(fin, 1):
                try:
                    record = json.loads(line)
                    hierarchical_record = self.process_pr_record(record)
                    fout.write(json.dumps(hierarchical_record) + '\n')
                    processed_count += 1

                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} records...")

                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    error_count += 1

        print(f"\nComplete!")
        print(f"  Processed: {processed_count}")
        print(f"  Errors: {error_count}")


def main():
    """Main entry point for plan extraction."""
    parser = argparse.ArgumentParser(
        description="Extract hierarchical plans from PR data"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file or directory"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file or directory"
    )

    args = parser.parse_args()

    extractor = PlanExtractor()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Handle directory input
    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

        for input_file in input_path.glob("*.jsonl"):
            print(f"\nProcessing {input_file.name}...")
            output_file = output_path / input_file.name
            extractor.process_file(str(input_file), str(output_file))

    # Handle single file input
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        extractor.process_file(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
