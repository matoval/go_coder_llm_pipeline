"""
Go Coder LLM Tokenizer

Wrapper around SentencePiece tokenizer with support for
hierarchical reasoning special tokens.
"""

import sentencepiece as spm
import json
from pathlib import Path
from typing import List, Union


class GoTokenizer:
    """Tokenizer for Go code with hierarchical reasoning support."""

    def __init__(self, model_path: str):
        """
        Initialize tokenizer.

        Args:
            model_path: Path to SentencePiece model file
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        # Load special tokens metadata
        tokenizer_dir = Path(model_path).parent
        special_tokens_path = tokenizer_dir / "special_tokens.json"

        if special_tokens_path.exists():
            with open(special_tokens_path, 'r') as f:
                self.special_tokens_metadata = json.load(f)
        else:
            self.special_tokens_metadata = {}

    def encode(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """
        Convert text to token IDs.

        Args:
            text: Single string or list of strings to encode

        Returns:
            List of token IDs or list of lists if input is a list
        """
        if isinstance(text, list):
            return [self.sp.encode_as_ids(t) for t in text]
        return self.sp.encode_as_ids(text)

    def decode(self, ids: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        Convert token IDs to text.

        Args:
            ids: List of token IDs or list of lists

        Returns:
            Decoded string or list of strings
        """
        if isinstance(ids[0], list):
            return [self.sp.decode_ids(id_list) for id_list in ids]
        return self.sp.decode_ids(ids)

    def tokenize(self, text: str) -> List[str]:
        """
        Get token pieces (for debugging/analysis).

        Args:
            text: String to tokenize

        Returns:
            List of token strings
        """
        return self.sp.encode_as_pieces(text)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        """Padding token ID."""
        return self.sp.pad_id()

    @property
    def eos_id(self) -> int:
        """End-of-sequence token ID."""
        return self.sp.eos_id()

    @property
    def bos_id(self) -> int:
        """Begin-of-sequence token ID."""
        return self.sp.bos_id()

    @property
    def unk_id(self) -> int:
        """Unknown token ID."""
        return self.sp.unk_id()

    def get_special_token_id(self, token: str) -> int:
        """
        Get ID for a specific special token.

        Args:
            token: Special token string (e.g., "<PLAN>")

        Returns:
            Token ID
        """
        return self.sp.piece_to_id(token)

    def is_special_token(self, token_id: int) -> bool:
        """
        Check if a token ID corresponds to a special token.

        Args:
            token_id: Token ID to check

        Returns:
            True if token is a special token
        """
        piece = self.sp.id_to_piece(token_id)
        return piece.startswith('<') and piece.endswith('>')

    def get_token_string(self, token_id: int) -> str:
        """
        Get string representation of a token ID.

        Args:
            token_id: Token ID

        Returns:
            Token string
        """
        return self.sp.id_to_piece(token_id)

    def encode_plan(self, steps: List[dict]) -> str:
        """
        Encode a structured plan into text format.

        Args:
            steps: List of plan steps, each with 'intent', 'target', 'description'

        Returns:
            Formatted plan string ready for tokenization

        Example:
            >>> steps = [
            ...     {"intent": "FIX", "target": "func:handle", "description": "Add nil check"},
            ...     {"intent": "PRESERVE", "description": "Keep existing logic"}
            ... ]
            >>> tokenizer.encode_plan(steps)
            '<PLAN>\\n<STEP> <INTENT:FIX> <TARGET:func> Add nil check\\n<STEP> <INTENT:PRESERVE> Keep existing logic\\n</PLAN>'
        """
        lines = ["<PLAN>"]

        for step in steps:
            parts = ["<STEP>"]

            if "intent" in step:
                parts.append(f"<INTENT:{step['intent']}>")

            if "target" in step:
                # Parse target like "func:handle" into "<TARGET:func> handle"
                if ":" in step["target"]:
                    target_type, target_name = step["target"].split(":", 1)
                    parts.append(f"<TARGET:{target_type}> {target_name}")
                else:
                    parts.append(step["target"])

            if "description" in step:
                parts.append(step["description"])

            lines.append(" ".join(parts))

        lines.append("</PLAN>")
        return "\n".join(lines)

    def encode_problem(self, title: str, body: str = "") -> str:
        """
        Encode a problem description.

        Args:
            title: Problem title
            body: Optional problem description

        Returns:
            Formatted problem string
        """
        lines = ["<PROBLEM>", title]
        if body:
            lines.append(body)
        lines.append("</PROBLEM>")
        return "\n".join(lines)

    def encode_context(self, file_path: str, code: str) -> str:
        """
        Encode code context.

        Args:
            file_path: Path to the file
            code: Code content

        Returns:
            Formatted context string
        """
        return f'<CONTEXT>\n<FILE path="{file_path}">\n{code}\n</FILE>\n</CONTEXT>'

    def encode_validation(self, syntax_ok: bool, tests_pass: bool = None) -> str:
        """
        Encode validation results.

        Args:
            syntax_ok: Whether syntax is valid
            tests_pass: Optional test results

        Returns:
            Formatted validation string
        """
        lines = ["<VALIDATE>"]
        lines.append(f"<SYNTAX_OK> {str(syntax_ok).lower()}")
        if tests_pass is not None:
            lines.append(f"<TEST_PASS> {str(tests_pass).lower()}")
        lines.append("</VALIDATE>")
        return "\n".join(lines)


def create_tokenizer(model_path: str = "tokenizer/go_coder_llm.model") -> GoTokenizer:
    """
    Create and return a tokenizer instance.

    Args:
        model_path: Path to SentencePiece model

    Returns:
        GoTokenizer instance
    """
    return GoTokenizer(model_path)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tokenizer.py <model_path>")
        print("\nExample:")
        print("  python tokenizer.py tokenizer/go_coder_llm.model")
        sys.exit(1)

    tokenizer = GoTokenizer(sys.argv[1])

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"PAD token: {tokenizer.pad_id}")
    print(f"BOS token: {tokenizer.bos_id}")
    print(f"EOS token: {tokenizer.eos_id}")
    print(f"UNK token: {tokenizer.unk_id}")
    print()

    # Test special tokens
    print("Special token IDs:")
    special_tokens = [
        "<PLAN>", "</PLAN>", "<STEP>",
        "<INTENT:FIX>", "<INTENT:ADD>",
        "<TARGET:func>", "<TARGET:type>",
        "<VALIDATE>", "<SYNTAX_OK>"
    ]

    for token in special_tokens:
        token_id = tokenizer.get_special_token_id(token)
        print(f"  {token}: {token_id}")

    print()

    # Test encoding
    code = "func main() {\n    fmt.Println(\"Hello\")\n}"
    tokens = tokenizer.tokenize(code)
    ids = tokenizer.encode(code)
    decoded = tokenizer.decode(ids)

    print("Test encoding:")
    print(f"  Code: {code}")
    print(f"  Tokens: {tokens}")
    print(f"  Token IDs: {ids}")
    print(f"  Decoded: {decoded}")
    print()

    # Test plan encoding
    steps = [
        {"intent": "FIX", "target": "func:handle", "description": "Add nil check"},
        {"intent": "PRESERVE", "description": "Keep existing logic"}
    ]
    plan_text = tokenizer.encode_plan(steps)
    print("Plan encoding:")
    print(plan_text)
