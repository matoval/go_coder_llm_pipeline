"""
Go Syntax Validator

Uses Go's parser to validate generated code syntax.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, List, Optional
import re


class GoSyntaxValidator:
    """Validates Go code syntax using the Go compiler."""

    def __init__(self, go_binary: str = "go"):
        """
        Initialize validator.

        Args:
            go_binary: Path to go binary (default: "go" from PATH)
        """
        self.go_binary = go_binary
        self._check_go_available()

    def _check_go_available(self):
        """Check if Go is available on the system."""
        try:
            result = subprocess.run(
                [self.go_binary, "version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"Go binary not working: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                f"Go binary not found: {self.go_binary}. "
                "Please install Go or specify correct path."
            )

    def validate(self, code: str, wrap_in_package: bool = True) -> bool:
        """
        Check if code is syntactically valid Go.

        Args:
            code: Go code string to validate
            wrap_in_package: If True, wrap code in package main if not present

        Returns:
            True if syntax is valid, False otherwise
        """
        is_valid, _ = self.validate_with_errors(code, wrap_in_package)
        return is_valid

    def validate_with_errors(
        self,
        code: str,
        wrap_in_package: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Check syntax and return detailed error messages.

        Args:
            code: Go code string to validate
            wrap_in_package: If True, wrap code in package main if not present

        Returns:
            Tuple of (is_valid, error_messages)
        """
        # Prepare code
        if wrap_in_package and not code.strip().startswith("package "):
            full_code = f"package main\n\n{code}"
        else:
            full_code = code

        # Write to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.go',
            delete=False
        ) as f:
            f.write(full_code)
            temp_path = f.name

        try:
            # Try to format with gofmt (this checks syntax)
            result = subprocess.run(
                [self.go_binary, "fmt", temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Parse errors if any
            if result.returncode != 0:
                errors = self._parse_errors(result.stderr, temp_path)
                return False, errors

            return True, []

        except subprocess.TimeoutExpired:
            return False, ["Validation timeout"]

        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    def _parse_errors(self, stderr: str, temp_path: str) -> List[str]:
        """
        Parse Go compiler error messages.

        Args:
            stderr: Error output from Go compiler
            temp_path: Path to temporary file (to strip from errors)

        Returns:
            List of cleaned error messages
        """
        errors = []

        for line in stderr.split('\n'):
            if not line.strip():
                continue

            # Remove temp file path from error messages
            line = line.replace(temp_path + ":", "")

            # Extract meaningful error message
            # Format: "line:col: error message"
            match = re.search(r'(\d+:\d+:?\s*.+)', line)
            if match:
                errors.append(match.group(1).strip())
            elif "syntax error" in line.lower() or "error" in line.lower():
                errors.append(line.strip())

        return errors if errors else ["Syntax error (details unavailable)"]

    def validate_tokens(
        self,
        token_ids: List[int],
        tokenizer,
        wrap_in_package: bool = True
    ) -> bool:
        """
        Validate tokenized code.

        Args:
            token_ids: List of token IDs
            tokenizer: Tokenizer instance with decode() method
            wrap_in_package: If True, wrap code in package main if needed

        Returns:
            True if syntax is valid
        """
        code = tokenizer.decode(token_ids)
        return self.validate(code, wrap_in_package)

    def get_error_feedback(
        self,
        code: str,
        wrap_in_package: bool = True
    ) -> str:
        """
        Get formatted error feedback for training.

        Args:
            code: Go code string
            wrap_in_package: If True, wrap in package main if needed

        Returns:
            Formatted error string suitable for model feedback
        """
        is_valid, errors = self.validate_with_errors(code, wrap_in_package)

        if is_valid:
            return "<SYNTAX_OK> true"

        error_text = " | ".join(errors[:3])  # Limit to top 3 errors
        return f"<SYNTAX_ERR> {error_text}"

    def quick_check(self, code: str) -> bool:
        """
        Quick syntax check without detailed errors.

        Faster than full validation, useful during generation.

        Args:
            code: Go code string

        Returns:
            True if likely valid (quick heuristic check)
        """
        # Quick heuristic checks
        if not code.strip():
            return False

        # Check for balanced braces
        if code.count('{') != code.count('}'):
            return False

        # Check for balanced parentheses
        if code.count('(') != code.count(')'):
            return False

        # Check for balanced brackets
        if code.count('[') != code.count(']'):
            return False

        # If quick checks pass, do full validation
        return self.validate(code, wrap_in_package=True)


class ValidationCache:
    """Cache for validation results to avoid redundant checks."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached results
        """
        self.cache = {}
        self.max_size = max_size
        self.validator = GoSyntaxValidator()

    def validate(self, code: str) -> bool:
        """
        Validate code with caching.

        Args:
            code: Go code string

        Returns:
            True if syntax is valid
        """
        # Use hash as cache key
        cache_key = hash(code)

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Validate and cache result
        is_valid = self.validator.validate(code)

        # Maintain cache size
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            self.cache.pop(next(iter(self.cache)))

        self.cache[cache_key] = is_valid
        return is_valid

    def clear(self):
        """Clear the cache."""
        self.cache.clear()


if __name__ == "__main__":
    # Example usage
    validator = GoSyntaxValidator()

    # Test valid code
    valid_code = """
    func main() {
        fmt.Println("Hello, World!")
    }
    """

    print("Testing valid code:")
    is_valid, errors = validator.validate_with_errors(valid_code)
    print(f"  Valid: {is_valid}")
    if errors:
        print(f"  Errors: {errors}")
    print()

    # Test invalid code
    invalid_code = """
    func main() {
        fmt.Println("Missing closing brace"
    }
    """

    print("Testing invalid code:")
    is_valid, errors = validator.validate_with_errors(invalid_code)
    print(f"  Valid: {is_valid}")
    if errors:
        print(f"  Errors:")
        for error in errors:
            print(f"    - {error}")
    print()

    # Test error feedback format
    feedback = validator.get_error_feedback(invalid_code)
    print(f"Error feedback: {feedback}")
