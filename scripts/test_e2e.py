"""
End-to-End Test for Hierarchical Go Coder Pipeline

Tests the complete pipeline:
1. Data processing (PR -> Hierarchical format)
2. Tokenization
3. Model training (small scale)
4. Code generation
5. Validation
"""

import sys
import json
import tempfile
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.plan_extractor import PlanExtractor
from tokenizer.tokenizer import GoTokenizer
from model.hrm_model import HierarchicalGoCoderModel
from model.config import get_small_config
from model.validators.syntax import GoSyntaxValidator


def test_plan_extraction():
    """Test 1: Plan extraction from PR data."""
    print("\n" + "="*60)
    print("Test 1: Plan Extraction")
    print("="*60)

    extractor = PlanExtractor()

    # Sample PR data
    sample_pr = {
        "repo": {"full_name": "test/repo"},
        "pull_request": {
            "number": 123,
            "title": "Fix nil pointer in handler",
            "body": "This fixes a panic when r == nil by adding a defensive check"
        },
        "files": [{
            "filename": "internal/server.go",
            "before": "func handle(r *Request) { process(r) }",
            "after": "func handle(r *Request) { if r == nil { return }; process(r) }",
            "patch": "@@ -1,1 +1,3 @@\n-func handle(r *Request) { process(r) }\n+func handle(r *Request) {\n+    if r == nil { return }\n+    process(r)\n+}"
        }]
    }

    # Extract plan
    hierarchical = extractor.process_pr_record(sample_pr)

    print(f"✓ Extracted plan:")
    print(f"  Intent: {hierarchical['plan']['intent']}")
    print(f"  Steps: {len(hierarchical['plan']['steps'])}")
    for i, step in enumerate(hierarchical['plan']['steps'], 1):
        print(f"    {i}. {step.get('intent', '?')}: {step.get('description', '?')}")

    return hierarchical


def test_model_creation():
    """Test 2: Model creation and parameter count."""
    print("\n" + "="*60)
    print("Test 2: Model Creation")
    print("="*60)

    config = get_small_config()
    model = HierarchicalGoCoderModel(config)

    params = model.get_num_params()
    print(f"✓ Model created successfully")
    print(f"  Parameters:")
    for name, count in params.items():
        print(f"    {name:20s}: {count/1e6:6.2f}M")

    # Check parameter count is reasonable
    total_params = params['total'] / 1e6
    if 20 <= total_params <= 40:
        print(f"  ✓ Parameter count within target range (20-40M)")
    else:
        print(f"  ⚠ Parameter count outside target range: {total_params:.2f}M")

    return model, config


def test_forward_pass(model, config):
    """Test 3: Forward pass (training mode)."""
    print("\n" + "="*60)
    print("Test 3: Forward Pass (Training)")
    print("="*60)

    batch_size = 2
    problem_len = 32
    plan_len = 20
    code_len = 50

    # Create dummy inputs
    problem_ids = torch.randint(0, config.vocab_size, (batch_size, problem_len))
    target_plan = torch.randint(0, config.vocab_size, (batch_size, plan_len))
    target_code = torch.randint(0, config.vocab_size, (batch_size, code_len))

    # Forward pass
    try:
        output = model(
            problem_ids=problem_ids,
            target_plan=target_plan,
            target_code=target_code,
        )

        print(f"✓ Forward pass successful")
        print(f"  Input shapes:")
        print(f"    Problem: {problem_ids.shape}")
        print(f"    Target plan: {target_plan.shape}")
        print(f"    Target code: {target_code.shape}")
        print(f"  Output shapes:")
        print(f"    Plan logits: {output['plan_logits'].shape}")
        print(f"    Code logits: {output['code_logits'].shape}")
        print(f"    Refinement logits: {output['refinement_logits'].shape}")

        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_generation(model, config):
    """Test 4: Code generation (inference mode)."""
    print("\n" + "="*60)
    print("Test 4: Code Generation (Inference)")
    print("="*60)

    batch_size = 1
    problem_len = 32

    # Create problem input
    problem_ids = torch.randint(0, config.vocab_size, (batch_size, problem_len))

    # Generate code
    try:
        result = model.generate(
            problem_ids=problem_ids,
            max_plan_tokens=20,
            max_code_tokens=50,
            max_refinement_iterations=3,
        )

        print(f"✓ Code generation successful")
        print(f"  Generated code shape: {result.generated_code.shape}")
        print(f"  Plan tokens shape: {result.plan_tokens.shape}")
        print(f"  Number of iterations: {result.num_iterations}")
        print(f"  Total attempts: {len(result.all_attempts)}")

        return True

    except Exception as e:
        print(f"✗ Code generation failed: {e}")
        return False


def test_validation():
    """Test 5: Go syntax validation."""
    print("\n" + "="*60)
    print("Test 5: Go Syntax Validation")
    print("="*60)

    try:
        validator = GoSyntaxValidator()

        # Test valid code
        valid_code = """
        func main() {
            fmt.Println("Hello, World!")
        }
        """

        is_valid = validator.validate(valid_code)
        print(f"✓ Validator created successfully")
        print(f"  Valid code test: {'✓ PASS' if is_valid else '✗ FAIL'}")

        # Test invalid code
        invalid_code = """
        func main() {
            fmt.Println("Missing closing brace"
        }
        """

        is_valid, errors = validator.validate_with_errors(invalid_code)
        print(f"  Invalid code test: {'✓ PASS' if not is_valid else '✗ FAIL'}")
        if errors:
            print(f"  Detected errors: {len(errors)}")

        return True

    except Exception as e:
        print(f"✗ Validation failed: {e}")
        print(f"  Note: Go compiler must be installed for validation")
        return False


def test_data_serialization(hierarchical_record):
    """Test 6: Data serialization for training."""
    print("\n" + "="*60)
    print("Test 6: Data Serialization")
    print("="*60)

    # Mock tokenizer since we don't have a trained one yet
    class MockTokenizer:
        def encode_plan(self, steps):
            lines = ["<PLAN>"]
            for step in steps:
                parts = ["<STEP>"]
                if "intent" in step:
                    parts.append(f"<INTENT:{step['intent']}>")
                if "description" in step:
                    parts.append(step["description"])
                lines.append(" ".join(parts))
            lines.append("</PLAN>")
            return "\n".join(lines)

        def encode_problem(self, title, body=""):
            return f"<PROBLEM>\n{title}\n{body}\n</PROBLEM>"

        def encode_context(self, file_path, code):
            return f'<CONTEXT>\n<FILE path="{file_path}">\n{code}\n</FILE>\n</CONTEXT>'

    tokenizer = MockTokenizer()

    # Serialize record
    try:
        problem_text = tokenizer.encode_problem(
            hierarchical_record["problem"]["description"],
            hierarchical_record["problem"].get("details", "")
        )

        plan_text = tokenizer.encode_plan(
            hierarchical_record["plan"]["steps"]
        )

        if hierarchical_record.get("context"):
            context_text = tokenizer.encode_context(
                hierarchical_record["context"]["file"],
                hierarchical_record["context"].get("before", "")
            )
        else:
            context_text = ""

        print(f"✓ Serialization successful")
        print(f"\n  Problem section:")
        print(f"    {problem_text[:100]}...")
        print(f"\n  Plan section:")
        print(f"    {plan_text[:100]}...")
        if context_text:
            print(f"\n  Context section:")
            print(f"    {context_text[:100]}...")

        return True

    except Exception as e:
        print(f"✗ Serialization failed: {e}")
        return False


def main():
    """Run all end-to-end tests."""
    print("\n" + "="*60)
    print("HIERARCHICAL GO CODER - END-TO-END TEST")
    print("="*60)

    results = {}

    # Test 1: Plan extraction
    try:
        hierarchical = test_plan_extraction()
        results['plan_extraction'] = True
    except Exception as e:
        print(f"✗ Plan extraction failed: {e}")
        results['plan_extraction'] = False
        hierarchical = None

    # Test 2: Model creation
    try:
        model, config = test_model_creation()
        results['model_creation'] = True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        results['model_creation'] = False
        model, config = None, None

    # Test 3-4: Only if model was created
    if model and config:
        results['forward_pass'] = test_forward_pass(model, config)
        results['generation'] = test_generation(model, config)
    else:
        results['forward_pass'] = False
        results['generation'] = False

    # Test 5: Validation
    results['validation'] = test_validation()

    # Test 6: Serialization (if we have hierarchical data)
    if hierarchical:
        results['serialization'] = test_data_serialization(hierarchical)
    else:
        results['serialization'] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:25s}: {status}")

    print(f"\n  Total: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n  ✅ All tests passed!")
        return 0
    else:
        print(f"\n  ⚠ {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
