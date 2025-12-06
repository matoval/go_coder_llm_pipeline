# HRM/TRM Integration Design

Detailed design for integrating Hierarchical Reasoning Modules (HRM) and Tree-based Recursive Modeling (TRM) into the Go Coder LLM Pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Research Foundation](#research-foundation)
3. [Architecture Design](#architecture-design)
4. [Data Transformation](#data-transformation)
5. [Training Methodology](#training-methodology)
6. [Implementation Details](#implementation-details)
7. [Evaluation Strategy](#evaluation-strategy)
8. [Migration Path](#migration-path)

---

## Overview

### Motivation

Traditional autoregressive language models (like GPT-2) treat code generation as a flat sequence prediction task. This approach has several limitations:

- **No explicit planning**: Model generates tokens without high-level strategy
- **No self-correction**: Errors compound without refinement
- **Parameter inefficient**: Needs massive models (100M-1B+ params) for good results
- **Black box**: Can't inspect reasoning process

### Solution: Hierarchical Recursive Architecture

We integrate insights from two recent papers:

1. **HRM** (arXiv:2506.21734): Two-module architecture with hierarchical reasoning
2. **TRM** (arXiv:2510.04871): Recursive refinement with tiny models

**Key Innovation**: Separate **planning** from **implementation** and add **recursive refinement**.

### Benefits

| Aspect | Traditional GPT-2 | HRM/TRM Approach | Improvement |
|--------|------------------|------------------|-------------|
| Model size | 125M params | 20-30M params | **6x smaller** |
| Training time | 5-7 days | 2-3 days | **3x faster** |
| VRAM usage | 10-11 GB | 3-4 GB | **3x less** |
| Accuracy | Baseline | Higher | **Self-correction** |
| Interpretability | Black box | Inspectable plans | **Transparent** |

---

## Research Foundation

### HRM: Hierarchical Reasoning Modules

**Paper**: [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)

**Core Concept**: Use two interdependent modules operating at different abstraction levels:

- **High-Level Module**: Strategic planning, slow reasoning
- **Low-Level Module**: Detailed execution, fast computation

**Key Results**:
- 27M parameters achieved strong performance on complex tasks
- No pre-training or Chain-of-Thought data required
- Outperformed larger models on ARC benchmark

**Application to Code**:
- High-level: Plan code structure, identify targets (functions/types)
- Low-level: Generate actual Go syntax and implementation

### TRM: Tiny Recursive Model

**Paper**: [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)

**Core Concept**: Single tiny network (7M params) with recursive refinement

**Key Results**:
- 45% accuracy on ARC-AGI-1 with only 7M parameters
- Outperformed LLMs 1000x larger through iteration
- Recursive improvement critical to success

**Application to Code**:
- Iterative refinement of code solutions
- Validation-driven recursion (syntax checks, tests)
- Learn when to continue vs when solution is complete

### Why This Combination Works for Go Code

1. **Natural fit**: Coding IS hierarchical (design → implement → refine)
2. **Validation available**: Can check syntax/tests during training
3. **Structured data**: PRs contain both plans (descriptions) and implementations (diffs)
4. **Efficiency matters**: Smaller models = faster iteration for developers

---

## Architecture Design

### System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│               Hierarchical Recursive Go Coder                │
└─────────────────────────────────────────────────────────────┘

                        INPUT
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
    Problem                             Context
  (PR title/desc)                   (Existing code)
        │                                   │
        └──────────┬───────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   Planning Module    │  ← High-level reasoning
        │   (10-12M params)    │
        │   3 layers, 256 dim  │
        └──────────┬───────────┘
                   │
                   │ Plan representation
                   │ (intents, targets, steps)
                   │
                   ▼
        ┌──────────────────────┐
        │  Generation Module   │  ← Low-level execution
        │   (10-12M params)    │
        │   3 layers, 256 dim  │
        │   + Cross-attention  │
        └──────────┬───────────┘
                   │
                   │ Generated code
                   │
                   ▼
        ┌──────────────────────┐
        │   Validation Layer   │  ← Syntax checker
        │   (Go parser)        │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Refinement Controller│  ← Decision: continue/refine/done
        │    (classifier)      │
        └──────────┬───────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
     Refine                 Done
        │                     │
        │                     ▼
        └────────────►    OUTPUT
     (loop back)         (Final code)
```

### Module Specifications

#### 1. Planning Module (High-Level)

```python
PlannerModule:
  - Input: Problem description + Context
  - Architecture: Transformer Encoder
  - Size: 3 layers, 256 dim, 8 heads
  - Parameters: ~10-12M
  - Output: Plan embedding (512 tokens max)

  Outputs:
    - Intent tokens: <INTENT:FIX>, <INTENT:ADD>, etc.
    - Target tokens: <TARGET:func>, <TARGET:type>, etc.
    - Step tokens: <STEP> descriptions
```

**Example Planning Output**:
```
<PLAN>
<STEP> <INTENT:FIX> Identify nil pointer vulnerability
<STEP> <TARGET:func> handle() in internal/server.go
<STEP> <INTENT:ADD> Add defensive nil check
<STEP> Preserve existing process() logic
</PLAN>
```

#### 2. Generation Module (Low-Level)

```python
GeneratorModule:
  - Input: Plan embedding + Context
  - Architecture: Transformer Decoder with Cross-Attention
  - Size: 3 layers, 256 dim, 8 heads
  - Parameters: ~10-12M
  - Output: Go code tokens (1024 tokens max)

  Cross-Attention:
    - Generator attends to Plan representations
    - Ensures implementation follows plan
```

**Example Generation Output**:
```go
func handle(r *Request) {
    if r == nil {
        return
    }
    process(r)
}
```

#### 3. Refinement Controller

```python
RefinementController:
  - Input: Plan embedding + Validation result
  - Architecture: Linear classifier
  - Output: 3-way decision

  Decisions:
    0: Continue (syntax error, try again)
    1: Refine (partial solution, improve plan)
    2: Done (solution correct)
```

### Recursive Loop

```python
def forward(problem, context, max_iterations=5):
    for iteration in range(max_iterations):
        # 1. High-level planning
        plan = planner(problem, context)

        # 2. Low-level generation (attends to plan)
        code = generator(plan, context)

        # 3. Validation
        syntax_ok = validate_syntax(code)
        tests_pass = run_basic_tests(code)

        # 4. Refinement decision
        decision = refinement_controller(
            plan,
            syntax_ok,
            tests_pass
        )

        if decision == DONE:
            break
        elif decision == REFINE:
            # Update context with partial solution
            context = context + code

    return code, plan, iteration
```

---

## Data Transformation

### Current PR Format

```json
{
  "repo": "owner/project",
  "pr_number": 123,
  "title": "Fix nil pointer in handler",
  "body": "This fixes a panic when r == nil",
  "files_changed": [
    {
      "path": "internal/server.go",
      "before": "func handle(r *Request) { process(r) }",
      "after": "func handle(r *Request) { if r == nil { return }; process(r) }"
    }
  ]
}
```

### Hierarchical Annotated Format

```json
{
  "repo": "owner/project",
  "pr_number": 123,

  "problem": {
    "description": "Fix nil pointer in handler",
    "details": "This fixes a panic when r == nil"
  },

  "context": {
    "file": "internal/server.go",
    "before": "func handle(r *Request) { process(r) }"
  },

  "plan": {
    "steps": [
      {
        "intent": "FIX",
        "target": "func:handle",
        "description": "Identify nil pointer vulnerability"
      },
      {
        "intent": "ADD",
        "target": "func:handle",
        "description": "Add defensive nil check at entry"
      },
      {
        "intent": "PRESERVE",
        "description": "Keep existing process() call"
      }
    ]
  },

  "solution": {
    "code": "func handle(r *Request) { if r == nil { return }; process(r) }",
    "validation": {
      "syntax": true,
      "tests": true
    }
  }
}
```

### Plan Extraction Algorithm

```python
# tokenizer/plan_extractor.py

def extract_plan(pr_title, pr_body, file_changes):
    """Extract high-level plan from PR information."""

    plan_steps = []

    # 1. Classify PR intent
    intent = classify_intent(pr_title, pr_body)
    # Returns: FIX, ADD, REFACTOR, OPTIMIZE, etc.

    # 2. Identify target entities
    targets = identify_targets(file_changes)
    # Returns: [(type, name), ...] e.g., [("func", "handle")]

    # 3. Extract action steps from PR body
    actions = extract_actions(pr_body)
    # Uses NLP to identify action verbs: "add check", "fix panic", etc.

    # 4. Construct structured plan
    for target_type, target_name in targets:
        plan_steps.append({
            "intent": intent,
            "target": f"{target_type}:{target_name}",
            "description": actions[0] if actions else f"{intent} {target_name}"
        })

    return {
        "steps": plan_steps,
        "intent": intent,
        "targets": targets
    }
```

### Serialization for Training

```
<PROBLEM>
Fix nil pointer in handler
This fixes a panic when r == nil
</PROBLEM>

<CONTEXT>
<FILE path="internal/server.go">
func handle(r *Request) {
    process(r)
}
</FILE>
</CONTEXT>

<PLAN>
<STEP> <INTENT:FIX> <TARGET:func:handle> Identify nil pointer vulnerability
<STEP> <INTENT:ADD> <TARGET:func:handle> Add defensive nil check at entry
<STEP> <INTENT:PRESERVE> Keep existing process() call
</PLAN>

<CODE>
func handle(r *Request) {
    if r == nil {
        return
    }
    process(r)
}
</CODE>

<VALIDATE>
<SYNTAX_OK> true
<TEST_PASS> true
</VALIDATE>
```

---

## Training Methodology

### Phase 1: Joint Hierarchical Training

**Objective**: Train both modules simultaneously to align planning and generation.

**Data**: 5000+ repos with hierarchically annotated PRs

**Multi-task Loss**:
```python
loss_total = (
    0.4 * loss_planning +      # Plan token prediction
    0.4 * loss_generation +    # Code token prediction
    0.2 * loss_refinement      # Refinement decision
)
```

**Planning Loss**:
```python
loss_planning = CrossEntropyLoss(
    predicted_plan_tokens,
    target_plan_tokens
)
# Measures: Can model predict correct intents/targets/steps?
```

**Generation Loss**:
```python
loss_generation = CrossEntropyLoss(
    predicted_code_tokens,
    target_code_tokens
)
# Measures: Can model generate correct Go code?
```

**Refinement Loss**:
```python
loss_refinement = CrossEntropyLoss(
    refinement_decision,
    expected_decision  # Based on validation
)
# Measures: Does model know when to stop/refine?
```

**Training Loop**:
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        problem, context, plan_target, code_target, validation = batch

        # Forward pass
        plan_pred = planner(problem, context)
        code_pred = generator(plan_pred, context)

        # Compute losses
        loss_plan = criterion(plan_pred, plan_target)
        loss_code = criterion(code_pred, code_target)

        # Simulate refinement decision
        should_refine = not validation["syntax"]
        loss_refine = refinement_loss(plan_pred, should_refine)

        # Combined loss
        loss = 0.4*loss_plan + 0.4*loss_code + 0.2*loss_refine

        # Backward
        loss.backward()
        optimizer_planner.step()
        optimizer_generator.step()
```

### Phase 2: Recursive Refinement Training

**Objective**: Teach model to iteratively improve solutions.

**Data Augmentation**:
1. Introduce deliberate syntax errors
2. Create partial solutions (incomplete implementations)
3. Add incorrect but valid Go code

**Training with Rollouts**:
```python
for batch in dataloader:
    problem, context, target_solution = batch

    # Initialize
    current_code = None
    refinement_steps = []

    # Rollout (up to 5 iterations)
    for iteration in range(5):
        # Generate plan
        plan = planner(problem, context, current_code)

        # Generate code
        code = generator(plan, context)

        # Validate
        syntax_ok = syntax_validator(code)
        tests_pass = test_runner(code, problem)

        # Refinement decision
        decision = refinement_controller(plan, syntax_ok, tests_pass)

        # Record step
        refinement_steps.append({
            "plan": plan,
            "code": code,
            "decision": decision
        })

        if decision == DONE or syntax_ok and tests_pass:
            break

        current_code = code  # Use as context for next iteration

    # Compute loss over entire refinement trajectory
    loss = trajectory_loss(refinement_steps, target_solution)
    loss.backward()
```

### Phase 3: Curriculum Learning

**Stage 1: Simple Patterns** (500M tokens)
- Basic error handling
- Nil checks
- Simple refactors

**Stage 2: Complex Changes** (2B tokens)
- Multi-function changes
- Interface modifications
- Concurrency patterns

**Stage 3: PR-Specific** (1.5B tokens)
- Full PR contexts
- Multi-file changes
- Integration with comments

---

## Implementation Details

### File Structure

```
model/
├── config.py                  # HRMConfig class
├── hrm_model.py              # Main HierarchicalGoCoderModel
├── modules/
│   ├── __init__.py
│   ├── planner.py            # PlannerModule
│   ├── generator.py          # GeneratorModule
│   └── refinement.py         # RefinementController
├── validators/
│   ├── __init__.py
│   ├── syntax.py             # Go syntax validator
│   └── tests.py              # Basic test runner
├── train.py                  # Training script
└── utils.py                  # Helper functions

tokenizer/
├── plan_extractor.py         # Extract plans from PRs
├── train_tokenizer.sh        # Train SentencePiece
└── special_tokens.json       # HRM-specific tokens

data/
├── raw/                      # Original GitHub data
├── processed/                # Standard format
└── hierarchical/             # HRM-annotated format
```

### Key Implementation: Cross-Attention

```python
# model/modules/generator.py

class GeneratorModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Standard transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config)
            for _ in range(config.n_layer)
        ])

        # Cross-attention to plan
        self.cross_attention = nn.ModuleList([
            MultiHeadAttention(config.n_embd, config.n_head)
            for _ in range(config.n_layer)
        ])

    def forward(self, plan_embedding, context):
        x = self.embed(context)

        for layer, cross_attn in zip(self.layers, self.cross_attention):
            # Self-attention on generated code
            x = layer(x)

            # Cross-attention to plan
            x = cross_attn(
                query=x,
                key=plan_embedding,
                value=plan_embedding
            )

        return self.lm_head(x)
```

### Validation Integration

```python
# model/validators/syntax.py

import go/parser  # Use Go's parser via subprocess

class GoSyntaxValidator:
    def validate(self, code_tokens):
        """Check if generated code is syntactically valid Go."""

        # Decode tokens to string
        code_str = tokenizer.decode(code_tokens)

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".go") as f:
            f.write(code_str.encode())
            f.flush()

            # Run go parser
            result = subprocess.run(
                ["go", "fmt", f.name],
                capture_output=True
            )

            return result.returncode == 0

    def get_errors(self, code_tokens):
        """Get specific syntax errors for training signal."""
        # Similar to above, but parse error messages
        # Return structured error info
        pass
```

---

## Evaluation Strategy

### Metrics

1. **Code Generation Accuracy**
   - Exact match with target code
   - AST similarity (structural match)
   - Syntax correctness rate

2. **Planning Quality**
   - Intent classification accuracy
   - Target identification F1 score
   - Plan-solution alignment

3. **Refinement Efficiency**
   - Average iterations to solution
   - Syntax error reduction per iteration
   - Premature stopping rate

4. **Comparison Baseline**
   - vs GPT-2 125M (standard approach)
   - vs GPT-2 30M (size-matched)
   - vs Code generation models (if available)

### Test Sets

1. **Held-out PRs**: 5% of repos for validation
2. **Synthetic tasks**: Generated Go problems
3. **Real-world scenarios**: Common bug patterns

### Benchmarks

```python
# scripts/evaluate.py

def evaluate_model(model, test_set):
    results = {
        "syntax_correctness": [],
        "exact_match": [],
        "iterations_needed": [],
        "plan_accuracy": [],
    }

    for example in test_set:
        # Generate solution
        code, plan, iters = model(
            example["problem"],
            example["context"]
        )

        # Evaluate
        results["syntax_correctness"].append(
            validate_syntax(code)
        )
        results["exact_match"].append(
            code == example["solution"]
        )
        results["iterations_needed"].append(iters)
        results["plan_accuracy"].append(
            plan_matches(plan, example["expected_plan"])
        )

    return aggregate_metrics(results)
```

---

## Migration Path

### From Current Pipeline to HRM

**Current State**:
- Data collection: ✅ Working
- Tokenizer: ✅ SentencePiece trained
- Model: ❌ Not implemented (placeholder)
- Training: ❌ Not implemented

**Migration Steps**:

#### Step 1: Data Annotation (1-2 days)
```bash
# Run plan extractor on existing data
python tokenizer/plan_extractor.py \
  --input data/processed/*.jsonl \
  --output data/hierarchical/
```

#### Step 2: Extended Tokenizer (1 day)
```bash
# Retrain tokenizer with hierarchical tokens
./tokenizer/train_tokenizer.sh --hierarchical
```

#### Step 3: Implement HRM Model (2-3 days)
```bash
# Implement core modules
# - model/config.py
# - model/hrm_model.py
# - model/modules/*.py
```

#### Step 4: Implement Validators (1 day)
```bash
# Implement syntax/test validators
# - model/validators/*.py
```

#### Step 5: Training Script (2 days)
```bash
# Implement hierarchical training loop
# - model/train.py
```

#### Step 6: Initial Training Run (3-4 days)
```bash
# Train on subset (500M tokens) to verify
python model/train.py --debug --tokens 500M
```

#### Step 7: Full Training (2-3 days)
```bash
# Full training on 4B tokens
python model/train.py --tokens 4B
```

**Total Timeline**: ~2-3 weeks development + 2-3 days training

### Backward Compatibility

**Option 1: Keep both approaches**
- Maintain GPT-2 config for comparison
- Add `--model-type hrm` flag to training script

**Option 2: Full migration**
- Replace model/ completely with HRM
- Update all docs to reflect new approach

**Recommendation**: Option 2 - HRM is strictly better for this use case.

---

## Appendix

### Special Tokens Reference

```python
HIERARCHICAL_TOKENS = {
    # Planning structure
    "<PLAN>": "Start of planning section",
    "</PLAN>": "End of planning section",
    "<STEP>": "Individual planning step",

    # Intents
    "<INTENT:FIX>": "Bug fix intent",
    "<INTENT:ADD>": "Feature addition intent",
    "<INTENT:REFACTOR>": "Code refactoring intent",
    "<INTENT:OPTIMIZE>": "Performance optimization",
    "<INTENT:UPDATE>": "Update existing code",
    "<INTENT:REMOVE>": "Remove code",

    # Targets
    "<TARGET:func>": "Target is a function",
    "<TARGET:type>": "Target is a type/struct",
    "<TARGET:interface>": "Target is an interface",
    "<TARGET:method>": "Target is a method",
    "<TARGET:var>": "Target is a variable",

    # Validation
    "<VALIDATE>": "Start validation section",
    "<SYNTAX_OK>": "Syntax is correct",
    "<SYNTAX_ERR>": "Syntax error present",
    "<TEST_PASS>": "Tests pass",
    "<TEST_FAIL>": "Tests fail",
    "<REFINE>": "Needs refinement",

    # Context
    "<CONTEXT>": "Code context section",
    "<PROBLEM>": "Problem description",
    "<CODE>": "Code implementation section",
}
```

### Research Paper References

- **HRM**: Hierarchical Reasoning Modules
  - Paper: https://arxiv.org/abs/2506.21734
  - Key insight: Separate strategic and tactical reasoning
  - Result: 27M params competitive with larger models

- **TRM**: Tiny Recursive Model
  - Paper: https://arxiv.org/abs/2510.04871
  - Key insight: Recursion > scale for reasoning
  - Result: 7M params outperforms LLMs 1000x larger

### Go Syntax Validation

```go
// validators/gosyntax/validator.go
package gosyntax

import (
    "go/parser"
    "go/token"
)

func ValidateCode(code string) (bool, []string) {
    fset := token.NewFileSet()
    _, err := parser.ParseFile(fset, "temp.go", code, 0)

    if err != nil {
        return false, parseErrors(err)
    }

    return true, nil
}
```

---

## Conclusion

The HRM/TRM integration transforms this project from a standard LLM approach to a novel hierarchical recursive reasoning system. Key advantages:

- **6x smaller model** (20-30M vs 125M)
- **3x faster training** (2-3 days vs 5-7 days)
- **Better accuracy** through recursive refinement
- **Interpretable** planning process
- **Validation-aware** architecture

This approach is particularly well-suited for code generation because:
1. Coding is naturally hierarchical
2. Validation is available during training
3. PR data contains both plans and implementations
4. Efficiency enables rapid iteration

Next steps: Implement core modules and begin initial training runs.
