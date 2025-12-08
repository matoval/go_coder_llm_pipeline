# Code Review - Issues Found

## Critical Bugs ⚠️

### 1. **Token ID Inconsistency in config.py** (CRITICAL)
**Location**: `model/config.py` lines 48-52 vs 121-158

**Issue**:
```python
# Lines 48-52
bos_token_id: int = 1  # <s>
eos_token_id: int = 2  # </s>
pad_token_id: int = 0  # <pad>

# Lines 121-158 - INCONSISTENT!
special_tokens: Dict[str, int] = {
    "<pad>": 0,     # ✓ Matches
    "<unk>": 1,     # ✗ Should be 0
    "<s>": 2,       # ✗ Should be 1 (BOS)
    "</s>": 3,      # ✗ Should be 2 (EOS)
    ...
}
```

**Impact**: Model will use wrong token IDs during training/inference
**Fix**: Align special_tokens dict with actual token IDs

### 2. **Attention Mask Handling Bug in planner.py**
**Location**: `model/modules/planner.py` lines 119-130

**Issue**:
```python
# Line 123: Converts mask to -inf values
attention_mask = (1.0 - attention_mask) * -1e9

# Line 129: But TransformerEncoder expects boolean mask!
src_key_padding_mask=attention_mask,  # ✗ Wrong format
```

**Impact**: Attention masking won't work correctly
**Fix**: Pass boolean mask or don't convert to -inf

## Code Quality Issues

### 3. **Unused Imports**
**Location**: `model/train.py`
- Line 10: `import os` - never used
- Line 17: `CosineAnnealingLR` - never used
- Line 30: `GoSyntaxValidator` - never used (imported but not implemented in training loop)

**Impact**: Code clutter, misleading
**Fix**: Remove unused imports or implement validation in training

### 4. **Missing Scheduler**
**Location**: `model/train.py` - HRMTrainer class

**Issue**: Imports `CosineAnnealingLR` but never creates or uses a learning rate scheduler

**Impact**: Training will use constant learning rate (less optimal)
**Fix**: Either use the scheduler or remove the import

### 5. **Naive Sequence Splitting in train.py**
**Location**: `model/train.py` lines 189-195

**Issue**:
```python
# Simple 1/3 splitting doesn't respect actual special tokens
problem_len = seq_len // 3
plan_len = seq_len // 3
```

**Impact**: May split in middle of tokens, breaking semantic boundaries
**Fix**: Use special tokens to properly split sequences

## Design Issues

### 6. **Refinement Loss Always Zero Target**
**Location**: `model/train.py` lines 217-223, 260-265

**Issue**:
```python
# Always assumes we want to continue (target = 0)
refinement_target = torch.zeros(
    refinement_logits.size(0),
    dtype=torch.long,
    device=self.device
)
```

**Impact**: Model will never learn proper refinement decisions
**Fix**: Calculate actual target based on validation feedback

### 7. **Validation Not Integrated in Training**
**Location**: `model/train.py`

**Issue**: `GoSyntaxValidator` imported but never used in training loop

**Impact**: Model doesn't receive validation feedback during training
**Fix**: Implement validation calls and incorporate into refinement loss

## Potential Issues

### 8. **Error Handling Missing**
**Location**: `scripts/process_data.py`, `tokenizer/plan_extractor.py`

**Issue**: Minimal error handling in data processing pipelines

**Impact**: Pipeline may crash on malformed data
**Fix**: Add try-catch blocks with better error messages

### 9. **No Gradient Accumulation**
**Location**: `model/train.py`

**Issue**: Config mentions gradient accumulation (line 97) but not implemented

**Impact**: Cannot train with larger effective batch sizes
**Fix**: Implement gradient accumulation or remove from config

### 10. **Missing EOS Token Handling in Generation**
**Location**: `model/modules/planner.py` line 180-182, `model/modules/generator.py` line 314

**Issue**: Generation loops don't check for EOS token to stop early

**Impact**: Always generates max_new_tokens even if EOS is generated
**Fix**: Add EOS detection and early stopping

## Documentation Issues

### 11. **Type Hints Incomplete**
**Location**: Various files

**Issue**: Some functions missing type hints (especially validators)

**Impact**: Harder to maintain, no static type checking
**Fix**: Add complete type annotations

### 12. **Docstring Inconsistencies**
**Location**: Various files

**Issue**: Some functions have detailed docstrings, others minimal

**Impact**: Inconsistent documentation quality
**Fix**: Standardize docstring format

## Testing Issues

### 13. **No Unit Tests**
**Location**: Repository root

**Issue**: Only end-to-end test, no unit tests for individual components

**Impact**: Hard to catch component-level bugs
**Fix**: Add pytest unit tests for each module

## Priority for Fixes

**MUST FIX (Breaks functionality):**
1. Token ID inconsistency (Critical)
2. Attention mask handling (Critical)
3. Refinement loss target calculation
4. Sequence splitting with special tokens

**SHOULD FIX (Reduces quality):**
5. Remove unused imports
6. Implement validation in training
7. Add EOS detection in generation
8. Better error handling

**NICE TO HAVE:**
9. Add gradient accumulation
10. Complete type hints
11. Add unit tests
12. Standardize documentation
