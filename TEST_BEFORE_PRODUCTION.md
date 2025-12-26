# Test Before Production - Validation Guide

This guide walks you through validating that all improvements are working correctly BEFORE starting the month-long data collection.

## Why Test First?

A month-long collection represents:
- **Time**: 30+ days of continuous running
- **API calls**: ~1-2 million GitHub API requests
- **Storage**: ~100-150GB of data
- **Money**: Potential GitHub API costs if using paid tier

If something is broken, you want to know in 30 minutes, not 30 days.

## Step-by-Step Testing Process

### Step 1: Run Small Test Collection (5-10 minutes)

```bash
# Set your GitHub token
export GITHUB_TOKEN="ghp_your_token_here"

# Create a test repository config (pick a high-quality Go repo)
echo '["kubernetes/kubernetes"]' > config/repos_test.json

# Run test collection - only 50 PRs
go run cmd/fetchdata/main.go \
  --max-prs=50 \
  --test=true
```

**Expected output:**
```
Fetching PRs from kubernetes/kubernetes...
  Found 150 PRs, applying quality filters...
  Progress: 10/150 PRs processed, 8 accepted, 2 filtered (rate limit: 4950)
  Progress: 20/150 PRs processed, 15 accepted, 5 filtered (rate limit: 4920)
  ...
  Completed: 50 PRs fetched, 100 filtered
  Filter breakdown:
    - not merged: 45
    - bot author: 30
    - file count (25): 15
    - additions (3500): 10
  Saved 50 PRs to data/raw/kubernetes_kubernetes.jsonl
```

**What to look for:**
- ✅ No errors or panics
- ✅ Filter breakdown shows filtering is working
- ✅ Final count is close to --max-prs (may be less if not enough quality PRs)
- ✅ Data file created in data/raw/

**Red flags:**
- ❌ Errors about missing fields
- ❌ All PRs filtered out
- ❌ Very slow (should process ~40 PRs/minute)
- ❌ No data file created

---

### Step 2: Run Automated Validation (2 minutes)

```bash
# Make the script executable
chmod +x scripts/validate_improvements.sh

# Run validation
./scripts/validate_improvements.sh
```

**Expected output:**
```
==============================================
Data Collection Improvements Validation
==============================================

1. Testing Full File Content Fetching
==============================================
✓ PASS: content_before populated: 142/156 files (91%)
✓ PASS: content_after populated: 150/156 files (96%)
✓ PASS: Average content_before size: 2845 bytes (reasonable)
✓ PASS: Average content_after size: 2912 bytes (reasonable)

2. Testing Go Syntax Validation
==============================================
✓ PASS: valid_syntax field present in all files
✓ PASS: Syntax validation: 148 valid, 8 invalid (95% valid)

3. Testing CI/CD Status Collection
==============================================
✓ PASS: ci_status field present in all PRs
✓ PASS: CI status known for 67% of PRs (34 success, 0 failure)

4. Testing Quality Filters
==============================================
✓ PASS: Merged filter working: 100% merged (50/50)
✓ PASS: Bot filter working: 0% bots (0/50)
✓ PASS: Average files per PR: 3 (within 1-20 range)
✓ PASS: Average additions per PR: 287 (within 10-2000 range)

...

==============================================
VALIDATION SUMMARY
==============================================
PASSED: 24
WARNINGS: 2
FAILED: 0

✓✓✓ ALL CHECKS PASSED ✓✓✓
```

**What to look for:**
- ✅ Most checks PASS (>20 passes)
- ✅ Few or no FAILURES (0-2 failures)
- ✅ Warnings are explainable

**Acceptable warnings:**
- ⚠️ "CI status only known for 30%" - Older PRs don't have CI data, this is normal
- ⚠️ "content_before only in 70% of files" - New files don't have "before", this is normal

**Unacceptable failures:**
- ❌ "No files have content_before populated" - Something is broken
- ❌ "valid_syntax field missing" - Code didn't compile correctly
- ❌ "base_sha/head_sha missing" - Data structure is wrong

---

### Step 3: Manual Spot Checks (5 minutes)

Even if automated tests pass, do some manual verification:

#### 3a. Inspect a Sample PR

```bash
# Look at the first PR in detail
head -1 data/raw/*.jsonl | jq . | less
```

**Check for:**
1. Does `pull_request.merged` = true?
2. Does `pull_request.base_sha` look like a git SHA (40 hex chars)?
3. Does `pull_request.ci_status` have a value (not null)?
4. Does `pull_request.is_bot` = false?
5. Do files have `content_before` with actual Go code?
6. Do files have `content_after` with actual Go code?
7. Does `files[0].valid_syntax` = true or false (not null)?

#### 3b. Verify Code Content is Real

```bash
# Extract and view some "before" code
jq -r '.files[0].content_before' data/raw/*.jsonl | head -1 | head -20
```

**Should look like:**
```go
package main

import (
    "fmt"
    "log"
)

func main() {
    // Real Go code here
    ...
}
```

**Should NOT look like:**
- Empty
- Just `@@ -1,3 +1,4 @@` (that's a diff, not content)
- Binary garbage
- JSON

#### 3c. Check Validation Fields

```bash
# Show validation data from a sample
jq -r '{
  syntax: .files[0].valid_syntax,
  ci: .pull_request.ci_status,
  merged: .pull_request.merged
}' data/raw/*.jsonl | head -5
```

**Expected:**
```json
{
  "syntax": true,
  "ci": "success",
  "merged": true
}
{
  "syntax": true,
  "ci": "unknown",
  "merged": true
}
```

**NOT expected:**
```json
{
  "syntax": null,
  "ci": null,
  "merged": null
}
```

---

### Step 4: Test Training Pipeline Integration (10 minutes)

The data is only useful if the training pipeline can consume it.

#### 4a. Run Plan Extraction

```bash
# Extract hierarchical plans from test data
python tokenizer/plan_extractor.py \
  --input data/raw/kubernetes_kubernetes.jsonl \
  --output data/test_hierarchical.jsonl
```

**Expected output:**
```
Processing PR records...
Processed 50 records...

Complete!
  Processed: 50
  Errors: 0
```

**Check the output:**
```bash
# View a hierarchical record
head -1 data/test_hierarchical.jsonl | jq . | less
```

**Should have:**
```json
{
  "repo": "kubernetes/kubernetes",
  "pr_number": 12345,
  "problem": {
    "description": "Fix memory leak in controller",
    "details": "..."
  },
  "context": {
    "file": "pkg/controller/manager.go",
    "before": "package controller\n\nfunc (m *Manager) Run() {\n..."
  },
  "plan": {
    "steps": [...],
    "intent": "FIX",
    "targets": ["func:Run"]
  },
  "solution": {
    "code": "package controller\n\nimport \"sync\"\n\nfunc (m *Manager) Run() {\n...",
    "validation": {
      "syntax": true,
      "tests": true
    }
  }
}
```

**Critical checks:**
- ✅ `context.before` has actual Go code (not empty, not diff)
- ✅ `solution.code` has actual Go code
- ✅ `solution.validation.syntax` is boolean (true/false), not null
- ✅ `solution.validation.tests` is boolean, not null

#### 4b. Test Tokenization

```bash
# Prepare training data
python scripts/prepare_training_data.py \
  --input_dir data \
  --output_dir data/test_tokenized \
  --tokenizer_path tokenizer/go_coder_llm.model \
  --max_length 1024 \
  --val_split 0.2
```

**Expected output:**
```
Loading hierarchical data from data/test_hierarchical.jsonl...
Loaded 50 samples
Tokenizing samples...
Processed 50 samples...

Splitting into train/val...
  Train: 40 samples
  Val: 10 samples

Saved to data/test_tokenized/
```

**Check tokenized output:**
```bash
# View tokenized data
head -1 data/test_tokenized/train.jsonl | jq . | less
```

**Should have:**
```json
{
  "input_ids": [1, 256, 789, ...],
  "text": "<PROBLEM>\nFix memory leak...\n<PLAN>\n<INTENT:FIX>\n...<VALIDATE>\n<SYNTAX_OK> true\n<TEST_PASS> true\n</VALIDATE>"
}
```

**Critical checks:**
- ✅ `text` contains `<SYNTAX_OK>` token
- ✅ `text` contains `<TEST_PASS>` token
- ✅ Values after tokens are "true" or "false", not just missing
- ✅ `input_ids` is array of integers

#### 4c. Quick Training Test (Optional, 15 minutes)

If you want to be absolutely sure, run a tiny training session:

```bash
# Train for just 10 steps to verify no errors
python model/train.py \
  --train_data_path data/test_tokenized/train.jsonl \
  --val_data_path data/test_tokenized/val.jsonl \
  --output_dir test_checkpoint \
  --num_epochs 1 \
  --batch_size 2 \
  --eval_every 5 \
  --save_every 1000
```

**Expected output:**
```
Loading model configuration...
Loading datasets...
  Train: 40 samples
  Val: 10 samples

Training...
Epoch 1/1:
  Step 5/20: loss=8.234, plan_loss=2.451, code_loss=4.892, refinement_loss=0.891
  Step 10/20: loss=7.892, plan_loss=2.234, code_loss=4.678, refinement_loss=0.980
  ...
```

**Critical checks:**
- ✅ No errors about missing fields
- ✅ Refinement loss is computed (not 0.0 or NaN)
- ✅ All three losses decrease (even slightly)
- ✅ No CUDA out of memory errors (if using GPU)

---

### Step 5: Verify Quality Metrics

Run this analysis to understand your test data quality:

```bash
# Create analysis script
cat > scripts/analyze_test_data.sh << 'EOF'
#!/bin/bash
FILE=data/raw/*.jsonl

echo "=== Data Quality Analysis ==="
echo ""

echo "Total PRs: $(cat $FILE | wc -l)"
echo "Total files: $(jq -r '.files | length' $FILE | awk '{sum+=$1} END {print sum}')"
echo ""

echo "Merged PRs: $(jq -r 'select(.pull_request.merged == true)' $FILE | wc -l)"
echo "Bot PRs: $(jq -r 'select(.pull_request.is_bot == true)' $FILE | wc -l)"
echo ""

echo "Files with valid syntax: $(jq -r '.files[] | select(.valid_syntax == true)' $FILE | wc -l)"
echo "Files with invalid syntax: $(jq -r '.files[] | select(.valid_syntax == false)' $FILE | wc -l)"
echo ""

echo "Files with content_before: $(jq -r '.files[] | select(.content_before != "")' $FILE | wc -l)"
echo "Files with content_after: $(jq -r '.files[] | select(.content_after != "")' $FILE | wc -l)"
echo ""

echo "CI Status:"
jq -r '.pull_request.ci_status' $FILE | sort | uniq -c
echo ""

echo "Size distribution (total additions per PR):"
jq -r '[.files[].additions] | add' $FILE | awk '{
  if ($1 < 50) small++;
  else if ($1 < 200) medium++;
  else if ($1 < 1000) large++;
  else huge++;
}
END {
  print "  Small (<50): " small
  print "  Medium (50-200): " medium
  print "  Large (200-1000): " large
  print "  Huge (>1000): " huge
}'
EOF

chmod +x scripts/analyze_test_data.sh
./scripts/analyze_test_data.sh
```

**Target metrics:**
- ✅ 95-100% merged PRs
- ✅ <5% bot PRs
- ✅ >80% files with valid syntax
- ✅ >70% files with content_before
- ✅ >80% files with content_after
- ✅ CI status known for >30% (more is better)

---

## Decision Matrix

Based on your test results:

### ✅ GREEN LIGHT - Proceed with Full Collection

**Criteria:**
- Automated validation shows 0 failures
- Manual spot checks look good
- Plan extraction works without errors
- Quality metrics meet targets
- Training test runs without errors

**Action:**
```bash
# You're good to go!
go run cmd/fetchdata/main.go --max-prs=1500
```

---

### ⚠️ YELLOW LIGHT - Investigate Warnings

**Criteria:**
- Automated validation shows 1-3 warnings
- Quality metrics slightly below targets
- Some fields missing but not critical

**Common warnings and fixes:**

| Warning | Likely Cause | Action |
|---------|--------------|--------|
| CI status mostly unknown | Old PRs or repo doesn't use GitHub Actions | Acceptable if <2 years old filter applied |
| content_before only 60% | Many new files in PRs | Acceptable if content_after >80% |
| Some invalid syntax | Test files, generated code | Acceptable if >80% valid |

**Action:**
- Review warnings carefully
- If explainable and acceptable, proceed
- If unclear, collect 100 more PRs and re-test

---

### ❌ RED LIGHT - Do NOT Proceed

**Criteria:**
- Automated validation shows >3 failures
- No content_before/content_after in files
- Missing critical fields (base_sha, valid_syntax)
- Plan extraction fails
- Training test errors out

**Action:**
1. Review error messages
2. Check GitHub token permissions
3. Verify code compiled: `go build ./cmd/fetchdata/...`
4. Re-read implementation files
5. Ask for help if stuck

**DO NOT start month-long collection until issues are fixed!**

---

## Quick Checklist

Before starting production collection, check:

- [ ] Test collection (50 PRs) completed successfully
- [ ] `./scripts/validate_improvements.sh` shows mostly PASS
- [ ] Manual spot check shows real Go code in content_before/after
- [ ] Plan extraction produces valid hierarchical format
- [ ] Quality metrics meet or exceed targets
- [ ] Training test runs without errors (optional but recommended)
- [ ] Disk space available (~150GB for full collection)
- [ ] GitHub token has adequate rate limit (check: `curl -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/rate_limit`)

---

## Time Investment

- Test collection: 5-10 minutes
- Automated validation: 2 minutes
- Manual spot checks: 5 minutes
- Pipeline integration test: 10-15 minutes
- **Total: 25-30 minutes**

This 30 minutes of testing can save you weeks of wasted effort!

---

## What If Tests Fail?

### Common Issues

**Issue: "No files have content_before"**
- **Cause**: `fetchFileContent` not being called
- **Fix**: Verify `fetchPRFilesWithContent` is used, not old `fetchPRFiles`
- **Check**: `internal/github/fetch.go` line 197 should call `fetchPRFilesWithContent`

**Issue: "valid_syntax field missing"**
- **Cause**: Types not updated or code not compiled
- **Fix**: Run `go build ./internal/github/...` and check for errors
- **Check**: `internal/types/types.go` should have `ValidSyntax bool` field

**Issue: "Plan extraction fails"**
- **Cause**: Field names don't match
- **Fix**: Verify `tokenizer/plan_extractor.py` uses `content_before` not `before`
- **Check**: Line 268 should be `first_file.get("content_before", "")`

**Issue: "All PRs filtered out"**
- **Cause**: Filters too strict or repo has unusual PRs
- **Fix**: Try different test repo (kubernetes, prometheus, etcd)
- **Alternative**: Temporarily relax filters for testing

---

## Success Stories

When tests pass, you should see:

```
✓✓✓ ALL CHECKS PASSED ✓✓✓

Your data collection improvements are working correctly!
You can proceed with the full data collection.
```

At this point, you can confidently invest in the month-long collection knowing:
- Data structure is correct
- Content is being fetched properly
- Validation signals are present
- Training pipeline can consume it
- Quality filters are working

Happy collecting! 🚀
