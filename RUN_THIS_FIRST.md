# ⚡ RUN THIS FIRST - Validation Instructions

## TL;DR - One Command to Test Everything

```bash
# Set your GitHub token
export GITHUB_TOKEN="ghp_your_token_here"

# Run complete validation (takes ~10-15 minutes)
./scripts/run_full_test.sh
```

This will:
1. ✅ Compile the Go code
2. ✅ Collect 50 test PRs from kubernetes/kubernetes
3. ✅ Run 24+ automated validation checks
4. ✅ Test plan extraction pipeline
5. ✅ Verify data structure is correct
6. ✅ Show you sample data
7. ✅ Give you a clear GO/NO-GO decision

---

## Expected Output (Success)

```
╔════════════════════════════════════════════════════════╗
║  Full Validation Test - Data Collection Improvements  ║
╚════════════════════════════════════════════════════════╝

✓ All prerequisites met

Step 1/6: Compiling Go code...
✓ Go code compiled successfully

Step 2/6: Running test data collection (50 PRs)...
✓ Collected 50 PRs

Step 3/6: Running automated validation tests...
✓ Automated validation passed

Step 4/6: Testing plan extraction pipeline...
✓ Plan extraction successful (50 records)

Step 5/6: Verifying hierarchical data structure...
  ✓ context.before exists (2845 bytes)
  ✓ solution.code exists (2912 bytes)
  ✓ validation.syntax = true (boolean)
  ✓ validation.tests = true (boolean)
✓ All hierarchical structure checks passed (4/4)

Step 6/6: Sample Data Inspection
[Shows you actual collected data]

╔════════════════════════════════════════════════════════╗
║                  ✓✓✓ ALL TESTS PASSED ✓✓✓             ║
╚════════════════════════════════════════════════════════╝

Your data collection improvements are working correctly!

You can now proceed with confidence to:
  go run cmd/fetchdata/main.go --max-prs=1500
```

---

## What If It Fails?

If the test fails, you'll see:

```
✗ VALIDATION FAILED
✗ [Specific error message]

DO NOT PROCEED - Check errors above
```

**Common issues:**
- Missing GITHUB_TOKEN
- jq not installed: `sudo apt-get install jq`
- Compilation error: Check Go code changes
- GitHub API rate limit: Wait an hour

---

## If You Want More Control

### Option 1: Just validate existing data

If you already collected test data:

```bash
./scripts/validate_improvements.sh
```

### Option 2: Step by step

```bash
# Step 1: Collect test data
export GITHUB_TOKEN="your_token"
go run cmd/fetchdata/main.go --max-prs=50 --test

# Step 2: Validate
./scripts/validate_improvements.sh

# Step 3: Manually inspect
jq . data/raw/*.jsonl | less
```

---

## What Gets Validated?

The test checks **everything**:

### Data Collection (6 checks)
- ✓ Files have `content_before` (full code, not diffs)
- ✓ Files have `content_after` (full code, not diffs)
- ✓ Content size is reasonable (>500 bytes avg)
- ✓ PRs have `base_sha` and `head_sha`
- ✓ PRs have `ci_status` field
- ✓ PRs have `is_bot` field

### Syntax Validation (3 checks)
- ✓ All files have `valid_syntax` field
- ✓ >70% of files are syntactically valid
- ✓ Invalid files are identified correctly

### Quality Filtering (5 checks)
- ✓ >80% PRs are merged (filter working)
- ✓ <20% PRs from bots (bot detection working)
- ✓ File count 1-20 per PR (size filter working)
- ✓ Additions 10-2000 per PR (size filter working)
- ✓ Filter statistics logged

### Pipeline Integration (6 checks)
- ✓ plan_extractor.py can read new format
- ✓ Hierarchical data has `context.before`
- ✓ Hierarchical data has `solution.code`
- ✓ Validation has `syntax` field
- ✓ Validation has `tests` field
- ✓ Validation values are booleans (not null)

### Data Structure (4 checks)
- ✓ All required PR fields present
- ✓ All required file fields present
- ✓ Field types are correct
- ✓ No null values where data should exist

---

## Time Investment

- **Test run**: 10-15 minutes
- **Reading results**: 2-3 minutes
- **Total**: ~15 minutes

This 15 minutes ensures your month-long collection will work correctly!

---

## After Tests Pass

Once you see "ALL TESTS PASSED", you can confidently run:

```bash
# Full production collection
go run cmd/fetchdata/main.go --max-prs=1500
```

Monitor with:
```bash
# In another terminal
watch -n 10 'tail -20 fetchdata.log'
```

Check progress:
```bash
# Count collected PRs
find data/raw -name "*.jsonl" -exec wc -l {} + | tail -1

# Check checkpoint
cat checkpoints/fetch_checkpoint.json | jq .
```

---

## Questions?

- **Test fails?** → Check error message, read `TEST_BEFORE_PRODUCTION.md`
- **Warnings but no failures?** → Usually OK, review `validate_improvements.sh` output
- **Want to understand the code?** → Read `DATA_COLLECTION_IMPROVEMENTS.md`
- **Need more details?** → See `QUICK_START_IMPROVED.md`

---

## Quick Reference

```bash
# Full automated test (recommended)
./scripts/run_full_test.sh

# Just validation (if data exists)
./scripts/validate_improvements.sh

# Manual collection
export GITHUB_TOKEN="..."
go run cmd/fetchdata/main.go --max-prs=50 --test

# Production collection (after tests pass)
go run cmd/fetchdata/main.go --max-prs=1500
```

---

**IMPORTANT**: Do NOT start the month-long collection until you see:

```
✓✓✓ ALL TESTS PASSED ✓✓✓
```

Good luck! 🚀
