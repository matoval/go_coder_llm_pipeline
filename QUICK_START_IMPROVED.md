# Quick Start Guide - Improved Data Collection

## Before You Start

**IMPORTANT**: These improvements significantly increase data quality but also increase:
- API calls per PR (due to fetching full file content)
- Data size (~10x larger due to full content vs diffs)
- Processing time (~2-3x slower due to validation)

Ensure you have:
- ✅ GitHub token with sufficient API rate limit
- ✅ Adequate disk space (~100KB per PR, so ~150GB for 1.5M PRs)
- ✅ Tested on a small dataset first

## Step 1: Test on Single Repository

Before running the month-long collection, test with a single repository:

```bash
# Create a test config with just one repo
echo '["golang/go"]' > config/repos_test.json

# Modify cmd/fetchdata/main.go temporarily to use repos_test.json
# OR just edit config/repos.json to have one repo for testing

# Fetch 50 PRs to test
export GITHUB_TOKEN="your_token_here"
go run cmd/fetchdata/main.go --max-prs=50 --test
```

### Verify Test Output

Check the collected data:
```bash
# View a sample PR
head -n 1 data/raw/golang_go.jsonl | jq . | less

# Check for new fields
jq '.pull_request | {merged, ci_status, is_bot, base_sha, head_sha}' data/raw/golang_go.jsonl | head -n 5

# Check file content fields
jq '.files[0] | {filename, valid_syntax, has_before: (.content_before != ""), has_after: (.content_after != "")}' data/raw/golang_go.jsonl | head -n 5

# Check validation statistics
echo "Total PRs:"
wc -l data/raw/golang_go.jsonl

echo "Merged PRs:"
jq -r 'select(.pull_request.merged == true)' data/raw/golang_go.jsonl | wc -l

echo "Bot PRs:"
jq -r 'select(.pull_request.is_bot == true)' data/raw/golang_go.jsonl | wc -l

echo "Files with valid syntax:"
jq -r '.files[] | select(.valid_syntax == true)' data/raw/golang_go.jsonl | wc -l

echo "CI status distribution:"
jq -r '.pull_request.ci_status' data/raw/golang_go.jsonl | sort | uniq -c
```

### Expected Test Results

For 50 PRs from golang/go, expect:
- ~40-45 merged PRs (most should be merged)
- ~0-2 bot PRs (Go repo has few bots)
- ~95% valid syntax (Go core is high quality)
- CI status: mix of success/unknown (older PRs lack CI)
- Each file should have `content_before` and `content_after` populated

## Step 2: Run Full Collection

Once test looks good:

```bash
# Reset to full repo list
# Make sure config/repos.json has all your target repos

# Run full collection with 1500 PRs per repo
export GITHUB_TOKEN="your_token_here"
go run cmd/fetchdata/main.go --max-prs=1500 --resume=true
```

### Monitor Progress

In another terminal:
```bash
# Watch progress
watch -n 10 'tail -n 20 fetchdata.log'

# Check checkpoint file
cat checkpoints/fetch_checkpoint.json | jq .

# Count collected PRs so far
find data/raw -name "*.jsonl" -exec wc -l {} \; | awk '{sum+=$1} END {print "Total PRs:", sum}'

# Monitor disk usage
du -h data/raw
```

## Step 3: Validate Collected Data

After collection completes:

```bash
# Create validation script
cat > scripts/validate_data.sh << 'EOF'
#!/bin/bash
echo "=== Data Collection Validation ==="
echo ""

echo "Total repositories:"
ls data/raw/*.jsonl | wc -l

echo "Total PRs collected:"
cat data/raw/*.jsonl | wc -l

echo "PRs with CI status:"
jq -r 'select(.pull_request.ci_status != "unknown")' data/raw/*.jsonl | wc -l

echo "PRs with valid syntax (all files):"
jq -r 'select(.files | all(.valid_syntax == true))' data/raw/*.jsonl | wc -l

echo "Average files per PR:"
jq -r '.files | length' data/raw/*.jsonl | awk '{sum+=$1; count++} END {print sum/count}'

echo "CI status breakdown:"
jq -r '.pull_request.ci_status' data/raw/*.jsonl | sort | uniq -c | sort -rn

echo "Bot vs Human PRs:"
echo "  Bots:"
jq -r 'select(.pull_request.is_bot == true)' data/raw/*.jsonl | wc -l
echo "  Humans:"
jq -r 'select(.pull_request.is_bot == false)' data/raw/*.jsonl | wc -l

echo "Merged vs Non-merged:"
echo "  Merged:"
jq -r 'select(.pull_request.merged == true)' data/raw/*.jsonl | wc -l
echo "  Not merged:"
jq -r 'select(.pull_request.merged == false)' data/raw/*.jsonl | wc -l

echo "Files with content:"
echo "  Has before content:"
jq -r '.files[] | select(.content_before != "")' data/raw/*.jsonl | wc -l
echo "  Has after content:"
jq -r '.files[] | select(.content_after != "")' data/raw/*.jsonl | wc -l
EOF

chmod +x scripts/validate_data.sh
./scripts/validate_data.sh
```

## Step 4: Process for Training

Run the updated pipeline:

```bash
# 1. Extract hierarchical plans (now uses content_before/after)
python tokenizer/plan_extractor.py

# 2. Prepare training data (now includes real validation signals)
python scripts/prepare_training_data.py \
  --input_dir data/hierarchical \
  --output_dir data/tokenized \
  --tokenizer_path tokenizer/go_coder_llm.model \
  --max_length 1024

# 3. Verify tokenized data has validation tokens
head -n 1 data/tokenized/train.jsonl | jq -r '.text' | grep -o '<SYNTAX_OK>\|<TEST_PASS>'
```

## Step 5: Train Model

The training now uses real validation signals:

```bash
python model/train.py \
  --train_data_path data/tokenized/train.jsonl \
  --val_data_path data/tokenized/val.jsonl \
  --output_dir checkpoints \
  --num_epochs 10 \
  --batch_size 4
```

### Monitor Training

Watch for improved refinement loss:
```bash
# Refinement loss should decrease steadily
# Before: would plateau or oscillate (bad heuristic)
# After: should decrease smoothly (real validation signals)
tail -f training.log | grep refinement_loss
```

## Common Issues & Solutions

### Issue: API Rate Limit Hit

**Symptoms**: Errors about rate limiting, fetching stops

**Solutions**:
- Wait for rate limit reset (check with: `curl -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/rate_limit`)
- Use GitHub Enterprise for higher limits
- Reduce concurrency or add delays

### Issue: Disk Space Full

**Symptoms**: Write errors, collection stops

**Solutions**:
- Data is ~100KB per PR, ensure you have enough space
- Delete old data if re-collecting
- Use compression: `gzip data/raw/*.jsonl`

### Issue: Many PRs Filtered Out

**Symptoms**: "Filtered: X, fetched: Y" where X >> Y

**Solutions**:
- Check filter breakdown in logs
- Adjust quality filters if too strict
- Some repos naturally have fewer merged PRs

### Issue: Missing content_before/after

**Symptoms**: Empty `content_before` or `content_after` fields

**Solutions**:
- Check GitHub API permissions
- Verify files exist at those SHAs
- May be legitimately empty for new/deleted files

### Issue: No CI Status

**Symptoms**: Most PRs have `ci_status: "unknown"`

**Solutions**:
- Older PRs don't have GitHub Checks API data
- Some repos don't use GitHub Actions
- This is expected, not an error

## Performance Expectations

### Collection Speed
- **Before**: ~100 PRs/minute (diffs only)
- **After**: ~30-40 PRs/minute (full content + validation)
- **For 1500 PRs/repo**: ~40-50 minutes per repo
- **For 100 repos**: ~2-3 days total

### Data Size
- **Before**: ~10KB per PR
- **After**: ~50-100KB per PR
- **For 150K PRs**: ~7-15GB total

### API Calls
- **Before**: ~3 calls per PR
- **After**: ~3 + N calls (N = # of Go files)
- **Average**: ~8-10 calls per PR
- **For 1500 PRs**: ~12K-15K API calls per repo

## Quality Metrics

Target metrics after collection:

✅ **>95% merged PRs** (quality filter working)
✅ **<5% bot PRs** (bot detection working)
✅ **>80% valid syntax** (code quality high)
✅ **>40% CI status known** (reasonable coverage)
✅ **>90% files have content** (fetching working)

## Next Steps After Collection

1. **Validate data quality** using the validation script
2. **Analyze distributions** to understand dataset
3. **Run plan extraction** to create hierarchical format
4. **Train model** with improved data
5. **Compare results** to baseline (before improvements)

## Getting Help

If issues persist:
1. Check `DATA_COLLECTION_IMPROVEMENTS.md` for detailed info
2. Review logs in `checkpoints/` directory
3. Inspect checkpoint file for progress
4. Verify test data structure matches expected format

## Success Checklist

Before starting month-long collection, ensure:

- [ ] Test collection on single repo completed successfully
- [ ] Data validation shows expected quality metrics
- [ ] Disk space sufficient (~150GB for full collection)
- [ ] GitHub API token has adequate rate limit
- [ ] Plan extraction works with new data format
- [ ] Training script uses new validation signals
- [ ] Monitoring scripts in place

Good luck with your data collection! 🚀
