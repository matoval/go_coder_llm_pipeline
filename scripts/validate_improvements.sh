#!/bin/bash
set -e

# Validation script to verify data collection improvements are working
# Run this on test data BEFORE starting month-long collection

echo "=============================================="
echo "Data Collection Improvements Validation"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0
WARN=0

check_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((PASS++))
}

check_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((FAIL++))
}

check_warn() {
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
    ((WARN++))
}

# Check if test data exists
if [ ! -d "data/raw" ] || [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
    echo -e "${RED}ERROR: No data found in data/raw/${NC}"
    echo "Please run: go run cmd/fetchdata/main.go --max-prs=50 --test"
    exit 1
fi

echo "Found data files:"
ls -lh data/raw/*.jsonl | head -5
echo ""

# Get first data file
SAMPLE_FILE=$(ls data/raw/*.jsonl | head -1)
TOTAL_PRS=$(cat data/raw/*.jsonl 2>/dev/null | wc -l)

echo "Analyzing $TOTAL_PRS PRs from collected data..."
echo ""

# ============================================
echo "1. Testing Full File Content Fetching"
echo "============================================"

# Check if content_before exists
HAS_BEFORE=$(jq -r '.files[]? | select(.content_before != null and .content_before != "")' data/raw/*.jsonl 2>/dev/null | wc -l)
TOTAL_FILES=$(jq -r '.files[]?' data/raw/*.jsonl 2>/dev/null | wc -l)

if [ "$HAS_BEFORE" -gt 0 ]; then
    PERCENT=$((HAS_BEFORE * 100 / TOTAL_FILES))
    if [ "$PERCENT" -gt 70 ]; then
        check_pass "content_before populated: $HAS_BEFORE/$TOTAL_FILES files ($PERCENT%)"
    else
        check_warn "content_before only in $PERCENT% of files (expected >70%)"
    fi
else
    check_fail "No files have content_before populated!"
fi

# Check if content_after exists
HAS_AFTER=$(jq -r '.files[]? | select(.content_after != null and .content_after != "")' data/raw/*.jsonl 2>/dev/null | wc -l)

if [ "$HAS_AFTER" -gt 0 ]; then
    PERCENT=$((HAS_AFTER * 100 / TOTAL_FILES))
    if [ "$PERCENT" -gt 80 ]; then
        check_pass "content_after populated: $HAS_AFTER/$TOTAL_FILES files ($PERCENT%)"
    else
        check_warn "content_after only in $PERCENT% of files (expected >80%)"
    fi
else
    check_fail "No files have content_after populated!"
fi

# Check that content is actual code, not empty
AVG_BEFORE_LENGTH=$(jq -r '.files[]? | .content_before | length' data/raw/*.jsonl 2>/dev/null | awk '{sum+=$1; count++} END {if(count>0) print int(sum/count); else print 0}')
AVG_AFTER_LENGTH=$(jq -r '.files[]? | .content_after | length' data/raw/*.jsonl 2>/dev/null | awk '{sum+=$1; count++} END {if(count>0) print int(sum/count); else print 0}')

if [ "$AVG_BEFORE_LENGTH" -gt 500 ]; then
    check_pass "Average content_before size: $AVG_BEFORE_LENGTH bytes (reasonable)"
else
    check_warn "Average content_before size: $AVG_BEFORE_LENGTH bytes (seems small)"
fi

if [ "$AVG_AFTER_LENGTH" -gt 500 ]; then
    check_pass "Average content_after size: $AVG_AFTER_LENGTH bytes (reasonable)"
else
    check_warn "Average content_after size: $AVG_AFTER_LENGTH bytes (seems small)"
fi

# Verify content looks like Go code
echo ""
echo "Sample content_before (first 10 lines):"
jq -r '.files[0]? | .content_before' "$SAMPLE_FILE" 2>/dev/null | head -10
echo ""

# ============================================
echo "2. Testing Go Syntax Validation"
echo "============================================"

# Check if valid_syntax field exists
HAS_SYNTAX_FIELD=$(jq -r '.files[]? | select(.valid_syntax != null)' data/raw/*.jsonl 2>/dev/null | wc -l)

if [ "$HAS_SYNTAX_FIELD" -eq "$TOTAL_FILES" ]; then
    check_pass "valid_syntax field present in all files"
else
    check_fail "valid_syntax field missing in some files ($HAS_SYNTAX_FIELD/$TOTAL_FILES)"
fi

# Check validation results
VALID_SYNTAX=$(jq -r '.files[]? | select(.valid_syntax == true)' data/raw/*.jsonl 2>/dev/null | wc -l)
INVALID_SYNTAX=$(jq -r '.files[]? | select(.valid_syntax == false)' data/raw/*.jsonl 2>/dev/null | wc -l)

if [ "$VALID_SYNTAX" -gt 0 ]; then
    PERCENT=$((VALID_SYNTAX * 100 / TOTAL_FILES))
    if [ "$PERCENT" -gt 70 ]; then
        check_pass "Syntax validation: $VALID_SYNTAX valid, $INVALID_SYNTAX invalid ($PERCENT% valid)"
    else
        check_warn "Only $PERCENT% files valid syntax (expected >70%)"
    fi
else
    check_fail "No files marked as valid_syntax:true"
fi

# Show sample of invalid syntax files
if [ "$INVALID_SYNTAX" -gt 0 ]; then
    echo ""
    echo "Sample files with invalid syntax:"
    jq -r '.files[]? | select(.valid_syntax == false) | .filename' data/raw/*.jsonl 2>/dev/null | head -3
fi

# ============================================
echo ""
echo "3. Testing CI/CD Status Collection"
echo "============================================"

# Check if ci_status field exists
HAS_CI=$(jq -r 'select(.pull_request.ci_status != null)' data/raw/*.jsonl 2>/dev/null | wc -l)

if [ "$HAS_CI" -eq "$TOTAL_PRS" ]; then
    check_pass "ci_status field present in all PRs"
else
    check_fail "ci_status field missing in some PRs ($HAS_CI/$TOTAL_PRS)"
fi

# Check CI status distribution
echo ""
echo "CI Status Distribution:"
jq -r '.pull_request.ci_status' data/raw/*.jsonl 2>/dev/null | sort | uniq -c | while read count status; do
    echo "  $status: $count"
done

CI_SUCCESS=$(jq -r 'select(.pull_request.ci_status == "success")' data/raw/*.jsonl 2>/dev/null | wc -l)
CI_FAILURE=$(jq -r 'select(.pull_request.ci_status == "failure")' data/raw/*.jsonl 2>/dev/null | wc -l)
CI_UNKNOWN=$(jq -r 'select(.pull_request.ci_status == "unknown")' data/raw/*.jsonl 2>/dev/null | wc -l)

PERCENT_KNOWN=$(( (CI_SUCCESS + CI_FAILURE) * 100 / TOTAL_PRS ))

if [ "$PERCENT_KNOWN" -gt 30 ]; then
    check_pass "CI status known for $PERCENT_KNOWN% of PRs (${CI_SUCCESS} success, ${CI_FAILURE} failure)"
elif [ "$PERCENT_KNOWN" -gt 10 ]; then
    check_warn "CI status known for only $PERCENT_KNOWN% of PRs (older repos may lack CI)"
else
    check_warn "CI status mostly unknown (${CI_UNKNOWN}/${TOTAL_PRS}) - this may be normal for older PRs"
fi

# ============================================
echo ""
echo "4. Testing Quality Filters"
echo "============================================"

# Check merged status
MERGED=$(jq -r 'select(.pull_request.merged == true)' data/raw/*.jsonl 2>/dev/null | wc -l)
NOT_MERGED=$(jq -r 'select(.pull_request.merged == false)' data/raw/*.jsonl 2>/dev/null | wc -l)

PERCENT_MERGED=$((MERGED * 100 / TOTAL_PRS))

if [ "$PERCENT_MERGED" -gt 80 ]; then
    check_pass "Merged filter working: $PERCENT_MERGED% merged ($MERGED/$TOTAL_PRS)"
else
    check_warn "Only $PERCENT_MERGED% PRs merged (filter may not be applied)"
fi

# Check bot detection
IS_BOT=$(jq -r 'select(.pull_request.is_bot == true)' data/raw/*.jsonl 2>/dev/null | wc -l)
IS_HUMAN=$(jq -r 'select(.pull_request.is_bot == false)' data/raw/*.jsonl 2>/dev/null | wc -l)

PERCENT_BOT=$((IS_BOT * 100 / TOTAL_PRS))

if [ "$PERCENT_BOT" -lt 20 ]; then
    check_pass "Bot filter working: $PERCENT_BOT% bots ($IS_BOT/$TOTAL_PRS)"
else
    check_warn "$PERCENT_BOT% PRs from bots (expected <20%)"
fi

# Check file count distribution
echo ""
echo "File count distribution:"
jq -r '.files | length' data/raw/*.jsonl 2>/dev/null | awk '
{
    count[$1]++
}
END {
    for (n in count) {
        print "  " n " files: " count[n] " PRs"
    }
}' | sort -n

AVG_FILES=$(jq -r '.files | length' data/raw/*.jsonl 2>/dev/null | awk '{sum+=$1; count++} END {print int(sum/count)}')

if [ "$AVG_FILES" -ge 1 ] && [ "$AVG_FILES" -le 20 ]; then
    check_pass "Average files per PR: $AVG_FILES (within 1-20 range)"
else
    check_warn "Average files per PR: $AVG_FILES (outside recommended 1-20 range)"
fi

# Check additions distribution
AVG_ADDITIONS=$(jq -r '[.files[].additions] | add' data/raw/*.jsonl 2>/dev/null | awk '{sum+=$1; count++} END {print int(sum/count)}')

if [ "$AVG_ADDITIONS" -ge 10 ] && [ "$AVG_ADDITIONS" -le 2000 ]; then
    check_pass "Average additions per PR: $AVG_ADDITIONS (within 10-2000 range)"
else
    check_warn "Average additions per PR: $AVG_ADDITIONS (outside recommended 10-2000 range)"
fi

# Check base_sha and head_sha exist
HAS_SHAS=$(jq -r 'select(.pull_request.base_sha != null and .pull_request.head_sha != null)' data/raw/*.jsonl 2>/dev/null | wc -l)

if [ "$HAS_SHAS" -eq "$TOTAL_PRS" ]; then
    check_pass "base_sha and head_sha present in all PRs"
else
    check_fail "base_sha/head_sha missing in some PRs ($HAS_SHAS/$TOTAL_PRS)"
fi

# ============================================
echo ""
echo "5. Testing Data Structure Completeness"
echo "============================================"

# Check all required fields exist in a sample
REQUIRED_PR_FIELDS=("number" "title" "merged" "base_sha" "head_sha" "ci_status" "is_bot")
REQUIRED_FILE_FIELDS=("filename" "patch" "content_before" "content_after" "valid_syntax")

echo "Checking required PR fields in sample..."
SAMPLE_PR=$(jq '.' "$SAMPLE_FILE" 2>/dev/null | head -1)

for field in "${REQUIRED_PR_FIELDS[@]}"; do
    if echo "$SAMPLE_PR" | jq -e ".pull_request.$field" >/dev/null 2>&1; then
        check_pass "PR field '$field' exists"
    else
        check_fail "PR field '$field' missing"
    fi
done

echo ""
echo "Checking required file fields in sample..."
for field in "${REQUIRED_FILE_FIELDS[@]}"; do
    if echo "$SAMPLE_PR" | jq -e ".files[0].$field" >/dev/null 2>&1; then
        check_pass "File field '$field' exists"
    else
        check_fail "File field '$field' missing"
    fi
done

# ============================================
echo ""
echo "6. Testing Training Pipeline Compatibility"
echo "============================================"

# Check if data can be processed by plan_extractor
echo "Testing plan_extractor.py compatibility..."

# Create a tiny test file
TEST_FILE="/tmp/test_pr.jsonl"
head -1 "$SAMPLE_FILE" > "$TEST_FILE"

# Try to process it
if python tokenizer/plan_extractor.py --input "$TEST_FILE" --output /tmp/test_hierarchical.jsonl 2>&1 | grep -q "Error"; then
    check_fail "plan_extractor.py failed to process data"
else
    check_pass "plan_extractor.py can process new data format"

    # Check if hierarchical output has expected fields
    if [ -f /tmp/test_hierarchical.jsonl ]; then
        if jq -e '.context.before' /tmp/test_hierarchical.jsonl >/dev/null 2>&1; then
            check_pass "Hierarchical data has 'context.before' field"
        else
            check_fail "Hierarchical data missing 'context.before' field"
        fi

        if jq -e '.solution.code' /tmp/test_hierarchical.jsonl >/dev/null 2>&1; then
            check_pass "Hierarchical data has 'solution.code' field"
        else
            check_fail "Hierarchical data missing 'solution.code' field"
        fi

        if jq -e '.solution.validation.syntax' /tmp/test_hierarchical.jsonl >/dev/null 2>&1; then
            check_pass "Hierarchical data has 'solution.validation.syntax' field"
        else
            check_fail "Hierarchical data missing 'solution.validation.syntax' field"
        fi

        if jq -e '.solution.validation.tests' /tmp/test_hierarchical.jsonl >/dev/null 2>&1; then
            check_pass "Hierarchical data has 'solution.validation.tests' field"
        else
            check_fail "Hierarchical data missing 'solution.validation.tests' field"
        fi

        # Show sample of validation values
        echo ""
        echo "Sample validation values:"
        jq -r '.solution.validation' /tmp/test_hierarchical.jsonl
    fi
fi

rm -f "$TEST_FILE" /tmp/test_hierarchical.jsonl 2>/dev/null

# ============================================
echo ""
echo "=============================================="
echo "VALIDATION SUMMARY"
echo "=============================================="
echo -e "${GREEN}PASSED: $PASS${NC}"
echo -e "${YELLOW}WARNINGS: $WARN${NC}"
echo -e "${RED}FAILED: $FAIL${NC}"
echo ""

# Final recommendation
if [ "$FAIL" -eq 0 ]; then
    if [ "$WARN" -eq 0 ]; then
        echo -e "${GREEN}✓✓✓ ALL CHECKS PASSED ✓✓✓${NC}"
        echo ""
        echo "Your data collection improvements are working correctly!"
        echo "You can proceed with the full data collection."
        echo ""
        echo "Recommended next steps:"
        echo "  1. Review the statistics above"
        echo "  2. Run: go run cmd/fetchdata/main.go --max-prs=1500"
        echo "  3. Monitor progress with: watch -n 10 'tail -20 fetchdata.log'"
        exit 0
    else
        echo -e "${YELLOW}⚠ CHECKS PASSED WITH WARNINGS ⚠${NC}"
        echo ""
        echo "Most checks passed but there are some warnings."
        echo "Review the warnings above and decide if they're acceptable."
        echo ""
        echo "Common reasons for warnings:"
        echo "  - Older PRs lack CI/CD data (normal)"
        echo "  - Some repos have different characteristics"
        echo "  - Small test sample may not be representative"
        echo ""
        echo "If warnings are acceptable, you can proceed with full collection."
        exit 0
    fi
else
    echo -e "${RED}✗✗✗ VALIDATION FAILED ✗✗✗${NC}"
    echo ""
    echo "Some critical checks failed. Please review failures above."
    echo ""
    echo "Common fixes:"
    echo "  1. Ensure you ran: go run cmd/fetchdata/main.go --max-prs=50 --test"
    echo "  2. Check GitHub API token permissions"
    echo "  3. Verify Go code compiled successfully"
    echo "  4. Review error messages in failures above"
    echo ""
    echo "DO NOT proceed with full collection until failures are fixed."
    exit 1
fi
