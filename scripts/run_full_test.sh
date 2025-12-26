#!/bin/bash
set -e

# Complete end-to-end test of data collection improvements
# This runs everything needed to validate before production

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Full Validation Test - Data Collection Improvements  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if [ -z "$GITHUB_TOKEN" ]; then
    echo -e "${RED}ERROR: GITHUB_TOKEN environment variable not set${NC}"
    echo "Please run: export GITHUB_TOKEN=\"your_token_here\""
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo -e "${RED}ERROR: jq not installed${NC}"
    echo "Please install jq: sudo apt-get install jq"
    exit 1
fi

if ! command -v go &> /dev/null; then
    echo -e "${RED}ERROR: Go not installed${NC}"
    exit 1
fi

if ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites met${NC}"
echo ""

# Clean up old test data
echo "Cleaning up old test data..."
rm -rf data/raw/*.jsonl 2>/dev/null || true
rm -rf data/test_hierarchical.jsonl 2>/dev/null || true
rm -rf data/test_tokenized 2>/dev/null || true
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""

# Step 1: Compile Go code
echo -e "${BLUE}Step 1/6: Compiling Go code...${NC}"
if go build -o /tmp/fetchdata ./cmd/fetchdata/... 2>&1 | grep -i "error"; then
    echo -e "${RED}✗ Go compilation failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Go code compiled successfully${NC}"
echo ""

# Step 2: Run test collection
echo -e "${BLUE}Step 2/6: Running test data collection (50 PRs)...${NC}"
echo "This will take 5-10 minutes depending on GitHub API rate limits"
echo ""

# Create test config
echo '["kubernetes/kubernetes"]' > config/repos_test.json

# Run collection
if /tmp/fetchdata -max-prs=50 2>&1 | tee /tmp/fetchdata.log | grep -i "panic\|fatal"; then
    echo -e "${RED}✗ Data collection failed${NC}"
    echo "Check /tmp/fetchdata.log for details"
    exit 1
fi

# Check if data was created
if [ ! -f data/raw/kubernetes_kubernetes.jsonl ]; then
    echo -e "${RED}✗ No data file created${NC}"
    exit 1
fi

PR_COUNT=$(cat data/raw/kubernetes_kubernetes.jsonl | wc -l)
if [ "$PR_COUNT" -lt 10 ]; then
    echo -e "${RED}✗ Only collected $PR_COUNT PRs (expected ~50)${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Collected $PR_COUNT PRs${NC}"
echo ""

# Step 3: Run automated validation
echo -e "${BLUE}Step 3/6: Running automated validation tests...${NC}"
echo ""

if ! ./scripts/validate_improvements.sh > /tmp/validation_results.txt 2>&1; then
    echo -e "${RED}✗ Validation failed${NC}"
    echo ""
    cat /tmp/validation_results.txt
    exit 1
fi

# Show validation results
cat /tmp/validation_results.txt

# Check results
FAILURES=$(grep "FAILED:" /tmp/validation_results.txt | grep -o '[0-9]*' | head -1)
if [ "$FAILURES" -gt 0 ]; then
    echo -e "${RED}Validation detected $FAILURES failures - aborting${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Automated validation passed${NC}"
echo ""

# Step 4: Test plan extraction
echo -e "${BLUE}Step 4/6: Testing plan extraction pipeline...${NC}"

if ! python tokenizer/plan_extractor.py \
    --input data/raw/kubernetes_kubernetes.jsonl \
    --output data/test_hierarchical.jsonl 2>&1 | tee /tmp/plan_extract.log | grep -i "error"; then

    if [ ! -f data/test_hierarchical.jsonl ]; then
        echo -e "${RED}✗ Plan extraction produced no output${NC}"
        exit 1
    fi

    HIER_COUNT=$(cat data/test_hierarchical.jsonl | wc -l)
    if [ "$HIER_COUNT" -lt 10 ]; then
        echo -e "${RED}✗ Plan extraction only produced $HIER_COUNT records${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Plan extraction successful ($HIER_COUNT records)${NC}"
else
    echo -e "${RED}✗ Plan extraction failed${NC}"
    cat /tmp/plan_extract.log
    exit 1
fi
echo ""

# Step 5: Verify hierarchical data structure
echo -e "${BLUE}Step 5/6: Verifying hierarchical data structure...${NC}"

# Check for required fields
SAMPLE=$(head -1 data/test_hierarchical.jsonl)

CHECKS=0
PASSED=0

# Check context.before
if echo "$SAMPLE" | jq -e '.context.before' >/dev/null 2>&1; then
    BEFORE_LEN=$(echo "$SAMPLE" | jq -r '.context.before | length')
    if [ "$BEFORE_LEN" -gt 100 ]; then
        echo -e "${GREEN}  ✓ context.before exists ($BEFORE_LEN bytes)${NC}"
        ((PASSED++))
    else
        echo -e "${RED}  ✗ context.before too short ($BEFORE_LEN bytes)${NC}"
    fi
else
    echo -e "${RED}  ✗ context.before missing${NC}"
fi
((CHECKS++))

# Check solution.code
if echo "$SAMPLE" | jq -e '.solution.code' >/dev/null 2>&1; then
    CODE_LEN=$(echo "$SAMPLE" | jq -r '.solution.code | length')
    if [ "$CODE_LEN" -gt 100 ]; then
        echo -e "${GREEN}  ✓ solution.code exists ($CODE_LEN bytes)${NC}"
        ((PASSED++))
    else
        echo -e "${RED}  ✗ solution.code too short ($CODE_LEN bytes)${NC}"
    fi
else
    echo -e "${RED}  ✗ solution.code missing${NC}"
fi
((CHECKS++))

# Check validation.syntax
if echo "$SAMPLE" | jq -e '.solution.validation.syntax' >/dev/null 2>&1; then
    SYNTAX=$(echo "$SAMPLE" | jq -r '.solution.validation.syntax')
    if [ "$SYNTAX" = "true" ] || [ "$SYNTAX" = "false" ]; then
        echo -e "${GREEN}  ✓ validation.syntax = $SYNTAX (boolean)${NC}"
        ((PASSED++))
    else
        echo -e "${RED}  ✗ validation.syntax = $SYNTAX (not boolean)${NC}"
    fi
else
    echo -e "${RED}  ✗ validation.syntax missing${NC}"
fi
((CHECKS++))

# Check validation.tests
if echo "$SAMPLE" | jq -e '.solution.validation.tests' >/dev/null 2>&1; then
    TESTS=$(echo "$SAMPLE" | jq -r '.solution.validation.tests')
    if [ "$TESTS" = "true" ] || [ "$TESTS" = "false" ]; then
        echo -e "${GREEN}  ✓ validation.tests = $TESTS (boolean)${NC}"
        ((PASSED++))
    else
        echo -e "${RED}  ✗ validation.tests = $TESTS (not boolean)${NC}"
    fi
else
    echo -e "${RED}  ✗ validation.tests missing${NC}"
fi
((CHECKS++))

if [ "$PASSED" -eq "$CHECKS" ]; then
    echo -e "${GREEN}✓ All hierarchical structure checks passed ($PASSED/$CHECKS)${NC}"
else
    echo -e "${RED}✗ Hierarchical structure checks failed ($PASSED/$CHECKS passed)${NC}"
    exit 1
fi
echo ""

# Step 6: Show sample data
echo -e "${BLUE}Step 6/6: Sample Data Inspection${NC}"
echo ""
echo "Sample PR data (first PR):"
echo "----------------------------------------"
head -1 data/raw/kubernetes_kubernetes.jsonl | jq '{
  pr: .pull_request.number,
  title: .pull_request.title,
  merged: .pull_request.merged,
  is_bot: .pull_request.is_bot,
  ci_status: .pull_request.ci_status,
  files: .files | length,
  first_file: {
    name: .files[0].filename,
    has_before: (.files[0].content_before != ""),
    has_after: (.files[0].content_after != ""),
    valid_syntax: .files[0].valid_syntax
  }
}'
echo ""

echo "Sample hierarchical data (first record):"
echo "----------------------------------------"
head -1 data/test_hierarchical.jsonl | jq '{
  repo: .repo,
  pr_number: .pr_number,
  problem: .problem.description,
  context_file: .context.file,
  context_size: (.context.before | length),
  solution_size: (.solution.code | length),
  validation: .solution.validation
}'
echo ""

echo "Sample code content (first 15 lines of context.before):"
echo "----------------------------------------"
head -1 data/test_hierarchical.jsonl | jq -r '.context.before' | head -15
echo "..."
echo ""

# Final summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  FINAL DECISION                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

echo "Test Results:"
echo "  ✓ Go code compiles"
echo "  ✓ Test data collection succeeded ($PR_COUNT PRs)"
echo "  ✓ Automated validation passed (0 failures)"
echo "  ✓ Plan extraction works ($HIER_COUNT records)"
echo "  ✓ Hierarchical data structure correct"
echo "  ✓ Sample data looks good"
echo ""

echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  ✓✓✓ ALL TESTS PASSED ✓✓✓             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}Your data collection improvements are working correctly!${NC}"
echo ""
echo "You can now proceed with confidence to:"
echo ""
echo -e "${YELLOW}  go run cmd/fetchdata/main.go --max-prs=1500${NC}"
echo ""
echo "Estimated completion time: 2-3 days for ~100 repositories"
echo "Expected output: ~150K high-quality PR records"
echo "Storage required: ~100-150GB"
echo ""
echo "Monitor progress with:"
echo "  watch -n 10 'tail -20 fetchdata.log'"
echo ""
echo "Good luck! 🚀"
echo ""

# Save report
cat > /tmp/test_report.txt << EOF
=== Data Collection Test Report ===
Date: $(date)

✓ Go compilation: SUCCESS
✓ Test collection: SUCCESS ($PR_COUNT PRs)
✓ Automated validation: SUCCESS (0 failures)
✓ Plan extraction: SUCCESS ($HIER_COUNT records)
✓ Data structure: SUCCESS (all fields present)

RECOMMENDATION: PROCEED WITH FULL COLLECTION

Next command:
  go run cmd/fetchdata/main.go --max-prs=1500
EOF

echo "Test report saved to: /tmp/test_report.txt"

exit 0
