#!/bin/bash
# Simple quick test without the full automation

echo "===== QUICK TEST ====="
echo ""
echo "This will collect 20 PRs from kubernetes/kubernetes"
echo "Expected time: 3-5 minutes"
echo ""

# Clean up old data
rm -rf data/raw/*.jsonl 2>/dev/null

# Run collection
echo "Starting collection..."
/tmp/fetchdata_fixed -max-prs=20 2>&1 | tee /tmp/quick_test.log

# Check results
if [ -f data/raw/kubernetes_kubernetes.jsonl ]; then
    COUNT=$(wc -l < data/raw/kubernetes_kubernetes.jsonl)
    echo ""
    echo "===== RESULTS ====="
    echo "Collected: $COUNT PRs"
    echo ""

    if [ "$COUNT" -gt 0 ]; then
        echo "Sample PR:"
        head -1 data/raw/kubernetes_kubernetes.jsonl | jq '{
            pr: .pull_request.number,
            merged: .pull_request.merged,
            ci: .pull_request.ci_status,
            bot: .pull_request.is_bot,
            files: .files | length,
            first_file_has_content: (.files[0].content_before != "")
        }'
        echo ""
        echo "✓ SUCCESS - Data collected!"
        echo "Run: ./scripts/validate_improvements.sh"
    else
        echo "✗ FAIL - No PRs collected"
        echo "Check /tmp/quick_test.log for errors"
    fi
else
    echo "✗ FAIL - No data file created"
fi
