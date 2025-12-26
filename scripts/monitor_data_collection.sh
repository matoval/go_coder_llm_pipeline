#!/bin/bash
# Monitor data collection progress

echo "========================================"
echo "Data Collection Monitor"
echo "========================================"
echo ""

# Check if fetchdata is running
if pgrep -f fetchdata > /dev/null; then
    echo "✓ Data collection is RUNNING"
else
    echo "✗ Data collection is NOT running"
fi
echo ""

# Show latest log lines
echo "Latest Progress:"
echo "----------------"
tail -10 fetchdata.log 2>/dev/null || echo "No log file yet"
echo ""

# Show data statistics
echo "Current Data Statistics:"
echo "------------------------"
echo "Repos processed: $(ls data/raw/*.jsonl 2>/dev/null | wc -l)"
echo "Total size: $(du -sh data/raw 2>/dev/null | cut -f1)"

# Count total PRs
python3 << 'EOF'
import os
total_prs = 0
try:
    for filename in os.listdir('data/raw'):
        if filename.endswith('.jsonl'):
            with open(os.path.join('data/raw', filename), 'r') as f:
                count = sum(1 for _ in f)
                total_prs += count
    print(f"Total PRs: {total_prs:,}")
except Exception as e:
    print(f"Error counting PRs: {e}")
EOF

echo ""
echo "To follow live progress:"
echo "  tail -f fetchdata.log"
echo ""
echo "To check GitHub rate limit:"
echo "  curl -H \"Authorization: token \$GITHUB_TOKEN\" https://api.github.com/rate_limit"
