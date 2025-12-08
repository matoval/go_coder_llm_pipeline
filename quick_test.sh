#!/bin/bash
#
# Quick Test Training Script
# Runs a minimal test to verify the pipeline works
#

set -e  # Exit on error

echo "============================================"
echo "  Go Coder LLM - Quick Test Training"
echo "============================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check dependencies
echo -e "${YELLOW}Step 1/4: Checking dependencies...${NC}"
if ! python -c "import torch" 2>/dev/null; then
    echo "❌ PyTorch not found. Installing dependencies..."
    pip install torch numpy sentencepiece tqdm
else
    echo "✓ PyTorch found"
fi

if ! python -c "import sentencepiece" 2>/dev/null; then
    echo "❌ SentencePiece not found. Installing..."
    pip install sentencepiece
else
    echo "✓ SentencePiece found"
fi

if ! python -c "import tqdm" 2>/dev/null; then
    echo "❌ tqdm not found. Installing..."
    pip install tqdm
else
    echo "✓ tqdm found"
fi

echo ""

# Step 2: Create synthetic test data
echo -e "${YELLOW}Step 2/4: Creating synthetic test data (100 samples)...${NC}"
python scripts/create_test_data.py --num-samples 100
echo ""

# Step 3: Test model architecture
echo -e "${YELLOW}Step 3/4: Testing model architecture...${NC}"
python -c "
from model.hrm_model import HierarchicalGoCoderModel
from model.config import get_tiny_config

config = get_tiny_config()
model = HierarchicalGoCoderModel(config)
params = model.get_num_params()

print(f'✓ Model created successfully')
print(f'  Total parameters: {params[\"total\"]/1e6:.1f}M')
"
echo ""

# Step 4: Run test training
echo -e "${YELLOW}Step 4/4: Running test training (1 epoch)...${NC}"
echo "This will take 1-3 minutes on CPU, 30-60 seconds on GPU"
echo ""

python model/train.py \
  --config tiny \
  --train-data data/test/train.jsonl \
  --val-data data/test/val.jsonl \
  --batch-size 4 \
  --epochs 1 \
  --lr 5e-4 \
  --output-dir checkpoints/test

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  ✅ Quick test completed successfully!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Results:"
echo "  - Checkpoint saved to: checkpoints/test/"
echo "  - Test data saved to: data/test/"
echo ""
echo "Next steps:"
echo "  1. Check training loss decreased"
echo "  2. Collect real GitHub data for full training"
echo "  3. See TRAINING.md for full training guide"
echo ""
