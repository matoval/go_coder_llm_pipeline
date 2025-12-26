#!/bin/bash
# Complete retraining pipeline for HRM Go Coder
# This script will:
# 1. Backup current data
# 2. Collect more training data (500 PRs per repo)
# 3. Process data into hierarchical format
# 4. Train tokenizer if needed
# 5. Prepare tokenized training data
# 6. Train model from scratch

set -e  # Exit on error

echo "========================================"
echo "HRM Go Coder - Complete Retraining Pipeline"
echo "========================================"
echo ""

# Configuration
MAX_PRS=500
MAX_SEQ_LENGTH=1024
EPOCHS=10
BATCH_SIZE=4
LEARNING_RATE=3e-4

echo "Configuration:"
echo "  Max PRs per repo: $MAX_PRS"
echo "  Max sequence length: $MAX_SEQ_LENGTH"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Step 1: Backup existing data
echo "Step 1: Backing up existing data..."
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -d "data/raw" ]; then
    echo "  Backing up data/raw to $BACKUP_DIR/raw"
    cp -r data/raw "$BACKUP_DIR/"
fi

if [ -d "data/processed" ]; then
    echo "  Backing up data/processed to $BACKUP_DIR/processed"
    cp -r data/processed "$BACKUP_DIR/"
fi

if [ -d "data/tokenized" ]; then
    echo "  Backing up data/tokenized to $BACKUP_DIR/tokenized"
    cp -r data/tokenized "$BACKUP_DIR/"
fi

if [ -d "checkpoints" ]; then
    echo "  Backing up checkpoints to $BACKUP_DIR/checkpoints"
    cp -r checkpoints "$BACKUP_DIR/"
fi

echo "  ✓ Backup complete: $BACKUP_DIR"
echo ""

# Step 2: Collect more data (incremental)
echo "Step 2: Collecting additional training data..."
echo "  Fetching up to $MAX_PRS PRs per repository (incremental)..."
echo "  Current data: $(ls data/raw/*.jsonl 2>/dev/null | wc -l) repos"
echo "  Starting at: $(date)"
echo ""

# Keep existing data, just fetch more PRs per repo
# The -resume flag (default: true) will append to existing files
./bin/fetchdata -max-prs $MAX_PRS -resume

echo ""
echo "  ✓ Data collection complete at: $(date)"
echo "  Total repos: $(ls data/raw/*.jsonl 2>/dev/null | wc -l)"
echo "  Raw data size: $(du -sh data/raw | cut -f1)"
echo ""

# Quick stats on how many PRs we have now
python3 << 'STATS_EOF'
import os
total_prs = 0
for filename in os.listdir('data/raw'):
    if filename.endswith('.jsonl'):
        with open(os.path.join('data/raw', filename), 'r') as f:
            count = sum(1 for _ in f)
            total_prs += count
print(f"  Total PRs: {total_prs:,}")
STATS_EOF
echo ""

# Step 3: Process data
echo "Step 3: Processing data to hierarchical format..."
source llm/bin/activate

python scripts/process_data.py hierarchical \
    --input data/raw \
    --output data/processed

echo "  ✓ Hierarchical processing complete"
echo "  Processed data: $(du -sh data/processed | cut -f1)"
echo ""

# Step 4: Create training corpus (for tokenizer if needed)
echo "Step 4: Creating training corpus..."
python scripts/process_data.py corpus \
    --input data/processed \
    --output data/corpus

echo "  ✓ Corpus creation complete"
echo "  Corpus size: $(du -sh data/corpus | cut -f1)"
echo ""

# Step 5: Check if we need to retrain tokenizer
echo "Step 5: Checking tokenizer..."
if [ ! -f "tokenizer/go_coder_llm.model" ]; then
    echo "  Tokenizer not found, training new tokenizer..."
    cd tokenizer
    python train_tokenizer.py
    cd ..
    echo "  ✓ Tokenizer training complete"
else
    echo "  ✓ Using existing tokenizer"
fi
echo ""

# Step 6: Prepare tokenized training data
echo "Step 6: Preparing tokenized training data..."
python scripts/prepare_training_data.py \
    --input data/processed \
    --output data/tokenized \
    --tokenizer tokenizer/go_coder_llm.model \
    --max-length $MAX_SEQ_LENGTH \
    --val-split 0.1

echo "  ✓ Tokenization complete"
echo ""

# Display dataset statistics
echo "Dataset Statistics:"
TRAIN_COUNT=$(wc -l < data/tokenized/train.jsonl)
VAL_COUNT=$(wc -l < data/tokenized/val.jsonl)
echo "  Training samples: $TRAIN_COUNT"
echo "  Validation samples: $VAL_COUNT"
echo "  Total samples: $((TRAIN_COUNT + VAL_COUNT))"
echo ""

# Estimate training time and token count
python3 << EOF
import json

# Sample first 1000 records to estimate average length
lengths = []
with open('data/tokenized/train.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        record = json.loads(line)
        lengths.append(len(record['input_ids']))

avg_length = sum(lengths) / len(lengths)
train_count = $TRAIN_COUNT
epochs = $EPOCHS
batch_size = $BATCH_SIZE

total_tokens = train_count * epochs * avg_length
steps_per_epoch = train_count // batch_size
total_steps = steps_per_epoch * epochs

print(f"Training Estimates:")
print(f"  Avg tokens/sample: {avg_length:.1f}")
print(f"  Total tokens: {total_tokens:,.0f} ({total_tokens/1e9:.2f}B)")
print(f"  Steps per epoch: {steps_per_epoch:,}")
print(f"  Total steps: {total_steps:,}")
print(f"  Estimated time: 2-3 days (if ~4B tokens)")
print()

# Check if we have enough data
if total_tokens < 3e9:
    print(f"⚠️  WARNING: Only {total_tokens/1e9:.2f}B tokens, need ~4B for good results")
    print(f"   Consider increasing --max-prs or running more epochs")
else:
    print(f"✓ Good data volume: {total_tokens/1e9:.2f}B tokens")
print()
EOF

# Step 7: Train model
echo "Step 7: Starting model training..."
echo "  This will take 2-3 days for 4B tokens"
echo "  Started at: $(date)"
echo ""
echo "  Training configuration:"
echo "    - Config: small (20-30M params)"
echo "    - Batch size: $BATCH_SIZE"
echo "    - Epochs: $EPOCHS"
echo "    - Learning rate: $LEARNING_RATE"
echo "    - Max sequence length: $MAX_SEQ_LENGTH"
echo ""

# Clear old checkpoints
rm -rf checkpoints
mkdir -p checkpoints

# Start training (this will take days)
python model/train.py \
    --config small \
    --train-data data/tokenized/train.jsonl \
    --val-data data/tokenized/val.jsonl \
    --output-dir checkpoints \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "  Finished at: $(date)"
echo "  Checkpoints saved in: checkpoints/"
echo "  Best model: checkpoints/best.pt"
echo ""
echo "Next steps:"
echo "  1. Test the model: python scripts/inference.py --checkpoint checkpoints/best.pt --problems test_problems.jsonl"
echo "  2. Export to GGUF: python model/export_gguf.py --checkpoint checkpoints/best.pt"
echo "  3. Deploy with Ollama: ollama create golang-llm -f Modelfile"
echo ""
