# Quick Test Training Guide

Fast test training to verify the pipeline works before committing to full training.

## Prerequisites

### 1. Install Dependencies (5 minutes)

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install torch numpy sentencepiece tqdm

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

**Expected time**: 5 minutes
**Disk space**: ~2-3 GB (for PyTorch + dependencies)

## Quick Test (No Real Data) - 2 minutes

### Option A: Test Model Architecture Only

```bash
# Test model creation and forward pass
python model/hrm_model.py
```

**What it tests**:
- Model initialization
- Parameter counts (should be ~31M)
- Forward pass (training mode)
- Generation (inference mode)

**Expected output**:
```
=== Hierarchical Recursive Go Coder Test ===
Config: ...
Parameter Counts:
  planner: 10.XX M
  generator: 10.XX M
  ...
✅ HRM Model Implementation Complete!
```

**Time**: ~30 seconds

### Option B: Test Individual Modules

```bash
# Test planner
python model/modules/planner.py

# Test generator
python model/modules/generator.py

# Test refinement
python model/modules/refinement.py
```

**Time**: ~1 minute total

## Minimal Training Test (Synthetic Data) - 5-10 minutes

### Step 1: Create Synthetic Test Data

```bash
python scripts/create_test_data.py --num-samples 100
```

This creates a small synthetic dataset in `data/test/`.

### Step 2: Run Test Training

```bash
python model/train.py \
  --config tiny \
  --train-data data/test/train.jsonl \
  --val-data data/test/val.jsonl \
  --batch-size 4 \
  --epochs 1 \
  --output-dir checkpoints/test
```

**Parameters**:
- `--config tiny` - Smallest model (~7-10M params)
- `--batch-size 4` - Small batch for quick iteration
- `--epochs 1` - Just one epoch to verify

**Expected time**:
- 100 samples, batch size 4 = 25 batches
- ~2-5 seconds per batch (CPU) or ~0.5-1 sec (GPU)
- **Total: 1-3 minutes** (CPU) or **30-60 seconds** (GPU)

**What to expect**:
```
INFO - Loading datasets...
INFO - Loaded 80 samples
INFO - Creating model...
INFO - Model parameters:
  planner: 7.2M
  generator: 7.5M
  total: 15.1M
INFO - Starting training...
Epoch 1/1: 100%|████████| 25/25 [00:45<00:00, loss=5.234]
INFO - Train metrics: {'loss': 5.234, 'plan_loss': 1.823, ...}
INFO - Checkpoints saved to checkpoints/test
```

## Real Data Mini-Test - 30-60 minutes

If you want to test with actual GitHub data (but minimal):

### Step 1: Collect Minimal Data (15-20 min)

```bash
# Fetch from just 3 popular Go repos
# This is already configured in the codebase
cd cmd/fetchdata
go run main.go
```

**Repos fetched** (from `main.go`):
- `golang/go`
- `spf13/cobra`
- `wailsapp/wails`

**Expected**: ~100-500 PRs depending on repo activity

### Step 2: Process Data (5-10 min)

```bash
# Convert to hierarchical format
python scripts/process_data.py hierarchical \
  --input data/processed/*.jsonl \
  --output data/hierarchical/repos.jsonl

# Create corpus for tokenizer
python scripts/process_data.py corpus \
  --input data/hierarchical/repos.jsonl \
  --output data/corpus/training_corpus.txt
```

### Step 3: Train Tokenizer (5-10 min)

```bash
# Train SentencePiece model
./tokenizer/train_tokenizer.sh
```

**Note**: With minimal data (~500 PRs), tokenizer will work but may not be optimal.

### Step 4: Tokenize Dataset (2-5 min)

```bash
# Tokenize for training
python scripts/process_data.py tokenize \
  --input data/hierarchical/repos.jsonl \
  --output data/tokenized/train.jsonl \
  --tokenizer tokenizer/go_coder_llm.model

# Create 80/20 train/val split
head -n 400 data/tokenized/train.jsonl > data/tokenized/train_split.jsonl
tail -n 100 data/tokenized/train.jsonl > data/tokenized/val_split.jsonl
```

### Step 5: Test Training (10-20 min)

```bash
python model/train.py \
  --config small \
  --train-data data/tokenized/train_split.jsonl \
  --val-data data/tokenized/val_split.jsonl \
  --batch-size 8 \
  --epochs 2 \
  --lr 5e-4 \
  --output-dir checkpoints/mini_test
```

**Expected time**:
- 400 samples, batch size 8 = 50 batches/epoch
- 2 epochs = 100 batches total
- ~5-10 seconds/batch (CPU) or ~1-2 sec/batch (GPU)
- **Total: 10-15 minutes** (CPU) or **2-4 minutes** (GPU)

**Memory requirements**:
- CPU: 4-8 GB RAM
- GPU: 4-6 GB VRAM (for small model)

## Expected Results

### What "Success" Looks Like

After test training, you should see:

1. **Loss decreases**:
   ```
   Epoch 1: loss=6.234 -> Epoch 2: loss=5.123
   ```
   Even minimal decrease shows model is learning.

2. **Checkpoint saved**:
   ```
   checkpoints/test/
   ├── epoch_1.pt
   ├── epoch_2.pt
   └── best.pt
   ```

3. **No crashes**: Training completes without errors

4. **Generation works** (optional test):
   ```python
   # Load checkpoint and test generation
   python -c "
   import torch
   from model.hrm_model import HierarchicalGoCoderModel
   from model.config import get_small_config

   config = get_small_config()
   model = HierarchicalGoCoderModel(config)

   # Load weights
   checkpoint = torch.load('checkpoints/test/best.pt', map_location='cpu')
   model.load_state_dict(checkpoint['model_state_dict'])

   # Test generation (random tokens, just to verify it runs)
   problem_ids = torch.randint(0, 50000, (1, 32))
   result = model.generate(problem_ids, max_refinement_iterations=2)
   print(f'Generated {result.num_iterations} iterations')
   print(f'Code shape: {result.generated_code.shape}')
   "
   ```

## Time Estimates Summary

| Test Type | Setup | Training | Total | What You Learn |
|-----------|-------|----------|-------|----------------|
| **Architecture Only** | 5 min | 30 sec | ~6 min | Model loads, forward pass works |
| **Synthetic Data** | 10 min | 1-3 min | ~15 min | Training loop works end-to-end |
| **Real Data Mini** | 30 min | 10-20 min | ~50 min | Full pipeline works with real PRs |

## Troubleshooting

### Out of Memory

If you get OOM errors:

```bash
# Reduce batch size
python model/train.py --batch-size 2 ...

# Or use tiny config
python model/train.py --config tiny ...
```

### Slow on CPU

CPU training is 5-10x slower than GPU. For quick tests:

```bash
# Reduce data
head -n 50 data/tokenized/train.jsonl > data/tokenized/mini.jsonl

# Train on minimal data
python model/train.py \
  --train-data data/tokenized/mini.jsonl \
  --batch-size 2 \
  --epochs 1
```

### Import Errors

```bash
# Make sure you're in the project root
cd /path/to/go_coder_llm_pipeline

# Verify Python can find modules
python -c "from model.config import get_small_config; print('OK')"
```

## What's NOT Tested in Quick Tests

These quick tests verify the pipeline works but don't validate:

- **Model quality**: Random/minimal data won't produce good code
- **Convergence**: Need full training to see actual learning
- **Validation accuracy**: Need proper held-out test set
- **Generation quality**: Need real evaluation metrics

For actual model quality, you need:
- 5000+ PRs (500M-1B tokens)
- Full training (10-20 hours on GPU)
- Proper evaluation suite

## Next Steps After Test

Once quick test passes:

1. ✅ Verify pipeline works
2. 📊 Collect full dataset (5000+ repos)
3. 🚀 Run full training (see TRAINING.md)
4. 📈 Evaluate on held-out test set
5. 🎯 Export to GGUF for deployment

## Recommended: Start with Synthetic Test

For fastest verification (15 minutes total):

```bash
# 1. Create synthetic data
python scripts/create_test_data.py --num-samples 100

# 2. Test training
python model/train.py \
  --config tiny \
  --train-data data/test/train.jsonl \
  --val-data data/test/val.jsonl \
  --batch-size 4 \
  --epochs 1 \
  --output-dir checkpoints/test

# 3. If that works, move to real data
```

This will catch any remaining bugs before committing to the longer real data pipeline.
