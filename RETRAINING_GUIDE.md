# Retraining Guide - Path A: Full Data Collection

This guide walks you through collecting more training data and retraining the model from scratch while keeping it tiny (20-30M parameters).

## Why Retrain?

Your current model completed in ~5 hours instead of 2-3 days because:
- **Only 0.31B tokens** trained (need 4B tokens)
- **50 PR limit per repo** - most repos hit this cap
- Model learned to repeat patterns, not generate code

## Solution: Incremental Data Collection

Good news! You already have 3,022 repos with ~50 PRs each (~127K PRs total). We'll **keep this data** and just fetch more PRs from the same repos.

### Estimated Timeline

| Step | Time | Description |
|------|------|-------------|
| 1. Additional Data Collection | 12-24 hours | Fetch 450 more PRs per repo (50→500) |
| 2. Data Processing | 1-2 hours | Convert to hierarchical format |
| 3. Tokenization | 30 min | Prepare training data |
| 4. Model Training | 2-3 days | Train on ~4B tokens |
| **Total** | **3-4 days** | Full pipeline |

## Step-by-Step Instructions

### Option A: Automated (Recommended)

Run the complete pipeline with one command:

```bash
# On 192.168.1.188
cd ~/go_coder_llm_pipeline

# Make script executable
chmod +x scripts/retrain_pipeline.sh

# Run full pipeline (will take 3-4 days total)
./scripts/retrain_pipeline.sh
```

This will:
1. ✓ Backup your current data and checkpoints
2. ✓ Fetch additional PRs (up to 500 per repo)
3. ✓ Process data to hierarchical format
4. ✓ Prepare tokenized training data (1024 token sequences)
5. ✓ Train model from scratch (10 epochs)

### Option B: Manual (Step-by-Step Control)

If you prefer to run each step manually:

#### Step 1: Backup Current Data

```bash
cd ~/go_coder_llm_pipeline

# Create backup
mkdir -p backup_$(date +%Y%m%d)
cp -r data backup_$(date +%Y%m%d)/
cp -r checkpoints backup_$(date +%Y%m%d)/
```

#### Step 2: Collect More PRs (12-24 hours)

```bash
# Fetch up to 500 PRs per repo (incremental - keeps existing data)
./bin/fetchdata -max-prs 500 -resume

# Check progress
ls data/raw/*.jsonl | wc -l  # Should show ~3022 repos
```

**Monitor progress:**
```bash
# In another terminal, watch the data grow
watch -n 60 'du -sh data/raw && ls data/raw/*.jsonl | wc -l'
```

#### Step 3: Process Data (1-2 hours)

```bash
source llm/bin/activate

# Convert to hierarchical format
python scripts/process_data.py hierarchical \
    --input data/raw \
    --output data/processed

# Create corpus (optional, for tokenizer)
python scripts/process_data.py corpus \
    --input data/processed \
    --output data/corpus
```

#### Step 4: Prepare Training Data (30 min)

```bash
# Tokenize with longer sequences
python scripts/prepare_training_data.py \
    --input data/processed \
    --output data/tokenized \
    --tokenizer tokenizer/go_coder_llm.model \
    --max-length 1024 \
    --val-split 0.1

# Check dataset size
wc -l data/tokenized/*.jsonl
```

**Expected output:**
- Train: ~1.2M - 1.5M samples
- Val: ~130K - 165K samples
- Total tokens: ~4B

#### Step 5: Train Model (2-3 days)

```bash
# Clear old checkpoints
rm -rf checkpoints
mkdir -p checkpoints

# Start training
python model/train.py \
    --config small \
    --train-data data/tokenized/train.jsonl \
    --val-data data/tokenized/val.jsonl \
    --output-dir checkpoints \
    --batch-size 4 \
    --epochs 10 \
    --lr 3e-4
```

**Run in background with nohup:**
```bash
nohup python model/train.py \
    --config small \
    --train-data data/tokenized/train.jsonl \
    --val-data data/tokenized/val.jsonl \
    --output-dir checkpoints \
    --batch-size 4 \
    --epochs 10 \
    --lr 3e-4 > training.log 2>&1 &

# Check progress
tail -f training.log
```

## Monitoring Training

### Real-Time Monitor

```bash
# In a separate terminal
source llm/bin/activate
python scripts/monitor_training.py --interval 30
```

This shows:
- Current step
- Training speed (steps/sec, tokens/sec)
- Runtime
- Best checkpoint

### Check Training Health

```bash
python scripts/monitor_training.py --health-check
```

This checks for:
- Overfitting (if best checkpoint is much earlier than current epoch)
- Training progress

### Manual Progress Check

```bash
# Check latest checkpoints
ls -lth checkpoints/ | head -20

# Count steps completed
ls checkpoints/step_*.pt | wc -l

# Check which epoch we're on
ls checkpoints/epoch_*.pt
```

## Expected Results

### Data Collection Success

After fetching 500 PRs per repo, you should have:
- **~1.5M PRs total** (vs current 127K)
- **~15 GB raw data** (vs current 1.5 GB)
- **~4B tokens** after tokenization

### Training Success Indicators

✅ **Good signs:**
- Loss decreasing steadily
- Best checkpoint around epoch 3-7
- Generated code has structure (not repetition)

⚠️ **Warning signs:**
- Loss stops decreasing after epoch 2
- Best checkpoint at epoch 1-2, training continues
- Still generating repetitive patterns

## Testing the Retrained Model

After training completes:

```bash
# Test on sample problems
python scripts/inference.py \
    --checkpoint checkpoints/best.pt \
    --problems test_problems.jsonl \
    --output results.json \
    --device cpu \
    --temperature 0.7

# Check results
cat results.json | python -m json.tool | less
```

**What to look for:**
- Generated code has Go syntax structure
- Includes function definitions, not just repetition
- Responds to problem description (even if not perfect)

## Troubleshooting

### Data Collection Stops/Fails

```bash
# Check if GitHub token is still valid
echo $GITHUB_TOKEN

# Check rate limit
curl -H "Authorization: token $GITHUB_TOKEN" \
    https://api.github.com/rate_limit

# Resume collection (automatically resumes from last repo)
./bin/fetchdata -max-prs 500 -resume
```

### Training Too Slow

```bash
# Check GPU usage (if using AMD GPU)
rocm-smi

# Check CPU/RAM usage
htop

# Reduce batch size if out of memory
python model/train.py --batch-size 2 ...
```

### Model Still Generating Garbage

If after retraining the model still doesn't work:

1. **Check data quality**:
   ```bash
   # Sample some training data
   head -5 data/tokenized/train.jsonl | python -m json.tool
   ```

2. **Verify token count**:
   - Should have ~4B tokens total
   - If much less, may need to increase --max-prs further

3. **Try smaller learning rate**:
   ```bash
   python model/train.py --lr 1e-4 ...
   ```

## Cost/Benefit Analysis

### Path A (Current Plan)
- **Time**: 3-4 days
- **Data**: 13x more training data
- **Model Size**: 20-30M params (stays tiny!)
- **Quality**: Should generate valid Go code structures

### Alternative: Pre-trained Model
- **Time**: Few hours
- **Model Size**: 125M+ params (6x larger)
- **Quality**: Better, but defeats "tiny model" goal

## Next Steps After Training

Once you have a working model:

1. **Export to GGUF** (for deployment):
   ```bash
   python model/export_gguf.py --checkpoint checkpoints/best.pt
   ```

2. **Deploy with Ollama**:
   ```bash
   ollama create golang-llm -f Modelfile
   ollama run golang-llm
   ```

3. **Evaluate systematically**:
   - Create test suite of Go problems
   - Measure success rate
   - Compare with GPT-2/other models

## Questions?

Common questions:

**Q: Can I use a GPU?**
A: Yes! Change `--device cpu` to `--device cuda`. Training will be 10-20x faster.

**Q: Will this definitely work?**
A: With 4B tokens and 10 epochs, the model should learn basic code structure. Quality depends on data quality and may need tuning.

**Q: Can I stop and resume training?**
A: Yes! Use `--resume checkpoints/latest.pt` to continue from last checkpoint.

**Q: What if I want the model even smaller?**
A: Use `--config tiny` instead of `--config small` for a 10-15M param model.
