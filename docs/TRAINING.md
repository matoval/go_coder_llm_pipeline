# Training Guide

Complete guide for training the Go Coder LLM on AMD RX 6700 XT with ROCm.

## Overview

Train a 125M parameter GPT-style model optimized for Go code understanding using PyTorch with ROCm on AMD GPU.

**Target**: 10B tokens in 5-7 days on RX 6700 XT (12 GB VRAM)

## Training Specifications

### Model Configuration

```python
# model/config.py
from transformers import GPT2Config

config = GPT2Config(
    vocab_size=50000,           # From tokenizer
    n_positions=1024,           # Context length
    n_embd=768,                 # Embedding dimension
    n_layer=12,                 # Transformer layers
    n_head=12,                  # Attention heads (n_embd // 64)
    n_inner=3072,               # FFN dimension (4 * n_embd)
    activation_function="gelu", # Activation
    resid_pdrop=0.1,           # Residual dropout
    embd_pdrop=0.1,            # Embedding dropout
    attn_pdrop=0.1,            # Attention dropout
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=3,
)

# Total parameters: ~125 million
```

### Training Hyperparameters

```python
# model/train.py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="checkpoints",
    overwrite_output_dir=True,

    # Batch configuration
    per_device_train_batch_size=2,      # Fits in 12 GB VRAM
    gradient_accumulation_steps=8,      # Effective batch: 16
    per_device_eval_batch_size=4,

    # Learning rate & schedule
    learning_rate=5e-4,
    lr_scheduler_type="cosine",
    warmup_steps=2000,
    weight_decay=0.1,

    # Training duration
    num_train_epochs=3,
    max_steps=195000,                   # ~10B tokens

    # Precision
    fp16=True,                          # ROCm supports FP16
    fp16_opt_level="O1",

    # Logging & evaluation
    logging_steps=50,
    eval_steps=500,
    save_steps=1000,
    save_total_limit=3,

    # Optimization
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,

    # Misc
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    report_to="tensorboard",
    load_best_model_at_end=False,
)
```

## Implementation

### Complete Training Script

Create or update `model/train.py`:

```python
#!/usr/bin/env python3
"""
Train Go Coder LLM on AMD RX 6700 XT with ROCm
"""

import os
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_gpu():
    """Verify GPU is available and recognized"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Check ROCm installation.")

    device = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    logger.info(f"GPU: {device}")
    logger.info(f"VRAM: {vram:.2f} GB")

    if vram < 11:
        logger.warning("Less than 12 GB VRAM detected. May need to reduce batch size.")

def load_tokenizer(path="tokenizer"):
    """Load trained tokenizer"""
    logger.info(f"Loading tokenizer from {path}")

    tokenizer = GPT2TokenizerFast(
        tokenizer_file=os.path.join(path, "go_coder_llm.model"),
        vocab_file=os.path.join(path, "go_coder_llm.vocab"),
        model_max_length=1024,
    )

    # Set special tokens
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.bos_token = "<s>"
    tokenizer.unk_token = "<unk>"

    return tokenizer

def create_model(tokenizer):
    """Initialize GPT-2 model from config"""
    logger.info("Creating model from config")

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = GPT2LMHeadModel(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    return model

def load_dataset(path="data/tokenized"):
    """Load preprocessed tokenized dataset"""
    logger.info(f"Loading dataset from {path}")

    dataset = load_from_disk(path)

    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Validation samples: {len(dataset['validation'])}")

    return dataset

def main():
    # Verify GPU
    verify_gpu()

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Create model
    model = create_model(tokenizer)

    # Load dataset
    dataset = load_dataset()

    # Data collator (for causal LM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="checkpoints",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=4,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=2000,
        weight_decay=0.1,
        num_train_epochs=3,
        max_steps=195000,
        fp16=True,
        logging_dir="logs",
        logging_steps=50,
        eval_steps=500,
        save_steps=1000,
        save_total_limit=3,
        evaluation_strategy="steps",
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="tensorboard",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model("models/go_coder_llm_final")
    tokenizer.save_pretrained("models/go_coder_llm_final")

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Training complete!")

if __name__ == "__main__":
    main()
```

Make executable:

```bash
chmod +x model/train.py
```

### Configuration File

Create `model/config.py`:

```python
"""Model configuration"""

from transformers import GPT2Config

def get_config(vocab_size=50000):
    """Get model configuration for 125M parameter model"""
    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
    )

# Alternative configurations

def get_config_tiny(vocab_size=50000):
    """50M params - for testing"""
    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=1024,
        n_embd=512,
        n_layer=8,
        n_head=8,
    )

def get_config_small(vocab_size=50000):
    """125M params - default"""
    return get_config(vocab_size)

def get_config_medium(vocab_size=50000):
    """350M params - requires batch_size=1"""
    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=1024,
        n_embd=1024,
        n_layer=16,
        n_head=16,
        n_inner=4096,
    )
```

## Dataset Preparation

### Convert JSONL to Hugging Face Dataset

```python
#!/usr/bin/env python3
"""
Prepare dataset for training
"""

import json
from datasets import Dataset, DatasetDict
from transformers import GPT2TokenizerFast
import sentencepiece as spm

def load_jsonl(path):
    """Load JSONL file"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def tokenize_function(examples, tokenizer):
    """Tokenize text examples"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt",
    )

def main():
    # Load tokenizer
    tokenizer = GPT2TokenizerFast(
        tokenizer_file="tokenizer/go_coder_llm.model",
        model_max_length=1024,
    )
    tokenizer.pad_token = "<pad>"

    # Load data
    train_data = load_jsonl("data/processed/train.jsonl")
    val_data = load_jsonl("data/processed/validation.jsonl")

    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # Tokenize
    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    val_tokenized = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    # Create dataset dict
    dataset_dict = DatasetDict({
        "train": train_tokenized,
        "validation": val_tokenized,
    })

    # Save
    dataset_dict.save_to_disk("data/tokenized")
    print("Dataset saved to data/tokenized")

if __name__ == "__main__":
    main()
```

## Running Training

### Start Training

```bash
# Activate virtual environment
source llm/bin/activate

# Set ROCm environment variables
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

# Run training
cd model
python train.py 2>&1 | tee ../logs/training.log
```

### Monitor Training

**Terminal 1 - Training logs**:
```bash
tail -f logs/training.log
```

**Terminal 2 - GPU monitoring**:
```bash
watch -n 1 rocm-smi
```

**Terminal 3 - TensorBoard**:
```bash
tensorboard --logdir logs --port 6006
```

Open browser: http://localhost:6006

## Memory Optimization

### If OOM (Out of Memory)

**Option 1: Reduce Batch Size**

```python
per_device_train_batch_size=1,
gradient_accumulation_steps=16,  # Keep effective batch size
```

**Option 2: Enable Gradient Checkpointing**

```python
# In train.py, before creating Trainer
model.gradient_checkpointing_enable()
```

**Option 3: Use 8-bit Optimizer**

```bash
pip install bitsandbytes
```

```python
# In TrainingArguments
optim="adamw_8bit",
```

**Option 4: Reduce Sequence Length**

```python
# In config
n_positions=512,  # Instead of 1024
```

## Performance Tuning

### Expected Performance

| Metric | Value |
|--------|-------|
| Throughput | ~5 tokens/sec |
| Batch time | ~200 ms/step |
| Epoch time | ~40-50 hours |
| Total time | 5-7 days (3 epochs) |
| VRAM usage | 10-11 GB |

### Optimization Tips

**1. Increase DataLoader Workers**

```python
dataloader_num_workers=8,  # More parallel data loading
```

**2. Enable Tensor Cores (if supported)**

```python
# In train.py
torch.set_float32_matmul_precision('medium')
```

**3. Use torch.compile (PyTorch 2.0+)**

```python
# In train.py, before training
model = torch.compile(model)
```

**4. Optimize Data Loading**

```python
# Pin memory for faster CPU->GPU transfer
dataloader_pin_memory=True,

# Prefetch batches
dataloader_prefetch_factor=2,
```

## Checkpointing & Resume

### Auto-Save Checkpoints

Checkpoints saved every 1000 steps:

```
checkpoints/
├── checkpoint-1000/
├── checkpoint-2000/
├── checkpoint-3000/
└── ...
```

### Resume Training

```python
# In train.py
trainer.train(resume_from_checkpoint="checkpoints/checkpoint-5000")
```

Or from command line:

```bash
python train.py --resume_from_checkpoint checkpoints/checkpoint-5000
```

## Evaluation

### During Training

Validation loss computed every 500 steps.

### Post-Training Evaluation

```python
# scripts/evaluate.py
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_from_disk

def calculate_perplexity(model, dataset, tokenizer):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataset:
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_tokens += batch["attention_mask"].sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

# Load model and dataset
model = GPT2LMHeadModel.from_pretrained("checkpoints/checkpoint-195000")
tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer")
dataset = load_from_disk("data/tokenized")

# Calculate perplexity
ppl = calculate_perplexity(model, dataset["validation"], tokenizer)
print(f"Perplexity: {ppl:.2f}")
```

### Code Generation Test

```python
from transformers import pipeline

# Load model
generator = pipeline(
    "text-generation",
    model="checkpoints/checkpoint-195000",
    tokenizer="tokenizer",
    device=0,
)

# Test prompts
prompts = [
    "func readFile(path string) error {",
    "// parseJSON parses a JSON string\nfunc parseJSON(data []byte)",
    "<PR_TITLE> Fix nil pointer dereference\n<PR_BODY>",
]

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    outputs = generator(
        prompt,
        max_length=200,
        num_return_sequences=1,
        temperature=0.8,
        top_p=0.95,
    )
    print(f"Generated: {outputs[0]['generated_text']}")
```

## Troubleshooting

### Training Loss Not Decreasing

**Possible causes**:
- Learning rate too high/low
- Data quality issues
- Model initialization problems

**Solutions**:
- Reduce learning rate to 1e-4
- Check dataset for corruption
- Verify tokenization quality

### GPU Utilization Low

**Symptoms**: <80% GPU usage

**Solutions**:
- Increase `dataloader_num_workers`
- Enable `dataloader_pin_memory`
- Increase batch size if VRAM allows

### Training Diverges (Loss → NaN)

**Symptoms**: Loss becomes NaN

**Solutions**:
- Reduce learning rate
- Enable gradient clipping (already set)
- Check for FP16 overflow (use FP32)

### Slow Training

**Symptoms**: <3 tokens/sec

**Solutions**:
- Update ROCm drivers
- Enable `torch.compile()`
- Reduce sequence length
- Check thermal throttling

## Advanced: Distributed Training

### Multi-GPU (if adding more GPUs)

Use Accelerate:

```bash
accelerate config
```

```yaml
# accelerate_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 2
gpu_ids: 0,1
mixed_precision: fp16
```

Run:

```bash
accelerate launch --config_file accelerate_config.yaml model/train.py
```

## Metrics to Track

### Training Metrics

- **Loss**: Should decrease steadily
- **Learning Rate**: Follows cosine schedule
- **Gradient Norm**: Should be <1.0 (clipped)
- **GPU Utilization**: >80%
- **VRAM Usage**: 10-11 GB

### Validation Metrics

- **Perplexity**: exp(val_loss), lower is better
- **Validation Loss**: Should track training loss

### Target Metrics

| Checkpoint | Perplexity | Validation Loss |
|------------|-----------|-----------------|
| 10K steps  | ~50-100   | ~4.0-4.5        |
| 50K steps  | ~20-40    | ~3.0-3.7        |
| 100K steps | ~10-20    | ~2.3-3.0        |
| 195K steps | ~8-15     | ~2.0-2.7        |

## Next Steps

After training completes:
1. Review [Deployment Guide](DEPLOYMENT.md) for export and serving
2. Run evaluation metrics
3. Test code generation quality
4. Export to GGUF for inference

## Resources

- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
