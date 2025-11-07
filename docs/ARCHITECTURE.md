# Architecture Overview

System design and component architecture for the Go Coder LLM pipeline.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Go Coder LLM Pipeline                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   GitHub     │───▶│  Tokenizer   │───▶│   Training   │───▶│  Deployment  │
│   Fetcher    │    │   Training   │    │   (PyTorch)  │    │   (GGUF)     │
│    (Go)      │    │ (SentencePiece)│   │   (ROCm)     │    │   (Ollama)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
      │                    │                    │                    │
      ▼                    ▼                    ▼                    ▼
  Raw JSONL          Corpus Text         Checkpoints           GGUF Model
```

## Component Architecture

### 1. Data Collection Layer (Go)

**Purpose**: Fetch and process GitHub repository data into structured format.

**Components**:

```
cmd/fetchdata/
    └── main.go                 # Entry point, orchestrates fetching

internal/github/
    ├── fetch.go                # GitHub API client
    └── parser.go               # PR/issue parsing logic

internal/tokenizer/
    └── prepare.go              # Corpus preparation

internal/types/
    └── types.go                # Shared data structures
```

**Data Flow**:
```
GitHub API → fetch.go → PRComment struct → prepare.go → corpus.txt
```

**Key Features**:
- Concurrent fetching with rate limiting
- Authenticated API requests
- License filtering
- Provenance tracking

### 2. Tokenization Layer (Python + SentencePiece)

**Purpose**: Train a BPE tokenizer that understands both Go code and natural language.

**Components**:

```
tokenizer/
    ├── train_tokenizer.sh      # Training script
    ├── go_coder_llm.model      # Trained SentencePiece model
    └── go_coder_llm.vocab      # Vocabulary (50K tokens)
```

**Tokenizer Strategy**:
- **Model Type**: BPE (Byte-Pair Encoding)
- **Vocab Size**: 50,000 tokens
- **Character Coverage**: 1.0 (full Unicode support)
- **Special Tokens**: `<pad>`, `<unk>`, `<s>`, `</s>`, `<REPO>`, `<PR_TITLE>`, `<CODE>`, etc.

**Why One Tokenizer**:
- Simpler pipeline (single vocabulary)
- No token-id mismatches
- Efficient for mixed code + natural language
- Modern tokenizers handle both well

### 3. Training Layer (PyTorch + ROCm)

**Purpose**: Train a GPT-style causal language model optimized for AMD GPUs.

**Components**:

```
model/
    ├── train.py                # Training loop (Hugging Face Trainer)
    ├── config.py               # Model configuration
    └── export_gguf.py          # GGUF export script

checkpoints/                    # Training checkpoints
logs/                          # Training logs and metrics
```

**Model Architecture** (GPT-2 style):

```python
{
    "vocab_size": 50000,
    "n_positions": 1024,        # Context length
    "n_embd": 768,              # Embedding dimension
    "n_layer": 12,              # Transformer blocks
    "n_head": 12,               # Attention heads
}
```

**Parameters**: ~125 million

**Memory Layout** (RX 6700 XT - 12 GB VRAM):
```
Model weights:        ~500 MB  (fp16)
Optimizer states:     ~1 GB    (AdamW)
Gradients:           ~500 MB
Activations:         ~8-9 GB  (batch_size=2, seq_len=1024)
─────────────────────────────
Total:               ~10-11 GB
```

### 4. Deployment Layer

**Purpose**: Export trained model for efficient CPU/GPU inference.

**Export Formats**:
- **GGUF**: For llama.cpp and Ollama
- **Safetensors**: For vLLM and OpenAI-compatible servers

**Inference Servers**:
- **Ollama**: Local inference with model management
- **LocalAI**: OpenAI-compatible API
- **vLLM**: High-throughput serving

## Data Schema

### GitHub Data Record

```go
type PRRecord struct {
    Repo         RepoMeta    // Repository metadata
    PRNumber     int         // Pull request number
    PRTitle      string      // PR title
    PRBody       string      // PR description
    PRAuthor     string      // Author username
    PRCreated    string      // Creation timestamp
    Comments     []Comment   // PR comments
    FilesChanged []FileChange // Code diffs
    FileTree     []string    // Repository structure
    Tags         []string    // Category tags (NAT/MIX/CODE)
}
```

### Text Serialization Format

Records are serialized for tokenizer training with special markers:

```
<REPO> owner/project
<PR_TITLE> Fix nil pointer in handler
<PR_BODY> This fixes a panic when r == nil
<FILES>
<FILE path="internal/server.go">
<BEFORE>
func handle(r *Request) {
    process(r)
}
</BEFORE>
<AFTER>
func handle(r *Request) {
    if r == nil {
        return
    }
    process(r)
}
</AFTER>
</FILE>
<COMMENTS>
reviewer: Consider checking nil
author: Thanks, updated
</COMMENTS>
```

## Data Categories

### NAT (Natural Language)
- README files
- Issue descriptions
- Documentation
- Comments without code

### MIX (Mixed Content)
- PR descriptions with code blocks
- Code review comments
- Technical discussions with inline code

### CODE (Code Only)
- Source files
- Diffs and patches
- Tests
- Configuration files

## Training Pipeline

### Stage 1: Data Collection

```
GitHub Repos (5000+)
    ↓
Filter by license (MIT, Apache, GPL, BSD)
    ↓
Fetch PRs, Issues, Code
    ↓
Parse and categorize (NAT/MIX/CODE)
    ↓
Serialize to JSONL
    ↓
data/processed/*.jsonl
```

### Stage 2: Tokenization

```
JSONL records
    ↓
Extract text fields
    ↓
Concatenate to corpus.txt
    ↓
Train SentencePiece (50K vocab)
    ↓
tokenizer/go_coder_llm.model
```

### Stage 3: Dataset Preparation

```
JSONL records + Tokenizer
    ↓
Tokenize text
    ↓
Pack into sequences (1024 tokens)
    ↓
Train/Val split (95/5 by repo)
    ↓
data/tokenized/train.bin
data/tokenized/val.bin
```

### Stage 4: Model Training

```
Tokenized dataset + Model config
    ↓
Initialize GPT-2 model (125M params)
    ↓
Train with AdamW + cosine LR
    ↓
Batch size: 2, Grad accum: 8
    ↓
FP16 training on ROCm
    ↓
Save checkpoints every 1000 steps
    ↓
checkpoints/step_N/
```

### Stage 5: Export & Serve

```
Final checkpoint
    ↓
Export to GGUF (quantized)
    ↓
Load in Ollama
    ↓
Inference API
```

## Training Strategy

### Curriculum Learning

**Option A: Single-Stage** (Recommended for POC)
- Train on mixed corpus (NAT + MIX + CODE) for 10B tokens
- Simple, fast validation of pipeline

**Option B: Multi-Stage** (For production)
1. **Stage 1**: General technical text (10-30B tokens)
2. **Stage 2**: Code-heavy pretraining (50-200B tokens)
3. **Stage 3**: Domain specialization on PRs (10-50B tokens)

### Hyperparameters (125M Model)

```python
{
    "learning_rate": 5e-4,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "effective_batch_size": 16,
    "max_steps": 195000,  # ~10B tokens
    "warmup_steps": 2000,
    "weight_decay": 0.1,
    "fp16": True,
    "save_steps": 1000,
    "eval_steps": 500,
    "logging_steps": 50,
}
```

### Optimization Techniques

1. **Gradient Accumulation**: Achieve larger effective batch size
2. **FP16 Training**: 2x faster, half memory
3. **Gradient Checkpointing**: Trade compute for memory
4. **8-bit Optimizer**: Reduce optimizer memory (bitsandbytes)

## Scalability

### Current Setup (RX 6700 XT)
- **Model**: 125M params
- **Context**: 1024 tokens
- **Batch**: 2 (effective: 16 with grad accum)
- **Speed**: ~5 tokens/sec
- **Time**: 5-7 days for 10B tokens

### Scaling Up

**To 350M params**:
- Increase `n_embd` to 1024, `n_layer` to 16
- Reduce batch to 1, increase grad accum to 16
- Training time: 2-3x longer

**Multi-GPU** (if adding GPUs):
- Use `accelerate` for distributed training
- Scale batch size proportionally
- Near-linear speedup

**Cloud Training** (alternative):
- Use AWS/Azure with AMD MI210/MI250
- 10-50x faster than single RX 6700 XT

## Monitoring & Logging

### Metrics Tracked
- **Loss**: Training and validation loss
- **Perplexity**: exp(loss)
- **Learning Rate**: Cosine decay schedule
- **GPU Utilization**: VRAM usage, compute %
- **Throughput**: Tokens/second

### Tools
- **TensorBoard**: Visualize training curves
- **WandB**: Experiment tracking (optional)
- **ROCm-SMI**: GPU monitoring

## Quality Assurance

### Data Quality
- License filtering (exclude proprietary)
- Deduplication (repo level)
- Filter generated/vendored code
- Validate UTF-8 encoding

### Model Quality
- Perplexity on validation set
- Code completion accuracy
- PR comment relevance
- Human evaluation

### Testing Strategy
1. **Unit Tests**: Data pipeline components
2. **Integration Tests**: End-to-end pipeline
3. **Smoke Tests**: Tiny model on small data
4. **Validation**: Full model on held-out repos

## Security & Privacy

### Data Handling
- Only public repositories
- No private/internal code
- License compliance checking
- Provenance tracking

### Model Safety
- No PII in training data
- Code vulnerability awareness
- Sanitize generated output
- Rate limiting on inference

## Performance Optimization

### Training Optimizations
1. Enable `torch.compile()` (PyTorch 2.1+)
2. Use `PYTORCH_HIP_ALLOC_CONF` for memory
3. Optimize data loading (prefetch, pin memory)
4. Use mixed precision (FP16)

### Inference Optimizations
1. Quantization (4-bit/8-bit GGUF)
2. KV-cache optimization
3. Batched inference
4. Flash Attention (if supported)

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data Collection | Go 1.24.9 | Fast, concurrent GitHub API client |
| API Client | net/http | GitHub REST API |
| Tokenization | SentencePiece | BPE tokenizer training |
| Training | PyTorch 2.1+ | Model training framework |
| GPU Support | ROCm 6.0 | AMD GPU acceleration |
| ML Framework | Transformers | GPT-2 implementation |
| Optimization | Accelerate | Distributed training |
| Logging | TensorBoard | Metrics visualization |
| Inference | Ollama | Model serving |
| Export | llama.cpp | GGUF conversion |

## Future Enhancements

1. **Multi-language Support**: Extend to Python, Rust, etc.
2. **Instruction Tuning**: Add SFT stage for chat format
3. **Code Execution**: Validate generated code
4. **RLHF**: Reinforcement learning from human feedback
5. **Larger Models**: Scale to 1B-7B parameters
6. **Streaming**: Real-time training data updates
7. **Fine-tuning**: Task-specific adaptation

## References

- [Implementation Plan](IMPLEMENTATION.md) - Detailed implementation guide
- [POC Guide](POC.md) - Proof of concept walkthrough
- [Setup Guide](SETUP.md) - Environment setup
- [Training Guide](TRAINING.md) - Training procedures
