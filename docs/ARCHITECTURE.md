# Architecture Overview

System design and component architecture for the Hierarchical Recursive Go Coder (HRGC) pipeline.

## System Overview

**Novel Approach**: This system uses Hierarchical Reasoning Modules (HRM) and Tree-based Recursive Modeling (TRM) to create a **tiny but highly accurate** code generation model specifically for Go.

```
┌─────────────────────────────────────────────────────────────────────┐
│            Hierarchical Recursive Go Coder Pipeline                 │
│         (HRM/TRM-based Recursive Reasoning Architecture)            │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   GitHub     │───▶│  Hierarchical│───▶│  Recursive   │───▶│  Deployment  │
│   Fetcher    │    │  Tokenizer   │    │  Training    │    │   (GGUF)     │
│    (Go)      │    │ (Plan+Code)  │    │   (HRM)      │    │   (Ollama)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
      │                    │                    │                    │
      ▼                    ▼                    ▼                    ▼
  Annotated          Planning+Code        Recursive Model       Tiny GGUF
  PR Tasks           Vocabulary           (20-30M params)       (~80MB)
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

### 2. Tokenization Layer (Python + SentencePiece + Hierarchical Extensions)

**Purpose**: Train a BPE tokenizer that understands Go code, natural language, AND hierarchical reasoning structures.

**Components**:

```
tokenizer/
    ├── train_tokenizer.sh      # Training script
    ├── go_coder_llm.model      # Trained SentencePiece model
    ├── go_coder_llm.vocab      # Vocabulary (50K tokens)
    └── plan_extractor.py       # Extracts high-level plans from PRs
```

**Tokenizer Strategy**:
- **Model Type**: BPE (Byte-Pair Encoding)
- **Vocab Size**: 50,000 tokens
- **Character Coverage**: 1.0 (full Unicode support)
- **Special Tokens** (Extended for HRM):
  - Base: `<pad>`, `<unk>`, `<s>`, `</s>`
  - Structural: `<REPO>`, `<PR_TITLE>`, `<CODE>`, `<CONTEXT>`
  - **NEW - Planning**: `<PLAN>`, `<STEP>`, `</PLAN>`
  - **NEW - Intents**: `<INTENT:FIX>`, `<INTENT:ADD>`, `<INTENT:REFACTOR>`, `<INTENT:OPTIMIZE>`
  - **NEW - Targets**: `<TARGET:func>`, `<TARGET:type>`, `<TARGET:interface>`
  - **NEW - Validation**: `<VALIDATE>`, `<SYNTAX_OK>`, `<TEST_PASS>`, `<REFINE>`

**Hierarchical Tokenization**:
- **Plan Level**: High-level intents and structural changes
- **Code Level**: Actual Go tokens and syntax
- **Validation Level**: Feedback tokens for recursive refinement

**Why Extended Vocabulary**:
- Enables explicit planning representation
- Supports recursive refinement loops
- Provides validation feedback signals
- Maintains backward compatibility with code/NL

### 3. Training Layer (PyTorch + ROCm + HRM/TRM)

**Purpose**: Train a hierarchical recursive reasoning model optimized for Go code generation.

**Novel Architecture**: Inspired by HRM (arXiv:2506.21734) and TRM (arXiv:2510.04871)

**Components**:

```
model/
    ├── hrm_model.py            # Hierarchical reasoning architecture
    ├── train.py                # Recursive training loop
    ├── config.py               # HRM configuration
    ├── validators.py           # Syntax/test validation hooks
    └── export_gguf.py          # GGUF export script

checkpoints/                    # Training checkpoints
logs/                          # Training logs and metrics
```

**Model Architecture** (Hierarchical Recursive):

```python
{
    # High-Level Planning Module (Abstract, Slow)
    "planner": {
        "vocab_size": 50000,
        "n_positions": 512,     # Plan length
        "n_embd": 256,          # Compact embedding
        "n_layer": 3,           # Lightweight
        "n_head": 8,
        "params": "~10-12M"
    },

    # Low-Level Generation Module (Detailed, Fast)
    "generator": {
        "vocab_size": 50000,
        "n_positions": 1024,    # Code length
        "n_embd": 256,          # Compact embedding
        "n_layer": 3,           # Lightweight
        "n_head": 8,
        "params": "~10-12M"
    },

    # Cross-Module Communication
    "cross_attention": "256 dim",
    "refinement_controller": "3-way classifier (continue/refine/done)",

    # Recursive Loop
    "max_refinement_iterations": 5,
    "early_stopping": "validation-based"
}
```

**Total Parameters**: ~20-30 million (**6x smaller than GPT-2 125M!**)

**Memory Layout** (RX 6700 XT - 12 GB VRAM):
```
Model weights:        ~80 MB   (fp16) - 6x smaller!
Optimizer states:     ~320 MB  (AdamW)
Gradients:           ~80 MB
Activations:         ~2-3 GB  (batch_size=8, hierarchical)
Validation overhead:  ~500 MB (syntax checker)
─────────────────────────────
Total:               ~3-4 GB  ← MUCH more headroom!
```

**Key Advantages**:
- **6x parameter reduction**: 20-30M vs 125M params
- **Recursive refinement**: Iteratively improves solutions
- **Hierarchical reasoning**: Separates planning from implementation
- **Validation-aware**: Built-in syntax/test checking
- **Faster training**: Smaller model = faster convergence

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

### Hierarchical Serialization Format

Records are serialized with hierarchical planning structure for HRM training:

```
<REPO> owner/project
<PR_TITLE> Fix nil pointer in handler
<PR_BODY> This fixes a panic when r == nil

<PLAN>
<STEP> <INTENT:FIX> Identify nil pointer vulnerability in handler function
<STEP> <TARGET:func> handle() in internal/server.go
<STEP> <INTENT:ADD> Add defensive nil check at function entry point
<STEP> Preserve existing process() logic
</PLAN>

<CONTEXT>
<FILE path="internal/server.go">
<BEFORE>
func handle(r *Request) {
    process(r)
}
</BEFORE>
</FILE>

<CODE>
<AFTER>
func handle(r *Request) {
    if r == nil {
        return
    }
    process(r)
}
</AFTER>
</CODE>

<VALIDATE>
<SYNTAX_OK> true
<TEST_PASS> true
</VALIDATE>

<COMMENTS>
reviewer: Consider checking nil
author: Thanks, updated
</COMMENTS>
```

**Hierarchical Structure**:
1. **PLAN**: High-level reasoning (used by planner module)
2. **CONTEXT**: Existing code (input to both modules)
3. **CODE**: Implementation (generated by generator module)
4. **VALIDATE**: Feedback (for recursive refinement)

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

## Training Strategy (HRM/TRM-based)

### Hierarchical Recursive Training

**Novel Approach**: Instead of traditional autoregressive training, we use a multi-task objective that trains planning and generation jointly with recursive refinement.

**Phase 1: Joint Hierarchical Training** (Recommended for POC)
- Train planner and generator modules simultaneously
- Multi-task loss: 40% planning + 40% code generation + 20% refinement
- Data: 5000+ repos with hierarchically annotated PRs
- Expected: 2-3 days on RX 6700 XT (vs 5-7 days for GPT-2)

**Phase 2: Recursive Refinement Training**
- Introduce deliberate errors and partial solutions
- Train model to detect issues and iteratively refine
- Use validation feedback (syntax errors, test failures) as training signal
- Learn when to continue vs when solution is complete

**Phase 3: Specialization** (For production)
1. **Stage 1**: Basic Go patterns (error handling, interfaces) - 1B tokens
2. **Stage 2**: Complex refactoring and bug fixes - 2B tokens
3. **Stage 3**: PR-specific reasoning - 1B tokens

**Total Training**: ~4B tokens (vs 10B for standard LLM approach)

### Hyperparameters (HRM 20-30M Model)

```python
{
    # Training config
    "learning_rate": 3e-4,        # Slightly lower for stability
    "batch_size": 8,              # 4x larger! (smaller model)
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 32,   # 2x larger effective batch
    "max_steps": 125000,          # ~4B tokens (vs 10B)
    "warmup_steps": 1000,
    "weight_decay": 0.1,
    "fp16": True,
    "save_steps": 1000,
    "eval_steps": 500,
    "logging_steps": 50,

    # HRM-specific
    "planner_lr_multiplier": 1.2,  # Planner trains slightly faster
    "generator_lr_multiplier": 0.8, # Generator more conservative
    "refinement_warmup": 5000,     # Start refinement training later
    "max_refinement_iterations": 5,
    "validation_frequency": 100,   # Run syntax checks every N steps

    # Multi-task loss weights
    "plan_loss_weight": 0.4,
    "code_loss_weight": 0.4,
    "refinement_loss_weight": 0.2,
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

1. **Multi-language Support**: Extend HRM to Python, Rust, etc.
2. **Deeper Recursion**: Increase refinement iterations with verification
3. **Code Execution**: Run tests in refinement loop
4. **Tree-based Search**: Add TRM-style tree search for complex problems
5. **Larger HRM Models**: Scale to 100M params (still small vs GPT)
6. **Online Learning**: Real-time updates from new PRs
7. **Fine-tuning**: Task-specific adaptation (bug fixes vs new features)
8. **Multi-step Planning**: Extend planner for multi-file changes

## Advantages Over Traditional LLMs

### Size & Efficiency
- **6x smaller**: 20-30M vs 125M params
- **3x faster training**: 2-3 days vs 5-7 days
- **Less data needed**: 4B vs 10B tokens
- **Fits in 3-4GB VRAM**: vs 10-11GB

### Accuracy & Quality
- **Recursive refinement**: Catches and fixes errors
- **Validation-aware**: Built-in syntax/test checking
- **Hierarchical reasoning**: Plans before implementing
- **Interpretable**: Can inspect planning steps

### Practical Benefits
- **Cheaper to train**: Less compute, less time
- **Easier to experiment**: Fast iteration cycles
- **More accessible**: Runs on consumer GPUs
- **Better for code**: Reasoning structure matches coding workflow

## References

### Internal Documentation
- [HRM/TRM Integration](HRM_TRM_INTEGRATION.md) - Detailed design of hierarchical recursive architecture
- [Implementation Plan](IMPLEMENTATION.md) - Detailed implementation guide
- [POC Guide](POC.md) - Proof of concept walkthrough
- [Setup Guide](SETUP.md) - Environment setup
- [Training Guide](TRAINING.md) - Training procedures

### Research Papers
- [HRM: Hierarchical Reasoning Modules](https://arxiv.org/abs/2506.21734) - Original HRM paper
- [TRM: Tiny Recursive Model](https://arxiv.org/abs/2510.04871) - Tree-based recursive reasoning
