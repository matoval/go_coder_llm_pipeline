# Go Coder LLM Pipeline

A specialized small language model focused on understanding Go code, pull requests, and developer communications. Designed to train on AMD RX 6700 XT GPU (12 GB VRAM) using ROCm.

## Overview

This project implements an end-to-end pipeline for training a code-focused language model:

1. **Data Collection** (Go) - Fetch GitHub repositories, PRs, issues, and code
2. **Tokenization** (SentencePiece) - Train a BPE tokenizer on Go code and natural language
3. **Training** (PyTorch + ROCm) - Train a GPT-style model optimized for AMD GPUs
4. **Export & Serving** (GGUF/Ollama) - Deploy for inference via Ollama or LocalAI

## Target Specifications

- **Model Size**: 125M parameters (fits in 12GB VRAM)
- **Context Length**: 1024 tokens
- **Vocab Size**: 50,000 tokens
- **Training Time**: ~5-7 days on RX 6700 XT for 10B tokens
- **GPU**: AMD RX 6700 XT (12 GB VRAM) with ROCm 6.0+

## Quick Start

### Prerequisites

- Go 1.24.9+
- Python 3.10+
- ROCm 6.0+ (for AMD GPU support)
- 12 GB VRAM (RX 6700 XT or equivalent)
- GitHub API token (for data collection)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/matoval/go_coder_llm_pipeline.git
cd go_coder_llm_pipeline

# 2. Set up Python environment (see docs/SETUP.md for ROCm)
python3 -m venv llm
source llm/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install transformers datasets accelerate sentencepiece

# 3. Build Go tools
go mod download
go build -o bin/fetchdata ./cmd/fetchdata

# 4. Set GitHub token
export GITHUB_TOKEN=your_token_here
```

### Running the Pipeline

```bash
# 1. Collect data from GitHub repositories
./bin/fetchdata

# 2. Train tokenizer
cd tokenizer
./train_tokenizer.sh

# 3. Train model
cd ../model
python train.py

# 4. Export to GGUF for inference
python export_gguf.py

# 5. Run with Ollama
ollama create golang-llm -f Modelfile
ollama run golang-llm
```

## Project Structure

```
go_coder_llm_pipeline/
├── cmd/
│   └── fetchdata/          # Main data collection binary
├── internal/
│   ├── github/             # GitHub API client and data fetching
│   ├── tokenizer/          # Corpus preparation utilities
│   └── types/              # Shared type definitions
├── model/
│   ├── train.py            # PyTorch training script
│   ├── config.py           # Model configuration
│   └── export_gguf.py      # GGUF export utility
├── tokenizer/
│   ├── train_tokenizer.sh  # SentencePiece training script
│   ├── go_coder_llm.model  # Trained tokenizer model
│   └── go_coder_llm.vocab  # Vocabulary file
├── scripts/
│   ├── run_all.sh          # End-to-end pipeline runner
│   └── evaluate.py         # Model evaluation utilities
└── docs/                   # Documentation
```

## Documentation

- [Setup Guide](docs/SETUP.md) - ROCm installation and environment setup for RX 6700 XT
- [Architecture](docs/ARCHITECTURE.md) - System design and component overview
- [Data Pipeline](docs/DATA_PIPELINE.md) - GitHub data collection and processing
- [Tokenizer](docs/TOKENIZER.md) - Tokenizer training and usage
- [Training](docs/TRAINING.md) - Model training guide and hyperparameters
- [Deployment](docs/DEPLOYMENT.md) - Export and serving options

## Features

- **Go-Optimized Tokenizer**: BPE tokenizer trained on Go code and developer language
- **PR-Aware Training**: Learns from pull request comments, diffs, and discussions
- **AMD GPU Support**: Optimized for ROCm and RX 6700 XT
- **Efficient Training**: Fits 125M model in 12GB VRAM with gradient accumulation
- **Production Ready**: Export to GGUF for Ollama/llama.cpp inference

## Data Sources

The model is trained on:
- Go repository code (filtered by license)
- Pull request titles, descriptions, and comments
- Code diffs and file changes
- README files and documentation
- Issue discussions

All data is collected from open-source repositories with permissive or copyleft licenses.

## Performance

On RX 6700 XT (12 GB VRAM):
- **Training Speed**: ~5 tokens/second
- **Batch Size**: 2 (with gradient accumulation of 8)
- **Effective Batch**: 16 sequences per step
- **Memory Usage**: ~10-11 GB VRAM
- **Training Time**: 5-7 days for 10B tokens

## License

This project respects the licenses of all source repositories. See [LICENSES.md](LICENSES.md) for details on dataset composition and provenance.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) with ROCm support
- [Transformers](https://huggingface.co/transformers/) by Hugging Face
- [SentencePiece](https://github.com/google/sentencepiece) for tokenization
- [Ollama](https://ollama.ai/) for inference serving

## References

- [Implementation Plan](docs/IMPLEMENTATION.md)
- [Proof of Concept Guide](docs/POC.md)
