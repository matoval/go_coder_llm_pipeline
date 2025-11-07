# Environment Setup Guide

Complete guide for setting up the Go Coder LLM pipeline on AMD RX 6700 XT with ROCm.

## System Requirements

### Hardware
- **GPU**: AMD RX 6700 XT (12 GB VRAM) or equivalent
- **RAM**: 32 GB recommended (16 GB minimum)
- **Storage**: 500 GB+ free space for datasets and checkpoints
- **CPU**: Multi-core processor (8+ cores recommended)

### Software
- **OS**: Fedora 40+, Ubuntu 22.04+, or Arch Linux
- **Kernel**: 6.8+ (for best ROCm compatibility)
- **Go**: 1.24.9+
- **Python**: 3.10+
- **ROCm**: 6.0+

## ROCm Installation

### Fedora Installation

```bash
# Install ROCm packages
sudo dnf install rocm-dev rocm-libs hipblas rocblas miopen-hip rocm-smi

# Add user to render and video groups
sudo usermod -a -G render,video $USER

# Reboot to apply group changes
sudo reboot
```

### Ubuntu Installation

```bash
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo dpkg -i amdgpu-install_6.0.60000-1_all.deb
sudo apt update

# Install ROCm
sudo amdgpu-install --usecase=rocm

# Add user to groups
sudo usermod -a -G render,video $USER
sudo reboot
```

### Verify ROCm Installation

After reboot, verify your GPU is detected:

```bash
# Check ROCm version
rocm-smi --showproductname

# Expected output: AMD Radeon RX 6700 XT

# Check ROCm info
rocminfo | grep "Name:"
```

## Python Environment Setup

### Create Virtual Environment

```bash
# Create and activate virtual environment
python3 -m venv llm
source llm/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Install PyTorch with ROCm Support

```bash
# Install PyTorch for ROCm 6.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

### Verify PyTorch GPU Support

```python
python - <<'PY'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
PY
```

Expected output:
```
PyTorch version: 2.x.x+rocm6.0
CUDA available: True
GPU: AMD Radeon RX 6700 XT
VRAM: 12.00 GB
```

### Install ML Dependencies

```bash
# Core ML libraries
pip install transformers==4.35.0
pip install datasets==2.14.0
pip install accelerate==0.24.0
pip install sentencepiece==0.1.99

# Training utilities
pip install tensorboard
pip install wandb  # optional, for experiment tracking

# Data processing
pip install jsonlines
pip install tqdm
```

## Go Environment Setup

### Install Go

```bash
# Download and install Go 1.24.9
wget https://go.dev/dl/go1.24.9.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.24.9.linux-amd64.tar.gz

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin

# Verify installation
go version
```

### Install Go Dependencies

```bash
cd go_coder_llm_pipeline
go mod download
go mod verify
```

## GitHub API Setup

### Create GitHub Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Select scopes:
   - `public_repo` (for public repository access)
   - `read:org` (if fetching from organizations)
4. Generate and copy the token

### Configure Token

```bash
# Set environment variable
export GITHUB_TOKEN=your_token_here

# Add to shell profile for persistence
echo 'export GITHUB_TOKEN=your_token_here' >> ~/.bashrc
source ~/.bashrc
```

## SentencePiece Installation

### Install from Package Manager

```bash
# Fedora
sudo dnf install sentencepiece

# Ubuntu
sudo apt install sentencepiece

# Or install via pip
pip install sentencepiece
```

### Verify Installation

```bash
spm_train --help
```

## Optional: GGUF Conversion Tools

For exporting to GGUF format for Ollama:

```bash
# Install llama.cpp (for GGUF conversion)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Add to PATH
export PATH=$PATH:$(pwd)
```

## Directory Structure Setup

```bash
# Create necessary directories
cd go_coder_llm_pipeline
mkdir -p data/{raw,processed,corpus}
mkdir -p checkpoints
mkdir -p models
mkdir -p logs
```

## Performance Tuning

### ROCm Environment Variables

Add these to your shell profile for optimal performance:

```bash
# Enable FP16 for faster training
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For RX 6700 XT

# Optimize memory allocation
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

# Enable TF32 for matrix operations (if supported)
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library/
```

### System Limits

```bash
# Increase file descriptor limits
ulimit -n 65535

# Check current limits
ulimit -a
```

## Verification Checklist

Before starting training, verify:

- [ ] ROCm installed and GPU detected (`rocm-smi`)
- [ ] PyTorch recognizes GPU (`torch.cuda.is_available()`)
- [ ] Go environment working (`go version`)
- [ ] GitHub token configured (`echo $GITHUB_TOKEN`)
- [ ] SentencePiece installed (`spm_train --help`)
- [ ] Python packages installed (`pip list`)
- [ ] Directories created
- [ ] ~12 GB VRAM available

## Troubleshooting

### GPU Not Detected

```bash
# Check GPU is visible
lspci | grep -i vga

# Verify kernel modules
lsmod | grep amdgpu

# Check ROCm stack
rocminfo

# Reinstall AMDGPU drivers if needed
sudo amdgpu-install --uninstall
sudo amdgpu-install --usecase=rocm
```

### PyTorch CUDA Not Available

```bash
# Verify ROCm version matches PyTorch
pip show torch | grep Version

# Reinstall PyTorch for correct ROCm version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

### Out of Memory Errors

If you encounter OOM during training:

1. Reduce batch size in `model/config.py`
2. Increase gradient accumulation steps
3. Enable gradient checkpointing
4. Use 8-bit optimizer (`pip install bitsandbytes`)

### GitHub API Rate Limits

```bash
# Check rate limit status
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit

# Use authenticated requests (automatically done if GITHUB_TOKEN is set)
```

## Next Steps

After completing setup:
1. Read [Data Pipeline](DATA_PIPELINE.md) to start collecting data
2. Review [Architecture](ARCHITECTURE.md) to understand the system
3. Follow [Training](TRAINING.md) for model training

## Useful Commands

```bash
# Monitor GPU usage during training
watch -n 1 rocm-smi

# Check GPU temperature and power
rocm-smi --showtemp --showpower

# Monitor training logs
tail -f logs/training.log

# Activate virtual environment
source llm/bin/activate
```

## Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Guide](https://pytorch.org/get-started/locally/)
- [GitHub API Documentation](https://docs.github.com/en/rest)
- [SentencePiece GitHub](https://github.com/google/sentencepiece)
