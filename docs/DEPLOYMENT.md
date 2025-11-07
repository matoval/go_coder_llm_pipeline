# Deployment Guide

Complete guide for exporting and serving the trained Go Coder LLM model.

## Overview

After training, export the model for efficient inference using GGUF format and serve with Ollama or LocalAI.

## Export Options

### Option 1: GGUF (Recommended)

- **Format**: GGUF (GPT-Generated Unified Format)
- **Use Cases**: Ollama, llama.cpp, LocalAI
- **Benefits**: Quantization, fast CPU/GPU inference, small file size
- **Platforms**: Cross-platform (Linux, macOS, Windows)

### Option 2: Hugging Face Format

- **Format**: Safetensors / PyTorch
- **Use Cases**: Transformers, vLLM, TGI (Text Generation Inference)
- **Benefits**: Native Python integration, GPU-optimized
- **Platforms**: Python environments

### Option 3: ONNX

- **Format**: ONNX Runtime
- **Use Cases**: Production deployments, edge devices
- **Benefits**: Hardware-agnostic, optimized inference
- **Platforms**: Any ONNX-compatible runtime

## Export to GGUF

### Step 1: Install llama.cpp

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with ROCm support
make LLAMA_HIPBLAS=1

# Or CPU only
make
```

### Step 2: Convert Model

Update `model/export_gguf.py`:

```python
#!/usr/bin/env python3
"""
Export trained model to GGUF format
"""

import sys
import os
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

def export_to_gguf(checkpoint_dir, output_path):
    """
    Export Hugging Face model to GGUF

    Args:
        checkpoint_dir: Path to trained model checkpoint
        output_path: Output GGUF file path
    """
    print(f"Loading model from {checkpoint_dir}")

    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)
    tokenizer = GPT2TokenizerFast.from_pretrained(checkpoint_dir)

    # Save in Hugging Face format first
    temp_dir = "temp_export"
    os.makedirs(temp_dir, exist_ok=True)

    model.save_pretrained(temp_dir, safe_serialization=True)
    tokenizer.save_pretrained(temp_dir)

    print(f"Model saved to {temp_dir}")
    print(f"\nTo convert to GGUF, run:")
    print(f"  cd llama.cpp")
    print(f"  python convert.py ../{temp_dir} --outtype f16 --outfile ../{output_path}")
    print(f"\nTo quantize (optional, reduces size):")
    print(f"  ./quantize ../{output_path} ../{output_path.replace('.gguf', '.q4_0.gguf')} q4_0")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python export_gguf.py <checkpoint_dir> <output_path>")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    output_path = sys.argv[2]

    export_to_gguf(checkpoint_dir, output_path)
```

Run export:

```bash
# Export checkpoint to Hugging Face format
python model/export_gguf.py checkpoints/checkpoint-195000 models/go_coder_llm.gguf

# Convert to GGUF (FP16)
cd llama.cpp
python convert.py ../temp_export --outtype f16 --outfile ../models/go_coder_llm_f16.gguf
```

### Step 3: Quantize (Optional)

Reduce model size with quantization:

```bash
cd llama.cpp

# 4-bit quantization (~3 GB)
./quantize ../models/go_coder_llm_f16.gguf ../models/go_coder_llm_q4_0.gguf q4_0

# 5-bit quantization (~4 GB, better quality)
./quantize ../models/go_coder_llm_f16.gguf ../models/go_coder_llm_q5_0.gguf q5_0

# 8-bit quantization (~6 GB, near-original quality)
./quantize ../models/go_coder_llm_f16.gguf ../models/go_coder_llm_q8_0.gguf q8_0
```

**Quantization Trade-offs**:

| Quantization | Size  | Quality | Speed |
|--------------|-------|---------|-------|
| F16          | ~12 GB | Best    | Base  |
| Q8_0         | ~6 GB  | Great   | 1.5x  |
| Q5_0         | ~4 GB  | Good    | 2x    |
| Q4_0         | ~3 GB  | OK      | 2.5x  |

## Serving with Ollama

### Install Ollama

```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
systemctl start ollama
```

### Create Modelfile

Create `Modelfile`:

```dockerfile
FROM ./models/go_coder_llm_q4_0.gguf

# Set parameters
PARAMETER temperature 0.8
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 1024

# Set system prompt
SYSTEM """
You are a Go programming expert. You help developers write, review, and understand Go code.
You provide clear explanations and follow Go best practices.
"""

# Template for chat format
TEMPLATE """
{{ if .System }}### System:
{{ .System }}
{{ end }}

### User:
{{ .Prompt }}

### Assistant:
"""
```

### Create Ollama Model

```bash
# Create model from Modelfile
ollama create golang-llm -f Modelfile

# List models
ollama list

# Test model
ollama run golang-llm
```

### Example Usage

**CLI**:

```bash
ollama run golang-llm "Write a Go function that reads a JSON file"
```

**API**:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "golang-llm",
  "prompt": "Write a Go function that validates email addresses",
  "stream": false
}'
```

**Python**:

```python
import requests

def query_ollama(prompt, model="golang-llm"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
    )
    return response.json()["response"]

# Usage
result = query_ollama("Explain defer in Go")
print(result)
```

## Serving with LocalAI

### Install LocalAI

```bash
# Using Docker
docker run -p 8080:8080 -v $PWD/models:/models localai/localai:latest

# Or build from source
git clone https://github.com/go-skynet/LocalAI
cd LocalAI
make build
./local-ai --models-path ./models
```

### Configure Model

Create `models/go_coder_llm.yaml`:

```yaml
name: go-coder-llm
backend: llama
parameters:
  model: go_coder_llm_q4_0.gguf
  context_size: 1024
  threads: 8
  temperature: 0.8
  top_p: 0.95
  top_k: 40
```

### Use OpenAI-Compatible API

```python
from openai import OpenAI

# Point to LocalAI
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

# Generate
response = client.completions.create(
    model="go-coder-llm",
    prompt="Write a Go HTTP server",
    max_tokens=200,
)

print(response.choices[0].text)
```

## LangChain Integration

### With Ollama

```python
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize LLM
llm = Ollama(model="golang-llm", base_url="http://localhost:11434")

# Create prompt template
template = """
You are a Go expert. Answer the following question:

Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
chain = LLMChain(llm=llm, prompt=prompt)

# Run
result = chain.run(question="How do goroutines work?")
print(result)
```

### With LocalAI (OpenAI-Compatible)

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Initialize
llm = ChatOpenAI(
    model_name="go-coder-llm",
    openai_api_base="http://localhost:8080/v1",
    openai_api_key="not-needed",
)

# Chat
messages = [
    SystemMessage(content="You are a Go programming expert."),
    HumanMessage(content="Explain context.Context in Go"),
]

response = llm(messages)
print(response.content)
```

## Go Integration

### Using Ollama from Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
)

type OllamaRequest struct {
    Model  string `json:"model"`
    Prompt string `json:"prompt"`
    Stream bool   `json:"stream"`
}

type OllamaResponse struct {
    Response string `json:"response"`
    Done     bool   `json:"done"`
}

func QueryOllama(prompt string) (string, error) {
    req := OllamaRequest{
        Model:  "golang-llm",
        Prompt: prompt,
        Stream: false,
    }

    body, _ := json.Marshal(req)
    resp, err := http.Post(
        "http://localhost:11434/api/generate",
        "application/json",
        bytes.NewBuffer(body),
    )
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()

    data, _ := io.ReadAll(resp.Body)
    var result OllamaResponse
    json.Unmarshal(data, &result)

    return result.Response, nil
}

func main() {
    response, err := QueryOllama("Write a Go function for reading CSV")
    if err != nil {
        panic(err)
    }
    fmt.Println(response)
}
```

## Performance Tuning

### GPU Acceleration

**Ollama with GPU**:

```bash
# Ollama automatically uses GPU if available
# Check GPU usage
watch -n 1 rocm-smi
```

**LocalAI with GPU**:

```yaml
# In model config
parameters:
  gpu_layers: 35  # Offload layers to GPU (adjust based on VRAM)
  use_mlock: true
```

### CPU Optimization

For CPU-only inference:

```bash
# Set thread count
export OMP_NUM_THREADS=8

# Run Ollama
ollama run golang-llm
```

### Batch Processing

```python
import requests

def batch_generate(prompts, model="golang-llm"):
    results = []
    for prompt in prompts:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        results.append(response.json()["response"])
    return results

prompts = [
    "Write a Go HTTP handler",
    "Explain Go interfaces",
    "Create a Go struct for user data",
]

outputs = batch_generate(prompts)
```

## Inference API Server

### Simple Flask API

```python
# server.py
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "golang-llm",
            "prompt": prompt,
            "stream": False,
        }
    )

    return jsonify(response.json())

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

Run:

```bash
pip install flask
python server.py
```

Test:

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Go function"}'
```

## Production Deployment

### Docker Deployment

**Dockerfile**:

```dockerfile
FROM ollama/ollama:latest

# Copy model
COPY models/go_coder_llm_q4_0.gguf /models/
COPY Modelfile /Modelfile

# Create model
RUN ollama create golang-llm -f /Modelfile

# Expose port
EXPOSE 11434

# Start Ollama
CMD ["ollama", "serve"]
```

Build and run:

```bash
docker build -t go-coder-llm .
docker run -p 11434:11434 go-coder-llm
```

### Kubernetes Deployment

**deployment.yaml**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: go-coder-llm
spec:
  replicas: 2
  selector:
    matchLabels:
      app: go-coder-llm
  template:
    metadata:
      labels:
        app: go-coder-llm
    spec:
      containers:
      - name: ollama
        image: go-coder-llm:latest
        ports:
        - containerPort: 11434
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: go-coder-llm-service
spec:
  selector:
    app: go-coder-llm
  ports:
    - protocol: TCP
      port: 80
      targetPort: 11434
  type: LoadBalancer
```

Deploy:

```bash
kubectl apply -f deployment.yaml
```

## Monitoring

### Ollama Metrics

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Monitor logs
journalctl -u ollama -f
```

### Prometheus Metrics

Expose metrics:

```python
# metrics.py
from prometheus_client import Counter, Histogram, start_http_server
import time

# Metrics
REQUEST_COUNT = Counter('llm_requests_total', 'Total requests')
REQUEST_DURATION = Histogram('llm_request_duration_seconds', 'Request duration')

@REQUEST_DURATION.time()
def generate(prompt):
    REQUEST_COUNT.inc()
    # Call Ollama...
    pass

# Start metrics server
start_http_server(9090)
```

## Troubleshooting

### Model Not Loading

**Issue**: "Model not found"

**Solution**:
```bash
# List models
ollama list

# Recreate model
ollama create golang-llm -f Modelfile
```

### Slow Inference

**Issue**: Generation taking >30s

**Solutions**:
- Use quantized model (Q4/Q5)
- Enable GPU acceleration
- Reduce context length
- Increase batch size

### Out of Memory

**Issue**: OOM during inference

**Solutions**:
- Use smaller quantization (Q4_0)
- Reduce `num_ctx` in Modelfile
- Decrease `gpu_layers` in LocalAI
- Add more RAM/swap

### Poor Generation Quality

**Issue**: Nonsensical output

**Solutions**:
- Adjust temperature (lower = more deterministic)
- Increase top_p/top_k
- Use less aggressive quantization
- Try different checkpoints

## Benchmarking

### Throughput Test

```python
import time
import requests

def benchmark(prompt, iterations=10):
    times = []
    for _ in range(iterations):
        start = time.time()
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "golang-llm", "prompt": prompt, "stream": False}
        )
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    throughput = 1 / avg_time

    print(f"Average time: {avg_time:.2f}s")
    print(f"Throughput: {throughput:.2f} req/s")

benchmark("Write a Go function")
```

### Quality Evaluation

```python
# Evaluate on held-out test set
test_prompts = [
    "Write a Go function that reads a file",
    "Explain defer in Go",
    # ... more prompts
]

for prompt in test_prompts:
    response = query_ollama(prompt)
    # Manual review or automated metrics (BLEU, ROUGE)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")
```

## Next Steps

After deployment:
1. Monitor performance and usage
2. Collect user feedback
3. Fine-tune based on real-world usage
4. Scale horizontally as needed
5. Consider continuous training pipeline

## Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [LocalAI](https://github.com/go-skynet/LocalAI)
- [LangChain](https://python.langchain.com/)
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
