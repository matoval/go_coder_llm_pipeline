# Goal

Train a **tiny code-focused GPT-style model (~125 M params)** on your GitHub PR / comment corpus using your AMD GPU.

---

## 1️⃣  Environment setup (ROCm / PyTorch)

> Tested on Fedora, Ubuntu 22.04+, Arch with kernel ≥ 6.8.

```bash
# ROCm and PyTorch (use ROCm wheels, not CUDA)
sudo dnf install rocm-dev rocm-libs hipblas rocblas miopen-hip

python3 -m venv llm
source llm/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install transformers datasets accelerate sentencepiece tiktoken bitsandbytes
```

Check GPU:

```python
python - <<'PY'
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

Expect `True`, and something like `AMD Radeon RX 6700 XT`.

---

## 2️⃣  Tokenizer

Use the corpus produced by your Go pipeline (plain-text or JSONL).
Train SentencePiece → 50 K vocab:

```bash
spm_train \
  --input=corpus.txt \
  --model_prefix=golang_llm \
  --vocab_size=50000 \
  --character_coverage=1.0 \
  --model_type=bpe
```

You’ll get:

```text
golang_llm.model
golang_llm.vocab
```

---

## 3️⃣  Dataset prep

Convert your serialized JSONL into plain text (prompt–response pairs or code–comment lines).
Then tokenize:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(".", model_max_length=1024, padding_side="right", truncation_side="right")
tok.add_special_tokens({"pad_token": "<|pad|>"})

ds = load_dataset("text", data_files="corpus.txt")
def tok_fn(batch): return tok(batch["text"], truncation=True, padding="max_length")
tok_ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
tok_ds.save_to_disk("tok_dataset")
```

---

## 4️⃣  Model config (125 M params)

```python
from transformers import GPT2Config, GPT2LMHeadModel

cfg = GPT2Config(
    vocab_size=50000,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
)
model = GPT2LMHeadModel(cfg)
model.save_pretrained("golang_gpt2_125m")
```

---

## 5️⃣  Training script

Create `train.py`:

```python
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_from_disk

model = GPT2LMHeadModel.from_pretrained("golang_gpt2_125m")
tok = AutoTokenizer.from_pretrained(".", model_max_length=1024)
tok.pad_token = "<|pad|>"

data = load_from_disk("tok_dataset")

args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=5e-4,
    fp16=True,                       # ROCm supports this
    save_total_limit=2,
    logging_steps=50,
    save_steps=1000,
    report_to="none",
)

collator = DataCollatorForLanguageModeling(tok, mlm=False)
trainer = Trainer(model=model, args=args, train_dataset=data["train"], data_collator=collator)
trainer.train()
```

Run it:

```bash
python train.py
```

> With batch = 2 × grad accum 8 and seq = 1024, fits in ≈ 10-11 GB VRAM.
> Expect **~5 tokens/sec → 5–7 days** for ~10 B tokens (scale down to 1 B tokens for ≈ 12 h run).

---

## 6️⃣  Evaluation / sampling

```python
from transformers import pipeline
gen = pipeline("text-generation", model="checkpoints", tokenizer=".")
out = gen("Write a Go function that reads a JSON file:", max_length=200)
print(out[0]["generated_text"])
```

---

## 7️⃣  Export to GGUF / Ollama (optional)

Once you’re happy with results:

```bash
python convert.py --from transformers --to gguf --model checkpoints
ollama create golang-llm -f Modelfile
```

Then you can use:

```bash
ollama run golang-llm
```

and call it from LangChain or your Go service.

---

## 8️⃣  What you’ll get

* A working tokenizer trained on your GitHub corpus
* A small 125 M GPT that understands Go code structure + PR language
* Full pipeline test (data → tokens → model → inference)

---

## 🧠  Optional optimizations

* Use `bitsandbytes` 8-bit optimizer: halves VRAM use.
* Use `torch.compile()` on PyTorch 2.1 + ROCm 6 for ≈ 10 % speedup.
* Increase `gradient_accumulation_steps` if you run out of VRAM.

---

## ✅  Summary

| Stage              | Tool                       | Runtime          |
| ------------------ | -------------------------- | ---------------- |
| Data → text        | Go pipeline                | CPU              |
| Tokenizer          | SentencePiece              | CPU              |
| Model train        | PyTorch (ROCm)             | GPU (RX 6700 XT) |
| Export → inference | convert → Ollama / LocalAI | CPU / GPU        |

Expected runtime (10 B tokens): **≈ 5–7 days**.
You can shorten dramatically by training on fewer tokens first to verify.

---
