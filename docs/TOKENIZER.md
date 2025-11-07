# Tokenizer Guide

Complete guide for training and using a BPE tokenizer for the Go Coder LLM.

## Overview

The tokenizer converts raw text (code + natural language) into numerical tokens that the model can process. We use **SentencePiece** with **BPE** (Byte-Pair Encoding) trained on mixed corpus.

## Why One Shared Tokenizer?

### Advantages

- **Simplicity**: Single vocabulary for entire model
- **No Mismatches**: Avoid token-id conflicts
- **Efficient**: Modern tokenizers handle code + text well
- **Unified**: Seamless transitions between code and comments

### Trade-offs

- Larger vocabulary needed (50K vs 32K for text-only)
- Some inefficiency on pure code vs specialized tokenizer

## Tokenizer Strategy

### Model Type: BPE (Byte-Pair Encoding)

- **Subword tokenization**: Breaks rare words into pieces
- **Vocabulary learning**: Iteratively merges common byte pairs
- **Handles OOV**: Unknown words split into known subwords
- **Good for code**: Preserves identifiers, operators

### Vocabulary Size: 50,000

- **Balance**: Coverage vs efficiency
- **Code-focused**: More tokens for syntax, keywords
- **Natural language**: Common words + programming terms

### Character Coverage: 1.0

- **Full Unicode**: Support all UTF-8 characters
- **International**: Handle non-ASCII in comments
- **Special chars**: Preserve operators, symbols

## Special Tokens

### Standard Tokens

```
<unk>   : Unknown token (ID: 0)
<s>     : Begin-of-sequence (ID: 1)
</s>    : End-of-sequence (ID: 2)
<pad>   : Padding token (ID: 3)
```

### Custom Tokens (for structured data)

```
<REPO>       : Repository name
<PR_TITLE>   : Pull request title
<PR_BODY>    : Pull request description
<FILE>       : File marker
<BEFORE>     : Code before change
<AFTER>      : Code after change
<COMMENTS>   : Comments section
<CODE>       : Code block
<STRUCTURE>  : Repository structure
```

## Training the Tokenizer

### Step 1: Prepare Corpus

Collect text from processed data:

```bash
# Concatenate all processed text files
cat data/processed/*.txt > data/corpus/training_corpus.txt

# Check corpus size
wc -l data/corpus/training_corpus.txt
```

**Recommended corpus size**: 10M-100M tokens (before tokenization)

### Step 2: Train with SentencePiece

Using the existing script (tokenizer/train_tokenizer.sh):

```bash
#!/bin/bash

# Navigate to tokenizer directory
cd tokenizer

# Train SentencePiece model
spm_train \
  --input=../data/corpus/training_corpus.txt \
  --model_prefix=go_coder_llm \
  --vocab_size=50000 \
  --character_coverage=1.0 \
  --model_type=bpe \
  --unk_id=0 \
  --bos_id=1 \
  --eos_id=2 \
  --pad_id=3 \
  --user_defined_symbols="<REPO>,<PR_TITLE>,<PR_BODY>,<FILE>,<BEFORE>,<AFTER>,<COMMENTS>,<CODE>,<STRUCTURE>" \
  --input_sentence_size=10000000 \
  --shuffle_input_sentence=true

echo "Tokenizer training complete!"
echo "Output files:"
echo "  - go_coder_llm.model"
echo "  - go_coder_llm.vocab"
```

Make executable:

```bash
chmod +x tokenizer/train_tokenizer.sh
```

Run:

```bash
./tokenizer/train_tokenizer.sh
```

### Step 3: Verify Tokenizer

Test tokenization:

```python
import sentencepiece as spm

# Load model
sp = spm.SentencePieceProcessor()
sp.load('tokenizer/go_coder_llm.model')

# Test Go code
code = """
func main() {
    fmt.Println("Hello, World!")
}
"""

# Tokenize
tokens = sp.encode_as_pieces(code)
print("Tokens:", tokens)
print("IDs:", sp.encode_as_ids(code))

# Test special tokens
text = "<REPO> golang/go\n<PR_TITLE> Fix nil pointer"
tokens = sp.encode_as_pieces(text)
print("Special tokens:", tokens)
```

Expected output:

```
Tokens: ['▁func', '▁main', '()', '▁{', '\n', '▁', '▁', '▁', '▁fmt', '.', 'Println', '("', 'Hello', ',', '▁World', '!")', '\n', '}']
IDs: [156, 287, 445, 89, 13, ...]
Special tokens: ['<REPO>', '▁golang', '/', 'go', '\n', '<PR_TITLE>', '▁Fix', '▁nil', '▁pointer']
```

## Using the Tokenizer

### Python (SentencePiece)

```python
import sentencepiece as spm

class GoTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text):
        """Convert text to token IDs"""
        return self.sp.encode_as_ids(text)

    def decode(self, ids):
        """Convert token IDs to text"""
        return self.sp.decode_ids(ids)

    def tokenize(self, text):
        """Get token pieces"""
        return self.sp.encode_as_pieces(text)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    @property
    def pad_id(self):
        return self.sp.pad_id()

    @property
    def eos_id(self):
        return self.sp.eos_id()

    @property
    def bos_id(self):
        return self.sp.bos_id()

# Usage
tokenizer = GoTokenizer('tokenizer/go_coder_llm.model')
ids = tokenizer.encode("func main() {}")
print(ids)
```

### Hugging Face Integration

Convert SentencePiece to Hugging Face format:

```python
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders

# Load SentencePiece model
sp_model = 'tokenizer/go_coder_llm.model'

# Create Hugging Face tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=None,
    model_max_length=1024,
    padding_side="right",
    truncation_side="right",
    pad_token="<pad>",
    eos_token="</s>",
    bos_token="<s>",
    unk_token="<unk>",
)

# Load from SentencePiece
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    ".",
    vocab_file="tokenizer/go_coder_llm.model",
    model_max_length=1024,
)

# Save in Hugging Face format
tokenizer.save_pretrained("tokenizer/hf_tokenizer")
```

## Dataset Tokenization

### Tokenize JSONL Records

```python
import json
import sentencepiece as spm
from tqdm import tqdm

def tokenize_jsonl(input_path, output_path, sp_model):
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model)

    with open(input_path, 'r') as fin, \
         open(output_path, 'w') as fout:

        for line in tqdm(fin):
            record = json.loads(line)

            # Serialize record to text (using format from DATA_PIPELINE.md)
            text = serialize_pr_record(record)

            # Tokenize
            token_ids = sp.encode_as_ids(text)

            # Save tokenized record
            output = {
                'input_ids': token_ids,
                'attention_mask': [1] * len(token_ids),
                'metadata': {
                    'repo': record['repo']['full_name'],
                    'pr_number': record['pull_request']['number'],
                }
            }
            fout.write(json.dumps(output) + '\n')

def serialize_pr_record(record):
    """Convert PRRecord to text format"""
    parts = []
    parts.append(f"<REPO> {record['repo']['full_name']}")
    parts.append(f"<PR_TITLE> {record['pull_request']['title']}")

    if record['pull_request']['body']:
        parts.append(f"<PR_BODY> {record['pull_request']['body']}")

    if record['files']:
        parts.append("<FILES>")
        for file in record['files']:
            parts.append(f"<FILE path=\"{file['filename']}\">")
            if file['patch']:
                parts.append(file['patch'])
            parts.append("</FILE>")
        parts.append("</FILES>")

    if record['comments']:
        parts.append("<COMMENTS>")
        for comment in record['comments']:
            parts.append(f"{comment['user']}: {comment['body']}")
        parts.append("</COMMENTS>")

    return '\n'.join(parts)

# Run
tokenize_jsonl(
    'data/processed/repos.jsonl',
    'data/tokenized/train.jsonl',
    'tokenizer/go_coder_llm.model'
)
```

### Sequence Packing

Pack tokenized sequences into fixed-length chunks:

```python
import numpy as np

def pack_sequences(token_ids_list, seq_len=1024, pad_id=3):
    """Pack variable-length sequences into fixed-length chunks"""
    buffer = []
    packed = []

    for token_ids in token_ids_list:
        buffer.extend(token_ids)
        buffer.append(2)  # EOS token

        # Pack into chunks
        while len(buffer) >= seq_len:
            chunk = buffer[:seq_len]
            packed.append(chunk)
            buffer = buffer[seq_len:]

    # Handle remainder
    if buffer:
        # Pad to seq_len
        padding = [pad_id] * (seq_len - len(buffer))
        packed.append(buffer + padding)

    return np.array(packed, dtype=np.int32)

# Usage
from datasets import load_dataset

# Load tokenized data
ds = load_dataset('json', data_files='data/tokenized/train.jsonl')

# Extract token IDs
all_token_ids = [item['input_ids'] for item in ds['train']]

# Pack sequences
packed = pack_sequences(all_token_ids, seq_len=1024)
print(f"Packed {len(all_token_ids)} sequences into {len(packed)} chunks")

# Save as binary
packed.tofile('data/tokenized/train.bin')
```

## Tokenizer Analysis

### Vocabulary Statistics

```python
import sentencepiece as spm
from collections import Counter

sp = spm.SentencePieceProcessor()
sp.load('tokenizer/go_coder_llm.model')

# Count token types
vocab = []
for i in range(sp.get_piece_size()):
    piece = sp.id_to_piece(i)
    vocab.append(piece)

# Analyze
print(f"Vocabulary size: {len(vocab)}")
print(f"Special tokens: {sum(1 for v in vocab if v.startswith('<'))}")
print(f"Byte tokens: {sum(1 for v in vocab if v.startswith('<0x'))}")

# Most common Go keywords
code_sample = open('data/corpus/training_corpus.txt').read(100000)
tokens = sp.encode_as_pieces(code_sample)
token_freq = Counter(tokens)

print("\nTop 20 tokens:")
for token, count in token_freq.most_common(20):
    print(f"  {token}: {count}")
```

### Token Efficiency

Measure average tokens per code sample:

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('tokenizer/go_coder_llm.model')

# Sample Go code
samples = [
    "func main() {}",
    "import \"fmt\"",
    "type User struct { Name string }",
    "for i := 0; i < 10; i++ {}",
]

for code in samples:
    tokens = sp.encode_as_pieces(code)
    compression = len(code) / len(tokens)
    print(f"Code: {code}")
    print(f"  Tokens: {len(tokens)}, Compression: {compression:.2f}x")
    print(f"  {tokens}\n")
```

Expected compression: 3-5x (3-5 characters per token)

## Best Practices

### 1. Corpus Preparation

- **Diverse sources**: Mix code, comments, documentation
- **Clean data**: Remove duplicates, invalid UTF-8
- **Representative**: Match target domain (Go code + PRs)
- **Large enough**: 10M+ sentences for good vocabulary

### 2. Training

- **Shuffle input**: `--shuffle_input_sentence=true`
- **Subsample**: `--input_sentence_size` for large corpus
- **Tuning vocab size**: Start 50K, adjust based on compression
- **Special tokens**: Add all custom markers

### 3. Testing

- **Coverage**: Test on held-out samples
- **Round-trip**: Encode then decode, check fidelity
- **Edge cases**: Unicode, long identifiers, operators
- **Special tokens**: Verify custom tokens work

### 4. Integration

- **Padding**: Use pad_id consistently
- **Attention masks**: 1 for real tokens, 0 for padding
- **Truncation**: Handle long sequences gracefully
- **EOS tokens**: Add between concatenated sequences

## Troubleshooting

### Poor Tokenization Quality

**Symptoms**: Many `<unk>` tokens, poor compression

**Solutions**:
- Increase character coverage to 1.0
- Check corpus for encoding issues (valid UTF-8)
- Increase vocabulary size
- Ensure corpus is representative

### Memory Issues During Training

**Symptoms**: OOM during `spm_train`

**Solutions**:
- Use `--input_sentence_size` to subsample
- Reduce vocabulary size
- Train on smaller corpus first

### Special Tokens Not Working

**Symptoms**: Special tokens split into pieces

**Solutions**:
- Add via `--user_defined_symbols`
- Quote properly: `"<TOKEN1>,<TOKEN2>"`
- Verify in .vocab file after training

### Incompatible with Hugging Face

**Symptoms**: Can't load in Transformers

**Solutions**:
- Use `PreTrainedTokenizerFast` wrapper
- Or convert to Hugging Face format
- Check special token mappings

## Performance Metrics

### Training Time

| Corpus Size | Vocab Size | Time (CPU) |
|-------------|------------|-----------|
| 10M lines   | 50K        | ~10 min   |
| 50M lines   | 50K        | ~30 min   |
| 100M lines  | 50K        | ~1 hour   |

### Tokenization Speed

- **Python**: ~1M tokens/sec (single core)
- **C++ (from Go)**: ~10M tokens/sec

### Compression Ratio

| Content Type | Characters/Token |
|--------------|------------------|
| Go code      | 4.5x             |
| Comments     | 4.0x             |
| English text | 4.2x             |
| Mixed        | 4.3x             |

## Advanced: Custom Tokenizer

For more control, implement custom BPE:

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Initialize BPE model
tokenizer = Tokenizer(models.BPE())

# Pre-tokenization (split on whitespace)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Trainer
trainer = trainers.BpeTrainer(
    vocab_size=50000,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<REPO>", "<PR_TITLE>"],
)

# Train
files = ["data/corpus/training_corpus.txt"]
tokenizer.train(files, trainer)

# Save
tokenizer.save("tokenizer/custom_bpe.json")
```

## Next Steps

After tokenizer is trained:
1. Review [Training Guide](TRAINING.md) for model training
2. Check [Data Pipeline](DATA_PIPELINE.md) for corpus generation
3. See [Architecture](ARCHITECTURE.md) for system overview

## Resources

- [SentencePiece GitHub](https://github.com/google/sentencepiece)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)
- [BPE Paper](https://arxiv.org/abs/1508.07909)
