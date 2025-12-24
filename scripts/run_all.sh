#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Go Coder LLM Pipeline - Full Run${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -f "llm/bin/activate" ]; then
    echo -e "${RED}Error: Virtual environment not found at llm/bin/activate${NC}"
    echo "Please create it first with: python -m venv llm"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}[1/5] Activating virtual environment...${NC}"
source llm/bin/activate

# Create necessary directories
echo -e "${GREEN}[2/5] Creating directories...${NC}"
mkdir -p data/processed data/corpus data/tok_dataset

# Count raw files
RAW_COUNT=$(find data/raw -name "*.jsonl" | wc -l)
echo -e "${BLUE}Found $RAW_COUNT raw JSONL files${NC}"
echo ""

# Step 1: Process raw data to hierarchical format
echo -e "${GREEN}[3/5] Processing raw data to hierarchical format...${NC}"
echo "This may take a while with $RAW_COUNT files..."
echo ""

python scripts/process_data.py hierarchical \
    --input data/raw \
    --output data/processed

if [ $? -ne 0 ]; then
    echo -e "${RED}Error processing raw data${NC}"
    exit 1
fi

echo ""

# Step 2: Create training corpus
echo -e "${GREEN}[4/5] Creating training corpus...${NC}"
echo "Combining all hierarchical files into training corpus..."
echo ""

python scripts/process_data.py corpus \
    --input data/processed \
    --output data/corpus

if [ $? -ne 0 ]; then
    echo -e "${RED}Error creating corpus${NC}"
    exit 1
fi

# Check corpus size
CORPUS_SIZE=$(du -h data/corpus/training_corpus.txt | cut -f1)
CORPUS_LINES=$(wc -l < data/corpus/training_corpus.txt)
echo ""
echo -e "${BLUE}Corpus created: ${CORPUS_SIZE} (${CORPUS_LINES} lines)${NC}"
echo ""

# Step 3: Train tokenizer
echo -e "${GREEN}[5/5] Training SentencePiece tokenizer...${NC}"
echo "This may take 10-30 minutes depending on corpus size..."
echo ""

cd tokenizer

# Check if sentencepiece is installed
if ! command -v spm_train &> /dev/null; then
    echo -e "${YELLOW}Warning: spm_train not found in PATH${NC}"
    echo "Installing sentencepiece..."
    pip install sentencepiece
fi

# Run tokenizer training
./train_tokenizer.sh

if [ $? -ne 0 ]; then
    echo -e "${RED}Error training tokenizer${NC}"
    exit 1
fi

cd ..

# Verify tokenizer was created
if [ ! -s tokenizer/go_coder_llm.model ]; then
    echo -e "${RED}Error: Tokenizer model was not created${NC}"
    exit 1
fi

if [ ! -s tokenizer/go_coder_llm.vocab ]; then
    echo -e "${RED}Error: Tokenizer vocab was not created${NC}"
    exit 1
fi

TOKENIZER_SIZE=$(du -h tokenizer/go_coder_llm.model | cut -f1)
VOCAB_SIZE=$(wc -l < tokenizer/go_coder_llm.vocab)

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Pipeline Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${BLUE}Summary:${NC}"
echo "  Raw files processed: $RAW_COUNT"
echo "  Corpus size: $CORPUS_SIZE"
echo "  Tokenizer model: $TOKENIZER_SIZE"
echo "  Vocabulary size: $VOCAB_SIZE tokens"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Verify tokenizer: ${BLUE}ls -lh tokenizer/go_coder_llm.{model,vocab}${NC}"
echo "  2. Train model: ${BLUE}cd model && python train.py${NC}"
echo ""
