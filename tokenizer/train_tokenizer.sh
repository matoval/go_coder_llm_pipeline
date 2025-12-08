#!/bin/bash

# Navigate to tokenizer directory
cd "$(dirname "$0")"

# Train SentencePiece model with HRM/TRM special tokens
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
  --user_defined_symbols="<REPO>,<PR_TITLE>,<PR_BODY>,<FILE>,<BEFORE>,<AFTER>,<COMMENTS>,<CODE>,<STRUCTURE>,<PLAN>,</PLAN>,<STEP>,<INTENT:FIX>,<INTENT:ADD>,<INTENT:REFACTOR>,<INTENT:OPTIMIZE>,<INTENT:UPDATE>,<INTENT:REMOVE>,<TARGET:func>,<TARGET:type>,<TARGET:interface>,<TARGET:method>,<TARGET:var>,<VALIDATE>,<SYNTAX_OK>,<SYNTAX_ERR>,<TEST_PASS>,<TEST_FAIL>,<REFINE>,<CONTEXT>,<PROBLEM>" \
  --input_sentence_size=10000000 \
  --shuffle_input_sentence=true

echo "Tokenizer training complete!"
echo "Output files:"
echo "  - go_coder_llm.model"
echo "  - go_coder_llm.vocab"
