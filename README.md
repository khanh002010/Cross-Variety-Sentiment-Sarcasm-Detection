# Multi-Task Learning for Sentiment & Sarcasm Detection (ALTA)

This repository contains the source code, datasets, and research papers for a Multi-Task Learning (MTL) framework focused on detecting Sentiment and Sarcasm across different varieties of English (specifically en-AU and en-UK).

## Structure

- **Code:**
  - mtl_qwen_qlora.py: A highly optimized implementation using **Qwen2.5-14B (14.7B params)** as the backbone with BF16 precision and LoRA for efficient fine-tuning on an A100 80GB GPU. Features Cross-Task Attention Interaction (CTAI) and Neural Tensor Network (NTN) fusion.
  - mtl_sentiment_sarcasm_besstie.py: The original implementation containing the full experimental pipeline.

- **Datasets:**
  - en-AU/train.csv: Training dataset for Australian English.
  - en-UK/train.csv: Training dataset for UK English.

- **Papers:**
  - The paper/ directory contains various research papers forming the theoretical foundation of this work.

