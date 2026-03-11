# Mamba-Transformer Hybrid Model Architecture

## Overview

A from-scratch **Mamba Encoder + Transformer Decoder** hybrid model designed for clinical note summarization (MIMIC-IV Brief Hospital Course generation).

The model efficiently encodes very long clinical notes (up to 4096 tokens) using Mamba's linear-time state space mechanism with memory token compression, then decodes summaries using a standard Transformer decoder with cross-attention.

---

## Architecture Diagram

```
Input Clinical Note (up to 4096 tokens)
         │
    ┌────▼────┐
    │ SentencePiece Embedding (vocab=16000, d_model=512) │
    └────┬────┘
         │
    ┌────▼────────────────────────────────────┐
    │ Chunking (chunk_size=256, stride=192)    │
    │  → Overlapping windows across input      │
    └────┬────────────────────────────────────┘
         │ (multiple chunks)
    ┌────▼────────────────────────────┐
    │ Mamba Encoder (6 layers)        │
    │  Per chunk:                     │
    │  - LayerNorm → Mamba SSM Block  │
    │  - Residual + Dropout           │
    │  - Final LayerNorm              │
    └────┬────────────────────────────┘
         │
    ┌────▼────────────────────────────────────┐
    │ Memory Token Compressor                  │
    │  - K=8 learned query vectors per chunk   │
    │  - Cross-attention pooling               │
    │  - LayerNorm + residual                  │
    │  → Each chunk → 8 memory tokens          │
    └────┬────────────────────────────────────┘
         │ Concatenate all memory tokens
         │ (n_chunks × 8 = total memory tokens)
    ┌────▼──────────────────────────────────┐
    │ Transformer Decoder (6 layers)        │
    │  Each layer:                          │
    │  - Causal Self-Attention (8 heads)    │
    │  - Cross-Attention to memory tokens   │
    │  - Feed-Forward (GELU, d_ff=2048)     │
    │  - LayerNorm (pre-norm) + Residual    │
    └────┬──────────────────────────────────┘
         │
    ┌────▼────────────────┐
    │ Output Projection    │
    │ (tied with tgt emb)  │
    │  → vocab logits      │
    └─────────────────────┘
```

---

## Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 16,000 | SentencePiece BPE vocabulary |
| `d_model` | 512 | Hidden dimension |
| `n_mamba_layers` | 6 | Mamba encoder depth |
| `n_decoder_layers` | 6 | Transformer decoder depth |
| `n_heads` | 8 | Attention heads |
| `d_ff` | 2,048 | Feed-forward inner dim |
| `dropout` | 0.1 | Dropout rate |
| `max_src_len` | 4,096 | Maximum source (input) length |
| `max_tgt_len` | 512 | Maximum target (output) length |
| `chunk_size` | 256 | Chunk window size |
| `stride` | 192 | Chunk stride (overlap = 64) |
| `n_memory_tokens` | 8 | Memory tokens per chunk |
| `mamba_d_state` | 16 | Mamba SSM state dimension |
| `mamba_d_conv` | 4 | Mamba convolution width |
| `mamba_expand` | 2 | Mamba expansion factor |
| `label_smoothing` | 0.1 | Label smoothing for loss |

## Parameter Count

- **Total Parameters**: 52,133,888 (~52M)
- **All Trainable** (no frozen layers)

---

## Component Details

### 1. Mamba Encoder
- Uses the [Mamba SSM](https://github.com/state-spaces/mamba) for O(n) linear-time sequence encoding
- Falls back to GRU if `mamba-ssm` is not installed
- 6 stacked blocks, each with: `LayerNorm → Mamba → Dropout + Residual`
- Processes each chunk of 256 tokens independently

### 2. Memory Token Compressor
- Compresses each chunk's 256 encoded tokens → 8 memory tokens
- Uses **learned query vectors** + multi-head cross-attention pooling
- Enables the decoder to attend to a compact representation of the full document
- For a 4096-token input: ~22 chunks × 8 = ~176 memory tokens

### 3. Transformer Decoder
- Standard 6-layer Transformer decoder
- **Causal self-attention** for autoregressive generation
- **Cross-attention** to concatenated memory tokens from encoder
- Pre-norm architecture with GELU activation in FFN
- Output projection weights **tied** with target embedding

### 4. Generation
- Autoregressive decoding with top-k sampling (k=50)
- Temperature-controlled sampling
- BOS/EOS token handling with early stopping

---

## Training Details

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (lr=1e-4, weight_decay=0.01) |
| Scheduler | Cosine annealing with linear warmup |
| Warmup Steps | 500 |
| Batch Size | 4 (× 4 gradient accumulation = 16 effective) |
| Mixed Precision | AMP fp16 |
| Gradient Clipping | max_norm=1.0 |
| Loss | Cross-entropy with label smoothing (0.1) |
| Evaluation Metric | ROUGE-L (F-measure) |

---

## Dataset

- **MIMIC-IV Brief Hospital Course** (mimic-iv-bhc.csv)
- ~256,529 training samples, ~13,502 validation samples
- Source: Full clinical discharge notes
- Target: Brief Hospital Course summaries
- Tokenizer: SentencePiece BPE (16,000 vocab)

---

## Training History (Main Run)

- **10,000 steps** completed on RTX 4070 Laptop GPU (8GB VRAM)
- Best ROUGE-L: **0.1866** (achieved at step ~3000)
- Final validation loss: ~5.36

---

## File Structure

```
workspace_mamba_longt5_v1/
├── src/
│   ├── model.py          # Model architecture (MambaTransformerModel)
│   ├── train.py           # Training loop with AMP, checkpointing, ROUGE eval
│   ├── evaluate.py        # Standalone evaluation script
│   ├── data_loader.py     # Dataset & DataLoader with dynamic padding
│   └── __init__.py
├── configs/
│   ├── main_train.yaml    # Main training config (10k steps)
│   ├── resume_train.yaml  # Resume training config
│   ├── default.yaml       # Default config template
│   └── rouge_target_train.yaml  # Target-based training (ROUGE-L ≥ 0.5)
├── data/
│   ├── mimic-iv-bhc.csv   # Dataset (not tracked in git)
│   └── tokenizer/
│       ├── spm.model      # SentencePiece model
│       └── spm.vocab      # SentencePiece vocabulary
├── checkpoints/           # Model checkpoints (not tracked in git)
├── requirements.txt       # Python dependencies
└── MODEL_ARCHITECTURE.md  # This file
```
