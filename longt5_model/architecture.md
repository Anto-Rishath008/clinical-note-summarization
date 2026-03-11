# LongT5-Inspired Model Architecture

## Overview

This document describes the architecture of our **Simplified LongT5** model, a Transformer-based encoder-decoder designed from scratch for clinical note summarization. The model handles long documents (up to 4096 tokens) efficiently using **Local-Global Attention**.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT: Clinical Note                               │
│                        (Up to 4096 tokens supported)                         │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SHARED EMBEDDING LAYER                              │
│                     vocab_size → d_model (16000 → 256)                       │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
┌─────────────────────────────┐    ┌─────────────────────────────────────────┐
│         ENCODER             │    │               DECODER                    │
│   (Local-Global Attention)  │    │  (Causal Self-Attn + Cross-Attn)        │
│      × 4 Layers             │    │           × 4 Layers                    │
└──────────────┬──────────────┘    └─────────────────────────┬───────────────┘
               │                                              │
               │              Encoder Hidden States           │
               └──────────────────────►───────────────────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LM HEAD                                         │
│                      d_model → vocab_size (256 → 16000)                      │
│                      (Tied with Embedding Weights)                           │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT: Summary Tokens                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 16,000 | SentencePiece vocabulary size |
| `d_model` | 256 | Hidden dimension |
| `d_ff` | 1,024 | Feed-forward intermediate dimension (4× d_model) |
| `num_encoder_layers` | 4 | Number of encoder layers |
| `num_decoder_layers` | 4 | Number of decoder layers |
| `num_heads` | 8 | Number of attention heads |
| `head_dim` | 32 | Dimension per head (d_model / num_heads) |
| `max_position_embeddings` | 4,096 | Maximum sequence length |
| `local_radius` | 127 | Local attention window radius |
| `global_block_size` | 16 | Interval for global tokens |
| `dropout` | 0.1 | Dropout probability |
| `label_smoothing` | 0.1 | Label smoothing for loss |

**Total Parameters**: ~12.5M (optimized for 8GB GPU)

---

## Component Details

### 1. Shared Embedding Layer

```python
nn.Embedding(vocab_size=16000, d_model=256)
```

- Converts token IDs to dense vectors
- **Shared** between encoder and decoder (parameter efficiency)
- **Tied** with LM Head output layer (weight tying)

---

### 2. Encoder

The encoder processes the full clinical note and produces contextualized representations.

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENCODER LAYER × 4                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input ──► LayerNorm ──► Local-Global Attention ──► Dropout    │
│    │                                                    │       │
│    └─────────────── Residual Connection ───────────────┘       │
│                              │                                  │
│                              ▼                                  │
│         ──► LayerNorm ──► Feed-Forward ──► Dropout              │
│    │                                            │               │
│    └────────────── Residual Connection ────────┘               │
│                              │                                  │
│                              ▼                                  │
│                           Output                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     Final LayerNorm
```

#### Local-Global Attention (Key Innovation)

This is the core mechanism that enables efficient long-sequence processing:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOCAL-GLOBAL ATTENTION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LOCAL ATTENTION (Window = 255 tokens):                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Token[i] attends to Token[i-127] ... Token[i+127]       │   │
│  │                                                          │   │
│  │  Example for token at position 500:                      │   │
│  │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐                  │   │
│  │  │373│374│...│499│500│501│...│626│627│                  │   │
│  │  └───┴───┴───┴───┴─▲─┴───┴───┴───┴───┘                  │   │
│  │                    │                                     │   │
│  │              Current Token                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  GLOBAL ATTENTION (Every 16th token):                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Token[0], Token[16], Token[32], ... attend GLOBALLY     │   │
│  │                                                          │   │
│  │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐              │   │
│  │  │ G │   │   │...│ G │   │   │...│ G │   │              │   │
│  │  │ 0 │ 1 │ 2 │   │16 │17 │18 │   │32 │33 │              │   │
│  │  └─┬─┴───┴───┴───┴─┬─┴───┴───┴───┴─┬─┴───┘              │   │
│  │    │               │               │                     │   │
│  │    └───────────────┼───────────────┘                     │   │
│  │          Global tokens attend to ALL tokens              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Complexity: O(n × w + n × g) instead of O(n²)                 │
│  Where: n=seq_len, w=window_size, g=num_global_tokens          │
└─────────────────────────────────────────────────────────────────┘
```

**Why Local-Global Attention?**
- Standard attention is O(n²) - impossible for 4096 tokens on 8GB GPU
- Local attention captures nearby context (important for clinical notes)
- Global tokens provide document-level information flow
- Inspired by LongT5's "Transient Global" attention

---

### 3. Decoder

The decoder generates the summary auto-regressively.

```
┌─────────────────────────────────────────────────────────────────┐
│                        DECODER LAYER × 4                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input ──► LayerNorm ──► Causal Self-Attention ──► Dropout     │
│    │                                                    │       │
│    └─────────────── Residual Connection ───────────────┘       │
│                              │                                  │
│                              ▼                                  │
│         ──► LayerNorm ──► Cross-Attention ──► Dropout          │
│    │                     (to Encoder)               │           │
│    └─────────────── Residual Connection ───────────┘           │
│                              │                                  │
│                              ▼                                  │
│         ──► LayerNorm ──► Feed-Forward ──► Dropout              │
│    │                                            │               │
│    └────────────── Residual Connection ────────┘               │
│                              │                                  │
│                              ▼                                  │
│                           Output                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     Final LayerNorm
```

#### Causal Self-Attention

Ensures decoder can only attend to previous tokens (auto-regressive):

```
                    Key Positions
              ┌───┬───┬───┬───┬───┐
              │ 0 │ 1 │ 2 │ 3 │ 4 │
          ┌───┼───┼───┼───┼───┼───┤
        0 │ ✓ │ ✗ │ ✗ │ ✗ │ ✗ │
Query   1 │ ✓ │ ✓ │ ✗ │ ✗ │ ✗ │
Pos     2 │ ✓ │ ✓ │ ✓ │ ✗ │ ✗ │
        3 │ ✓ │ ✓ │ ✓ │ ✓ │ ✗ │
        4 │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │
          └───┴───┴───┴───┴───┴───┘
          
✓ = Can attend    ✗ = Masked (−∞)
```

#### Cross-Attention

Allows decoder to attend to all encoder positions:

```
    Decoder                        Encoder
    Query                          Key/Value
      │                               │
      ▼                               ▼
  ┌───────┐                     ┌───────────┐
  │Summary│ ─────────────────► │Clinical   │
  │Token i│     Attention       │Note       │
  └───────┘                     │(All 4096  │
                                │ tokens)   │
                                └───────────┘
```

---

### 4. Relative Position Bias

Instead of absolute position embeddings, we use **T5-style relative position biases**:

```
┌─────────────────────────────────────────────────────────────────┐
│                   RELATIVE POSITION BIAS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Position Difference → Bucket → Learnable Bias                  │
│                                                                 │
│  Buckets (32 total):                                            │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Exact positions: -16 to +16 (linear mapping)           │    │
│  │ Far positions: logarithmic buckets up to ±128          │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Benefits:                                                      │
│  • Generalizes to unseen sequence lengths                       │
│  • Captures relative relationships (distance matters)           │
│  • Shared across all attention heads                            │
└─────────────────────────────────────────────────────────────────┘
```

---

### 5. Feed-Forward Network

Each layer has a position-wise feed-forward network:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEED-FORWARD NETWORK                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (d_model=256)                                            │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                           │
│  │ Linear(256→1024)│  Expand to 4× dimension                   │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │      GELU       │  Non-linearity (smoother than ReLU)       │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │    Dropout(0.1) │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Linear(1024→256)│  Project back to model dimension          │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  Output (d_model=256)                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

### 6. LM Head (Output Layer)

```python
nn.Linear(d_model=256, vocab_size=16000, bias=False)
# Weights TIED with embedding layer for parameter efficiency
```

---

## Generation Methods

### Greedy Decoding

```
Step 1: Input [BOS] → Model → argmax → "Patient"
Step 2: Input [BOS, "Patient"] → Model → argmax → "was"  
Step 3: Input [BOS, "Patient", "was"] → Model → argmax → "admitted"
...
Until [EOS] token generated or max_length reached
```

### Beam Search

```
Beam Width = 4

Step 1:           [BOS]
                    │
         ┌────┬────┼────┬────┐
         ▼    ▼    ▼    ▼    ▼
       "The" "A" "Patient" "He" ...
        │
    Keep top 4 scoring beams
        │
Step 2: ┌──────────────────┐
        │ Expand each beam │
        │ Keep top 4 total │
        └──────────────────┘
...
Until all beams finish with [EOS]
Select highest scoring complete sequence
```

**Beam Search Features:**
- `length_penalty`: Adjusts preference for longer/shorter outputs
- `no_repeat_ngram_size=3`: Prevents repetitive text
- `min_length=50`: Ensures minimum summary length
- `early_stopping`: Stops when all beams complete

---

## Data Flow Summary

```
1. TOKENIZATION
   "Patient admitted with chest pain..." → [234, 892, 123, 456, ...]

2. EMBEDDING
   [234, 892, ...] → [[0.12, -0.34, ...], [0.56, 0.78, ...], ...]
                         ↑ d_model=256 dimensions per token

3. ENCODING (4 layers)
   Embed vectors → Contextualized representations
   Each position now "knows" about the full document

4. DECODING (4 layers, auto-regressive)
   [BOS] → "Patient"
   [BOS, "Patient"] → "presented"
   [BOS, "Patient", "presented"] → "with"
   ... until [EOS]

5. OUTPUT
   "Patient presented with acute chest pain. Cardiac workup negative..."
```

---

## Memory Efficiency

| Approach | Memory for 4096 tokens |
|----------|------------------------|
| Standard Attention | O(n²) = 16M attention weights |
| Local-Global | O(n×w + n×g) ≈ 1M attention weights |

**Result**: Can process 4096-token documents on 8GB GPU!

---

## Comparison with Original LongT5

| Feature | Google LongT5 | Our Implementation |
|---------|---------------|-------------------|
| Pretrained | Yes (16B tokens) | No (from scratch) |
| Parameters | 220M-3B | ~12.5M |
| Attention | TGlobal | Simplified Local-Global |
| Position | Relative Bias | Relative Bias |
| Tokenizer | T5 SentencePiece | Custom SentencePiece |
| Max Length | 16K | 4K |

---

## File Structure

```
longt5_model/
├── model.py          # This architecture (940 lines)
│   ├── LongT5Config         # Configuration dataclass
│   ├── RelativePositionBias # T5-style position encoding  
│   ├── LocalGlobalAttention # Efficient long attention
│   ├── CrossAttention       # Decoder cross-attention
│   ├── FeedForward          # MLP layer
│   ├── EncoderLayer         # Single encoder block
│   ├── DecoderLayer         # Single decoder block
│   ├── Encoder              # Full encoder stack
│   ├── Decoder              # Full decoder stack
│   └── SimplifiedLongT5     # Main model class
├── data_loader.py    # Streaming data loading
├── train.py          # Training script
├── inference.py      # Generation script
├── config.yaml       # Hyperparameters
└── README.md         # Usage guide
```

---

## References

1. **LongT5**: Guo et al., "LongT5: Efficient Text-To-Text Transformer for Long Sequences" (2022)
2. **T5**: Raffel et al., "Exploring the Limits of Transfer Learning" (2020)
3. **Attention is All You Need**: Vaswani et al. (2017)
4. **LED**: Beltagy et al., "Longformer: The Long-Document Transformer" (2020)
