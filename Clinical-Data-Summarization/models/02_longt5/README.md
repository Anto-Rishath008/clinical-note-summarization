# Model 2: LongT5 — Local-Global Attention Transformer

<p align="center"><strong>Resolves Issue #1 — Supports 4,096 Token Input via Efficient Attention</strong></p>

---

## Overview

LongT5 replaces the Pointer-Generator's BiLSTM with a **T5-inspired Transformer** featuring **Local-Global Attention** in the encoder. This enables processing clinical notes up to **4,096 tokens** (5× improvement over PG's 768-token limit) while maintaining manageable memory usage on a single 8 GB GPU.

### Why LongT5 over Pointer-Generator?

| Limitation in PG | LongT5 Solution |
|------------------|-----------------|
| Max 768 tokens (truncation) | 4,096 tokens via local-global attention |
| Coverage tensor OOM | No coverage needed — self-attention handles it |
| Chunking artifacts at boundaries | Seamless local+global attention spans |
| Hardcoded CUDA devices | Device-agnostic GradScaler, multi-GPU support |
| No restart safety | Robust checkpoint management |

### Key Characteristics

| Property | Value |
|----------|-------|
| **Architecture** | Encoder-Decoder Transformer with Local-Global Attention |
| **Parameters** | 12.5M |
| **d_model** | 256 |
| **d_ff** | 1,024 |
| **Encoder / Decoder Layers** | 4 / 4 |
| **Attention Heads** | 8 (head_dim = 32) |
| **Max Input Length** | 4,096 tokens |
| **Local Window Radius** | 127 tokens |
| **Global Block Size** | 16 tokens |

### ROUGE Scores

| Metric | Score | Improvement vs PG |
|--------|-------|-------------------|
| ROUGE-1 | 0.3517 | +93.9% |
| ROUGE-2 | 0.1232 | +4,006% |
| ROUGE-L | 0.2185 | +155.6% |

---

## Architecture

```
Input Clinical Note [up to 4,096 tokens]
       │
       ▼
SentencePiece Tokenization [16K vocab]
       │
       ▼
Token Embedding + Relative Position Bias
       │
       ▼
┌─────────────────────────────────────────┐
│  Encoder Layer ×4                       │
│  ┌─────────────────────────────────┐    │
│  │ Local-Global Self-Attention     │    │
│  │  • Local: window ±127 tokens    │    │
│  │  • Global: every 16th token     │    │
│  │  • Complexity: O(n·255) ≈ O(n)  │    │
│  └────────────────┬────────────────┘    │
│  Pre-LN → FFN (GELU) → Residual        │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Decoder Layer ×4                       │
│  • Causal Self-Attention                │
│  • Cross-Attention to encoder output    │
│  • FFN (GELU) with residual             │
│  • Weight-tied output projection        │
└───────────────────┬─────────────────────┘
                    │
                    ▼
Output Logits [16,000 vocab] → Summary
```

### Local-Global Attention

```
Local Attention:    Each token attends to ±127 neighbors (window = 255)
Global Attention:   Every 16th token attends to ALL tokens
Combined:           O(n × 255) instead of O(n²) — ~16× more efficient at 4K tokens
```

---

## File Structure

```
02_longt5/
├── src/
│   ├── model.py                       # Full LongT5 model (local-global attn)
│   ├── train.py                       # Training with streaming data loader
│   ├── inference.py                   # Beam search inference
│   ├── data_loader.py                 # IterableDataset for memory efficiency
│   ├── train_longt5.py                # Alternative training entry point
│   ├── inference_longt5.py            # Batched inference + ROUGE eval
│   ├── longt5_model_simplified.py     # Simplified model variant
│   └── visualize_architecture.py      # Generate architecture diagrams
│
├── configs/
│   ├── config.yaml                    # Primary LongT5 configuration
│   └── longt5_config.yaml             # Alternative configuration
│
├── scripts/
│   ├── test_longt5.py                 # Model functionality tests
│   └── decode_sanity.py               # Decoding sanity checks
│
├── docs/
│   ├── README.md                      # Quick start guide
│   ├── architecture.md                # Detailed architecture documentation
│   └── issue1_resolution.md           # How LongT5 resolves PG issues
│
└── visualizations/                    # 11 architecture diagrams
    ├── longt5_architecture.dot        # Full architecture (GraphViz)
    ├── longt5_architecture.pdf        # Rendered PDF
    ├── longt5_architecture_graphviz.dot
    ├── longt5_architecture_graphviz.pdf
    ├── longt5_architecture_matplotlib.png
    ├── longt5_compute_graph.dot
    ├── longt5_compute_graph.pdf
    ├── attention_patterns_graphviz.dot
    ├── attention_patterns_graphviz.pdf
    ├── attention_pattern_heatmap.png
    ├── layer_details_graphviz.dot
    └── layer_details_graphviz.pdf
```

---

## Training

```bash
python src/train.py --config configs/config.yaml
```

### Inference

```bash
python src/inference.py                  # Single-sample inference
python src/inference_longt5.py           # Batched evaluation with ROUGE
```

---

## Remaining Limitations

1. **Standard O(n²) cross-attention** in decoder — bottleneck for long sequences
2. **No copy mechanism** — cannot directly copy medical terms (drug names, dosages)
3. **Small model** (12.5M params) — limited capacity for complex medical language
4. **Trained from scratch** — no pre-trained medical knowledge
5. **GELU activation** — less efficient than modern alternatives (SwiGLU)

> These limitations motivated the transition to **Mamba-Transformer Hybrid** (Models 3 & 4).
