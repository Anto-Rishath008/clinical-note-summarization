# Architecture: Mamba–Transformer Hybrid for Clinical Note Summarization

> **Model name:** `MambaTransformerModel`  
> **Task:** Automatic summarization of MIMIC-IV discharge notes → Brief Hospital Course (BHC)  
> **Total parameters:** 52,146,176  
> **Active experiment:** `exp_memory32` — Memory Token Isolation (n_memory_tokens = 32)

---

## 1. Overview

This model is a **from-scratch hybrid encoder–decoder** that combines:

- A **Mamba SSM encoder** for linear-time long-sequence encoding
- A **learned memory token compressor** to form a compact, fixed-size cross-attention context
- A **standard Transformer decoder** to generate summaries autoregressively

The key insight is that clinical notes can be extremely long (thousands of tokens), and standard self-attention scales quadratically. Mamba provides $\mathcal{O}(n)$ sequence modeling, while the memory compressor distills each chunk down to $K$ vectors, giving the decoder a fixed-size "memory" regardless of source length.

```
Input tokens  ──► Chunk ──► Mamba Encoder ──► Memory Compressor
                  ×N_chunks                        │
                                            K memory tokens/chunk
                                                   │
                                            Concatenate all chunks
                                                   │
Target tokens ──► Embedding ──► Transformer Decoder (cross-attn to memory)
                                                   │
                                            Output Projection ──► Vocabulary
```

---

## 2. Model Configuration (exp_memory32)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `vocab_size` | 16,000 | SentencePiece BPE tokenizer |
| `d_model` | 512 | Embedding / hidden dimension |
| `n_mamba_layers` | 6 | Mamba encoder blocks |
| `n_decoder_layers` | 6 | Transformer decoder layers |
| `n_heads` | 8 | Multi-head attention heads |
| `d_ff` | 2,048 | Feed-forward inner dimension |
| `dropout` | 0.15 | Applied in all sub-layers |
| `chunk_size` | 256 | Tokens per encoder chunk |
| `stride` | 192 | Stride between chunks (64-token overlap) |
| `n_memory_tokens` | **32** | Memory tokens per chunk (experiment variable) |
| `max_src_len` | 4,096 | Maximum source tokens |
| `max_tgt_len` | 512 | Maximum target tokens |
| `mamba_d_state` | 16 | Mamba SSM state dimension |
| `mamba_d_conv` | 4 | Mamba convolution kernel size |
| `mamba_expand` | 2 | Mamba expansion factor |
| `label_smoothing` | 0.15 | Cross-entropy smoothing |

---

## 3. Component Breakdown

### 3.1 Tokenizer

- **Type:** SentencePiece unigram model (`data/tokenizer/spm.model`)
- **Vocabulary:** 16,000 subword tokens
- **Special tokens:** `<bos>=1`, `<eos>=2`, `<pad>=3`

---

### 3.2 Embeddings

```
src_embedding : Embedding(16000, 512)   ← source input tokens
tgt_embedding : Embedding(16000, 512)   ← target output tokens (shared with output projection)
pos_encoding  : Sinusoidal PE (max 8192 positions)
```

Embeddings are Xavier-initialized. Output projection weight is **tied** to `tgt_embedding` (weight sharing reduces parameters and improves generation quality).

---

### 3.3 Chunking Strategy

Before encoding, the source sequence is sliced into overlapping windows:

```
source: [t_0, t_1, ..., t_L]

chunk_0 : tokens [0   : 256 ]
chunk_1 : tokens [192 : 448 ]   ← 64-token overlap with chunk_0
chunk_2 : tokens [384 : 640 ]
...
```

- **chunk_size = 256 tokens** — fits within Mamba's efficient processing range
- **stride = 192** (overlap = 64) — ensures continuity at chunk boundaries
- Chunks are zero-padded to `chunk_size` if the last chunk is shorter

For a 4096-token input: `⌈(4096 - 256) / 192⌉ + 1 = 21 chunks`

---

### 3.4 Mamba Encoder

Each chunk is independently encoded by a stack of **6 Mamba blocks**:

```
MambaEncoder
└─ MambaBlock × 6
   ├─ LayerNorm
   ├─ Mamba SSM  (d_model=512, d_state=16, d_conv=4, expand=2)
   ├─ Dropout(0.15)
   └─ Residual connection
└─ Final LayerNorm
```

**Mamba SSM** ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752)) is a State Space Model that:
- Processes sequences in $\mathcal{O}(n)$ time and memory (vs $\mathcal{O}(n^2)$ for attention)
- Uses selective state transitions controlled by the input itself
- Has a hardware-aware parallel scan for fast GPU training

Each chunk `[batch, 256, 512]` → encoded `[batch, 256, 512]`

---

### 3.5 Memory Token Compressor

This is the **architectural centrepiece** — it converts a variable-length chunk representation into exactly $K$ fixed-size memory tokens via learned cross-attention:

```
MemoryTokenCompressor
├─ memory_queries : nn.Parameter(K=32, d_model=512)   ← learned
├─ cross_attn     : MultiheadAttention(512, 8 heads)
├─ LayerNorm
└─ Dropout(0.15)

Forward:
  queries [B, 32, 512] × key/value [B, 256, 512]  →  memory_tokens [B, 32, 512]
```

**Why this matters:**
- Each chunk (256 tokens) is compressed to 32 memory tokens
- **Compression ratio: 256 → 32 = 8× per chunk (12.5% retention)**
- The baseline used K=8 (3.1% retention), which caused information loss
- With K=32, the decoder gets 4× more information per chunk

For 21 chunks: `21 × 32 = 672 memory tokens` total context for the decoder.

---

### 3.6 Transformer Decoder

The decoder is a standard pre-norm Transformer with 6 layers:

```
TransformerDecoder
└─ TransformerDecoderLayer × 6
   ├─ LayerNorm → Masked Self-Attention → Dropout + Residual
   ├─ LayerNorm → Cross-Attention (→ memory tokens) → Dropout + Residual
   ├─ LayerNorm → Feed-Forward (512→2048→512, GELU) → Residual
   └─ (causal mask enforced in self-attention)
└─ Final LayerNorm
```

**Cross-attention:** decoder tokens attend to all `N_chunks × K` memory tokens. This gives the decoder full access to the compressed source representation regardless of source length.

---

### 3.7 Output Projection

```
Linear(512, 16000, bias=False)   ← weight tied to tgt_embedding
```

Followed by cross-entropy loss with label smoothing (ε = 0.15).

---

## 4. Information Flow

```
Source (up to 4096 tokens)
        │
        ▼
   Chunk into 21 windows of 256 tokens (stride 192)
        │
        ▼  For each chunk:
   ┌─────────────────────────────────────┐
   │  src_embedding + pos_encoding       │
   │  → MambaEncoder (6 layers)          │  [B, 256, 512]
   │  → MemoryCompressor (K=32 queries)  │  [B, 32, 512]
   └─────────────────────────────────────┘
        │
        ▼
   Concatenate all chunks → memory [B, 672, 512]
        │
        ▼
   TransformerDecoder (6 layers) ────────────────── cross-attn to memory
        │
        ▼
   Output projection → logits [B, tgt_len, 16000]
        │
        ▼
   Autoregressive generation (greedy or beam search)
```

---

## 5. Training Setup (exp_memory32)

| Setting | Value |
|---------|-------|
| Dataset | MIMIC-IV Brief Hospital Course (BHC) |
| Training samples | 256,529 |
| Validation samples | 13,502 |
| Total steps | 50,000 |
| Warmup steps | 2,000 (cosine LR) |
| Learning rate | 5 × 10⁻⁵ |
| Batch size | 2 × grad_accum 8 = **16 effective** |
| Optimizer | AdamW (weight_decay=0.03) |
| Gradient clipping | max_norm = 1.0 |
| Mixed precision | AMP (float16) |
| Checkpointing | Every 200 steps |
| Evaluation | Every 1,000 steps (ROUGE-1/2/L + Val Loss) |
| GPU | NVIDIA RTX 4070 Laptop (8.59 GB VRAM) |
| Framework | PyTorch 2.5.1 + CUDA 12.1 |

---

## 6. Experiment: Memory Token Isolation

The `exp_memory32` config is a **controlled ablation** to test the memory bottleneck hypothesis:

| | Baseline (`full_train`) | **Memory32** (`exp_memory32`) |
|---|---|---|
| `n_memory_tokens` | 8 | **32** |
| Information retention | 3.1% per chunk | **12.5% per chunk** |
| Total memory tokens | ~168 | **~672** |
| All other settings | — | **IDENTICAL** |

**Hypothesis:** The K=8 baseline suffers from template hallucination and clinical entity loss because the MemoryTokenCompressor cannot encode sufficient source fidelity at 3.1% retention. K=32 (4× more) should yield improved ROUGE scores and clinical accuracy.

**Best ROUGE-L so far (step ~3400):** 0.1814

---

## 7. File Structure

```
workspace_mamba_longt5_v1/
├── src/
│   ├── model.py           # MambaTransformerModel — full architecture
│   ├── train.py           # Training loop (AMP, cosine LR, auto-resume)
│   ├── data_loader.py     # MIMIC-IV CSV loader + SentencePiece tokenizer
│   ├── clinical_eval.py   # Clinical evaluation (greedy + beam, 3 gates)
│   ├── evaluate.py        # ROUGE + Val loss evaluation
│   └── __init__.py
├── configs/
│   ├── exp_memory32.yaml  # Active experiment config (K=32)
│   ├── full_train.yaml    # Baseline config (K=8) for comparison
│   └── default.yaml       # Base defaults
├── gated_eval_v2.py       # Gated evaluation: Safety / Quality / Similarity
├── fast_eval.py           # Quick ROUGE evaluation on checkpoints
├── monitor_training.py    # Live training log monitor
├── scripts/
│   └── batch_eval.py      # Batch evaluation across multiple checkpoints
├── requirements.txt
├── .gitignore
└── ARCHITECTURE.md        # This file
```

---

## 8. Dependencies

```
torch>=2.0.0
mamba-ssm>=1.0.1          # Mamba SSM CUDA kernels
sentencepiece>=0.1.99     # Tokenizer
rouge-score               # ROUGE evaluation
PyYAML>=6.0
tqdm
numpy
```

See [requirements.txt](requirements.txt) for pinned versions.

---

## 9. Reproducing the Experiment

```bash
# Install dependencies
pip install -r requirements.txt

# Train (auto-resumes from latest checkpoint if interrupted)
python src/train.py --config configs/exp_memory32.yaml

# Monitor live training
python monitor_training.py

# Evaluate a checkpoint
python fast_eval.py --checkpoint checkpoints/memory32_run/best_model.pt \
                    --config configs/exp_memory32.yaml \
                    --max_samples 200

# Full clinical evaluation with beam search
python src/clinical_eval.py \
    --checkpoint checkpoints/memory32_run/best_model.pt \
    --config configs/exp_memory32.yaml \
    --max_samples 100 --beam_size 4 --compare_greedy
```

---

## 10. References

- **Mamba:** Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- **MIMIC-IV:** Johnson, A., et al. (2023). *MIMIC-IV, a freely accessible electronic health record dataset.* Scientific Data.
- **Transformer:** Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
- **Memory Tokens / Perceiver:** Jaegle, A., et al. (2021). *Perceiver: General Perception with Iterative Attention.* ICML.
