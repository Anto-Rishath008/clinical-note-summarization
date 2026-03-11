# Model 4: Mamba-Transformer Hybrid V2

<p align="center">
  <strong>★ Best Model — BiMamba + Gated Cross-Attention + SwiGLU + Source Bypass</strong><br/>
  <em>79.4M Parameters • 150K Training Steps • ROUGE-L: 0.2296</em>
</p>

---

## Overview

Mamba V2 is the **final and best-performing model** in our progressive architecture study. It addresses every issue identified in V1 through five key innovations: **additive gated cross-attention**, **SwiGLU activation**, **RMSNorm**, **source bypass**, and **cosine warm restarts**. Trained for 150K steps on a single RTX 4070 (8 GB), it surpasses all previous models and the Lead-150 extractive baseline.

### Performance Summary

| Metric | Score | vs Pointer-Gen | vs LongT5 | vs Mamba V1 | vs Lead-150 |
|--------|-------|----------------|-----------|-------------|-------------|
| **ROUGE-1** | **0.3840** | +111.7% | +9.2% | +10.2% | +52.2% |
| **ROUGE-2** | **0.1386** | +4,520% | +12.5% | +35.0% | — |
| **ROUGE-L** | **0.2296** | +168.5% | +5.1% | +17.6% | +52.1% |
| **Val Loss** | **2.56** | -46.7% | -39.0% | -21.2% | — |

---

## What Changed from V1 to V2

| Issue in V1 | V2 Fix | Impact |
|-------------|--------|--------|
| Cross-attention suppression (weights → 0) | **Additive gated cross-attention** with bias=3.0 | Decoder properly attends to encoder output |
| GELU activation (no gating) | **SwiGLU** (learnable gate) | Richer feature interaction, used in LLaMA/PaLM |
| LayerNorm (redundant mean shift) | **RMSNorm** (15-20% faster) | Faster training per step |
| No fine-grained detail preservation | **Source bypass** (every 8th token) | Decoder accesses raw token detail |
| Step-based LR schedule | **Cosine warm restarts** (3 cycles) | Escapes local minima during 150K steps |
| LayerNorm (batch=1 instability risk) | **RMSNorm** | More stable with effective batch=1 |

---

## Architecture

```
Input Clinical Note [4,096 tokens]
       │
       ▼
SentencePiece Tokenization [16K vocab]
       │
       ▼
Chunking: 16 chunks × 256 tokens (stride=256)
       │
       ▼
┌──────────────────────────────────────────────────┐
│  BiMamba Encoder ×6                              │
│  ┌────────────────────────────────────────────┐  │
│  │ Forward SSM:  y_t→ = MambaSSM(x₁, ..., xₜ)│  │
│  │ Backward SSM: y_t← = MambaSSM(xₜ, ..., x₁)│  │
│  │ Gated Fusion:                               │  │
│  │   g_t = σ(W_g [y_t→; y_t←])               │  │
│  │   y_t = g_t · y_t→ + (1 - g_t) · y_t←    │  │
│  └───────────────────┬────────────────────────┘  │
│  RMSNorm → SwiGLU FFN → Residual                │
└─────────────┬────────────────┬───────────────────┘
              │                │
              ▼                ▼
    Memory Compressor     Source Bypass
    [32 queries/chunk     [every 8th encoder
     → attn pooling]      token saved]
              │                │
              ▼                │
    Cross-Chunk Attention ×2   │
    [memory intercommunication]│
              │                │
              ▼────────────────┘
    Concat: Memory (512) + Bypass (~64) tokens
              │
              ▼
┌──────────────────────────────────────────────────┐
│  Transformer Decoder ×6                          │
│  • Causal Self-Attention                         │
│  • ★ Additive Gated Cross-Attention              │
│      gate = σ(W·x + bias)   [bias=3.0]          │
│      out = gate · cross_attn(x, mem) + (1-gate)·x│
│  • SwiGLU FFN + RMSNorm + Residual              │
└─────────────┬────────────────────────────────────┘
              │
              ▼
    Output Projection → Logits [16,000] → Summary
```

---

## Key Innovations

### 1. Additive Gated Cross-Attention
```python
# The critical fix — prevents cross-attention suppression
gate = sigmoid(W_gate @ x + bias)  # bias=3.0 → σ(3.0)=0.953
output = gate * cross_attention(query, memory) + (1 - gate) * query
```
**Why bias=3.0?** Initializes gates mostly open (95.3%), ensuring the decoder uses encoder information from the start of training.

### 2. SwiGLU Activation
```python
# Replaces GELU — learnable gating for richer features
SwiGLU(x) = (Swish(x @ W1)) ⊙ (x @ W2)
```
Used in LLaMA, PaLM — proven superior to GELU/ReLU for deep networks.

### 3. RMSNorm
```python
# Replaces LayerNorm — no mean subtraction, 15-20% faster
RMSNorm(x) = x / RMS(x) * γ
```

### 4. Source Bypass
Every 8th encoder token is directly passed to the decoder, bypassing compression. This preserves fine-grained detail (drug names, dosages) that memory compression might lose.

### 5. Cosine Warm Restarts
```
η_t = η_min + (η_max - η_min)/2 · (1 + cos(π · t_local / T_cycle))
```
3 cycles over 150K steps with 0.7× peak decay per restart.

---

## File Structure

```
04_mamba_v2/
├── src/
│   ├── model.py                   # ★ Full V2 model (79.4M params)
│   ├── train.py                   # Training pipeline (150K steps)
│   ├── data_loader.py             # Streaming chunked data loader
│   ├── evaluate.py                # ROUGE evaluation
│   ├── clinical_eval.py           # 3-tier clinical evaluation
│   └── __init__.py
│
├── configs/
│   ├── full_train.yaml            # ★ Primary training config (150K steps)
│   ├── default.yaml               # Default configuration
│   └── exp_memory32.yaml          # Memory token experiment config
│
├── scripts/
│   ├── comprehensive_eval.py      # Full evaluation suite
│   ├── fast_eval.py               # Quick evaluation
│   ├── gated_eval_v2.py           # Gated clinical metrics
│   ├── monitor_training.py        # Real-time training monitor
│   ├── run_train.py               # Resilient auto-restart trainer
│   ├── batch_eval.py              # Batch evaluation
│   ├── verify_v3.py               # V3 feature verification
│   ├── start_training.bat         # Windows training launcher
│   ├── restart_train.bat          # Windows restart script
│   └── patches/                   # Bug fixes & debugging
│       ├── patch1.py
│       ├── patch2.py
│       ├── patch3.py
│       ├── patch_oom.py
│       ├── patch_train.py
│       ├── fix_clean.py
│       ├── fix_dup.py
│       └── fix_try.py
│
├── checkpoints/
│   ├── best_model.pt              # ★ Best model (by ROUGE-L)
│   ├── final_model.pt             # Final model at step 150K
│   ├── safety_backup_best_model.pt
│   ├── emergency_step_150000.pt   # Emergency save
│   └── checkpoint_step_*.pt       # Milestone checkpoints (50K, 75K, 100K, 125K, 130K, 150K)
│
├── logs/
│   ├── train.log                  # Primary training log
│   ├── train_out.log              # stdout
│   ├── train_err.log              # stderr
│   ├── train_stdout.log
│   ├── train_stderr.log
│   ├── train_restart_out.log
│   ├── train_restart_err.log
│   └── train_status.txt           # Current status
│
├── docs/
│   ├── ARCHITECTURE.md            # ★ Complete architecture reference
│   ├── CHANGELOG_V3.md            # V3 improvements changelog
│   ├── CLINICAL_METRICS.md        # Clinical evaluation metrics
│   ├── MODEL_LEARNING.md          # Learning insights & analysis
│   └── MODEL_MATHEMATICS.md       # Mathematical foundations
│
├── webapp/
│   ├── app.py                     # Gradio web interface
│   ├── server.py                  # Flask web server
│   ├── launch.bat                 # Windows launcher
│   └── templates/
│       └── index.html             # Flask HTML template
│
└── requirements.txt               # Model-specific dependencies
```

---

## Training

### Full Training (150K steps)
```bash
python src/train.py --config configs/full_train.yaml
```

### Resilient Training (auto-restart on crash)
```bash
python scripts/run_train.py
```

### Training Configuration
| Setting | Value |
|---------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.999, λ=0.03) |
| Learning Rate | 3×10⁻⁴ (peak) |
| Scheduler | Cosine with warm restarts (3 cycles) |
| Batch Size | 1 (grad accum: 16 → effective: 16) |
| Mixed Precision | FP16 AMP |
| Gradient Clipping | max_norm = 1.0 |
| Total Steps | 150,000 |
| Eval Frequency | Every 5,000 steps |
| Throughput | ~1.4 s/batch, 1,500 tok/s, 3.6 GB peak |

### Training Robustness Features
- **Trigger-file save:** Touch `SAVE_NOW` for on-demand checkpoint
- **Emergency save:** Signal handlers (SIGTERM, SIGABRT)
- **OOM protection:** Skip batch + clear CUDA cache
- **NaN/Inf detection:** Gradient sanity checks
- **Atomic saves:** temp → rename (no corruption)

---

## Evaluation

```bash
# Quick evaluation
python scripts/fast_eval.py

# Comprehensive evaluation with clinical metrics
python scripts/comprehensive_eval.py

# Gated clinical evaluation (Safety → Quality → Similarity)
python scripts/gated_eval_v2.py
```

### Inference with Beam Search
- **Width:** 4
- **Length Penalty:** 0.8
- **N-gram Blocking:** n=3 (prevents trigram repetition)
- **Min Output:** 200 tokens

---

## Web Interface

### Gradio UI
```bash
python webapp/app.py
# Opens browser at http://localhost:7860
```

### Flask UI
```bash
python webapp/server.py
# Opens browser at http://localhost:5000
```

Both interfaces support:
- Text input of clinical notes
- Configurable generation parameters
- Real-time summary generation
- ROUGE score display

---

## Clinical Evaluation Framework

The model includes a proposed **4-gate clinical evaluation framework**:

| Gate | Focus | Metrics |
|------|-------|---------|
| **Gate 1** | Safety | PHI leakage (0%), negation consistency (<2%), contradiction rate (<15%) |
| **Gate 2** | Clinical Accuracy | MEDCON (UMLS F1), Medical Entity F1, ICD-10 coverage |
| **Gate 3** | Semantic Quality | BERTScore (Bio_ClinicalBERT), BARTScore |
| **Gate 4** | Surface Overlap | ROUGE-1/2/L, repetition rate, length ratio |
