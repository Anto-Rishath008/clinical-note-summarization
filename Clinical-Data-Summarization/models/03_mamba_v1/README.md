# Model 3: Mamba-Transformer Hybrid V1

<p align="center"><strong>Linear-Time SSM Encoder + Transformer Decoder — First Hybrid Attempt</strong></p>

---

## Overview

Mamba V1 replaces LongT5's quadratic attention encoder with a **Mamba State Space Model (SSM)** that processes sequences in **O(n) linear time**. The encoder uses a bidirectional Mamba (BiMamba) architecture with gated fusion, while the decoder retains cross-attention for generation.

### Why Mamba over LongT5?

| Limitation in LongT5 | Mamba V1 Solution |
|----------------------|-------------------|
| O(n²) cross-attention in decoder | Memory token compression (4K → 512 tokens) |
| No copy mechanism | Infrastructure for copy mechanism |
| Small model (12.5M) | Scaled to 52.1M parameters (d=512) |
| GELU activation | Foundation for modern alternatives |
| O(n·255) encoder attention | True O(n) linear SSM scan |

### Key Characteristics

| Property | Value |
|----------|-------|
| **Architecture** | BiMamba SSM Encoder + Transformer Decoder |
| **Parameters** | 52.1M |
| **d_model** | 512 |
| **Encoder Layers** | 6 (BiMamba) |
| **Decoder Layers** | 6 (Transformer) |
| **Max Input Length** | 4,096 tokens |
| **Memory Tokens** | 32 per chunk (512 total) |
| **Training Steps** | 40,000 |

### ROUGE Scores

| Metric | Score | vs LongT5 |
|--------|-------|-----------|
| ROUGE-1 | 0.3484 | -0.9% |
| ROUGE-2 | 0.1027 | -16.6% |
| ROUGE-L | 0.1952 | -10.7% |

> **Note:** V1 underperforms LongT5 despite 4× more parameters, due to a critical **cross-attention suppression bug** identified and fixed in V2.

---

## Architecture

```
Input Clinical Note [4,096 tokens]
       │
       ▼
SentencePiece Tokenization [16K vocab]
       │
       ▼
Chunking: 16 chunks × 256 tokens
       │
       ▼
┌─────────────────────────────────────────┐
│  BiMamba Encoder Layer ×6               │
│  ┌─────────────────────────────────┐    │
│  │ Forward SSM:  y_t→ = SSM(x₁..xₜ)│   │
│  │ Backward SSM: y_t← = SSM(xₜ..x₁)│   │
│  │ Gated Fusion:                    │    │
│  │   g_t = σ(W_g[y_t→; y_t←])     │    │
│  │   y_t = g_t·y_t→ + (1-g_t)·y_t←│    │
│  └────────────────┬────────────────┘    │
│  LayerNorm → FFN (GELU) → Residual     │
└───────────────────┬─────────────────────┘
                    │
                    ▼
Memory Token Compression [32 per chunk → 512 total]
                    │
                    ▼
Transformer Decoder ×6 [Causal Self-Attn + Cross-Attn + FFN]
                    │
                    ▼
Output → Logits [16,000] → Summary
```

### Mamba SSM Core Equations

**Continuous-time SSM:**
```
h'(t) = A·h(t) + B·x(t)
y(t) = C·h(t)
```

**Discrete-time (after ZOH discretization):**
```
h_t = Ā·h_{t-1} + B̄·x_t
y_t = C·h_t + D·x_t
```

**Selective Mechanism (what makes it "Mamba"):**
```
Δ_t = softplus(x_t · W_Δ)     ← Input-dependent step size
B_t = x_t · W_B                ← Input-dependent input matrix
C_t = x_t · W_C                ← Input-dependent output matrix
```

---

## File Structure

```
03_mamba_v1/
├── src/
│   ├── model.py               # Mamba-Transformer hybrid model
│   ├── train.py               # Training pipeline with AMP
│   ├── data_loader.py         # Streaming data loader with chunking
│   ├── evaluate.py            # ROUGE evaluation
│   └── __init__.py
│
├── configs/
│   ├── default.yaml           # V1 default configuration
│   └── exp_memory32.yaml      # Experimental memory token config
│
├── scripts/
│   ├── batch_eval.py          # Batch evaluation
│   ├── launch_train.py        # Training launcher
│   ├── check_env.py           # Environment validation
│   ├── debug_check.py         # Debug utilities
│   └── verify_fixes.py        # Fix verification
│
├── checkpoints/
│   ├── best_model.pt          # Best BART baseline checkpoint
│   └── bart_step_13000.pt     # BART reference checkpoint
│
└── logs/
    ├── train.log              # Training log
    ├── crash_report.log       # Crash diagnostics
    └── process_check.txt      # Process monitoring
```

---

## Critical Bug: Cross-Attention Suppression

The key finding from V1 was a **cross-attention suppression bug** where the decoder's cross-attention weights were being driven to near-zero, effectively ignoring the encoder output:

```
Cross-attention output ≈ 0 → Decoder relies only on self-attention → Poor generation
```

**Root cause:** The standard cross-attention mechanism in the decoder was not properly gated, causing the signal from compressed memory tokens to be suppressed during training.

**Fix in V2:** Additive gated cross-attention with learnable skip connection:
```python
# V2 fix: Additive gated cross-attention
gate = sigmoid(linear(x))  # Learnable gate
cross_out = gate * cross_attention(x, memory) + (1 - gate) * x
```

> This bug discovery directly motivated all the improvements in **Mamba V2** (Model 4).

---

## Training

```bash
python src/train.py --config configs/default.yaml
```

## Evaluation

```bash
python src/evaluate.py
python scripts/batch_eval.py
```
