# Model 1: Pointer-Generator Network

<p align="center"><strong>Baseline Model — BiLSTM Encoder + Attention + Copy Mechanism</strong></p>

---

## Overview

The Pointer-Generator Network serves as the **baseline model** in our progressive architecture study. It combines a bidirectional LSTM encoder with Bahdanau attention and a pointer-generator mechanism that can either generate words from the vocabulary or copy them directly from the source document.

### Key Characteristics

| Property | Value |
|----------|-------|
| **Architecture** | BiLSTM Encoder → Attention → Pointer-Generator Decoder |
| **Parameters** | 22.5M |
| **Max Input Length** | 768 tokens (severe limitation for clinical notes) |
| **Max Target Length** | 192 tokens |
| **Embedding Dim** | 128 |
| **Hidden Dim** | 256 |
| **Encoder Layers** | 2 (Bidirectional LSTM) |
| **Training Steps** | 500 / 50,000 (severely undertrained) |
| **Best Val Loss** | 4.80 |

### ROUGE Scores

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.1814 |
| ROUGE-2 | 0.0030 |
| ROUGE-L | 0.0855 |

---

## Architecture

```
Input Clinical Note (truncated to 768 tokens)
       │
       ▼
Token Embedding (128-dim)
       │
       ▼
Bidirectional LSTM ×2 (hidden=256, output=512)
       │
       ├─── Chunked Processing (6 chunks × 128 tokens)
       │
       ▼
Bahdanau Attention
       │
       ├── Attention Distribution α_t
       ├── Context Vector c_t
       │
       ▼
Pointer-Generator Switch (p_gen)
       │
       ├── Generate: P_vocab(w) from vocabulary
       ├── Copy: P_copy(w) from source attention
       │
       ▼
Final: P(w) = p_gen · P_vocab + (1-p_gen) · P_copy
       │
       ▼
Coverage Mechanism (prevents repetition)
```

### Mathematical Formulation

**Encoder:**
```
h_t^enc = [LSTM_fwd(e_t, h_{t-1}); LSTM_bwd(e_t, h_{t+1})]
```

**Bahdanau Attention:**
```
e_t^i = v^T tanh(W_h · h_i^enc + W_d · h_t^dec + b)
α_t = softmax(e_t)
c_t = Σ α_t^i · h_i^enc
```

**Pointer-Generator:**
```
p_gen = σ(w^T [h_t^dec; c_t; e_{y_{t-1}}] + b)
P(w) = p_gen · P_vocab(w) + (1 - p_gen) · P_copy(w)
```

---

## File Structure

```
01_pointer_generator/
├── src/
│   ├── core.py                    # Complete model implementation
│   ├── data_loader.py             # Data loading utilities
│   ├── train.py                   # Training script
│   ├── evaluate.py                # ROUGE evaluation
│   └── inference_visualization.py # Inference + visualization
│
├── configs/
│   ├── default.yaml               # Default configuration
│   ├── rtx4070_8gb.yaml          # GPU-optimized config
│   ├── stage1_fromscratch.yaml   # From-scratch training
│   ├── resume_config.yaml        # Resume interrupted training
│   └── full_training_final.yaml  # Final training config
│
├── scripts/
│   ├── baseline_eval.py           # Lead-K baseline evaluation
│   ├── demo_inference.py          # Demo with sample outputs
│   ├── eval_pipeline.py           # ROUGE tracking pipeline
│   ├── post_training_analysis.py  # Post-training analysis
│   ├── overfit_test.py            # Overfitting sanity check
│   └── single_batch_step_test.py  # Single batch test
│
├── checkpoints/                   # Saved model weights
│   ├── best_model.pt              # Best validation loss
│   └── checkpoint_step_*.pt       # Step-wise checkpoints
│
├── logs/                          # Training logs
│   ├── training_log.txt
│   ├── training_stdout*.txt
│   ├── training_stderr*.txt
│   └── metrics.csv
│
└── visualizations/                # Architecture diagrams
    ├── pointer_generator_architecture.dot
    ├── pointer_generator_architecture_v2.dot
    └── pointer_generator_modular.dot
```

---

## Training

```bash
python src/train.py --config configs/default.yaml
```

### Training Configuration
- **Optimizer:** Adam (lr=3×10⁻⁴)
- **Batch Size:** 1 (gradient accumulation: 16 → effective batch: 16)
- **Mixed Precision:** FP16
- **Gradient Clipping:** max_norm = 1.0
- **Scheduler:** Cosine LR with warmup (1,000 steps)
- **Loss:** NLL + Coverage Loss (λ_cov weighted)

---

## Identified Issues

1. **Max 768 tokens** — Clinical notes are 1K–8K tokens; severe information loss
2. **Coverage tensor OOM** — Memory explosion on 8 GB GPU
3. **Chunking artifacts** — 128-token chunks cause discontinuities at boundaries
4. **Hardcoded CUDA** — No device-agnostic support
5. **Severely undertrained** — Only 500/50,000 steps completed

> These limitations directly motivated the transition to **LongT5** (Model 2).

---

## Evaluation

```bash
python src/evaluate.py
python scripts/baseline_eval.py      # Compare with Lead-K baselines
python scripts/demo_inference.py     # Generate sample summaries
```
