# V3 Changelog — Mamba-Transformer Hybrid for Clinical Note Summarization

> **Branch:** `feature/v3-source-bypass-training-improvements`
> **Date:** March 3, 2026
> **Resume checkpoint:** `checkpoint_step_35000.pt` → currently training at step ~39,300
> **Best ROUGE-L so far:** 0.1947 (step 34,000) — training ongoing toward 0.40+ target

---

## Summary of Changes

This update introduces **V3 architectural improvements** (Source Bypass, Copy Mechanism infrastructure), **critical training bug fixes**, **resilient checkpoint management**, and **hyperparameter tuning** to push the from-scratch Mamba-Transformer model beyond the ROUGE-L plateau at ~0.19.

---

## 1. Architecture Changes (`src/model.py`)

### 1.1 Source Bypass (NEW — V3)
- **What:** Decoder gets direct access to downsampled encoder output (every 8th token) concatenated with memory tokens
- **Why:** The memory compression bottleneck (4096 src tokens → 32 memory tokens per chunk) loses fine-grained detail. Source bypass gives the decoder direct access to source positions, enabling better content selection
- **How:** During `encode()`, every `source_bypass_stride`-th encoded vector is collected and concatenated with memory tokens before cross-attention
- **Config:** `source_bypass_stride: 8` (active), can be disabled by setting to 0

### 1.2 Copy Mechanism / Pointer-Generator (NEW — V3, currently disabled)
- **What:** Full `CopyMechanism` class implementing pointer-generator network
- **Why:** Clinical summaries often need to copy exact medical terms, drug names, lab values from source notes
- **How:** `p_gen * vocabulary_distribution + (1 - p_gen) * copy_distribution` where copy distribution scatters attention weights over source token IDs
- **Gate initialization:** Bias = 3.0 → sigmoid = 0.953, so model starts by mostly generating (safe default)
- **Status:** Class is fully implemented and integrated in `forward()`, `generate()`, and `_beam_search()`, but **disabled** in config (`use_copy_mechanism: false`) for training speed. Will be enabled for fine-tuning later
- **Speed impact when enabled:** ~8.85s/batch (vs ~1.4s without)

### 1.3 Gated Cross-Attention Fix (BUG FIX)
- **Before (V2):** Multiplicative gating `sigmoid(gate) * cross_attn_output` — initial gate bias of 0.0 → sigmoid(0.0) = 0.5, which halves the cross-attention signal. Over multiple layers, this compounds and suppresses source information
- **After (V3):** Additive gating `(1 + sigmoid(gate)) * cross_attn_output` — range [1.0, 2.0], initial value ~1.73x. Cross-attention signal is always at least 1x (never suppressed), with capacity to amplify up to 2x
- **Impact:** This was a root cause of ROUGE-L stagnation — the model was penalizing its own cross-attention

### 1.4 Forward Pass Returns Raw Logits
- **Before:** `forward()` returned `log_softmax(logits)` — problematic with AMP fp16 (log of small softmax values → NaN)
- **After:** `forward()` returns raw logits; loss is computed with `F.cross_entropy()` which is numerically stable under AMP (fused log_softmax + NLL internally)

### 1.5 Beam Search Implementation
- Full `_beam_search()` method (~180 lines) with:
  - Length penalty and n-gram blocking
  - Copy mechanism support (expanded source states per beam)
  - Configurable beam_size, temperature, repetition_penalty

---

## 2. Training Pipeline Changes (`src/train.py`)

### 2.1 Loss Function Fix
- **Before:** `LabelSmoothingLoss` with `label_smoothing=0.15` + `log_softmax` — label smoothing was reducing generation sharpness (distributing probability mass uniformly across vocabulary tokens)
- **After:** `F.cross_entropy(logits, targets)` with `label_smoothing=0.0` — clean cross-entropy with AMP-safe numerics

### 2.2 Evaluation Parameters
- **min_length = 200:** Forces model to generate at least 200 tokens during eval (clinical summaries are typically 200-400 tokens). Prevents artificially short summaries that hurt ROUGE
- **repetition_penalty = 1.0:** Removed penalty (was 1.3). With n-gram blocking active, additional repetition penalty was causing incoherent outputs
- **greedy decode for eval:** Deterministic, reproducible, fast

### 2.3 Trigger-File Save Mechanism (NEW)
- **What:** Training loop checks for a `SAVE_NOW` file in the workspace root every gradient step
- **When triggered:** Saves `manual_checkpoint_step_XXXXX.pt`, then deletes the trigger file
- **Purpose:** On-demand checkpoint saves without stopping training
- **Usage:** `New-Item SAVE_NOW` → checkpoint saved within ~22 seconds

### 2.4 Emergency Save on Crash/Exit (NEW)
- **What:** Saves `emergency_step_XXXXX.pt` when training crashes, receives kill signals, or exits
- **How:** 
  - Signal handlers for SIGTERM, SIGABRT, SIGBREAK (Windows)
  - `atexit` hook
  - Exception handler in `__main__`
  - `builtins._emergency_state` dict updated every gradient step with current model/optimizer/step/best_rouge_l
- **Purpose:** Never lose training progress to unexpected crashes

### 2.5 OOM Protection
- Forward/backward pass wrapped in `try/except RuntimeError`
- On OOM: skip batch, clear CUDA cache, continue training
- Prevents full training crashes from occasional long sequences

### 2.6 NaN/Inf Detection
- Loss checked for NaN/Inf after each forward pass
- Gradient norm checked before optimizer step
- Invalid batches are skipped with warnings

### 2.7 Checkpoint Loading Improvements
- `find_latest_checkpoint()`: Auto-finds latest `checkpoint_step_*.pt` by step number, falls back to `best_model.pt`
- `load_checkpoint()`: Flexible key matching (handles extra/missing keys), optional optimizer reset, optional gate bias boosting
- `reset_optimizer=True` on resume: Fresh LR warm restart from latest checkpoint

### 2.8 Cosine Schedule with Warm Restarts
- `num_lr_restarts: 2` — 3 cosine cycles over 150K steps
- Peak LR decays 0.7x per restart (1e-4 → 7e-5 → 4.9e-5)
- Floor at 10% of base LR
- Purpose: Escape loss plateaus with periodic LR re-warming

---

## 3. Configuration Changes (`configs/full_train.yaml`)

| Parameter | V2 Value | V3 Value | Rationale |
|---|---|---|---|
| `dropout` | 0.15 | **0.05** | Less regularization for faster convergence from scratch |
| `stride` | 192 | **256** | No overlap (16 chunks vs 21) → 30% faster throughput |
| `label_smoothing` | 0.15 | **0.0** | Removed — hurts generation quality |
| `source_bypass_stride` | — | **8** | NEW: Direct source access for decoder |
| `use_copy_mechanism` | — | **false** | NEW: Disabled for speed |
| `batch_size` | 2 | **1** | Reduced for VRAM headroom |
| `gradient_accumulation` | 8 | **16** | Same effective batch size (16) |
| `max_steps` | 50,000 | **150,000** | 3x longer for 10+ epochs on 270K samples |
| `warmup_steps` | 2,000 | **500** | Faster ramp-up |
| `learning_rate` | 5e-5 | **1e-4** | 2x higher for from-scratch training |
| `eval_steps` | 1,000 | **5,000** | Less overhead (eval takes ~10 min) |
| `save_steps` | 200 | **5,000** | Less I/O (each save = 917MB) |
| `num_lr_restarts` | — | **2** | Cosine warm restarts |

---

## 4. Evaluation Changes (`src/evaluate.py`)

- `max_gen_len`: 256 → **512** (clinical summaries can be long)
- Generation: greedy → **beam search** (beam_size=4, length_penalty=0.8, no_repeat_ngram_size=3, repetition_penalty=1.2)

---

## 5. New Files Added

| File | Purpose |
|---|---|
| `CLINICAL_METRICS.md` | 643-line guide on clinical evaluation metrics beyond ROUGE |
| `MODEL_LEARNING.md` | 644-line explanation of how the model learns |
| `MODEL_MATHEMATICS.md` | 657-line mathematical derivation of every model component |
| `app.py` | Gradio web application for inference with text + speech input |
| `fast_eval.py` | Fast greedy-first clinical evaluation script |
| `gated_eval_v2.py` | Full 3-gate clinical evaluation (Safety → Quality → Similarity) |
| `monitor_training.py` | Real-time training progress monitor |
| `scripts/batch_eval.py` | Batch evaluation across multiple checkpoints |
| `start_training.bat` | Windows batch file to launch training |

---

## 6. Documentation Fixes

- **ARCHITECTURE.md:** Fixed corruption where training log lines leaked into section 15 (Weight Tying)

---

## 7. Training Progress

| Step | Loss | ROUGE-L | Notes |
|---|---|---|---|
| 34,000 | ~3.4 | **0.1947** | Best model so far (saved as `best_model.pt`) |
| 35,000 | ~3.3 | — | Auto-checkpoint saved |
| 38,506 | ~3.24 | — | Manual trigger-file checkpoint |
| 39,300 | ~3.21 | — | Currently training (eval at step 40,000) |

**Loss trend:** Steadily decreasing from 3.4 → 3.2 (healthy convergence)
**Gradient norms:** 1.2-2.0 (healthy, no explosion/vanishing)
**Speed:** ~1.4s/batch, ~1,500 tok/s, ~22s per gradient step
**GPU:** 3.6GB peak / 8GB available

---

## 8. Checkpoint Files (not in git — too large)

| File | Size | Step | Description |
|---|---|---|---|
| `best_model.pt` | 920 MB | 34,000 | Best ROUGE-L = 0.1947 |
| `checkpoint_step_35000.pt` | 917 MB | 35,000 | Auto-save at 35K |
| `manual_checkpoint_step_38506.pt` | 917 MB | 38,506 | Manual trigger-file save |
| `safety_backup_best_model.pt` | 920 MB | 34,000 | Safety copy |
| `safety_backup_step_35000.pt` | 917 MB | 35,000 | Safety copy |

---

## 9. Model Architecture Summary

```
Input: Clinical Note (up to 4096 tokens)
  ↓
[SentencePiece Tokenizer] → 16,000 vocab
  ↓
[Token Embedding + Positional Encoding] → (seq_len, 512)
  ↓
[Chunking] → 16 chunks of 256 tokens each (stride=256)
  ↓ per chunk
[BiMamba Encoder × 6] → Forward + Backward SSM with gated fusion
  ↓ per chunk
[Memory Token Compressor] → 32 learned queries attend to chunk → (32, 512)
  ↓ source bypass: every 8th encoded token saved
[Cross-Chunk Attention × 2] → Memory tokens attend across all chunks
  ↓
[Concat: Memory (32×16=512 tokens) + SourceBypass (~64 tokens)]
  ↓
[Transformer Decoder × 6]
  ├── Causal Self-Attention (8 heads)
  ├── Additive Gated Cross-Attention to encoder memory+bypass
  └── SwiGLU FFN
  ↓
[Output Projection] → (vocab_size=16000) with weight tying
  ↓
Output: Brief Hospital Course Summary
```

**Total Parameters:** 79,410,694 (all trainable, from scratch)
