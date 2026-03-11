# ROUGE Optimization Plan: Clinical Note Summarization

**Date:** January 13, 2026  
**Model Lead:** GitHub Copilot  
**Objective:** Maximize ROUGE (especially ROUGE-L) with practical, executable strategies

---

## Executive Summary

**Current Status:**
- **Model:** Pointer-Generator with Coverage (1-layer LSTM, 384 hidden dim)
- **Training:** Only 500 steps (severely undertrained)
- **Performance:** ROUGE-L = 0.0855 (WORSE than Lead-150 baseline = 0.1510)
- **Critical Issue:** Model outputs are incoherent (dots, fragments, low diversity)

**Decision:** **PATH B (Pretrained Transformer)** is strongly recommended for highest ROUGE.
- PATH A (improve pointer-gen) can reach ~0.25-0.30 ROUGE-L with extensive training
- PATH B (pretrained) can realistically reach ~0.35-0.45 ROUGE-L with fine-tuning
- Given compute constraints (8-12GB GPU), we'll use BART-base + LoRA/gradient checkpointing

---

## Phase 0: Reality Check — Evaluation Hygiene ✅

### 0.1 Current Model Baseline ✅

**Evaluation Results (500 val examples):**

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| **Current Pointer-Gen** | 0.1814 | 0.0030 | **0.0855** |
| Lead-100 (extractive) | 0.1848 | 0.0705 | 0.1186 |
| Lead-150 (extractive) | **0.2530** | **0.0858** | **0.1510** |
| Random (sanity) | 0.0164 | 0.0001 | 0.0118 |

**Key Findings:**
- ❌ Current model is WORSE than simple extractive baseline
- ❌ ROUGE-2 = 0.0030 indicates almost no bigram overlap
- ❌ Model was only trained for 500 steps (max_steps=600 in config)

### 0.2 Decode Quality Analysis ✅

**Sanity Check Results (20 examples):**

| Metric | Value | Status |
|--------|-------|--------|
| Avg generation length | 217.6 tokens | ✓ OK |
| Trigram repetition rate | 0.0000 | ✓ Good (no repetition) |
| Unique token ratio | 0.3336 | ⚠️ **LOW** (33% diversity) |
| Copy ratio | 1.0000 | ⚠️ **Suspicious** (100% copied but garbage output) |
| Early EOS rate | 0% | ✓ Good |

**Sample Output:**
```
Reference: The patient presented to orthopaedic trauma clinic on ___ and was evaluated...

Generated: ___... ___. ___ ___. ...,.. 11:.. the.. of.. was..AM.. with.. 15.. and.. to.. <.. a.. wound..>.. left..-.. AKA.. He.. infection.. The.. for..#.. tissue...
```

**Diagnosis:**
- 🔴 **Broken tokenization/decoding**: Outputs are dots and fragments
- 🔴 **Severe undertraining**: Model hasn't learned language modeling
- 🔴 **Low diversity**: Only using 33% of generated vocabulary

### 0.3 ROUGE Computation Verification ✅

**Implementation Review:**
- ✅ Using `rouge_scorer` with stemming enabled
- ✅ Special tokens (BOS/EOS/PAD) correctly filtered
- ✅ Evaluation on consistent val split
- ✅ No data leakage detected

---

## Phase 1: Data & Preprocessing Improvements

### 1.1 Current Preprocessing Analysis

**Max Input Length:**
- Config: `chunk_len=256 * num_chunks=6 = 1536 tokens`
- Actual data: Truncated to 1024 tokens in preprocessing
- **Issue:** Mismatch between preprocessing (1024) and model config (1536)

**Truncation Strategy:**
- Current: Simple head truncation (keep first 1024 tokens)
- **Problem:** Medical notes often have key info in later sections (Assessment, Plan, Discharge)

**Vocabulary:**
- SentencePiece vocab_size = 16,000
- Trained on clinical corpus (appears correct)
- Special tokens: BOS=1, EOS=2, PAD=3, UNK=0

### 1.2 Proposed Improvements

**Priority 1: Fix Preprocessing/Config Mismatch**
```yaml
# In preprocessing (preprocess_data.py):
max_src_len: 1536  # Match model config

# OR in model config:
model:
  chunk_len: 256
  num_chunks: 6  # = 1536 total, matches preprocessing
```

**Priority 2: Section-Aware Truncation** (Optional, medium ROI)
```python
def section_aware_truncate(text, max_len=1536):
    """Keep important sections: HPI, Assessment/Plan, Hospital Course"""
    # Heuristic: keep first 512 tokens + sections with keywords
    # + last 256 tokens (often contains discharge summary)
    ...
```

**Priority 3: Better Tokenization** (Low priority unless issues found)
- Current SentencePiece vocab=16K is reasonable for medical domain
- Could expand to 32K if vocabulary coverage is poor

---

## Phase 2: Model Strategy Decision

### Decision: **PATH B — Pretrained Transformer** ✅

**Rationale:**

| Factor | PATH A (Pointer-Gen) | PATH B (Pretrained) |
|--------|---------------------|-------------------|
| **ROUGE ceiling** | ~0.25-0.30 | ~0.35-0.45 |
| **Training time** | ~50K-100K steps | ~10K-20K steps |
| **Implementation effort** | Medium (fix existing code) | High (new pipeline) |
| **Compute required** | ~20-40 GPU hours | ~10-20 GPU hours (with LoRA) |
| **Risk** | Low (known architecture) | Medium (new codebase) |

**Why PATH B:**
1. ✅ **Higher ROUGE ceiling**: Pretrained models have learned medical language
2. ✅ **Faster convergence**: Fine-tuning vs training from scratch
3. ✅ **Modern best practices**: Transformers are SOTA for summarization
4. ✅ **Efficient**: LoRA + gradient checkpointing fits 8-12GB GPU

### PATH B: Implementation Plan

**Model Choice:** BART-base (facebook/bart-base, 139M parameters)
- **Alternatives:** LED (long-context), LongT5, BigBird-Pegasus
- **Chosen BART because:** 
  - Proven for summarization
  - Good balance of size/performance
  - Well-supported in HuggingFace
  - Can handle 1024 tokens (our data is truncated to 1536 but most important info is in first 1024)

**Memory Optimization:**
```python
- FP16/BF16 mixed precision: ~2x memory reduction
- Gradient checkpointing: ~30-40% memory reduction
- LoRA (rank=8-16): ~90% parameter reduction
- Gradient accumulation: effective batch size without OOM
```

**Expected Memory:**
- BART-base: ~550MB (FP16) + ~1.5GB activations = **~2GB**
- LoRA adapters: ~10MB trainable
- Batch size 1-2 with grad accumulation: **fits 8GB GPU** ✅

---

## Phase 3: Experiment Plan

### Stage 1: Sanity Check (Quick validation)
**Objective:** Verify training loop works, model learns something

| Config | Details |
|--------|---------|
| **Steps** | 500-1000 |
| **Data** | Full train set |
| **Batch size** | 1-2 (+ grad accumulation 8-16) |
| **Eval freq** | Every 200 steps |
| **Expected ROUGE-L** | > 0.15 (better than current) |
| **Time** | ~1-2 hours |

### Stage 2: Hyperparameter Search (Find best config)
**Objective:** Find optimal LR, batch size, LoRA rank, max_length

| Experiment | Learning Rate | LoRA Rank | Max Length | Decoding |
|------------|--------------|-----------|------------|----------|
| exp_1_baseline | 5e-5 | 8 | 512 | beam=4, len_pen=1.0 |
| exp_2_higher_lr | 1e-4 | 8 | 512 | beam=4, len_pen=1.0 |
| exp_3_larger_lora | 5e-5 | 16 | 512 | beam=4, len_pen=1.0 |
| exp_4_longer_output | 5e-5 | 8 | 768 | beam=4, len_pen=1.0 |
| exp_5_decode_tuning | 5e-5 | 8 | 512 | beam=6, len_pen=1.5 |
| exp_6_low_lr | 3e-5 | 8 | 512 | beam=4, len_pen=1.0 |

**Steps per experiment:** 2000-3000 steps  
**Time per experiment:** ~2-3 hours  
**Total time:** ~15-20 hours for 6 experiments

### Stage 3: Full Training (Best config)
**Objective:** Train to convergence with early stopping

| Config | Details |
|--------|---------|
| **Steps** | 10K-20K (early stopping) |
| **Patience** | 5 evaluations without improvement |
| **Eval freq** | Every 500 steps |
| **Checkpointing** | Save best + every 1000 steps |
| **Expected ROUGE-L** | **0.35-0.40** (target) |
| **Time** | ~8-12 hours |

### Stage 4: Final Evaluation
**Objective:** Test set evaluation, generate predictions

- Evaluate best model on full test set
- Generate predictions for error analysis
- Compute final ROUGE-1/2/L scores
- Save predictions to `experiments/final_predictions.csv`

---

## Phase 4: Implementation Tasks

### Task 4.1: Create HuggingFace Training Pipeline ⏳

**Files to create:**
1. `scripts/hf_train.py` — Training script with BART + LoRA
2. `scripts/hf_eval.py` — Evaluation script
3. `configs/hf_baseline.yaml` — Base configuration
4. `configs/hf_experiments/*.yaml` — Experiment configs

**Key components:**
```python
# Using PEFT for LoRA
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BartForConditionalGeneration, BartTokenizer

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,  # LoRA rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model = get_peft_model(model, lora_config)
```

### Task 4.2: Experiment Tracking ⏳

**Create:** `experiments/results.csv`

| run_name | model | lora_rank | lr | steps | rouge1 | rouge2 | rougeL | notes |
|----------|-------|-----------|----|-|--------|--------|--------|-------|
| exp_1_baseline | bart-base | 8 | 5e-5 | 2000 | 0.28 | 0.09 | 0.22 | baseline config |
| exp_2_higher_lr | bart-base | 8 | 1e-4 | 2000 | 0.30 | 0.10 | 0.24 | converged faster |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

**Tracking script:**
```python
# Append results after each run
import pandas as pd
results = pd.DataFrame([{
    'run_name': 'exp_1_baseline',
    'model': 'bart-base',
    'lora_rank': 8,
    'lr': 5e-5,
    'steps': 2000,
    'rouge1': 0.28,
    'rouge2': 0.09,
    'rougeL': 0.22,
    'notes': 'baseline config'
}])
results.to_csv('experiments/results.csv', mode='a', header=False, index=False)
```

---

## Expected Outcomes

### ROUGE Targets

| Milestone | ROUGE-1 | ROUGE-2 | ROUGE-L | Timeline |
|-----------|---------|---------|---------|----------|
| **Current Model** | 0.18 | 0.00 | **0.09** | Baseline |
| Lead-150 Baseline | 0.25 | 0.09 | 0.15 | Already computed |
| After Stage 1 (sanity) | 0.22 | 0.05 | 0.16 | +2 hours |
| After Stage 2 (search) | 0.30 | 0.11 | 0.25 | +20 hours |
| After Stage 3 (full) | **0.35** | **0.14** | **0.30** | +32 hours |
| **Stretch Goal** | 0.38 | 0.16 | 0.32 | +40 hours |

### ROI Analysis

| Improvement | Effort (hours) | ROUGE-L Gain | ROI |
|-------------|----------------|--------------|-----|
| Fix current pointer-gen (PATH A) | 40-60 | +0.15 (0.09→0.24) | Low |
| BART fine-tuning (PATH B) | 30-40 | +0.21 (0.09→0.30) | **High** ✅ |
| Section-aware preprocessing | 4-6 | +0.02-0.03 | Medium |
| Decoding optimization | 2-4 | +0.01-0.02 | High |
| Ensemble models | 8-12 | +0.02-0.04 | Low |

**Recommendation:** Prioritize BART fine-tuning (PATH B) first, then optimize decoding/preprocessing.

---

## Risk Mitigation

### Risk 1: GPU Memory OOM
**Probability:** Medium  
**Mitigation:**
- Start with batch_size=1 + gradient_accumulation=16
- Enable gradient checkpointing
- Use LoRA rank=8 (can reduce to 4 if needed)
- Fall back to BART-small (54M params) if BART-base doesn't fit

### Risk 2: Poor Convergence
**Probability:** Low  
**Mitigation:**
- Use proven hyperparameters from literature
- Cosine LR schedule with warmup
- Early stopping to avoid overfitting
- Monitor train/val loss curves closely

### Risk 3: Time Constraints
**Probability:** Medium  
**Mitigation:**
- Stage 1 (sanity check) is quick validation
- Can skip some Stage 2 experiments if time-limited
- Use smaller data subset (10K examples) for hyperparameter search

---

## Deliverables Checklist

### Code
- [x] `scripts/check_checkpoint.py` — Checkpoint inspection
- [x] `scripts/evaluate_baselines.py` — Lead-K baselines
- [x] `scripts/decode_sanity.py` — Quality analysis
- [ ] `scripts/hf_train.py` — HuggingFace training
- [ ] `scripts/hf_eval.py` — HuggingFace evaluation
- [ ] `configs/hf_baseline.yaml` — BART config
- [ ] `configs/hf_experiments/` — Experiment configs

### Documentation
- [x] `docs/rouge_optimization_plan.md` — This document
- [ ] `README.md` updates — Training/eval instructions
- [ ] `experiments/results.csv` — Tracking sheet

### Experiments
- [ ] Stage 1: Sanity check (500-1000 steps)
- [ ] Stage 2: Hyperparameter search (6 experiments)
- [ ] Stage 3: Full training (best config)
- [ ] Stage 4: Final evaluation (test set)

---

## Quick Start Commands

### 1. Evaluate Current Baselines
```bash
# Already completed ✅
python scripts/evaluate_baselines.py \
  --tokenized_dir data/tokenized/kaggle_upload_full \
  --num_examples 500
```

### 2. Run Sanity Check (Stage 1)
```bash
# After implementing hf_train.py
python scripts/hf_train.py \
  --config configs/hf_baseline.yaml \
  --run_name stage1_sanity \
  --max_steps 1000 \
  --eval_every 200
```

### 3. Hyperparameter Search (Stage 2)
```bash
# Run experiments in sequence
for config in configs/hf_experiments/*.yaml; do
  python scripts/hf_train.py --config $config
done
```

### 4. Full Training (Stage 3)
```bash
python scripts/hf_train.py \
  --config configs/hf_best.yaml \
  --run_name final_training \
  --max_steps 20000 \
  --early_stopping_patience 5
```

---

## Appendix: Alternative Approaches

### If PATH B Fails (Fallback to PATH A)

**Pointer-Generator Improvements:**
1. Train for 50K-100K steps (current: 500 steps)
2. Increase model capacity: 2-layer LSTM, 512 hidden dim
3. Tune coverage_lambda: sweep [0.1, 0.5, 1.0, 2.0]
4. Tune label_smoothing: sweep [0.0, 0.05, 0.1]
5. Better LR schedule: cosine decay with 2K warmup
6. Decoding: tune length_penalty [0.6, 1.0, 1.5], min_length [50, 75, 100]

**Expected outcome:** ROUGE-L ~0.24-0.28 (still lower than PATH B)

### If Memory is Still Constrained

**Ultra-low memory options:**
- BART-small (54M params): ~300MB
- LoRA rank=4: ~5MB trainable
- Batch size=1, grad_accum=32
- Quantization: 8-bit LoRA (bitsandbytes)

---

## Conclusion

**Recommended Path:** PATH B (BART fine-tuning with LoRA)

**Expected Timeline:**
- Week 1: Implement HuggingFace pipeline + Stage 1 sanity check
- Week 2: Stage 2 hyperparameter search (6 experiments)
- Week 3: Stage 3 full training + Stage 4 evaluation

**Expected Final Performance:**
- **ROUGE-L: 0.30-0.35** (vs current 0.09)
- **ROUGE-1: 0.35-0.40** (vs current 0.18)
- **ROUGE-2: 0.12-0.16** (vs current 0.00)

**Success Criteria:**
- ✅ ROUGE-L > 0.25 (beats Lead-150 baseline of 0.15)
- ✅ ROUGE-2 > 0.10 (shows bigram understanding)
- ✅ Coherent outputs (qualitative check)
- ✅ Training pipeline reproducible and documented

---

**Next Steps:** Proceed to Phase 4.1 - Implement HuggingFace training pipeline.
