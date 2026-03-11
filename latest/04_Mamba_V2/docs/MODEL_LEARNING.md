# How the Model Learns from the Dataset

> **Model:** Mamba-Transformer Hybrid V2  
> **Dataset:** MIMIC-IV Brief Hospital Course (BHC) — ~270K clinical note–summary pairs  
> **Task:** Clinical note → Brief Hospital Course summarization

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Data Loading Pipeline](#2-data-loading-pipeline)
3. [Tokenization — SentencePiece](#3-tokenization--sentencepiece)
4. [Train/Val Split Strategy](#4-trainval-split-strategy)
5. [Batching & Dynamic Padding](#5-batching--dynamic-padding)
6. [How the Model Sees Each Sample](#6-how-the-model-sees-each-sample)
7. [Teacher Forcing — How the Decoder Learns](#7-teacher-forcing--how-the-decoder-learns)
8. [Loss Function — Label Smoothing Cross-Entropy](#8-loss-function--label-smoothing-cross-entropy)
9. [Chunking Strategy — Handling Long Documents](#9-chunking-strategy--handling-long-documents)
10. [Memory Compression — Information Bottleneck](#10-memory-compression--information-bottleneck)
11. [Gradient Flow Through the Full Model](#11-gradient-flow-through-the-full-model)
12. [Optimization — AdamW with Cosine Warmup](#12-optimization--adamw-with-cosine-warmup)
13. [Mixed Precision Training (AMP fp16)](#13-mixed-precision-training-amp-fp16)
14. [Gradient Accumulation — Simulating Larger Batches](#14-gradient-accumulation--simulating-larger-batches)
15. [Evaluation Loop — ROUGE-Based Feedback](#15-evaluation-loop--rouge-based-feedback)
16. [Checkpoint Strategy — Saving Progress](#16-checkpoint-strategy--saving-progress)
17. [End-to-End Learning Flow (Summary)](#17-end-to-end-learning-flow-summary)

---

## 1. Dataset Overview

The model learns from the **MIMIC-IV Brief Hospital Course (BHC)** dataset, stored in `data/mimic-iv-bhc.csv`.

| Property | Value |
|---|---|
| **Total samples** | ~270,000 clinical note–summary pairs |
| **Source column** | `input` — full clinical note (admission notes, progress notes, labs, vitals, etc.) |
| **Target column** | `target` — Brief Hospital Course (BHC) summary written by physicians |
| **Source length** | Typically 1,000–8,000+ tokens (long clinical documents) |
| **Target length** | Typically 50–400 tokens (concise clinical summary) |
| **Compression ratio** | ~10–20x (the model must learn extreme compression) |

### What the Data Looks Like

```
SOURCE (input):
"ADMISSION DATE: 2023-01-15. CHIEF COMPLAINT: Chest pain and shortness of breath.
HISTORY OF PRESENT ILLNESS: 72-year-old male with PMH of HTN, DM2, CAD s/p CABG
2019, presented to ED with acute onset substernal chest pain radiating to left arm...
LABS: Troponin 0.45 (elevated), BNP 890, WBC 12.3, Hgb 11.2, Cr 1.8 (baseline 1.2)...
HOSPITAL COURSE: Patient was admitted to CCU... Started on heparin drip...
Echo showed EF 35%... Cardiology consulted... Underwent cardiac cath..."

TARGET (summary):
"72M with CAD s/p CABG, HTN, DM2 presented with NSTEMI. Troponin peaked at 0.45.
Started on heparin, aspirin, clopidogrel. Cardiac cath showed 90% LAD stenosis,
underwent PCI with DES. Post-procedure course uncomplicated. Discharged on
dual antiplatelet therapy, statin, beta-blocker. Follow-up with cardiology in 2 weeks."
```

The model must learn to:
- **Extract key clinical entities** (diagnoses, medications, lab values)
- **Preserve medical accuracy** (correct dosages, negations)
- **Compress** thousands of tokens into a concise, clinically useful summary
- **Follow clinical narrative structure** (presentation → workup → treatment → disposition)

---

## 2. Data Loading Pipeline

The data flows through this pipeline:

```
CSV File (data/mimic-iv-bhc.csv)
        │
        ▼  _read_csv_once()
Read ALL rows once → list of (source_text, target_text) tuples
        │
        ▼  random.shuffle(seed=42)
Deterministically shuffle all samples
        │
        ▼  95/5 split
Train: first 95% of shuffled samples (~256,500 samples)
Val:   last 5% of shuffled samples   (~13,500 samples)
        │
        ▼  PreloadedClinicalDataset
Wrap samples into PyTorch Dataset (tokenizes on-the-fly)
        │
        ▼  DataLoader(shuffle=True, batch_size=2, collate_fn=...)
Create batches with dynamic padding
```

### Key Design Decisions

1. **Single CSV read**: The CSV is read exactly once and split in memory, avoiding the overhead of reading a large file twice.
2. **On-the-fly tokenization**: Raw text is stored in memory; tokenization happens in `__getitem__()` when a sample is accessed. This avoids pre-tokenizing 270K samples upfront.
3. **Deterministic split**: `random.seed(42)` ensures the same train/val split every time, guaranteeing reproducibility.

---

## 3. Tokenization — SentencePiece

The model uses a **SentencePiece unigram** tokenizer trained on clinical text, stored at `data/tokenizer/spm.model`.

| Property | Value |
|---|---|
| **Algorithm** | Unigram (subword) |
| **Vocab size** | 16,000 tokens |
| **BOS token** | ID 1 (Beginning-of-Sequence) |
| **EOS token** | ID 2 (End-of-Sequence) |
| **PAD token** | ID 3 (Padding) |

### Why SentencePiece for Clinical Text?

- **Handles OOV (out-of-vocabulary)**: Medical terms like "cardiomyopathy" or "omeprazole" are split into subwords (e.g., `▁cardi`, `omy`, `opathy`), so no word is ever unknown.
- **Compact vocab**: 16K tokens is enough to cover common clinical vocabulary without excessive memory usage.
- **Language-agnostic**: Works directly on raw text without pre-tokenization rules.

### Tokenization Example

```python
text = "Patient admitted with chest pain and elevated troponin"
token_ids = tokenizer.EncodeAsIds(text)
# → [245, 1032, 87, 441, 2201, 56, 892, 3401]
```

---

## 4. Train/Val Split Strategy

```python
# In _read_csv_once():
random.seed(42)             # Deterministic
random.shuffle(samples)     # Shuffle all ~270K samples
split_idx = int(len(samples) * 0.95)  # 95/5 split

train_samples = samples[:split_idx]   # ~256,500 samples
val_samples = samples[split_idx:]     # ~13,500 samples
```

- **No separate test set** in the main pipeline — the gated evaluation (`clinical_eval.py`) uses the validation set for comprehensive assessment.
- **Stratification is not used** — clinical notes are assumed to be sufficiently diverse after shuffling.
- **Seed 42** ensures identical splits across runs, enabling checkpoint resumption.

---

## 5. Batching & Dynamic Padding

Each batch is assembled with **dynamic padding** — sequences are padded only to the length of the longest sample in that batch, not to some global maximum:

```python
def collate_fn(batch, pad_id=3):
    max_src_len = max(item['src'].size(0) for item in batch)  # e.g., 3200
    max_tgt_len = max(item['tgt_input'].size(0) for item in batch)  # e.g., 180

    # Pad all sequences to batch-local maximum
    src = torch.full((batch_size, max_src_len), pad_id)  # filled with 3 (PAD)
    # ... fill in actual tokens ...

    # Create pad masks (True = padded position = IGNORE)
    src_mask[i, :src_len] = False   # Real tokens → False (attend to these)
    src_mask[i, src_len:] = True    # Padding → True (ignore these)
```

### Training Batch Configuration

| Setting | Value |
|---|---|
| **Batch size** | 2 (physical — fits in 8GB VRAM) |
| **Gradient accumulation** | 8 steps |
| **Effective batch size** | 2 × 8 = **16** |
| **Shuffle** | Yes (train only) |
| **Drop last** | Yes (avoid incomplete batches) |
| **Pin memory** | Yes (faster CPU→GPU transfer) |

---

## 6. How the Model Sees Each Sample

For each training sample, the data loader produces:

```python
{
    'src':        [245, 1032, 87, 441, ..., 3, 3, 3],   # Source tokens + PAD
    'tgt_input':  [1, 72, 441, 2201, ...],               # [BOS, tok1, tok2, ...]
    'tgt_output': [72, 441, 2201, ..., 2],                # [tok1, tok2, ..., EOS]
    'src_mask':   [False, False, ..., True, True, True],  # True = padded (ignore)
    'tgt_mask':   [False, False, ..., True, True, True],  # True = padded (ignore)
}
```

### Source-Target Alignment

```
tgt_input:   [BOS]  [72]  [441]  [2201]  ...  [tok_N-1]
                ↓     ↓      ↓       ↓            ↓
Model predicts: [72]  [441]  [2201]  [tok4]  ...  [EOS]   ← tgt_output
```

The model learns to predict **each next token** given all previous tokens + the encoded source. This is called **teacher forcing**.

---

## 7. Teacher Forcing — How the Decoder Learns

During training, the decoder receives the **ground-truth target** shifted right by one position as input, and tries to predict the original target:

```
Decoder Input  (tgt_input):   BOS  tok₁  tok₂  tok₃  ...  tok_{N-1}
Expected Output (tgt_output):      tok₁  tok₂  tok₃  ...  tok_N   EOS
```

### Why Teacher Forcing?

- **Stable training**: The decoder always sees correct previous tokens, preventing error accumulation during training.
- **Faster convergence**: The model gets clear supervisory signal at every position.
- **Exposure bias**: The model never sees its own mistakes during training, which can cause issues at inference. This is partially mitigated by **label smoothing** (below).

### Causal Masking

The decoder uses a **causal (upper-triangular) attention mask** to prevent each position from seeing future tokens:

```
Position:    1    2    3    4
        1  [ 0   -∞   -∞   -∞ ]
        2  [ 0    0   -∞   -∞ ]
        3  [ 0    0    0   -∞ ]
        4  [ 0    0    0    0  ]
```

This ensures that when predicting `tok₃`, the model can only attend to `BOS`, `tok₁`, `tok₂` — not `tok₃` or beyond.

---

## 8. Loss Function — Label Smoothing Cross-Entropy

The model uses **cross-entropy loss with label smoothing** (ε = 0.15):

### Standard Cross-Entropy

For each token position, the model outputs a probability distribution over 16,000 vocab tokens. Standard cross-entropy computes:

$$\mathcal{L}_{\text{CE}} = -\log P(\text{correct token})$$

### With Label Smoothing (ε = 0.15)

Instead of putting 100% probability on the correct token, the target distribution becomes:

$$q(k) = \begin{cases}
1 - \varepsilon = 0.85 & \text{if } k = \text{correct token} \\
\frac{\varepsilon}{V - 1} = \frac{0.15}{15999} \approx 0.0000094 & \text{otherwise}
\end{cases}$$

The smoothed loss:

$$\mathcal{L}_{\text{smooth}} = -\sum_{k=1}^{V} q(k) \cdot \log P(k)$$

### Why Label Smoothing?

1. **Prevents overconfidence**: The model doesn't learn to output 99.99% probability for one token, making it more robust.
2. **Better generalization**: Acts as implicit regularization.
3. **Especially important for clinical text**: Multiple valid ways to phrase the same clinical finding — e.g., "HTN" vs "hypertension" vs "elevated blood pressure".

### Ignoring Padding

Padding tokens (ID=3) are **excluded from the loss** via a mask:

```python
mask = (targets != self.ignore_index)  # True for real tokens
loss = (loss * mask.float()).sum() / mask.sum()
```

This ensures the model never learns to "predict padding".

---

## 9. Chunking Strategy — Handling Long Documents

Clinical notes can be 4,000–8,000+ tokens, but processing them in one shot would exhaust GPU memory. The model uses **overlapping chunking**:

```
Input: [tok₁, tok₂, tok₃, ..., tok₄₀₉₆]

Chunk 1: tokens[0:256]      (positions 0–255)
Chunk 2: tokens[192:448]    (positions 192–447) — 64 token overlap with Chunk 1
Chunk 3: tokens[384:640]    (positions 384–639) — 64 token overlap with Chunk 2
   ...
Chunk N: tokens[start:start+256]
```

| Parameter | Value |
|---|---|
| **Chunk size** | 256 tokens |
| **Stride** | 192 tokens |
| **Overlap** | 64 tokens (256 - 192) |
| **Chunks for 4096 tokens** | ceil((4096-256)/192)+1 = **21 chunks** |

### Why Overlapping?

- **Context at boundaries**: Without overlap, information at chunk boundaries would be lost. A clinical finding spanning positions 250–260 would be split across chunks.
- **Smooth information flow**: The 64-token overlap ensures each chunk has context from neighboring sections.
- **Memory efficiency**: Each chunk is only 256 tokens, which is manageable for the Mamba SSM encoder.

---

## 10. Memory Compression — Information Bottleneck

After encoding each 256-token chunk through 6 Mamba layers, the result is compressed to just **32 memory tokens** per chunk using learned attention pooling:

```
Chunk encoded: [batch, 256, 512]
                    ↓
        MemoryTokenCompressor
        (32 learned query vectors)
                    ↓
Memory tokens: [batch, 32, 512]
```

### How Compression Works

1. **32 learned query vectors** (randomly initialized, learned during training) act as "questions" that attend to the chunk representations.
2. **Cross-attention**: Each query attends to all 256 positions in the chunk, learning which information is most relevant.
3. **Output**: 32 memory vectors, each capturing a different aspect of the chunk's content.

### Why This Matters for Learning

- **Forces abstraction**: The model must learn which information is important enough to preserve in 32 tokens (down from 256). This is a 8x compression.
- **Learns to prioritize**: Over training, the 32 queries specialize — some learn to capture diagnoses, others medications, others lab values.
- **Global representation for decoder**: For a 4096-token input (21 chunks × 32 tokens), the decoder sees 672 memory tokens that capture the entire document.

### Cross-Chunk Memory Attention (V2)

After compression, memory tokens from different chunks attend to each other through 2 cross-chunk attention layers:

```
Memory: [batch, 672, 512]  (21 chunks × 32 tokens)
        ↓
CrossChunkAttention × 2
        ↓
Memory: [batch, 672, 512]  (now globally informed)
```

This allows information from the beginning of the note (e.g., admission diagnosis) to connect with information at the end (e.g., discharge medications).

---

## 11. Gradient Flow Through the Full Model

During backpropagation, gradients flow backward through the entire model:

```
Loss (scalar)
        ↑
Output Projection (vocab logits → loss)
        ↑
Transformer Decoder (6 layers: self-attn → cross-attn → FFN)
        ↑                    ↑
 Causal self-attn      Cross-attention to memory
        ↑                    ↑
Target embeddings      Encoder memory tokens
                             ↑
                  Cross-Chunk Attention × 2
                             ↑
                  MemoryTokenCompressor (learned queries)
                             ↑
                  MambaEncoder (6 bidirectional Mamba layers)
                             ↑
                  Source embeddings + Position encoding
                             ↑
                  Source token embeddings
```

### What Each Component Learns

| Component | What It Learns |
|---|---|
| **Source embeddings** | Semantic representations for 16K clinical tokens |
| **Positional encoding** | Position-awareness within each 256-token chunk |
| **Mamba encoder** | Sequential patterns in clinical text (temporal flow, entity relationships) |
| **Bidirectional gates** | How much to weight forward vs. backward context |
| **Memory queries** | Which aspects of each chunk to compress/preserve |
| **Cross-chunk attention** | How to connect information across distant parts of the note |
| **Decoder self-attention** | Language model — how to generate fluent clinical summaries |
| **Decoder cross-attention** | Where to look in the source for each summary token |
| **Gated cross-attention** | How much source info to use at each decoder layer |
| **Output projection** | Final token prediction (weight-tied with target embeddings) |

---

## 12. Optimization — AdamW with Cosine Warmup

### Optimizer: AdamW

| Parameter | Value |
|---|---|
| **Type** | AdamW (Adam + decoupled weight decay) |
| **Learning rate** | 5 × 10⁻⁵ |
| **Weight decay** | 0.03 |
| **β₁, β₂** | 0.9, 0.999 (PyTorch defaults) |

### Learning Rate Schedule: Cosine Warmup

```
LR Schedule:
  │    ╱⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺╲
  │   ╱                      ╲
  │  ╱                        ╲
  │ ╱                          ╲
  │╱                            ╲___
  └────────────────────────────────── steps
  0    2000                      50000
     warmup        cosine decay
```

1. **Warmup (steps 0–2000)**: LR linearly increases from 0 to 5×10⁻⁵. This prevents early instability when loss gradients are large and model weights are randomly initialized.
2. **Cosine decay (steps 2000–50000)**: LR follows a cosine curve down to 10% of peak (min_lr_ratio = 0.1 → final LR ≈ 5×10⁻⁶).

### Gradient Clipping

All gradients are clipped to **max norm = 1.0** before each optimizer step:

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm=1.0)
```

This prevents gradient explosion, which is common in Transformer training especially with long sequences.

---

## 13. Mixed Precision Training (AMP fp16)

The model trains in **mixed precision** (AMP — Automatic Mixed Precision) to fit in 8GB VRAM:

```python
with autocast('cuda', enabled=True):
    logits = model(src, tgt_input, src_mask, tgt_mask)  # Forward pass in fp16
    loss = criterion(logits, tgt_output)                 # Loss in fp16

scaler.scale(loss).backward()   # Scaled backward pass
scaler.step(optimizer)          # Optimizer step in fp32
scaler.update()                 # Adjust loss scale
```

### How It Works

1. **Forward pass**: Computed in fp16 (half precision) — uses ~50% less memory and is faster on NVIDIA Tensor Cores.
2. **Loss scaling**: GradScaler multiplies the loss by a large factor (e.g., 65536) before backward pass, preventing gradients from underflowing to zero in fp16.
3. **Backward pass**: Gradients computed in fp16 with scaled loss.
4. **Optimizer step**: GradScaler unscales gradients back to fp32, then AdamW updates weights in fp32 (full precision for numerical stability).

---

## 14. Gradient Accumulation — Simulating Larger Batches

The physical batch size is 2 (limited by 8GB VRAM), but the model simulates an effective batch size of 16:

```python
for batch_idx, batch in enumerate(train_loader):
    loss = criterion(logits, tgt_output)
    loss = loss / gradient_accumulation_steps  # Divide by 8

    scaler.scale(loss).backward()  # Accumulate gradients

    if (batch_idx + 1) % gradient_accumulation_steps == 0:
        # Only update weights every 8 batches
        scaler.step(optimizer)
        optimizer.zero_grad()
```

### Why?

- **Batch size 2** processes only 2 samples → noisy gradient estimates
- **Accumulating over 8 steps** averages gradients from 16 samples → smoother, more stable updates
- **Identical effect** to batch size 16, but uses 8x less memory

---

## 15. Evaluation Loop — ROUGE-Based Feedback

Every **1,000 steps**, the model is evaluated on 50 validation samples:

### Evaluation Process

1. **Switch to eval mode**: `model.eval()` — disables dropout, uses running stats for normalization.
2. **Compute validation loss**: Forward pass with teacher forcing (same as training, but no gradient update).
3. **Generate summaries**: Autoregressive greedy decoding (no teacher forcing — model generates freely).
4. **Compute ROUGE scores**: Compare generated summaries to reference summaries.

```
Step 1000 — Evaluation:
  Val Loss: 4.2134
  ROUGE-1: 0.2845    ← unigram overlap
  ROUGE-2: 0.0923    ← bigram overlap
  ROUGE-L: 0.2156    ← longest common subsequence
```

### Best Model Selection

The checkpoint with the **highest ROUGE-L** score is saved as `best_model.pt`:

```python
if eval_results['rougeL'] > best_rouge_l:
    best_rouge_l = eval_results['rougeL']
    save_checkpoint_atomic(model, ..., 'best_model.pt')  # Save new best
```

---

## 16. Checkpoint Strategy — Saving Progress

| Checkpoint Type | Frequency | Purpose |
|---|---|---|
| **Periodic** | Every 200 steps | Resume training after interruption |
| **Best model** | When ROUGE-L improves | Best model for inference |
| **Final model** | At training completion | Last state of training |

### Atomic Saves (Crash-Safe)

Checkpoints are saved atomically to prevent corruption:

```python
torch.save(checkpoint, temp_path)       # Write to temporary file
shutil.move(temp_path, checkpoint_path)  # Atomic rename (OS-level)
```

### What's Saved

Each checkpoint contains:
- `model_state_dict` — all model weights
- `optimizer_state_dict` — Adam moments (m, v) for all parameters
- `scheduler_state_dict` — current learning rate schedule position
- `scaler_state_dict` — AMP loss scale factor
- `step` — current training step
- `best_rouge_l` — best ROUGE-L achieved so far
- `config` — full model/data/training configuration

### Auto-Resume

On restart, the training script automatically finds the latest checkpoint:

```python
resume_from = find_latest_checkpoint(checkpoint_dir)
# Finds checkpoint_step_6600.pt (highest step number)
```

---

## 17. End-to-End Learning Flow (Summary)

### Single Training Step

```
1. LOAD BATCH
   (src, tgt_input, tgt_output, masks) ← DataLoader

2. ENCODE SOURCE
   src [batch=2, ~3000 tokens]
     → chunk into 16 × [batch=2, 256]
     → each chunk through 6 BiMamba layers → [batch=2, 256, 512]
     → compress each to [batch=2, 32, 512]
     → concat all → [batch=2, 512, 512]  (16 chunks × 32 memory tokens)
     → 2 CrossChunkAttention layers → [batch=2, 512, 512]

3. DECODE TARGET (teacher forcing)
   tgt_input [batch=2, ~150 tokens]
     → embed + positional encoding → [batch=2, 150, 512]
     → 6 decoder layers (self-attn + cross-attn to memory + FFN)
     → [batch=2, 150, 512]

4. COMPUTE LOGITS
   → output projection → [batch=2, 150, 16000]  (vocab probabilities)

5. COMPUTE LOSS
   → label-smoothed cross-entropy(logits, tgt_output) / 8  (grad accum)

6. BACKWARD PASS
   → compute gradients for all ~79M parameters

7. ACCUMULATE (repeat steps 1–6 for 8 micro-batches)

8. CLIP & UPDATE
   → clip gradients to norm ≤ 1.0
   → AdamW optimizer step
   → cosine LR scheduler step
   → zero gradients
```

### Training Timeline

```
Steps 0–2000:       Warmup phase — LR increases linearly
                    Loss drops rapidly from ~10 to ~4
                    Model learns basic word embeddings and simple patterns

Steps 2000–10000:   Early training — loss stabilizes around 3.5–4.0
                    Model learns to copy key entities from source
                    ROUGE-1 reaches 0.25+

Steps 10000–30000:  Mid training — loss slowly decreases
                    Model learns clinical narrative structure
                    ROUGE-L reaches 0.25+

Steps 30000–50000:  Late training — LR decaying via cosine
                    Model refines fluency and factual accuracy
                    Target: ROUGE-L 0.30+
```

### What Learning Looks Like at Different Stages

**Early (Step 500):**
```
Source: "72M with HTN, DM2, CAD presented with chest pain..."
Output: "patient the the the admitted to the hospital the was"
```
→ Model has learned common words but no coherent structure.

**Mid (Step 5000):**
```
Source: "72M with HTN, DM2, CAD presented with chest pain..."
Output: "patient admitted with chest pain. started on heparin. discharged home."
```
→ Model captures key events but misses specific details.

**Late (Step 30000+):**
```
Source: "72M with HTN, DM2, CAD presented with chest pain..."
Output: "72M with CAD, HTN, DM2 presented with NSTEMI. Troponin 0.45.
         Started heparin, aspirin. Cath showed LAD stenosis. PCI performed.
         Discharged on dual antiplatelet therapy."
```
→ Model preserves entities, dosages, temporal flow, and clinical accuracy.

---

## Key Takeaways

1. **The model learns by minimizing the cross-entropy between its predicted next token and the ground-truth next token**, across all positions in the summary.
2. **Teacher forcing** gives the model a clear learning signal but creates exposure bias — at inference, the model must deal with its own mistakes.
3. **The chunking + compression architecture** forces the model to learn an **information bottleneck** — only the most clinically relevant information survives the 256→32 compression.
4. **Cross-chunk attention** lets the model learn **long-range dependencies** across the full clinical note, even though each chunk is processed independently.
5. **Label smoothing + dropout + weight decay** provide regularization that prevents overfitting to the training set's specific phrasings.
6. **ROUGE-L as the selection criterion** guides the model toward producing summaries that share long subsequences with the reference summaries, encouraging factual coverage and structural similarity.
