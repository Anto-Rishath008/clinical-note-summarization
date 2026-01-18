# Pointer-Generator Network Architecture

**Complete Technical Documentation**

---

## Table of Contents
1. [Overview](#overview)
2. [Data Pipeline](#data-pipeline)
3. [Tokenization Process](#tokenization-process)
4. [Architecture Components](#architecture-components)
5. [Encoder Module](#encoder-module)
6. [Attention Mechanisms](#attention-mechanisms)
7. [Decoder Module](#decoder-module)
8. [Pointer-Generator Mechanism](#pointer-generator-mechanism)
9. [Coverage Mechanism](#coverage-mechanism)
10. [Word Generation Process](#word-generation-process)
11. [Training Process](#training-process)
12. [Beam Search Decoding](#beam-search-decoding)
13. [Model Specifications](#model-specifications)
14. [Mathematical Formulations](#mathematical-formulations)

---

## Overview

The **Pointer-Generator Network** is a hybrid sequence-to-sequence model designed for abstractive text summarization of clinical notes. It combines the strengths of:

- **Generation**: Creating new words from a learned vocabulary
- **Copying**: Extracting relevant words directly from the source text
- **Coverage**: Preventing repetition by tracking attention history

### Key Features

✅ **Hierarchical Encoder**: Processes long clinical notes (up to 768 tokens) using chunked bidirectional LSTM  
✅ **Additive Attention**: Bahdanau-style attention with coverage mechanism  
✅ **Pointer-Generator**: Dynamic switching between vocabulary generation and source copying  
✅ **Coverage Loss**: Penalizes repeated attention to reduce redundancy  
✅ **Mixed Precision Training**: FP16 training for memory efficiency on 8GB GPU  

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT: Clinical Note                        │
│                     (Tokenized: up to 768 tokens)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL ENCODER                          │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Word Embedding (vocab_size=16000, emb_dim=128)       │    │
│  └──────────────────────────┬─────────────────────────────┘    │
│                             │                                    │
│  ┌──────────────────────────▼─────────────────────────────┐    │
│  │  Bidirectional LSTM (hidden_dim=256, num_layers=2)    │    │
│  │  • Forward LSTM  (256 dim)  ──────────────────────┐   │    │
│  │  • Backward LSTM (256 dim)  ──────────────────────┤   │    │
│  │  • Concatenated Output: 512 dim                   │   │    │
│  └──────────────────────────┬─────────────────────────┬───┘    │
│                             │                         │          │
└─────────────────────────────┼─────────────────────────┼──────────┘
                              │                         │
                    Encoder Outputs              Final Hidden
                   (batch, seq, 512)            State (h, c)
                              │                         │
                              └──────────┬──────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DECODER (Time Step t)                         │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Previous Token Embedding (emb_dim=128)                 │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                        │
│  ┌──────────────────────▼──────────────────────────────────┐   │
│  │  Concatenate: [Embedding (128) + Context (512)]        │   │
│  │  = 640 dim input                                        │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                        │
│  ┌──────────────────────▼──────────────────────────────────┐   │
│  │  Unidirectional LSTM (hidden_dim=256, num_layers=2)    │   │
│  │  Output: (batch, 256)                                   │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                        │
│                         ▼                                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │          ATTENTION MECHANISM (with Coverage)           │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │  Inputs:                                      │    │    │
│  │  │  • Decoder hidden: h_t (256)                 │    │    │
│  │  │  • Encoder outputs: (seq_len, 512)          │    │    │
│  │  │  • Coverage vector: c_t (seq_len)           │    │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │  Attention Score Computation:                │    │    │
│  │  │  e_t^i = v^T * tanh(W_h*h_s^i + W_d*h_t +  │    │    │
│  │  │                     W_c*c_t^i + b)          │    │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │  Attention Weights: α_t = softmax(e_t)      │    │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │  Context Vector:                             │    │    │
│  │  │  context_t = Σ(α_t^i * h_s^i)              │    │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │  Update Coverage:                            │    │    │
│  │  │  c_{t+1} = c_t + α_t                        │    │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  └────────────────────────┬───────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌────────────────────────────────────────────────────────┐   │
│  │       POINTER-GENERATOR MECHANISM                      │   │
│  │  ┌──────────────────────────────────────────────┐    │   │
│  │  │  Concatenate Features:                       │    │   │
│  │  │  [context_t (512) + h_t (256) + emb_t (128)]│    │   │
│  │  │  = 896 dim                                   │    │   │
│  │  └──────────────────┬───────────────────────────┘    │   │
│  │                     │                                 │   │
│  │  ┌──────────────────▼───────────────────────────┐    │   │
│  │  │  Vocabulary Distribution P_vocab:            │    │   │
│  │  │  Linear(896 → 16000) + Softmax              │    │   │
│  │  └──────────────────────────────────────────────┘    │   │
│  │                                                       │   │
│  │  ┌──────────────────────────────────────────────┐    │   │
│  │  │  Generation Probability p_gen:               │    │   │
│  │  │  sigmoid(Linear(896 → 1))                   │    │   │
│  │  └──────────────────────────────────────────────┘    │   │
│  │                                                       │   │
│  │  ┌──────────────────────────────────────────────┐    │   │
│  │  │  Copy Distribution P_copy:                   │    │   │
│  │  │  Attention weights α_t                       │    │   │
│  │  └──────────────────────────────────────────────┘    │   │
│  │                                                       │   │
│  │  ┌──────────────────────────────────────────────┐    │   │
│  │  │  Final Distribution:                         │    │   │
│  │  │  P_final = p_gen * P_vocab +                │    │   │
│  │  │            (1 - p_gen) * P_copy             │    │   │
│  │  └──────────────────────────────────────────────┘    │   │
│  └────────────────────┬───────────────────────────────────┘   │
│                       │                                        │
└───────────────────────┼────────────────────────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  Next Token Prediction │
              │  (Argmax or Sampling) │
              └─────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  Repeat until <EOS>  │
              └─────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  Generated Summary   │
              │  (up to 192 tokens) │
              └─────────────────────┘
```

---

## Data Pipeline

### End-to-End Data Flow

```
Raw CSV File (MIMIC-IV BHC)
         ↓
┌────────────────────────────────────────────────────────────┐
│  PREPROCESSING (preprocess_data.py)                        │
│                                                             │
│  1. Load CSV with columns: [input, target]                 │
│  2. Clean data (remove null, short texts)                  │
│  3. Train SentencePiece tokenizer (if needed)              │
│  4. Tokenize both source and target                        │
│  5. Add BOS/EOS tokens                                     │
│  6. Split into train/val/test (80/10/10)                   │
│  7. Save as Parquet files                                  │
│                                                             │
└────────────────────────────────────────────────────────────┘
         ↓
   Parquet Files
   ├── train.parquet (4000 samples)
   ├── val.parquet (500 samples)
   └── test.parquet (500 samples)
         ↓
┌────────────────────────────────────────────────────────────┐
│  DATALOADER (TokenizedDataset + collate_fn)               │
│                                                             │
│  1. Read parquet file row by row                           │
│  2. Extract src_ids and tgt_ids (list of integers)        │
│  3. Batch examples together                                │
│  4. Pad sequences to same length within batch              │
│  5. Create attention masks (1=valid, 0=pad)                │
│  6. Convert to PyTorch tensors                             │
│                                                             │
└────────────────────────────────────────────────────────────┘
         ↓
   Training Batch
   {
     'src_ids': tensor(batch, src_len),    # Source token IDs
     'tgt_ids': tensor(batch, tgt_len),    # Target token IDs
     'src_lens': tensor(batch),            # Actual lengths
     'tgt_lens': tensor(batch)
   }
         ↓
┌────────────────────────────────────────────────────────────┐
│  MODEL FORWARD PASS                                         │
│  (See detailed architecture below)                          │
└────────────────────────────────────────────────────────────┘
         ↓
   Predicted Summary
```

### Detailed Preprocessing Steps

#### Step 1: Raw Data Loading
```python
# Input CSV format
input,target
"Patient admitted with chest pain...", "Patient has acute coronary syndrome..."
```

#### Step 2: Data Cleaning
```python
# Remove invalid examples
df = df.dropna(subset=['note', 'summary'])
df = df[df['note'].str.len() > 50]        # Min 50 chars for source
df = df[df['summary'].str.len() > 10]     # Min 10 chars for target

# Result: 5000 valid examples (from original dataset)
```

#### Step 3: SentencePiece Tokenization
```python
# Tokenizer training (done once)
spm.SentencePieceTrainer.train(
    input='all_text.txt',
    model_prefix='spm',
    vocab_size=16000,
    model_type='unigram',
    character_coverage=0.9995,
    pad_id=3, unk_id=0, bos_id=1, eos_id=2
)

# Creates: spm.model, spm.vocab
```

**Vocabulary Structure** (16,000 tokens):
- **Special tokens**: `<unk>=0`, `<s>=1`, `</s>=2`, `<pad>=3`
- **Common words**: `the`, `patient`, `with`, `and`, etc.
- **Medical terms**: `hypertension`, `diabetes`, `thrombocytopenia`
- **Subwords**: `▁cardio`, `vascular`, `itis` (enables handling rare words)

#### Step 4: Tokenization Process

**Example Clinical Note**:
```
Original: "Patient has thrombocytopenia and requires monitoring."
```

**Tokenization Steps**:
```python
# 1. Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load('artifacts/tokenizer/spm.model')

# 2. Encode text to token IDs
token_ids = sp.encode("Patient has thrombocytopenia and requires monitoring.")
# Result: [245, 89, 12453, 21, 1876, 3421]

# 3. Add BOS and EOS
token_ids = [1] + token_ids + [2]
# Result: [1, 245, 89, 12453, 21, 1876, 3421, 2]

# Token mapping:
# 1: <s> (BOS)
# 245: "▁Patient"
# 89: "▁has"
# 12453: "▁thrombocytopenia"
# 21: "▁and"
# 1876: "▁requires"
# 3421: "▁monitoring"
# 2: </s> (EOS)
```

#### Step 5: Data Splitting

```python
# Split ratios
train: 80% → 4000 samples
val:   10% → 500 samples  
test:  10% → 500 samples

# Random shuffle with seed=42 for reproducibility
df = df.sample(frac=1, random_state=42)
```

#### Step 6: Parquet Storage

**Why Parquet?**
- Columnar storage format
- Efficient compression
- Fast random access
- Preserves data types (lists of integers)

**Stored Format**:
```python
{
  'src_ids': [1, 245, 89, ...],  # List of int
  'tgt_ids': [1, 156, 78, ...],  # List of int
  'src_len': 512,                 # Int
  'tgt_len': 64                   # Int
}
```

---

## Tokenization Process

### SentencePiece Unigram Model

**What is SentencePiece?**
- Unsupervised text tokenizer
- Works directly on raw text (language-agnostic)
- Handles out-of-vocabulary words via subword units
- No preprocessing required (no lowercasing, punctuation removal)

### Tokenization Algorithm

**Unigram Language Model**:
1. Start with large vocabulary of all substrings
2. Iteratively remove tokens to minimize loss
3. Final vocabulary: 16,000 most important subwords

**Subword Segmentation Example**:
```
Word: "thrombocytopenia" (rare medical term)

Possible segmentations:
1. "thrombocytopenia" (if in vocab)
2. "thrombo" + "cyto" + "penia" (subwords)
3. "th" + "rom" + "bo" + "cy" + "to" + "pe" + "nia" (fallback)

SentencePiece selects: Most probable segmentation
→ Likely: "▁thrombo" + "cytopenia" (2 tokens)
```

**Why Subwords Matter**:
- **Common words**: Kept as single tokens ("patient", "has")
- **Rare words**: Split into meaningful parts ("cardio" + "vascular")
- **Unknown words**: Can always be represented (no true OOV)
- **Medical terminology**: Often compositional ("hyper" + "tension")

### Token IDs to Embeddings

```python
# Input: Token ID
token_id = 245  # "▁Patient"

# Embedding Lookup
embedding_layer = nn.Embedding(vocab_size=16000, emb_dim=128)
embedding_vector = embedding_layer(token_id)  # Shape: (128,)

# Result: Dense vector
# [0.234, -0.567, 0.891, ..., 0.123]  # 128 values
# Learned during training to capture semantic meaning
```

**Embedding Properties**:
- Similar words have similar vectors (cosine similarity)
- "patient" and "individual" → Close in embedding space
- "thrombocytopenia" and "platelet disorder" → Related vectors
- Learned from clinical text patterns

---

## Architecture Components

### 1. Model Hierarchy

```
PointerGeneratorModel
├── HierarchicalEncoder (ChunkedEncoder)
│   ├── Embedding Layer
│   └── Bidirectional LSTM
├── PointerGeneratorDecoder
│   ├── Embedding Layer
│   ├── Unidirectional LSTM
│   ├── Attention Module (CoverageAttention / AdditiveAttention)
│   ├── Vocabulary Projection
│   └── Pointer-Generator Switch
└── Loss Computation
    ├── Negative Log-Likelihood Loss
    └── Coverage Loss
```

---

## Encoder Module

### Hierarchical Encoder Architecture

**Purpose**: Process long clinical notes efficiently using bidirectional LSTM

**Components**:

#### 1. **Embedding Layer**
```python
nn.Embedding(vocab_size=16000, emb_dim=128, padding_idx=3)
```
- **Input**: Token IDs `(batch, seq_len)`
- **Output**: Dense vectors `(batch, seq_len, 128)`
- **Initialization**: Uniform distribution [-0.1, 0.1]
- **Padding**: Zero vector for `pad_id=3`

#### 2. **Bidirectional LSTM**
```python
nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=0.3
)
```

**Forward Pass**:
- **Forward LSTM**: Processes tokens left-to-right
- **Backward LSTM**: Processes tokens right-to-left
- **Concatenation**: `[h_forward; h_backward]` → 512 dim output

**Packed Sequences**:
```python
packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
outputs, (h_n, c_n) = lstm(packed)
outputs, _ = pad_packed_sequence(outputs, batch_first=True)
```
- Skips padding for computational efficiency
- Handles variable-length sequences in same batch

#### 3. **Chunked Processing**
```python
ChunkedEncoder(chunk_len=128, num_chunks=6)
```
- **Maximum Input**: 128 × 6 = **768 tokens**
- **Purpose**: Handle long clinical notes exceeding typical LSTM limits
- **Strategy**: Split into 128-token chunks, process with shared LSTM

**Output**:
- **Encoder Outputs**: `(batch, seq_len, 512)` - All hidden states
- **Final Hidden**: `(h_n, c_n)` where `h_n, c_n: (num_layers × 2, batch, 256)`

---

## Attention Mechanisms

### Additive Attention (Bahdanau)

**Formula**:
```
e_t^i = v^T * tanh(W_h * h_s^i + W_d * h_t + b)
α_t = softmax(e_t)
context_t = Σ α_t^i * h_s^i
```

**Components**:
- **W_h**: `Linear(512 → 256)` - Transforms encoder outputs
- **W_d**: `Linear(256 → 256)` - Transforms decoder hidden state
- **v**: `Linear(256 → 1)` - Attention scoring vector
- **Activation**: `tanh`

**Implementation**:
```python
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim=256):
        self.W_h = nn.Linear(512, 256, bias=False)  # Encoder: 256*2
        self.W_d = nn.Linear(256, 256, bias=False)  # Decoder: 256
        self.v = nn.Linear(256, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs, mask):
        # decoder_hidden: (batch, 256)
        # encoder_outputs: (batch, seq_len, 512)
        
        # Expand decoder hidden: (batch, seq_len, 256)
        h_t_expanded = decoder_hidden.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Compute energy: (batch, seq_len, 256)
        energy = torch.tanh(self.W_h(encoder_outputs) + self.W_d(h_t_expanded))
        
        # Attention scores: (batch, seq_len)
        scores = self.v(energy).squeeze(2)
        
        # Mask padding positions
        scores = scores.masked_fill(mask == 0, -1e4)
        
        # Attention weights: (batch, seq_len)
        attn_weights = F.softmax(scores, dim=1)
        
        # Context vector: (batch, 512)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attn_weights
```

### Coverage Attention

**Purpose**: Prevent repetitive attention by tracking cumulative attention history

**Additional Formula**:
```
c_t = Σ_{t'=0}^{t-1} α_t'  (coverage vector)
e_t^i = v^T * tanh(W_h * h_s^i + W_d * h_t + W_c * c_t^i + b)
```

**Extra Component**:
- **W_c**: `Linear(1 → 256)` - Coverage feature transformation

**Coverage Update**:
```python
coverage_{t+1} = coverage_t + attention_weights_t
```

**Coverage Loss** (prevents re-attending):
```python
coverage_loss = Σ_i min(α_t^i, c_t^i)
```

---

## Decoder Module

### Pointer-Generator Decoder

**Purpose**: Generate summary tokens by combining vocabulary generation and source copying

#### Architecture Details

**1. Token Embedding**
```python
nn.Embedding(vocab_size=16000, emb_dim=128, padding_idx=3)
```

**2. LSTM Decoder**
```python
nn.LSTM(
    input_size=128 + 512,  # embedding + context
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.3
)
```

**Input Concatenation**:
```
LSTM Input = [Token Embedding (128) ⊕ Previous Context (512)] = 640 dim
```

**3. Output Projection**
```python
nn.Linear(896 → 16000)  # hidden(256) + context(512) + emb(128) = 896
```

#### Decoding Step Process

```python
def forward(self, input_token, last_hidden, encoder_outputs, ...):
    # 1. Embed input token
    embedded = self.embedding(input_token)  # (batch, 1, 128)
    
    # 2. Concatenate with previous context
    lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # (batch, 1, 640)
    
    # 3. LSTM step
    lstm_output, hidden = self.lstm(lstm_input, last_hidden)  # (batch, 1, 256)
    
    # 4. Attention
    context, attn_weights, coverage = self.attention(
        lstm_output.squeeze(1), encoder_outputs, coverage, mask
    )
    
    # 5. Vocabulary distribution
    out_features = torch.cat([lstm_output.squeeze(1), context, embedded.squeeze(1)], dim=1)
    vocab_logits = self.out_proj(out_features)  # (batch, 16000)
    vocab_dist = F.softmax(vocab_logits, dim=1)
    
    # 6. Pointer-generator (next section)
    ...
```

---

## Pointer-Generator Mechanism

### Core Concept

**Hybrid Generation Strategy**:
- **Generation Mode**: Sample from vocabulary (handles common words, grammar)
- **Copy Mode**: Copy rare/OOV words directly from source (handles names, technical terms)

### Mathematical Formulation

**Generation Probability**:
```
p_gen = σ(W_gen * [context_t; h_t; emb_t] + b_gen)
```
where σ is sigmoid function.

**Final Distribution**:
```
P_final(w) = p_gen * P_vocab(w) + (1 - p_gen) * P_copy(w)

P_vocab(w) = softmax(W_out * [h_t; context_t; emb_t])
P_copy(w) = Σ_{i: x_i=w} α_t^i
```

**Extended Vocabulary**: Union of fixed vocabulary + source tokens

### Implementation

```python
class PointerGeneratorDecoder:
    def __init__(self, ...):
        # p_gen computation
        self.p_gen_linear = nn.Linear(896, 1)  # context(512) + hidden(256) + emb(128)
    
    def forward(self, ...):
        # ... (previous steps)
        
        # Compute p_gen
        p_gen_input = torch.cat([context, lstm_output.squeeze(1), embedded.squeeze(1)], dim=1)
        p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))  # (batch, 1)
        
        # Mix distributions
        final_dist = p_gen * vocab_dist  # (batch, vocab_size)
        
        # Add copy distribution
        copy_dist = (1 - p_gen) * attn_weights  # (batch, src_len)
        
        # Scatter copy probabilities to vocab positions
        final_dist = final_dist.scatter_add(
            dim=1,
            index=encoder_input_ids,  # source token IDs
            src=copy_dist
        )
        
        return final_dist, ...
```

### Example Scenario

**Source**: "Patient has **thrombocytopenia** and requires monitoring."

**Target**: "Patient diagnosed with **thrombocytopenia**."

- **"Patient"**: Common word → **Generate** from vocabulary (p_gen ≈ 0.9)
- **"thrombocytopenia"**: Rare medical term → **Copy** from source (p_gen ≈ 0.1)
- **"diagnosed"**: Abstract term → **Generate** (p_gen ≈ 0.8)

---

## Coverage Mechanism

### Problem: Repetition

Without coverage, attention may repeatedly focus on same source positions, causing:
- Repeated phrases: "patient has... patient has..."
- Redundant information

### Solution: Coverage Vector

**Track Attention History**:
```
c_t = Σ_{t'=0}^{t-1} α_t'
```

**Coverage Loss**:
```
L_coverage = Σ_t Σ_i min(α_t^i, c_t^i)
```

**Interpretation**:
- Penalizes attending to positions with high cumulative attention
- Encourages diverse attention across source
- Weight: `coverage_lambda = 1.0` (can tune)

### Impact on Training

**Loss Function**:
```
L_total = L_NLL + λ_coverage * L_coverage
```

**Trade-off**:
- Higher λ: Less repetition, may miss important info
- Lower λ: More coverage, may repeat

**Current Setting**: `coverage_lambda = 0.0` initially, enable after model stabilizes

---

## Word Generation Process

### Complete Word Generation Pipeline

Every word in the summary is generated through this detailed process:

#### **Overview**
```
Current Decoder State (h_t, c_t)
         +
Previous Context Vector (context_{t-1})
         +
Previous Token Embedding (emb_{t-1})
         ↓
     LSTM Step
         ↓
   New Hidden State (h_t)
         ↓
    ATTENTION
         ↓
  Context Vector (context_t)
         ↓
POINTER-GENERATOR
         ↓
  Final Distribution P(w)
         ↓
   Sample/Argmax
         ↓
    NEXT WORD
```

---

## Word Generation Process

### Complete Word Generation Pipeline

Every word in the summary is generated through this detailed process:

#### **Overview**
```
Current Decoder State (h_t, c_t)
         +
Previous Context Vector (context_{t-1})
         +
Previous Token Embedding (emb_{t-1})
         ↓
     LSTM Step
         ↓
   New Hidden State (h_t)
         ↓
    ATTENTION
         ↓
  Context Vector (context_t)
         ↓
POINTER-GENERATOR
         ↓
  Final Distribution P(w)
         ↓
   Sample/Argmax
         ↓
    NEXT WORD
```

### Step-by-Step Generation (Time Step t)

#### **Input Preparation**
```python
# At time step t, we have:
input_token_id = 245  # "Patient" (previous token or BOS)
decoder_hidden = (h_{t-1}, c_{t-1})  # Previous LSTM state
context_prev = context_{t-1}  # Previous context vector (512 dim)
coverage = coverage_{t-1}  # Cumulative attention (src_len,)
```

#### **Step 1: Token Embedding**
```python
# Convert token ID to dense vector
embedding = self.embedding(input_token_id)  # (1, 128)

# Result: Dense representation of "Patient"
# embedding = [0.234, -0.567, 0.891, ..., 0.123]  # 128 values
```

#### **Step 2: Concatenate with Context**
```python
# Combine embedding with previous context
lstm_input = torch.cat([embedding, context_prev], dim=1)
# Shape: (1, 128 + 512) = (1, 640)

# This tells LSTM:
# - What word we just generated (embedding)
# - What source content we were focusing on (context)
```

#### **Step 3: LSTM Processing**
```python
# Feed through 2-layer LSTM
lstm_output, (h_t, c_t) = self.lstm(lstm_input, (h_{t-1}, c_{t-1}))
# lstm_output: (1, 256) - Current decoder state
# h_t, c_t: Hidden and cell states for next step

# LSTM captures:
# - What has been generated so far
# - What should come next
# - Grammar and coherence patterns
```

#### **Step 4: Attention Mechanism**

**Purpose**: Decide which source words to focus on for current generation

```python
# Compute attention scores for each source position
for i in range(src_len):
    # Energy function
    e_i = v^T * tanh(
        W_h * encoder_outputs[i] +  # Source word i representation
        W_d * lstm_output +          # Current decoder state
        W_c * coverage[i]            # How much we've attended before
    )

# Convert to probabilities
attention_weights = softmax(e_1, e_2, ..., e_src_len)
# Example: [0.05, 0.12, 0.48, 0.23, 0.08, 0.04]
#           Position 3 gets highest attention (0.48)

# Create context vector (weighted sum of source)
context_t = Σ attention_weights[i] * encoder_outputs[i]
# Shape: (512,) - Represents relevant source information

# Update coverage
coverage_t = coverage_{t-1} + attention_weights
# Tracks cumulative attention to prevent repetition
```

**Attention Example**:
```
Source: "Patient has thrombocytopenia and requires monitoring"
         ↑       ↑    ↑↑↑↑↑↑↑↑↑↑↑↑↑↑   ↑    ↑        ↑
Attention: [0.05, 0.12, 0.48,        0.23, 0.08,    0.04]

Current generation: "thrombocytopenia"
→ Model attends strongly to source position 3 (the medical term itself)
```

#### **Step 5: Vocabulary Distribution**

**Generate P_vocab**: Probability over all 16,000 vocabulary words

```python
# Concatenate features
features = torch.cat([lstm_output, context_t, embedding], dim=1)
# Shape: (1, 256 + 512 + 128) = (1, 896)

# Project to vocabulary
vocab_logits = self.out_proj(features)  # (1, 16000)
# Each value represents unnormalized probability for that word

# Convert to probabilities
P_vocab = softmax(vocab_logits)
# Shape: (1, 16000)
# P_vocab[0] = 0.0001  # <unk>
# P_vocab[245] = 0.032  # "Patient"
# P_vocab[89] = 0.018   # "has"
# ...
# P_vocab[12453] = 0.0005  # "thrombocytopenia" (rare, low prob)
```

#### **Step 6: Pointer-Generator Switch**

**Decide**: Should we generate from vocabulary or copy from source?

```python
# Compute generation probability
p_gen = sigmoid(W_gen * features)  # Scalar in [0, 1]
# Example: p_gen = 0.15 (low → prefer copying)

# Why low p_gen?
# - Current source word "thrombocytopenia" is rare
# - Model learned: rare medical terms should be copied
# - High attention on source position → copying mode
```

#### **Step 7: Copy Distribution**

**Create P_copy**: Probability of copying each source word

```python
P_copy = attention_weights  # Already computed in Step 4
# Shape: (src_len,)
# Example: [0.05, 0.12, 0.48, 0.23, 0.08, 0.04]

# Map source positions to vocabulary IDs
source_ids = [1, 245, 89, 12453, 21, 1876, 3421, 2]
#              ↑   ↑    ↑    ↑    ↑    ↑     ↑    ↑
#             <s> Patient has thrombo and requires monitoring </s>
```

#### **Step 8: Final Distribution**

**Combine generation and copying**:

```python
# Initialize with generation probabilities
P_final = p_gen * P_vocab  # Shape: (1, 16000)

# Add copy probabilities
for i, src_token_id in enumerate(source_ids):
    copy_prob = (1 - p_gen) * attention_weights[i]
    P_final[0, src_token_id] += copy_prob

# Example calculation for "thrombocytopenia" (token_id=12453):
# P_vocab[12453] = 0.0005 (low, rare word)
# attention_weights[3] = 0.48 (high attention on source)
# p_gen = 0.15 (prefer copying)
# 
# P_final[12453] = 0.15 * 0.0005 + (1 - 0.15) * 0.48
#                = 0.000075 + 0.408
#                = 0.408075
# → High probability! Model will likely copy "thrombocytopenia"
```

**Visual Representation**:
```
Vocabulary Probability vs Copy Probability for "thrombocytopenia":

P_vocab (generate):  ▏ (0.05%)
                     
P_copy (from source): ████████████████████████ (40.8%)
                     
Final P_final:       ████████████████████████ (40.8%)
                     ↑ Dominated by copying!
```

#### **Step 9: Token Selection**

**Training**: Use ground truth (teacher forcing)
```python
# Ground truth says next word is "thrombocytopenia" (id=12453)
next_token = target_ids[t]  # Use ground truth
```

**Inference**: Sample from distribution
```python
# Greedy decoding
next_token = argmax(P_final)  # Token with highest probability
# Result: 12453 ("thrombocytopenia")

# OR Beam search (keep top-k hypotheses)
# See detailed beam search section below
```

#### **Step 10: Update State**
```python
# Update for next time step
decoder_hidden = (h_t, c_t)
context_prev = context_t
coverage_prev = coverage_t
input_token = next_token

# Repeat Steps 1-10 until:
# - Generate </s> (EOS token)
# - Reach max_length (384 tokens)
```

### Example Generation Trace

**Source**: "Patient has thrombocytopenia"

**Generation Steps**:

| Step | Input Token | Attention Focus | p_gen | Action | Output Token |
|------|------------|----------------|-------|---------|--------------|
| 1 | `<s>` | "Patient" | 0.82 | Generate | "Patient" |
| 2 | "Patient" | "has" | 0.89 | Generate | "has" |
| 3 | "has" | "thrombocytopenia" | 0.15 | **Copy** | "thrombocytopenia" |
| 4 | "thrombocytopenia" | - | 0.91 | Generate | `</s>` |

**Output**: "Patient has thrombocytopenia"

**Key Insights**:
- **Common words** ("Patient", "has"): High p_gen → Generate from vocabulary
- **Rare medical term** ("thrombocytopenia"): Low p_gen → Copy from source
- **Grammar and flow**: LSTM maintains coherence
- **Attention**: Dynamically focuses on relevant source positions

### Word Selection Strategies

#### **Greedy Decoding**
```python
next_token = argmax(P_final)
```
- **Pros**: Fast, deterministic
- **Cons**: May miss better global sequences

#### **Beam Search** (Used in our model)
```python
# Keep top-4 hypotheses at each step
beams = [
    ("Patient has", prob=0.78),
    ("Patient diagnosed", prob=0.72),
    ("Patient presents", prob=0.68),
    ("Individual has", prob=0.65)
]

# Expand each beam, keep top-4 overall
```
- **Pros**: Better quality, explores alternatives
- **Cons**: Slower (4x more computation)
- **Used during evaluation**: beam_size=4

#### **Sampling** (Alternative)
```python
next_token = sample(P_final, temperature=0.8)
```
- **Pros**: More diverse outputs
- **Cons**: Less reliable, may produce errors
- **Not used in our model** (prefer deterministic beam search)

---

## Training Process

### Forward Pass

```python
def forward(self, src_ids, src_lengths, tgt_ids, teacher_forcing_ratio=1.0):
    # 1. Encode source
    encoder_outputs, encoder_hidden = self.encoder(src_ids, src_lengths)
    
    # 2. Initialize decoder
    decoder_hidden = self._bridge_encoder_hidden(encoder_hidden)
    context = torch.zeros(batch, 512)
    coverage = torch.zeros(batch, src_len)
    
    # 3. Decode step-by-step with teacher forcing
    outputs = []
    input_token = tgt_ids[:, 0]  # <BOS>
    
    for t in range(1, tgt_len):
        # Decoder step
        final_dist, decoder_hidden, context, attn, p_gen, coverage = self.decoder(
            input_token, decoder_hidden, encoder_outputs, mask, context, coverage, src_ids
        )
        
        outputs.append(final_dist)
        
        # Teacher forcing: use ground truth
        if random() < teacher_forcing_ratio:
            input_token = tgt_ids[:, t]
        else:
            input_token = final_dist.argmax(dim=1)
    
    return torch.stack(outputs, dim=1)  # (batch, tgt_len-1, vocab_size)
```

### Loss Computation

**Combined Loss**:
```python
def compute_loss(self, outputs, targets, attentions, coverages):
    # 1. Negative Log-Likelihood
    if self.use_pointer_gen:
        # Outputs are probabilities
        log_probs = torch.log(outputs.clamp(min=1e-8))
        nll_loss = F.nll_loss(log_probs.reshape(-1, vocab_size), 
                               targets.reshape(-1),
                               ignore_index=pad_id)
    else:
        # Outputs are logits
        nll_loss = F.cross_entropy(outputs.reshape(-1, vocab_size),
                                    targets.reshape(-1),
                                    ignore_index=pad_id,
                                    label_smoothing=0.0)
    
    # 2. Coverage Loss
    prev_coverage = torch.cat([torch.zeros(batch, 1, src_len), coverages[:, :-1]], dim=1)
    coverage_loss = torch.min(attentions, prev_coverage).sum(dim=2).mean()
    
    # 3. Total Loss
    total_loss = nll_loss + coverage_lambda * coverage_loss
    
    return total_loss, nll_loss, coverage_loss
```

### Optimization

**Optimizer**: Adam
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4
)
```

**Learning Rate Schedule**: Cosine with Warmup
```python
# Warmup: 0 → peak_lr over 1000 steps
# Decay: peak_lr → 0 over 45000 steps using cosine annealing
```

**Gradient Accumulation**: 16 steps (effective batch size = 16)

**Gradient Clipping**: `max_norm = 1.0`

**Mixed Precision (FP16)**:
```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(src_ids, src_lengths, tgt_ids)
    loss = compute_loss(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Model Specifications

### Current Configuration

| **Component**          | **Specification**                    |
|------------------------|--------------------------------------|
| **Vocabulary Size**    | 16,000 (SentencePiece)              |
| **Embedding Dim**      | 128                                  |
| **Encoder Hidden Dim** | 256 (Bidirectional → 512 output)    |
| **Decoder Hidden Dim** | 256                                  |
| **LSTM Layers**        | 2 (both encoder & decoder)          |
| **Dropout**            | 0.3                                  |
| **Max Source Length**  | 768 tokens (128 × 6 chunks)         |
| **Max Target Length**  | 192 tokens                           |
| **Attention Type**     | Additive (Bahdanau) with Coverage   |
| **Pointer-Gen**        | Enabled                              |
| **Coverage**           | Disabled initially (λ = 0.0)        |

### Model Size

**Total Parameters**: **22,459,137**

**Breakdown**:
```
Encoder:
  - Embedding: 16,000 × 128 = 2,048,000
  - BiLSTM: 2 layers × 2 directions × (4 × (128 + 256) × 256) ≈ 1,574,912
  
Decoder:
  - Embedding: 16,000 × 128 = 2,048,000
  - LSTM: 2 layers × (4 × (640 + 256) × 256) ≈ 1,835,008
  - Attention: 512 × 256 + 256 × 256 + 256 × 1 ≈ 197,120
  - Output Projection: 896 × 16,000 = 14,336,000
  - p_gen Linear: 896 × 1 = 896
```

### Hardware Requirements

**Training Configuration** (RTX 4070 8GB):
- **Batch Size**: 1
- **Gradient Accumulation**: 16 steps
- **Effective Batch Size**: 16
- **Memory Usage**: ~7.5 GB VRAM
- **Training Speed**: ~1.4 sec/iteration
- **Estimated Time**: 5-6 hours for 10,000 steps

---

## Mathematical Formulations

### Complete Mathematical Model

#### **1. Encoder**

**Embedding**:
```
e_i = E[x_i]  where E ∈ ℝ^{V×d_emb}
```

**Bidirectional LSTM**:
```
→h_t, →c_t = →LSTM(e_t, →h_{t-1}, →c_{t-1})
←h_t, ←c_t = ←LSTM(e_t, ←h_{t+1}, ←c_{t+1})
h_t^enc = [→h_t; ←h_t]  ∈ ℝ^{2d_h}
```

#### **2. Attention with Coverage**

**Coverage Vector**:
```
c_t = Σ_{t'=0}^{t-1} α_{t'}  ∈ ℝ^{T_src}
```

**Attention Score**:
```
e_t^i = v^T tanh(W_h h_i^enc + W_d h_t^dec + W_c c_t^i + b_attn)
```

**Attention Weights**:
```
α_t = softmax(e_t)  ∈ ℝ^{T_src}
```

**Context Vector**:
```
context_t = Σ_{i=1}^{T_src} α_t^i h_i^enc  ∈ ℝ^{2d_h}
```

#### **3. Decoder LSTM**

**Input**:
```
lstm_input_t = [E[y_{t-1}]; context_{t-1}]  ∈ ℝ^{d_emb + 2d_h}
```

**Hidden State Update**:
```
h_t^dec, c_t^dec = LSTM(lstm_input_t, h_{t-1}^dec, c_{t-1}^dec)
```

#### **4. Pointer-Generator**

**Vocabulary Distribution**:
```
P_vocab(w) = softmax(W_out [h_t^dec; context_t; E[y_{t-1}]])
```

**Generation Probability**:
```
p_{gen} = σ(w_gen^T [h_t^dec; context_t; E[y_{t-1}]] + b_gen)
```

**Copy Distribution**:
```
P_copy(w) = Σ_{i: x_i = w} α_t^i
```

**Final Distribution**:
```
P(w) = p_{gen} P_vocab(w) + (1 - p_{gen}) P_copy(w)
```

#### **5. Loss Functions**

**Negative Log-Likelihood**:
```
L_NLL = -Σ_{t=1}^{T_tgt} log P(y_t^* | y_{<t}, x)
```

**Coverage Loss**:
```
L_cov = Σ_{t=1}^{T_tgt} Σ_{i=1}^{T_src} min(α_t^i, c_t^i)
```

**Total Loss**:
```
L = L_NLL + λ_cov L_cov
```

---

## Beam Search Decoding

### Algorithm

**Purpose**: Find high-probability sequences instead of greedy decoding

**Process**:
1. Start with beam_size=4 hypotheses (all <BOS>)
2. At each step, expand each hypothesis with top-k tokens
3. Keep top beam_size hypotheses by score
4. Stop when <EOS> generated or max_length reached

**Scoring**:
```
score = log P(y_1, ..., y_t) / length_penalty^t
```

**Length Penalty**:
```
lp(t) = ((5 + t) / (5 + 1))^α  where α = 1.0
```

**Constraints**:
- **Min Length**: 50 tokens (force continuation)
- **Max Length**: 256 tokens
- **No Repeat N-gram**: Block trigrams (n=3)

---

## Training Metrics

### Monitoring

**Primary Metrics**:
- **Loss**: NLL + Coverage Loss
- **ROUGE-1**: Unigram overlap F1
- **ROUGE-2**: Bigram overlap F1  
- **ROUGE-L**: Longest Common Subsequence F1

**Secondary Metrics**:
- **Perplexity**: exp(NLL)
- **Gradient Norm**: Monitor for explosion
- **Learning Rate**: Track schedule

### Evaluation

**Validation Every 500 Steps**:
```python
# Generate summaries for validation set
predictions = model.generate(val_src, beam_size=4)

# Compute ROUGE scores
rouge_scores = compute_rouge(predictions, val_targets)

# Save best model
if rouge_scores['rougeL'] > best_score:
    save_checkpoint('best_model.pt')
```

---

## References

### Key Papers

1. **Pointer-Generator Networks** (See et al., 2017)
   - Original paper introducing pointer-generator mechanism
   - [ACL Anthology](https://aclanthology.org/P17-1099/)

2. **Sequence-to-Sequence Learning** (Sutskever et al., 2014)
   - Foundation for seq2seq models
   - [NeurIPS 2014](https://arxiv.org/abs/1409.3215)

3. **Attention Mechanism** (Bahdanau et al., 2015)
   - Additive attention for NMT
   - [ICLR 2015](https://arxiv.org/abs/1409.0473)

4. **Coverage Mechanism** (Tu et al., 2016)
   - Coverage for preventing repetition
   - [ACL 2016](https://aclanthology.org/P16-1008/)

### Implementation Details

- **Framework**: PyTorch 2.0+
- **Tokenizer**: SentencePiece (Unigram model)
- **Dataset**: MIMIC-IV BHC (Clinical Notes)
- **Hardware**: NVIDIA RTX 4070 8GB Laptop GPU

---

## Summary

The Pointer-Generator Network combines:

✅ **Hierarchical Encoding**: Processes long clinical notes (768 tokens) with bidirectional LSTM  
✅ **Additive Attention**: Focuses on relevant source positions dynamically  
✅ **Pointer-Generator**: Switches between vocabulary generation and source copying  
✅ **Coverage Mechanism**: Tracks attention history to prevent repetition  
✅ **Beam Search**: Finds high-quality summaries during inference  

**Key Innovation**: Hybrid generation strategy handles both common language patterns (generation) and rare clinical terminology (copying), making it ideal for medical text summarization.

**Training Strategy**: Mixed precision training with gradient accumulation enables training on consumer GPUs while maintaining model quality.

---

**Document Version**: 1.0  
**Last Updated**: January 17, 2026  
**Model Checkpoint**: `artifacts/checkpoints/full_training_restart/best_model.pt`
