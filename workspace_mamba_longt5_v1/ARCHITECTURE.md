# Architecture: Mamba-Transformer Hybrid V2 for Clinical Note Summarization

> **Model name:** `MambaTransformerModel` (V2)  
> **Task:** Automatic summarization of MIMIC-IV discharge notes -> Brief Hospital Course (BHC)  
> **V1 parameters:** 52,146,176 | **V2 parameters:** ~79,398,406 (all features enabled)  
> **Active V1 experiment:** `exp_memory32` - Memory Token Isolation (n_memory_tokens = 32)  
> **V2 config:** `full_train.yaml` - All V2 improvements enabled

---

## 1. Overview

This model is a **from-scratch hybrid encoder-decoder** that combines:

- A **Mamba SSM encoder** for linear-time long-sequence encoding
- A **learned memory token compressor** to form a compact, fixed-size cross-attention context
- A **standard Transformer decoder** to generate summaries autoregressively

### V2 Improvements (inspired by Mamba-2, RetNet, LLaMA/PaLM)

| Feature | Source | What it does | Why it helps clinical summarization |
|---------|--------|-------------|-------------------------------------|
| **RMSNorm** | Mamba-2, LLaMA | Replaces LayerNorm with Root Mean Square Normalization | Faster computation, more stable gradients during long training |
| **SwiGLU FFN** | LLaMA, PaLM | Gated Linear Unit with SiLU activation replaces GELU FFN | Better feature mixing - proven 2-5% improvement on language tasks |
| **Bidirectional Mamba** | Mamba-2 | Runs forward + backward SSM with learned gating fusion | Full-context chunk encoding: clinical notes reference future/past findings |
| **Cross-chunk memory attention** | RetNet | Self-attention across memory tokens from all chunks | Connects information across document sections (admission -> treatment -> outcome) |
| **Gated cross-attention** | RetNet retention | Learned scalar gate on decoder cross-attention | Controls how much source info flows to each decoder layer - reduces hallucination |

The key insight is that clinical notes can be extremely long (thousands of tokens), and standard self-attention scales quadratically. Mamba provides O(n) sequence modeling, while the memory compressor distills each chunk down to K vectors, giving the decoder a fixed-size "memory" regardless of source length. V2 adds cross-chunk interaction so this memory is globally coherent.

```
Input tokens  --> Chunk --> Mamba Encoder (BiMamba V2) --> Memory Compressor
                  xN_chunks                                     |
                                                        K memory tokens/chunk
                                                                |
                                                   Cross-Chunk Attention (V2)
                                                                |
                                                        Concatenate all chunks
                                                                |
Target tokens --> Embedding --> Transformer Decoder (Gated cross-attn V2)
                                                                |
                                                        Output Projection --> Vocabulary
```

---

## 2. Model Configuration

### V1 Baseline (exp_memory32)

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
| `n_memory_tokens` | **32** | Memory tokens per chunk |
| `max_src_len` | 4,096 | Maximum source tokens |
| `max_tgt_len` | 512 | Maximum target tokens |
| `mamba_d_state` | 16 | Mamba SSM state dimension |
| `mamba_d_conv` | 4 | Mamba convolution kernel size |
| `mamba_expand` | 2 | Mamba expansion factor |
| `label_smoothing` | 0.15 | Cross-entropy smoothing |

### V2 Feature Flags (full_train.yaml)

| Flag | Default | V2 Value | Effect |
|------|---------|----------|--------|
| `use_rmsnorm` | `false` | `true` | RMSNorm replaces LayerNorm everywhere |
| `use_swiglu` | `false` | `true` | SwiGLU FFN replaces GELU FFN in decoder + cross-chunk layers |
| `use_bidirectional_mamba` | `false` | `true` | Bidirectional Mamba encoder with gated fusion |
| `use_cross_chunk_attn` | `false` | `true` | Cross-chunk memory attention after compression |
| `use_gated_cross_attn` | `false` | `true` | Learned gates on decoder cross-attention |
| `n_cross_chunk_layers` | `2` | `2` | Number of cross-chunk attention layers |

---

## 3. Component Breakdown

### 3.1 Tokenizer

- **Type:** SentencePiece unigram model (`data/tokenizer/spm.model`)
- **Vocabulary:** 16,000 subword tokens
- **Special tokens:** `<bos>=1`, `<eos>=2`, `<pad>=3`

---

### 3.2 Embeddings

```
src_embedding : Embedding(16000, 512)   <- source input tokens
tgt_embedding : Embedding(16000, 512)   <- target output tokens (shared with output projection)
pos_encoding  : Sinusoidal PE (max 8192 positions)
```

Embeddings are Xavier-initialized. Output projection weight is **tied** to `tgt_embedding` (weight sharing reduces parameters and improves generation quality).

---

### 3.3 Chunking Strategy

Before encoding, the source sequence is sliced into overlapping windows:

```
source: [t_0, t_1, ..., t_L]

chunk_0 : tokens [0   : 256 ]
chunk_1 : tokens [192 : 448 ]   <- 64-token overlap with chunk_0
chunk_2 : tokens [384 : 640 ]
...
```

- **chunk_size = 256 tokens** - fits within Mamba's efficient processing range
- **stride = 192** (overlap = 64) - ensures continuity at chunk boundaries
- Chunks are zero-padded to `chunk_size` if the last chunk is shorter

For a 4096-token input: ceil((4096 - 256) / 192) + 1 = 21 chunks

---

### 3.4 Mamba Encoder

#### V1 (Unidirectional)

Each chunk is independently encoded by a stack of **6 Mamba blocks**:

```
MambaEncoder
+-- MambaBlock x 6
|   +-- LayerNorm
|   +-- Mamba SSM  (d_model=512, d_state=16, d_conv=4, expand=2)
|   +-- Dropout(0.15)
|   +-- Residual connection
+-- Final LayerNorm
```

#### V2 (Bidirectional - inspired by Mamba-2)

```
MambaEncoder
+-- BidirectionalMambaBlock x 6
|   +-- RMSNorm (or LayerNorm)
|   +-- Mamba SSM Forward  (d_model=512, d_state=16, d_conv=4, expand=2)
|   +-- Mamba SSM Backward (same config, processes reversed sequence)
|   +-- Gated Fusion: gate = sigmoid(Linear([fwd; bwd])) -> gate*fwd + (1-gate)*bwd
|   +-- Dropout(0.15)
|   +-- Residual connection
+-- Final RMSNorm
```

**Why bidirectional?** In clinical notes, a diagnosis mentioned at the end of a section may contextualize findings mentioned earlier. Bidirectional encoding captures both forward and backward dependencies within each chunk.

**Mamba SSM** ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752)) is a State Space Model that:
- Processes sequences in O(n) time and memory (vs O(n^2) for attention)
- Uses selective state transitions controlled by the input itself
- Has a hardware-aware parallel scan for fast GPU training

Each chunk `[batch, 256, 512]` -> encoded `[batch, 256, 512]`

---

### 3.5 Memory Token Compressor

This is the **architectural centrepiece** - it converts a variable-length chunk representation into exactly K fixed-size memory tokens via learned cross-attention:

```
MemoryTokenCompressor
+-- memory_queries : nn.Parameter(K=32, d_model=512)   <- learned
+-- cross_attn     : MultiheadAttention(512, 8 heads)
+-- RMSNorm (V2) or LayerNorm (V1)
+-- Dropout(0.15)

Forward:
  queries [B, 32, 512] x key/value [B, 256, 512]  ->  memory_tokens [B, 32, 512]
```

**Why this matters:**
- Each chunk (256 tokens) is compressed to 32 memory tokens
- **Compression ratio: 256 -> 32 = 8x per chunk (12.5% retention)**
- The baseline used K=8 (3.1% retention), which caused information loss
- With K=32, the decoder gets 4x more information per chunk

For 21 chunks: `21 x 32 = 672 memory tokens` total context for the decoder.

---

### 3.6 Cross-Chunk Memory Attention (V2 - RetNet-inspired)

**This is the key V2 innovation for clinical summarization.**

After all chunks are compressed to memory tokens, they are concatenated into a single sequence. The Cross-Chunk Attention layers then allow memory tokens from different chunks to attend to each other:

```
CrossChunkAttention x 2
+-- RMSNorm -> Self-Attention (all memory tokens attend to all) -> Dropout + Residual
+-- RMSNorm -> SwiGLU FFN -> Residual
```

**Why this matters for clinical text:**
- Chunk 0 might contain "Patient admitted with chest pain"
- Chunk 5 might contain "Troponin elevated, started heparin drip"
- Chunk 15 might contain "Cardiac catheterization performed"
- Cross-chunk attention allows these related concepts to interact BEFORE the decoder sees them
- Without this, the decoder must independently discover cross-chunk relationships through cross-attention alone

**Inspiration from RetNet:** RetNet's retention mechanism uses a decay factor to weight nearby tokens more. Our cross-chunk attention provides similar global context mixing but through explicit self-attention over the compressed memory tokens, which is computationally efficient since the memory sequence is much shorter than the original input.

---

### 3.7 Transformer Decoder

The decoder is a pre-norm Transformer with 6 layers:

#### V1 (Standard)

```
TransformerDecoder
+-- TransformerDecoderLayer x 6
|   +-- LayerNorm -> Masked Self-Attention -> Dropout + Residual
|   +-- LayerNorm -> Cross-Attention (-> memory tokens) -> Dropout + Residual
|   +-- LayerNorm -> Feed-Forward (512->2048->512, GELU) -> Residual
|   +-- (causal mask enforced in self-attention)
+-- Final LayerNorm
```

#### V2 (with SwiGLU, RMSNorm, Gated Cross-Attention)

```
TransformerDecoder
+-- TransformerDecoderLayer x 6
|   +-- RMSNorm -> Masked Self-Attention -> Dropout + Residual
|   +-- RMSNorm -> Cross-Attention (-> memory) -> Gated: sigmoid(gate) * output -> Residual
|   +-- RMSNorm -> SwiGLU FFN (512->2048->512) -> Residual
|   +-- (causal mask enforced in self-attention)
+-- Final RMSNorm
```

**Gated cross-attention:** Each decoder layer has a learned scalar gate (initialized at sigmoid(0)=0.5) that controls how much source information flows into the decoder. This prevents early layers from being overwhelmed by noisy source representations and allows the model to learn the optimal information flow.

**SwiGLU FFN:** Instead of `GELU(xW1)W2`, uses `SiLU(xW1) * (xW3) * W2`. The extra gate projection (W3) allows the model to selectively filter features, yielding better representation learning.

**Cross-attention:** decoder tokens attend to all `N_chunks x K` memory tokens. This gives the decoder full access to the compressed source representation regardless of source length.

---

### 3.8 Output Projection

```
Linear(512, 16000, bias=False)   <- weight tied to tgt_embedding
```

Followed by cross-entropy loss with label smoothing (epsilon = 0.15).

---

## 4. Information Flow

### V1 Flow

```
Source (up to 4096 tokens)
        |
        v
   Chunk into 21 windows of 256 tokens (stride 192)
        |
        v  For each chunk:
   +-------------------------------------+
   |  src_embedding + pos_encoding       |
   |  -> MambaEncoder (6 layers)         |  [B, 256, 512]
   |  -> MemoryCompressor (K=32 queries) |  [B, 32, 512]
   +-------------------------------------+
        |
        v
   Concatenate all chunks -> memory [B, 672, 512]
        |
        v
   TransformerDecoder (6 layers) -------------- cross-attn to memory
        |
        v
   Output projection -> logits [B, tgt_len, 16000]
        |
        v
   Autoregressive generation (greedy or sampling)
```

### V2 Flow (all improvements enabled)

```
Source (up to 4096 tokens)
        |
        v
   Chunk into 21 windows of 256 tokens (stride 192)
        |
        v  For each chunk:
   +----------------------------------------------+
   |  src_embedding + pos_encoding                |
   |  -> BiMamba Encoder (6 layers, fwd+bwd+gate) |  [B, 256, 512]
   |  -> MemoryCompressor (K=32, RMSNorm)         |  [B, 32, 512]
   +----------------------------------------------+
        |
        v
   Concatenate all chunks -> memory [B, 672, 512]
        |
        v
   Cross-Chunk Attention (2 layers, RMSNorm + SwiGLU)
        |                  ^--- memory tokens from different chunks interact
        v
   TransformerDecoder (6 layers, RMSNorm + SwiGLU + Gated Cross-Attn)
        |                  ^--- learned gates control source info flow
        v
   Output projection -> logits [B, tgt_len, 16000]
        |
        v
   Autoregressive generation (greedy or sampling)
```

---

## 5. Training Setup

### V1 (exp_memory32 - current)

| Setting | Value |
|---------|-------|
| Dataset | MIMIC-IV Brief Hospital Course (BHC) |
| Training samples | 256,529 |
| Validation samples | 13,502 |
| Total steps | 50,000 |
| Warmup steps | 2,000 (cosine LR) |
| Learning rate | 5 x 10^-5 |
| Batch size | 2 x grad_accum 8 = **16 effective** |
| Optimizer | AdamW (weight_decay=0.03) |
| Gradient clipping | max_norm = 1.0 |
| Mixed precision | AMP (float16) |
| Checkpointing | Every 200 steps |
| Evaluation | Every 1,000 steps (ROUGE-1/2/L + Val Loss) |
| GPU | NVIDIA RTX 4070 Laptop (8.59 GB VRAM) |
| Framework | PyTorch 2.5.1 + CUDA 12.1 |

### V2 (full_train.yaml)

Same training schedule as V1, with the V2 model architecture enabled. The V2 model uses ~79M parameters vs ~52M for V1, but fits within 8GB VRAM with batch_size=2 and gradient_accumulation=8.

---

## 6. V2 Technical Deep Dive

### 6.1 RMSNorm vs LayerNorm

**LayerNorm (V1):**
```
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
```

**RMSNorm (V2):**
```
RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps)
```

RMSNorm removes the mean-centering step, which:
- Reduces computation by ~15%
- Provides more stable gradients (no mean subtraction noise)
- Used in LLaMA, Mamba-2, Gemma, and most modern LLMs

### 6.2 SwiGLU FFN vs GELU FFN

**GELU FFN (V1):**
```
FFN(x) = Dropout(Linear_2(GELU(Linear_1(x))))
```

**SwiGLU FFN (V2):**
```
FFN(x) = Dropout(W2 * (SiLU(W1 * x) * W3 * x))
```

SwiGLU adds a gate projection (W3) that learns to selectively pass features. This was shown to improve perplexity by 2-5% in PaLM and LLaMA papers. The trade-off is 50% more FFN parameters, but the quality improvement is consistent.

### 6.3 Bidirectional Mamba Encoding

Standard Mamba processes sequences left-to-right only. For encoder tasks (where we need full context), bidirectional processing is critical:

```
Forward:  [t1] -> [t1,t2] -> [t1,t2,t3] -> ... -> [t1,...,tn]
Backward: [tn] -> [tn,tn-1] -> [tn,tn-1,tn-2] -> ... -> [tn,...,t1]
Fusion:   gate = sigmoid(Linear([fwd_out; bwd_out]))
          output = gate * fwd_out + (1 - gate) * bwd_out
```

The learned gate allows each position to dynamically choose how much forward vs backward context to use. This is especially important for clinical notes where:
- Lab results may precede or follow their interpretation
- Medications may be mentioned before their indication
- Temporal relationships are often non-linear

### 6.4 Cross-Chunk Memory Attention

Without cross-chunk attention, each chunk's K memory tokens are independently compressed and the decoder must discover relationships between chunks through its cross-attention layers. With cross-chunk attention, the memory tokens can interact before the decoder sees them:

```
Before cross-chunk attn:       After cross-chunk attn:
[chunk_0 memory] | isolated    [chunk_0 memory] | mixed with all
[chunk_1 memory] | isolated    [chunk_1 memory] | mixed with all
...                            ...
[chunk_20 memory]| isolated    [chunk_20 memory]| mixed with all
```

This is inspired by RetNet's retention mechanism, which uses exponential decay for long-range mixing. Our approach uses explicit self-attention over the compressed memory, which is computationally cheap (672 tokens vs 4096 original).

### 6.5 Gated Cross-Attention

Each decoder layer learns a scalar gate that controls source information flow:

```
cross_out = CrossAttention(decoder_state, memory)
output = sigmoid(gate) * dropout(cross_out) + residual
```

- `gate` is initialized to 0 (sigmoid(0) = 0.5), starting balanced
- During training, the model learns per-layer optimal gate values
- Lower layers may learn smaller gates (rely more on decoder self-attention)
- Higher layers may learn larger gates (incorporate more source info for generation)

This prevents template hallucination by giving the model fine-grained control over how much to copy from the source vs generate from learned patterns.

---

## 7. Experiment: Memory Token Isolation (V1)

The `exp_memory32` config is a **controlled ablation** to test the memory bottleneck hypothesis:

| | Baseline (`full_train` V1) | **Memory32** (`exp_memory32`) |
|---|---|---|
| `n_memory_tokens` | 8 | **32** |
| Information retention | 3.1% per chunk | **12.5% per chunk** |
| Total memory tokens | ~168 | **~672** |
| All other settings | -- | **IDENTICAL** |

**Hypothesis:** The K=8 baseline suffers from template hallucination and clinical entity loss because the MemoryTokenCompressor cannot encode sufficient source fidelity at 3.1% retention. K=32 (4x more) should yield improved ROUGE scores and clinical accuracy.

**Best ROUGE-L so far (step ~3400):** 0.1814

---

## 8. V1 vs V2 Comparison

| Aspect | V1 (baseline) | V2 (improved) |
|--------|--------------|---------------|
| Encoder | Unidirectional Mamba | Bidirectional Mamba (fwd+bwd+gate) |
| Normalization | LayerNorm | RMSNorm |
| FFN | GELU (Linear->GELU->Linear) | SwiGLU (Linear->SiLU * Linear->Linear) |
| Memory interaction | None (chunks isolated) | Cross-Chunk Attention (2 layers) |
| Decoder cross-attn | Standard additive | Gated (learned sigmoid gate) |
| Parameters | ~52M | ~79M |
| Expected benefit | Baseline | +5-15% ROUGE improvement |
| Hallucination | Subject to template hallucination | Reduced via gated cross-attn |
| Long-range coherence | Limited by chunk isolation | Improved via cross-chunk attention |

---

## 9. File Structure

```
workspace_mamba_longt5_v1/
+-- src/
|   +-- model.py           # MambaTransformerModel V2 - full architecture
|   +-- train.py           # Training loop (AMP, cosine LR, auto-resume)
|   +-- data_loader.py     # MIMIC-IV CSV loader + SentencePiece tokenizer
|   +-- clinical_eval.py   # Clinical evaluation (greedy + beam, 3 gates)
|   +-- evaluate.py        # ROUGE + Val loss evaluation
|   +-- __init__.py
+-- configs/
|   +-- exp_memory32.yaml  # V1 experiment config (K=32, V2 flags commented out)
|   +-- full_train.yaml    # V2 config (all improvements enabled)
|   +-- default.yaml       # Base defaults
+-- gated_eval_v2.py       # Gated evaluation: Safety / Quality / Similarity
+-- fast_eval.py           # Quick ROUGE evaluation on checkpoints
+-- monitor_training.py    # Live training log monitor
+-- scripts/
|   +-- batch_eval.py      # Batch evaluation across multiple checkpoints
+-- requirements.txt
+-- ARCHITECTURE.md        # This file
```

---

## 10. Dependencies

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

## 11. Reproducing Experiments

### V1 (current training)

```bash
# Train (auto-resumes from latest checkpoint if interrupted)
python src/train.py --config configs/exp_memory32.yaml

# Monitor live training
python monitor_training.py
```

### V2 (improved architecture)

```bash
# Train V2 from scratch (all improvements enabled)
python src/train.py --config configs/full_train.yaml

# Evaluate a checkpoint
python fast_eval.py --checkpoint checkpoints/v2_run/best_model.pt \
                    --config configs/full_train.yaml \
                    --max_samples 200

# Full clinical evaluation with beam search
python src/clinical_eval.py \
    --checkpoint checkpoints/v2_run/best_model.pt \
    --config configs/full_train.yaml \
    --max_samples 100 --beam_size 4 --compare_greedy
```

---

## 12. References

- **Mamba:** Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- **Mamba-2:** Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- **RetNet:** Sun, Y., et al. (2023). *Retentive Network: A Successor to Transformer for Large Language Models.* [arXiv:2307.08621](https://arxiv.org/abs/2307.08621)
- **SwiGLU:** Shazeer, N. (2020). *GLU Variants Improve Transformer.* [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- **LLaMA:** Touvron, H., et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.* [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- **MIMIC-IV:** Johnson, A., et al. (2023). *MIMIC-IV, a freely accessible electronic health record dataset.* Scientific Data.
- **Transformer:** Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
- **Memory Tokens / Perceiver:** Jaegle, A., et al. (2021). *Perceiver: General Perception with Iterative Attention.* ICML.
- **RMSNorm:** Zhang, B., & Sennrich, R. (2019). *Root Mean Square Layer Normalization.* NeurIPS.
