# Model Mathematics — Mamba-Transformer Hybrid V2

> **Model:** `MambaTransformerModel` (V2)  
> **Parameters:** 79,410,694  
> **Architecture:** Mamba SSM Encoder + Transformer Decoder with memory compression

---

## Table of Contents

1. [Notation](#1-notation)
2. [Embedding Layer](#2-embedding-layer)
3. [Sinusoidal Positional Encoding](#3-sinusoidal-positional-encoding)
4. [Chunking Formulation](#4-chunking-formulation)
5. [State Space Model (SSM) — Mamba Core](#5-state-space-model-ssm--mamba-core)
6. [Bidirectional Mamba Block (V2)](#6-bidirectional-mamba-block-v2)
7. [RMSNorm (V2)](#7-rmsnorm-v2)
8. [SwiGLU Feed-Forward Network (V2)](#8-swiglu-feed-forward-network-v2)
9. [Memory Token Compression](#9-memory-token-compression)
10. [Multi-Head Attention](#10-multi-head-attention)
11. [Cross-Chunk Memory Attention (V2)](#11-cross-chunk-memory-attention-v2)
12. [Transformer Decoder Layer](#12-transformer-decoder-layer)
13. [Gated Cross-Attention (V2)](#13-gated-cross-attention-v2)
14. [Output Projection & Weight Tying](#14-output-projection--weight-tying)
15. [Loss Function — Label-Smoothed Cross-Entropy](#15-loss-function--label-smoothed-cross-entropy)
16. [AdamW Optimizer](#16-adamw-optimizer)
17. [Cosine Schedule with Linear Warmup](#17-cosine-schedule-with-linear-warmup)
18. [Gradient Clipping](#18-gradient-clipping)
19. [Autoregressive Generation](#19-autoregressive-generation)
20. [Beam Search](#20-beam-search)
21. [Parameter Count Derivation](#21-parameter-count-derivation)

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $L_s$ | Source sequence length (≤ 4096 tokens) |
| $L_t$ | Target sequence length (≤ 512 tokens) |
| $d$ | Model dimension ($d_{\text{model}} = 512$) |
| $V$ | Vocabulary size (16,000) |
| $H$ | Number of attention heads (8) |
| $d_h$ | Head dimension ($d / H = 64$) |
| $d_{ff}$ | Feed-forward inner dimension (2048) |
| $C$ | Chunk size (256 tokens) |
| $S$ | Stride (192 tokens) |
| $K$ | Memory tokens per chunk (32) |
| $N_c$ | Number of chunks per sample |
| $N_e$ | Number of encoder layers (6) |
| $N_d$ | Number of decoder layers (6) |
| $\varepsilon$ | Label smoothing factor (0.15) |

---

## 2. Embedding Layer

The embedding function maps discrete token IDs to dense vectors:

$$\mathbf{E} \in \mathbb{R}^{V \times d}$$

For a token sequence $\mathbf{x} = [x_1, x_2, \ldots, x_L]$ where $x_i \in \{0, 1, \ldots, V-1\}$:

$$\text{Embed}(\mathbf{x})_i = \mathbf{E}[x_i] \in \mathbb{R}^d$$

The output is:

$$\mathbf{X} = \text{Embed}(\mathbf{x}) \in \mathbb{R}^{L \times d}$$

The model uses **separate source and target embedding matrices**:
- $\mathbf{E}_{\text{src}} \in \mathbb{R}^{V \times d}$ — for the encoder
- $\mathbf{E}_{\text{tgt}} \in \mathbb{R}^{V \times d}$ — for the decoder (**weight-tied** with output projection)

---

## 3. Sinusoidal Positional Encoding

Since the model has no inherent notion of position, sinusoidal positional encodings are added:

$$\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

where $pos$ is the position index and $i$ is the dimension index.

The final embedded representation with position:

$$\hat{\mathbf{X}} = \text{Dropout}\left(\mathbf{X} + \text{PE}[:L]\right)$$

with dropout rate $p = 0.15$.

### Properties

- Positions up to 8192 are supported (pre-computed buffer).
- Each chunk gets positional encoding relative to the chunk (position 0–255), not the global document position.

---

## 4. Chunking Formulation

Given a source sequence of length $L_s$, it is divided into $N_c$ overlapping chunks:

$$N_c = \left\lceil \frac{L_s - C}{S} \right\rceil + 1$$

For default parameters ($C = 256$, $S = 192$, $L_s = 4096$):

$$N_c = \left\lceil \frac{4096 - 256}{192} \right\rceil + 1 = \left\lceil 20 \right\rceil + 1 = 21$$

Chunk $j$ spans positions:

$$\text{Chunk}_j = \mathbf{x}[j \cdot S : j \cdot S + C] \quad \text{for } j = 0, 1, \ldots, N_c - 1$$

The **overlap** between consecutive chunks is:

$$O = C - S = 256 - 192 = 64 \text{ tokens}$$

If the last chunk has fewer than $C$ tokens, it is **right-padded** with pad token (ID=3).

---

## 5. State Space Model (SSM) — Mamba Core

Mamba is based on **Structured State Space Sequences (S4)**. The continuous-time SSM is:

$$\dot{\mathbf{h}}(t) = \mathbf{A} \cdot \mathbf{h}(t) + \mathbf{B} \cdot x(t)$$

$$y(t) = \mathbf{C} \cdot \mathbf{h}(t) + \mathbf{D} \cdot x(t)$$

where:
- $\mathbf{h}(t) \in \mathbb{R}^{N}$ is the hidden state ($N = d_{\text{state}} = 16$)
- $\mathbf{A} \in \mathbb{R}^{N \times N}$ is the state transition matrix
- $\mathbf{B} \in \mathbb{R}^{N \times 1}$ is the input projection
- $\mathbf{C} \in \mathbb{R}^{1 \times N}$ is the output projection
- $\mathbf{D} \in \mathbb{R}$ is the feedthrough (skip connection)

### Discretization (Zero-Order Hold)

The continuous system is discretized with step size $\Delta$:

$$\bar{\mathbf{A}} = \exp(\Delta \cdot \mathbf{A})$$

$$\bar{\mathbf{B}} = (\Delta \cdot \mathbf{A})^{-1} (\exp(\Delta \cdot \mathbf{A}) - \mathbf{I}) \cdot \Delta \cdot \mathbf{B}$$

The discrete recurrence becomes:

$$\mathbf{h}_t = \bar{\mathbf{A}} \cdot \mathbf{h}_{t-1} + \bar{\mathbf{B}} \cdot x_t$$

$$y_t = \mathbf{C} \cdot \mathbf{h}_t + \mathbf{D} \cdot x_t$$

### Mamba's Selective Mechanism

Unlike standard S4 where $\mathbf{B}$, $\mathbf{C}$, and $\Delta$ are time-invariant, **Mamba makes them input-dependent**:

$$\mathbf{B}_t = \text{Linear}_B(\mathbf{x}_t) \in \mathbb{R}^{N}$$

$$\mathbf{C}_t = \text{Linear}_C(\mathbf{x}_t) \in \mathbb{R}^{N}$$

$$\Delta_t = \text{softplus}(\text{Linear}_\Delta(\mathbf{x}_t)) \in \mathbb{R}^{+}$$

This **selectivity** allows the model to dynamically decide what information to remember or forget at each time step — analogous to the gating mechanism in LSTMs but more efficient.

### Mamba Block with Expansion

The full Mamba block expands the hidden dimension by factor $E = 2$:

$$\mathbf{z} = \text{Linear}_{\text{in}}(\mathbf{x}) \in \mathbb{R}^{L \times Ed}$$

$$\mathbf{z}_{\text{conv}} = \text{DepthwiseConv1D}(\mathbf{z}, \text{kernel}=4)$$

$$\mathbf{z}_{\text{act}} = \text{SiLU}(\mathbf{z}_{\text{conv}})$$

$$\mathbf{y}_{\text{ssm}} = \text{SSM}(\mathbf{z}_{\text{act}}) \quad \text{(selective scan)}$$

$$\mathbf{y} = \text{Linear}_{\text{out}}(\mathbf{y}_{\text{ssm}} \odot \text{SiLU}(\mathbf{z}_{\text{gate}})) \in \mathbb{R}^{L \times d}$$

### MambaBlock with Residual (as implemented)

$$\mathbf{x}_{\text{out}} = \text{Dropout}\left(\text{Mamba}\left(\text{LayerNorm}(\mathbf{x})\right)\right) + \mathbf{x}$$

**Parameters per Mamba block** (with $d = 512$, $N = 16$, $E = 2$):
- $\text{Linear}_{\text{in}}: d \to 2 \cdot Ed$ — $2 \times 512 \times 1024$ params
- Conv1D: $Ed \times d_{\text{conv}}$ — $1024 \times 4$ params
- SSM projections ($\mathbf{B}$, $\mathbf{C}$, $\Delta$): $\sim Ed \times (2N + 1)$ params
- $\text{Linear}_{\text{out}}: Ed \to d$ — $1024 \times 512$ params
- LayerNorm: $2 \times d$ params

---

## 6. Bidirectional Mamba Block (V2)

The V2 bidirectional block processes input in both directions:

**Forward SSM:**
$$\mathbf{h}^{\rightarrow}_t = \bar{\mathbf{A}}^{\rightarrow} \cdot \mathbf{h}^{\rightarrow}_{t-1} + \bar{\mathbf{B}}^{\rightarrow}_t \cdot x_t$$

$$y^{\rightarrow}_t = \mathbf{C}^{\rightarrow}_t \cdot \mathbf{h}^{\rightarrow}_t$$

**Backward SSM** (processes reversed input):
$$\mathbf{h}^{\leftarrow}_t = \bar{\mathbf{A}}^{\leftarrow} \cdot \mathbf{h}^{\leftarrow}_{t+1} + \bar{\mathbf{B}}^{\leftarrow}_t \cdot x_t$$

$$y^{\leftarrow}_t = \mathbf{C}^{\leftarrow}_t \cdot \mathbf{h}^{\leftarrow}_t$$

**Gated Fusion:**
$$\mathbf{g}_t = \sigma\left(\mathbf{W}_g \left[\mathbf{y}^{\rightarrow}_t \| \mathbf{y}^{\leftarrow}_t\right]\right) \in \mathbb{R}^d$$

$$\mathbf{y}_t = \mathbf{g}_t \odot \mathbf{y}^{\rightarrow}_t + (1 - \mathbf{g}_t) \odot \mathbf{y}^{\leftarrow}_t$$

where:
- $\mathbf{W}_g \in \mathbb{R}^{d \times 2d}$ is the gating weight (no bias)
- $\sigma$ is sigmoid activation
- $\|$ denotes concatenation
- $\odot$ is element-wise multiplication

**Full block with residual:**

$$\mathbf{x}_{\text{out}} = \text{Dropout}(\mathbf{y}) + \mathbf{x}$$

---

## 7. RMSNorm (V2)

RMSNorm (Root Mean Square Normalization) simplifies LayerNorm by removing the mean-centering step:

### Standard LayerNorm

$$\text{LayerNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu = \frac{1}{d}\sum_{i=1}^d x_i$, $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$

### RMSNorm

$$\text{RMSNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}$$

where:

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}$$

- **No mean subtraction** ($\mu$ not computed)
- **No additive bias** ($\beta$ not used)
- Parameters: only $\gamma \in \mathbb{R}^d$ (learnable scale)
- $\epsilon = 10^{-8}$ for numerical stability

### Advantages

- ~15% faster than LayerNorm (one fewer pass over the data)
- Mathematically simpler while achieving comparable or better training stability
- Used in LLaMA, Mamba-2, and other modern architectures

---

## 8. SwiGLU Feed-Forward Network (V2)

### Standard FFN (GELU)

$$\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$

where $\mathbf{W}_1 \in \mathbb{R}^{d \times d_{ff}}$, $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d}$

### SwiGLU FFN (V2)

$$\text{SwiGLU}(\mathbf{x}) = \left(\text{SiLU}(\mathbf{x} \mathbf{W}_1) \odot (\mathbf{x} \mathbf{W}_3)\right) \mathbf{W}_2$$

where:
- $\mathbf{W}_1 \in \mathbb{R}^{d \times d_{ff}}$ — activation projection (no bias)
- $\mathbf{W}_3 \in \mathbb{R}^{d \times d_{ff}}$ — gate projection (no bias)
- $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d}$ — output projection (no bias)
- $\text{SiLU}(x) = x \cdot \sigma(x)$ — Sigmoid Linear Unit

### SiLU Activation

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

This is smooth, non-monotonic, and allows small negative values through — unlike ReLU which hard-zeros negative inputs.

### Why SwiGLU > Standard FFN

The gating mechanism ($\mathbf{W}_3$) provides a learned multiplicative interaction:
- Each dimension of the intermediate representation has an independently learned "gate"
- The network can selectively amplify or suppress features
- Empirically: +1–2% performance improvement in language modeling benchmarks

---

## 9. Memory Token Compression

Each 256-token chunk representation is compressed to $K = 32$ memory tokens using learned cross-attention pooling.

### Learned Query Vectors

$$\mathbf{Q}_{\text{mem}} \in \mathbb{R}^{K \times d}$$

initialized as $\mathcal{N}(0, 0.02^2)$ and learned during training.

### Cross-Attention Pooling

$$\mathbf{M} = \text{Attn}(\mathbf{Q}_{\text{mem}}, \mathbf{Z}_{\text{chunk}}, \mathbf{Z}_{\text{chunk}})$$

where $\mathbf{Z}_{\text{chunk}} \in \mathbb{R}^{C \times d}$ is the chunk's encoded representation.

Expanded explicitly:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{W}_Q (\mathbf{K} \mathbf{W}_K)^\top}{\sqrt{d_h}}\right) \mathbf{V} \mathbf{W}_V$$

For multi-head attention with $H = 8$ heads and $d_h = 64$:

$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_Q^{(i)}, \mathbf{K}\mathbf{W}_K^{(i)}, \mathbf{V}\mathbf{W}_V^{(i)})$$

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \mathbf{W}_O$$

### With Residual and Normalization

$$\mathbf{M}_{\text{out}} = \text{Norm}\left(\text{Dropout}(\mathbf{M}_{\text{attn}}) + \mathbf{Q}_{\text{mem}}\right)$$

**Input:** $\mathbf{Z}_{\text{chunk}} \in \mathbb{R}^{B \times 256 \times 512}$  
**Output:** $\mathbf{M} \in \mathbb{R}^{B \times 32 \times 512}$  
**Compression ratio:** $256 / 32 = 8\times$

### Concatenation Across All Chunks

$$\mathbf{M}_{\text{total}} = \text{Concat}(\mathbf{M}_1, \mathbf{M}_2, \ldots, \mathbf{M}_{N_c}) \in \mathbb{R}^{B \times (N_c \cdot K) \times d}$$

For default parameters: $\mathbf{M}_{\text{total}} \in \mathbb{R}^{B \times 672 \times 512}$ (21 chunks × 32 tokens)

---

## 10. Multi-Head Attention

Used in memory compression, cross-chunk attention, and all decoder layers.

### Scaled Dot-Product Attention

For queries $\mathbf{Q} \in \mathbb{R}^{L_q \times d_h}$, keys $\mathbf{K} \in \mathbb{R}^{L_k \times d_h}$, values $\mathbf{V} \in \mathbb{R}^{L_k \times d_h}$:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}} + \mathbf{M}_{\text{mask}}\right)\mathbf{V}$$

where $\mathbf{M}_{\text{mask}}$ is:
- **Zero** for positions that should be attended to
- $-\infty$ for positions that should be ignored (padding or future tokens)

### Multi-Head Projection

$$\mathbf{Q}^{(i)} = \mathbf{X} \mathbf{W}_Q^{(i)}, \quad \mathbf{K}^{(i)} = \mathbf{X} \mathbf{W}_K^{(i)}, \quad \mathbf{V}^{(i)} = \mathbf{X} \mathbf{W}_V^{(i)}$$

where $\mathbf{W}_Q^{(i)}, \mathbf{W}_K^{(i)}, \mathbf{W}_V^{(i)} \in \mathbb{R}^{d \times d_h}$

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = [\text{head}_1; \ldots; \text{head}_H] \mathbf{W}_O$$

where $\mathbf{W}_O \in \mathbb{R}^{d \times d}$

**Per-layer parameters:** $4 \times d \times d = 4 \times 512 \times 512 = 1,048,576$

---

## 11. Cross-Chunk Memory Attention (V2)

After all chunks are compressed, 2 layers of self-attention let memory tokens communicate across chunks:

### Layer $l$ ($l = 1, 2$):

**Self-attention sub-layer:**

$$\mathbf{M}' = \text{Dropout}\left(\text{MultiHead}(\text{Norm}_1(\mathbf{M}), \text{Norm}_1(\mathbf{M}), \text{Norm}_1(\mathbf{M}))\right) + \mathbf{M}$$

**Feed-forward sub-layer (SwiGLU):**

$$\mathbf{M}_{\text{out}} = \text{SwiGLU}(\text{Norm}_2(\mathbf{M}')) + \mathbf{M}'$$

**Input/Output:** $\mathbb{R}^{B \times 672 \times 512} \to \mathbb{R}^{B \times 672 \times 512}$

This is **critical** because without this, memory tokens from different chunks would never interact. A medication mentioned in the first chunk couldn't be linked to a side effect in the last chunk.

---

## 12. Transformer Decoder Layer

Each of the 6 decoder layers has three sub-layers:

### Sub-layer 1: Causal Self-Attention

$$\mathbf{h}' = \text{Dropout}\left(\text{MultiHead}_{\text{self}}(\text{Norm}_1(\mathbf{h}), \text{Norm}_1(\mathbf{h}), \text{Norm}_1(\mathbf{h}); \mathbf{M}_{\text{causal}})\right) + \mathbf{h}$$

The causal mask $\mathbf{M}_{\text{causal}} \in \mathbb{R}^{L_t \times L_t}$:

$$\mathbf{M}_{\text{causal}}[i, j] = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

### Sub-layer 2: Cross-Attention to Memory

$$\mathbf{h}'' = \text{Dropout}\left(\text{MultiHead}_{\text{cross}}(Q{=}\text{Norm}_2(\mathbf{h}'), K{=}\mathbf{M}_{\text{enc}}, V{=}\mathbf{M}_{\text{enc}})\right) + \mathbf{h}'$$

The decoder attends to $\mathbf{M}_{\text{enc}} \in \mathbb{R}^{B \times 672 \times 512}$ — the full encoder memory.

### Sub-layer 3: Feed-Forward (SwiGLU)

$$\mathbf{h}_{\text{out}} = \text{SwiGLU}(\text{Norm}_3(\mathbf{h}'')) + \mathbf{h}''$$

**Pre-norm architecture**: Normalization is applied *before* each sub-layer (not after), following the convention from LLaMA and GPT-3.

---

## 13. Gated Cross-Attention (V2)

The V2 gated cross-attention modifies sub-layer 2 of the decoder with a learned scalar gate:

$$\alpha = \sigma(g) \quad \text{where } g \in \mathbb{R} \text{ is a learnable parameter}$$

$$\mathbf{h}'' = \alpha \cdot \text{Dropout}\left(\text{MultiHead}_{\text{cross}}(\ldots)\right) + \mathbf{h}'$$

- $g$ is initialized to 0.0, so $\alpha = \sigma(0) = 0.5$ initially (equal weight to residual and cross-attention).
- During training, each decoder layer learns its own $\alpha$, allowing the model to decide how much source information to incorporate at each layer.
- **Inspired by RetNet's retention mechanism**: lower layers may need more source info (facts), upper layers may need less (fluency).

---

## 14. Output Projection & Weight Tying

The decoder output is projected to vocabulary logits:

$$\mathbf{logits} = \mathbf{h}_{\text{out}} \cdot \mathbf{E}_{\text{tgt}}^\top \in \mathbb{R}^{B \times L_t \times V}$$

**Weight tying**: The output projection matrix $\mathbf{W}_{\text{out}}$ is set equal to the target embedding matrix $\mathbf{E}_{\text{tgt}}$:

$$\mathbf{W}_{\text{out}} = \mathbf{E}_{\text{tgt}} \in \mathbb{R}^{V \times d}$$

This:
- Reduces parameters by $V \times d = 16000 \times 512 = 8.2M$
- Creates a shared semantic space between input and output representations
- Improves generalization (proven in practice)

---

## 15. Loss Function — Label-Smoothed Cross-Entropy

### Probability Distribution

$$P(y_t = k \mid y_{<t}, \mathbf{x}) = \frac{\exp(\text{logits}_{t,k})}{\sum_{k'=1}^V \exp(\text{logits}_{t,k'})}$$

### Standard Cross-Entropy

$$\mathcal{L}_{\text{CE}} = -\frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \log P(y_t = y_t^* \mid y_{<t}, \mathbf{x})$$

where $\mathcal{T}$ is the set of non-padding target positions and $y_t^*$ is the ground-truth token.

### Label Smoothing ($\varepsilon = 0.15$)

The hard one-hot target $\mathbf{q}^{\text{hard}} \in \{0, 1\}^V$ is replaced with a soft distribution:

$$q(k) = \begin{cases}
1 - \varepsilon & \text{if } k = y_t^* \\
\frac{\varepsilon}{V - 1} & \text{otherwise}
\end{cases}$$

The smoothed loss:

$$\mathcal{L}_{\text{smooth}} = -\frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \sum_{k=1}^V q(k) \cdot \log P(y_t = k)$$

This can be decomposed:

$$\mathcal{L}_{\text{smooth}} = (1 - \varepsilon) \cdot \mathcal{L}_{\text{CE}} + \varepsilon \cdot \mathcal{L}_{\text{uniform}}$$

where $\mathcal{L}_{\text{uniform}} = -\frac{1}{V} \sum_k \log P(k)$ is the cross-entropy with uniform distribution (penalizes overconfidence).

### Padding Exclusion

$$\mathcal{L} = \frac{\sum_{t=1}^{L_t} \ell_t \cdot \mathbb{1}[y_t^* \neq \text{pad}]}{\sum_{t=1}^{L_t} \mathbb{1}[y_t^* \neq \text{pad}]}$$

---

## 16. AdamW Optimizer

### Adam Update Rule

For parameter $\theta$, gradient $g_t = \nabla_\theta \mathcal{L}$ at step $t$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment / mean)}$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment / variance)}$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(bias correction)}$$

### AdamW (Decoupled Weight Decay)

$$\theta_{t+1} = \theta_t - \eta_t \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

where:
- $\eta_t$ — learning rate at step $t$ (from scheduler)
- $\lambda = 0.03$ — weight decay
- $\beta_1 = 0.9$, $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$

### Why AdamW vs. Adam?

Standard Adam applies weight decay to the gradient, which interacts with the adaptive learning rate. AdamW **decouples** weight decay — it shrinks weights directly, independent of the gradient. This is more principled and leads to better generalization.

---

## 17. Cosine Schedule with Linear Warmup

### Warmup Phase ($t < t_w = 2000$)

$$\eta_t = \eta_{\max} \cdot \frac{t}{t_w}$$

### Cosine Decay Phase ($t \geq t_w$)

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \cdot \frac{t - t_w}{T - t_w}\right)\right)$$

where:
- $\eta_{\max} = 5 \times 10^{-5}$
- $\eta_{\min} = 0.1 \times \eta_{\max} = 5 \times 10^{-6}$ (min_lr_ratio = 0.1)
- $T = 50000$ (total training steps)
- $t_w = 2000$ (warmup steps)

---

## 18. Gradient Clipping

Before each optimizer step, all parameter gradients are clipped to have a maximum L2 norm of $\gamma = 1.0$:

$$\hat{\mathbf{g}} = \begin{cases}
\mathbf{g} & \text{if } \|\mathbf{g}\|_2 \leq \gamma \\
\gamma \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|_2} & \text{if } \|\mathbf{g}\|_2 > \gamma
\end{cases}$$

where $\|\mathbf{g}\|_2 = \sqrt{\sum_i g_i^2}$ is computed across **all** model parameters jointly.

---

## 19. Autoregressive Generation

At inference, tokens are generated one at a time:

$$y_t = \arg\max_{k} P(y_t = k \mid y_1, \ldots, y_{t-1}, \mathbf{x}) \quad \text{(greedy)}$$

### With Temperature and Top-K Sampling

$$P_{\text{temp}}(k) = \frac{\exp(\text{logits}_k / \tau)}{\sum_{k'} \exp(\text{logits}_{k'} / \tau)}$$

For top-$k$: set $P(k) = 0$ for all $k$ not in the top-$k$ largest logits, then renormalize.

### Repetition Penalty

For tokens already generated $\mathcal{G} = \{y_1, \ldots, y_{t-1}\}$:

$$\text{logits}'_k = \begin{cases}
\text{logits}_k / \alpha & \text{if } k \in \mathcal{G} \text{ and } \text{logits}_k > 0 \\
\text{logits}_k \times \alpha & \text{if } k \in \mathcal{G} \text{ and } \text{logits}_k \leq 0 \\
\text{logits}_k & \text{otherwise}
\end{cases}$$

where $\alpha > 1$ is the repetition penalty factor.

### No-Repeat N-gram Blocking

For $n = 3$: collect all existing trigrams in $\{y_1, \ldots, y_{t-1}\}$. If the last 2 tokens match the first 2 tokens of any existing trigram, set the completing token's logit to $-\infty$.

---

## 20. Beam Search

Beam search maintains $b$ candidate sequences (beams) and expands each at every step:

### Step $t$:

For each beam $j$ with log-probability $\text{LP}_j$ and sequence $\mathbf{y}_j$:

1. Compute $\text{logits} = \text{Model}(\mathbf{y}_j, \text{memory})$
2. Get top-$b$ tokens: $\{(k_1, \log P(k_1)), \ldots, (k_b, \log P(k_b))\}$
3. Create $b$ new candidates: $\text{LP}_j + \log P(k_i)$

### Selection with Length Penalty

$$\text{Score}(\mathbf{y}) = \frac{\log P(\mathbf{y})}{|\mathbf{y}|^{\alpha}}$$

where $\alpha = 0.6$ (length penalty). This discourages both too-short and too-long outputs.

### Minimum Length Constraint

EOS token logit is set to $-\infty$ for the first $\text{min\_len} = 20$ steps, ensuring minimum summary length.

---

## 21. Parameter Count Derivation

### Embeddings

| Component | Parameters |
|---|---|
| Source embedding ($V \times d$) | $16000 \times 512 = 8,192,000$ |
| Target embedding ($V \times d$, shared with output) | $16000 \times 512 = 8,192,000$ |
| **Total embedding** | **16,384,000** |

### Mamba Encoder (6 × BidirectionalMambaBlock)

Each BiMamba block has forward + backward Mamba + gate + normalization:

| Component | Parameters |
|---|---|
| Mamba forward ($d{=}512, N{=}16, E{=}2$) | $\approx 2,106,880$ |
| Mamba backward (identical) | $\approx 2,106,880$ |
| Gate ($\mathbf{W}_g: 2d \to d$) | $2 \times 512 \times 512 = 524,288$ |
| RMSNorm ($d$) | $512$ |
| **Per layer** | **$\approx 4,738,560$** |
| **6 layers + final norm** | **$\approx 28,431,872$** |

### Memory Compressor

| Component | Parameters |
|---|---|
| Learned queries ($K \times d$) | $32 \times 512 = 16,384$ |
| Cross-attention ($4 \times d \times d$) | $1,048,576$ |
| RMSNorm ($d$) | $512$ |
| **Total** | **$\approx 1,065,472$** |

### Cross-Chunk Attention (2 layers)

| Component | Params per layer |
|---|---|
| Self-attention ($4 \times d \times d$) | $1,048,576$ |
| SwiGLU FFN ($3 \times d \times d_{ff}$) | $3 \times 512 \times 2048 = 3,145,728$ |
| 2 × RMSNorm ($d$) | $1,024$ |
| **Per layer** | **$\approx 4,195,328$** |
| **2 layers** | **$\approx 8,390,656$** |

### Transformer Decoder (6 layers)

| Component | Params per layer |
|---|---|
| Self-attention ($4 \times d \times d$) | $1,048,576$ |
| Cross-attention ($4 \times d \times d$) | $1,048,576$ |
| SwiGLU FFN ($3 \times d \times d_{ff}$) | $3,145,728$ |
| 3 × RMSNorm ($d$) | $1,536$ |
| Gate scalar | $1$ |
| **Per layer** | **$\approx 5,244,417$** |
| **6 layers + final norm** | **$\approx 31,467,014$** |

### Output Projection

Weight-tied with target embedding: **0 additional parameters**.

### Total

$$\text{Total} = 16,384,000 + 28,431,872 + 1,065,472 + 8,390,656 + 31,467,014 \approx \mathbf{79,410,694}$$

---

## Summary of Key Mathematical Properties

| Property | Description |
|---|---|
| **O(L) encoder complexity** | Mamba SSM is linear in sequence length (vs. O(L²) for Transformers) |
| **O(T² · M) decoder complexity** | Decoder is O(T²) for self-attention + O(T · M) for cross-attention |
| **Compression ratio** | 8x per chunk (256 → 32 tokens) |
| **Total compression** | ~6x overall (4096 tokens → 672 memory tokens) |
| **Selective state space** | Input-dependent gating in Mamba (content-aware memory) |
| **Bidirectional encoding** | Full context within each chunk (both directions) |
| **Global memory** | Cross-chunk attention enables full-document reasoning |
