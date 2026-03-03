"""
Mamba-Transformer Hybrid Model V2
=================================

Improved from-scratch implementation for clinical note summarization.
Combines ideas from Mamba-2, RetNet, and modern LLMs (LLaMA, PaLM).

V2 Improvements:
- RMSNorm (from Mamba-2/LLaMA) - faster, more stable normalization
- SwiGLU feed-forward (from LLaMA/PaLM) - gated activation for better feature mixing
- Bidirectional Mamba encoder (from Mamba-2) - forward+backward SSM with gated fusion
- Cross-chunk memory attention (RetNet-inspired) - lets memory tokens interact across chunks
- Gated cross-attention in decoder (RetNet retention) - learned gates for source-target alignment

All V2 features are backward-compatible via config flags (default=False).

Architecture:
1. Token Embeddings (SentencePiece vocab)
2. Chunk long input into fixed length (chunk_size=256, stride=192)
3. Mamba encoder stack per chunk (optionally bidirectional)
4. Compress each chunk to K memory tokens via learned queries + attention pooling
5. Cross-chunk memory attention (optional) - memory tokens attend across chunks
6. Concatenate memory tokens across all chunks
7. Transformer decoder with cross-attention to memory tokens (optionally gated)
8. Output projection to vocab
"""

import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("WARNING: mamba-ssm not installed. Will use fallback.")


@dataclass
class MambaTransformerConfig:
    """Model configuration"""
    vocab_size: int = 32000
    d_model: int = 512
    n_mamba_layers: int = 6
    n_decoder_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_src_len: int = 8192
    max_tgt_len: int = 512
    chunk_size: int = 256
    stride: int = 192
    n_memory_tokens: int = 8  # K memory tokens per chunk
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    pad_id: int = 3
    bos_id: int = 1
    eos_id: int = 2
    label_smoothing: float = 0.1

    # V2 improvements (inspired by Mamba-2, RetNet, LLaMA/PaLM)
    use_rmsnorm: bool = False             # RMSNorm instead of LayerNorm
    use_swiglu: bool = False              # SwiGLU feed-forward instead of GELU
    use_bidirectional_mamba: bool = False  # Bidirectional Mamba encoding
    use_cross_chunk_attn: bool = False    # Cross-chunk memory attention
    use_gated_cross_attn: bool = False    # Gated cross-attention in decoder
    n_cross_chunk_layers: int = 2         # Number of cross-chunk attention layers

    # V3 additions - source bypass + copy mechanism
    source_bypass_stride: int = 0         # >0 enables source bypass (stride for downsampling)
    use_copy_mechanism: bool = False       # Pointer-generator for copying source tokens

    @classmethod
    def from_dict(cls, d: dict) -> 'MambaTransformerConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================================
# V2 Components: RMSNorm, SwiGLU, factory helpers
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Mamba-2 / LLaMA).
    Faster and more stable than LayerNorm — removes mean-centering,
    keeps only scale normalization."""
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU Feed-Forward Network (LLaMA / PaLM).
    Uses gated linear units with SiLU activation — proven to outperform
    standard GELU feed-forward in language modeling benchmarks."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # gate projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def create_norm(d_model: int, use_rmsnorm: bool = False) -> nn.Module:
    """Factory for normalization layers"""
    return RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)


def create_ffn(d_model: int, d_ff: int, dropout: float = 0.1, use_swiglu: bool = False) -> nn.Module:
    """Factory for feed-forward networks"""
    if use_swiglu:
        return SwiGLUFeedForward(d_model, d_ff, dropout)
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model),
        nn.Dropout(dropout),
    )


class MambaFallback(nn.Module):
    """Fallback module when mamba-ssm is not available - uses GRU"""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=False)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.norm(out)


class MambaBlock(nn.Module):
    """Single Mamba block with residual connection"""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba = MambaFallback(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x) + residual
        return x


class BidirectionalMambaBlock(nn.Module):
    """Bidirectional Mamba block (inspired by Mamba-2).
    Processes input both forward and backward through separate Mamba SSMs,
    then uses a learned gate to fuse both directions. This provides
    full-context chunk encoding — critical for clinical notes where
    diagnoses may reference earlier or later findings."""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1, use_rmsnorm: bool = False):
        super().__init__()
        self.norm = create_norm(d_model, use_rmsnorm)
        if MAMBA_AVAILABLE:
            self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mamba_fwd = MambaFallback(d_model, d_state, d_conv, expand)
            self.mamba_bwd = MambaFallback(d_model, d_state, d_conv, expand)
        self.gate = nn.Linear(d_model * 2, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        fwd_out = self.mamba_fwd(x)
        bwd_out = self.mamba_bwd(x.flip(1)).flip(1)
        gate_val = torch.sigmoid(self.gate(torch.cat([fwd_out, bwd_out], dim=-1)))
        x = gate_val * fwd_out + (1 - gate_val) * bwd_out
        return self.dropout(x) + residual


class MambaEncoder(nn.Module):
    """Stack of Mamba blocks for encoding.
    Supports both unidirectional (V1) and bidirectional (V2) Mamba blocks."""
    def __init__(self, config: MambaTransformerConfig):
        super().__init__()
        if config.use_bidirectional_mamba:
            self.layers = nn.ModuleList([
                BidirectionalMambaBlock(
                    d_model=config.d_model,
                    d_state=config.mamba_d_state,
                    d_conv=config.mamba_d_conv,
                    expand=config.mamba_expand,
                    dropout=config.dropout,
                    use_rmsnorm=config.use_rmsnorm,
                )
                for _ in range(config.n_mamba_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                MambaBlock(
                    d_model=config.d_model,
                    d_state=config.mamba_d_state,
                    d_conv=config.mamba_d_conv,
                    expand=config.mamba_expand,
                    dropout=config.dropout,
                )
                for _ in range(config.n_mamba_layers)
            ])
        self.final_norm = create_norm(config.d_model, config.use_rmsnorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class MemoryTokenCompressor(nn.Module):
    """
    Compress chunk representations to K memory tokens using learned queries and attention pooling.
    """
    def __init__(self, d_model: int, n_memory_tokens: int = 8, n_heads: int = 8,
                 dropout: float = 0.1, use_rmsnorm: bool = False):
        super().__init__()
        self.n_memory_tokens = n_memory_tokens
        self.d_model = d_model
        
        # Learned query vectors for memory tokens
        self.memory_queries = nn.Parameter(torch.randn(n_memory_tokens, d_model) * 0.02)
        
        # Cross-attention: queries attend to chunk representations
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = create_norm(d_model, use_rmsnorm)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, chunk_repr: torch.Tensor, chunk_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            chunk_repr: [batch, chunk_len, d_model]
            chunk_mask: [batch, chunk_len] - True for padded positions
        Returns:
            memory_tokens: [batch, n_memory_tokens, d_model]
        """
        batch_size = chunk_repr.size(0)
        
        # Expand queries for batch
        queries = self.memory_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Handle fully-masked chunks (all padding) - return just the queries
        if chunk_mask is not None:
            # Check if any sample has all tokens masked
            all_masked = chunk_mask.all(dim=1)  # [batch]
            if all_masked.any():
                # For fully-masked chunks, just return normalized queries (no attention needed)
                # This prevents NaN from attention over empty sequences
                return self.norm(queries)
        
        # Cross-attention: memory queries attend to chunk representations
        attn_out, _ = self.cross_attn(
            query=queries,
            key=chunk_repr,
            value=chunk_repr,
            key_padding_mask=chunk_mask,
        )
        
        memory_tokens = self.norm(self.dropout(attn_out) + queries)
        return memory_tokens


class CrossChunkAttention(nn.Module):
    """Cross-chunk memory interaction layer (inspired by RetNet's retention mixing).
    After each chunk is compressed to K memory tokens, this layer allows
    memory tokens from *different chunks* to attend to each other.
    This is critical for long clinical documents where information from
    early sections (e.g., admission diagnosis) must connect with later
    sections (e.g., treatment outcomes)."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, use_rmsnorm: bool = False,
                 use_swiglu: bool = False):
        super().__init__()
        self.norm1 = create_norm(d_model, use_rmsnorm)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = create_norm(d_model, use_rmsnorm)
        self.ff = create_ffn(d_model, d_ff, dropout, use_swiglu)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """memory: [batch, total_memory_tokens, d_model]"""
        residual = memory
        memory = self.norm1(memory)
        memory, _ = self.attn(memory, memory, memory)
        memory = self.dropout(memory) + residual

        residual = memory
        memory = self.norm2(memory)
        memory = self.ff(memory) + residual
        return memory


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention and cross-attention.
    V2: supports SwiGLU FFN, RMSNorm, and gated cross-attention."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 use_rmsnorm: bool = False, use_swiglu: bool = False,
                 use_gated_cross_attn: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ff = create_ffn(d_model, d_ff, dropout, use_swiglu)

        self.norm1 = create_norm(d_model, use_rmsnorm)
        self.norm2 = create_norm(d_model, use_rmsnorm)
        self.norm3 = create_norm(d_model, use_rmsnorm)
        self.dropout = nn.Dropout(dropout)

        # Gated cross-attention (RetNet-inspired): learned gate controls
        # ADDITIONAL boost of source information on top of full cross-attention.
        # Formula: x = (1 + sigmoid(gate)) * cross_out + residual
        # This ensures full cross-attention always flows through (min 1.0x),
        # with the gate providing up to 2.0x boost. Prevents source suppression.
        self.use_gated_cross_attn = use_gated_cross_attn
        if use_gated_cross_attn:
            self.cross_gate = nn.Parameter(torch.tensor(1.0))  # sigmoid(1)≈0.73 → 1.73x initial

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with causal mask
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.dropout(x) + residual

        # Cross-attention to memory tokens (optionally gated)
        # Additive gating: full cross-attention ALWAYS flows through.
        # Gate provides additional boost: (1 + sigmoid(gate)) * cross_out
        residual = x
        x = self.norm2(x)
        cross_out, _ = self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask)
        if self.use_gated_cross_attn:
            gate_val = 1.0 + torch.sigmoid(self.cross_gate)  # range [1.0, 2.0]
            x = gate_val * self.dropout(cross_out) + residual
        else:
            x = self.dropout(cross_out) + residual

        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.ff(x) + residual

        return x


class TransformerDecoder(nn.Module):
    """Stack of Transformer decoder layers (V2: supports RMSNorm, SwiGLU, gated cross-attn)"""
    def __init__(self, config: MambaTransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                use_rmsnorm=config.use_rmsnorm,
                use_swiglu=config.use_swiglu,
                use_gated_cross_attn=config.use_gated_cross_attn,
            )
            for _ in range(config.n_decoder_layers)
        ])
        self.final_norm = create_norm(config.d_model, config.use_rmsnorm)
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.final_norm(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CopyMechanism(nn.Module):
    """Pointer-generator network for copy mechanism.
    Combines vocab generation distribution with source copy distribution.
    Allows the decoder to directly copy tokens from the source document.
    Initialized with balanced p_gen (~0.5) to force active copy usage."""
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_linear = nn.Linear(d_model, 1)
        nn.init.constant_(self.gate_linear.bias, 3.0)  # sigmoid(3) = 0.95, mostly-generate start
        self.copy_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, decoder_out, source_repr, source_ids, vocab_logits, source_mask=None):
        """Returns log-probabilities [B, tgt_len, vocab_size]."""
        B, tgt_len, d = decoder_out.shape
        vocab_size = vocab_logits.size(-1)
        p_gen = torch.sigmoid(self.gate_linear(decoder_out))
        gen_probs = F.softmax(vocab_logits, dim=-1)
        copy_query = self.copy_proj(decoder_out)
        copy_scores = torch.bmm(copy_query, source_repr.transpose(1, 2)) / math.sqrt(d)
        if source_mask is not None:
            copy_scores = copy_scores.masked_fill(source_mask.unsqueeze(1), float('-inf'))
        copy_attn = F.softmax(copy_scores, dim=-1)
        copy_probs = torch.zeros(B, tgt_len, vocab_size, device=decoder_out.device)
        source_ids_expanded = source_ids.unsqueeze(1).expand(-1, tgt_len, -1)
        copy_probs.scatter_add_(2, source_ids_expanded, copy_attn)
        combined = p_gen * gen_probs + (1.0 - p_gen) * copy_probs
        return torch.log(combined + 1e-12)


class MambaTransformerModel(nn.Module):
    """
    Mamba-Transformer Hybrid Model V2 for Clinical Summarization

    Architecture:
    1. Embed input tokens
    2. Chunk input sequence
    3. Encode each chunk with Mamba (optionally bidirectional)
    4. Compress each chunk to K memory tokens
    5. Source bypass: concatenate downsampled encoded chunks with memory
    6. Cross-chunk memory attention (optional)
    7. Decode with Transformer decoder cross-attending to memory (optionally gated)
    8. Copy mechanism: pointer-generator for copying source tokens
    9. Project to vocabulary (combined with copy distribution)
    """
    def __init__(self, config: MambaTransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.src_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.tgt_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.pos_encoding = PositionalEncoding(config.d_model, max_len=max(config.max_src_len, config.max_tgt_len))
        
        # Encoder (Mamba)
        self.encoder = MambaEncoder(config)
        
        # Memory compression
        self.memory_compressor = MemoryTokenCompressor(
            d_model=config.d_model,
            n_memory_tokens=config.n_memory_tokens,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_rmsnorm=config.use_rmsnorm,
        )
        
        # Cross-chunk memory interaction (V2: RetNet-inspired)
        self.cross_chunk_attn = None
        if config.use_cross_chunk_attn:
            self.cross_chunk_attn = nn.ModuleList([
                CrossChunkAttention(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    use_rmsnorm=config.use_rmsnorm,
                    use_swiglu=config.use_swiglu,
                )
                for _ in range(config.n_cross_chunk_layers)
            ])
        
        # Decoder (Transformer)
        self.decoder = TransformerDecoder(config)
        
        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie embeddings
        self.output_proj.weight = self.tgt_embedding.weight

        # Copy mechanism (pointer-generator)
        self.copy_mechanism = None
        if getattr(config, 'use_copy_mechanism', False):
            self.copy_mechanism = CopyMechanism(config.d_model)

        # Internal state for source bypass + copy mechanism
        self._source_detail = None
        self._source_ids = None
        self._source_mask = None
        self._last_decoded = None

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _chunk_input(self, src: torch.Tensor, src_mask: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Chunk input sequence into overlapping windows.
        
        Args:
            src: [batch, seq_len]
            src_mask: [batch, seq_len] - True for padded positions
        
        Returns:
            chunks: list of [batch, chunk_size] tensors
            chunk_masks: list of [batch, chunk_size] tensors
        """
        batch_size, seq_len = src.shape
        chunk_size = self.config.chunk_size
        stride = self.config.stride
        
        chunks = []
        chunk_masks = []
        
        start = 0
        while start < seq_len:
            end = min(start + chunk_size, seq_len)
            chunk = src[:, start:end]
            mask = src_mask[:, start:end]
            
            # Pad if needed
            if chunk.size(1) < chunk_size:
                pad_len = chunk_size - chunk.size(1)
                chunk = F.pad(chunk, (0, pad_len), value=self.config.pad_id)
                mask = F.pad(mask, (0, pad_len), value=True)
            
            chunks.append(chunk)
            chunk_masks.append(mask)
            
            start += stride
            if end >= seq_len:
                break
        
        return chunks, chunk_masks
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode source with chunking, Mamba encoding, memory compression,
        and optional source bypass for richer decoder input.
        """
        chunks, chunk_masks = self._chunk_input(src, src_mask)
        bypass_stride = getattr(self.config, 'source_bypass_stride', 0)

        all_memory = []
        all_source_detail = []
        all_source_ids = []
        all_source_masks = []

        for chunk, mask in zip(chunks, chunk_masks):
            chunk_emb = self.src_embedding(chunk)
            chunk_emb = self.pos_encoding(chunk_emb)
            chunk_encoded = self.encoder(chunk_emb)
            memory_tokens = self.memory_compressor(chunk_encoded, mask)
            all_memory.append(memory_tokens)

            if bypass_stride > 0:
                all_source_detail.append(chunk_encoded[:, ::bypass_stride, :])
                all_source_ids.append(chunk[:, ::bypass_stride])
                all_source_masks.append(mask[:, ::bypass_stride])

        memory = torch.cat(all_memory, dim=1)

        if self.cross_chunk_attn is not None:
            for layer in self.cross_chunk_attn:
                memory = layer(memory)

        if all_source_detail:
            self._source_detail = torch.cat(all_source_detail, dim=1)
            self._source_ids = torch.cat(all_source_ids, dim=1)
            self._source_mask = torch.cat(all_source_masks, dim=1)
            memory = torch.cat([memory, self._source_detail], dim=1)
        else:
            self._source_detail = None
            self._source_ids = None
            self._source_mask = None

        return memory
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode target sequence with cross-attention to memory.
        
        Args:
            tgt: [batch, tgt_len]
            memory: [batch, memory_len, d_model]
            tgt_mask: [tgt_len, tgt_len] causal mask
            tgt_key_padding_mask: [batch, tgt_len] - True for padded positions
        
        Returns:
            logits: [batch, tgt_len, vocab_size]
        """
        # Embed target
        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Decode
        decoded = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        self._last_decoded = decoded

        # Project to vocab
        logits = self.output_proj(decoded)
        return logits
    
    @staticmethod
    def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            src: [batch, src_len] source token ids
            tgt: [batch, tgt_len] target token ids (teacher forcing)
            src_mask: [batch, src_len] - True for padded positions
            tgt_mask: [batch, tgt_len] - True for padded positions
        
        Returns:
            logits: [batch, tgt_len, vocab_size]
        """
        if src_mask is None:
            src_mask = (src == self.config.pad_id)
        if tgt_mask is None:
            tgt_mask = (tgt == self.config.pad_id)
        
        # Encode source
        memory = self.encode(src, src_mask)
        
        # Generate causal mask for decoder
        causal_mask = self.generate_causal_mask(tgt.size(1), tgt.device)
        
        # Decode
        logits = self.decode(tgt, memory, causal_mask, tgt_mask)

        # Apply copy mechanism or return raw logits
        if self.copy_mechanism is not None and self._source_detail is not None:
            return self.copy_mechanism(
                self._last_decoded, self._source_detail,
                self._source_ids, logits, self._source_mask,
            )
        return logits
    
    def _block_repeated_ngrams(self, next_logits: torch.Tensor, generated: torch.Tensor, n: int) -> torch.Tensor:
        """Block any token that would create a repeated n-gram."""
        if n <= 0 or generated.size(1) < n:
            return next_logits
        batch_size = generated.size(0)
        for batch_idx in range(batch_size):
            gen = generated[batch_idx].tolist()
            # Build set of existing n-grams (using last n-1 tokens + candidate)
            if len(gen) < n - 1:
                continue
            # Collect all existing n-grams
            ngrams = set()
            for i in range(len(gen) - n + 1):
                ngram = tuple(gen[i:i + n])
                ngrams.add(ngram)
            # The partial n-gram is the last (n-1) tokens
            partial = tuple(gen[-(n - 1):])
            # Block any token that would complete a repeated n-gram
            for ngram in ngrams:
                if ngram[:-1] == partial:
                    next_logits[batch_idx, ngram[-1]] = float('-inf')
        return next_logits

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        max_len: int = 512,
        min_len: int = 50,
        greedy: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.2,
        beam_size: int = 4,
        length_penalty: float = 0.8,
    ) -> torch.Tensor:
        """
        Autoregressive generation with beam search and repetition suppression.
        
        Args:
            src: [batch, src_len] source token ids
            src_mask: [batch, src_len] - True for padded positions
            max_len: maximum generation length (default 512)
            min_len: minimum generation length before EOS is allowed (default 50)
            greedy: if True, use greedy decoding (fastest, lower quality)
            temperature: sampling temperature (only used if greedy=False and beam_size<=1)
            top_k: top-k sampling (only used if greedy=False and beam_size<=1)
            no_repeat_ngram_size: block repeated n-grams of this size (0 to disable)
            repetition_penalty: penalize already-generated tokens (1.0 = no penalty)
            beam_size: number of beams for beam search (default 4, set 1 to disable)
            length_penalty: length normalization penalty for beam search (default 0.8)
        
        Returns:
            generated: [batch, gen_len] generated token ids
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        if src_mask is None:
            src_mask = (src == self.config.pad_id)
        
        # Encode source (shared across all beams)
        memory = self.encode(src, src_mask)
        
        # Use beam search for better generation quality
        if not greedy and beam_size > 1:
            return self._beam_search(
                memory, batch_size, device, max_len, min_len,
                beam_size, length_penalty, no_repeat_ngram_size, repetition_penalty,
            )
        
        # Fallback: greedy / sampling decode
        generated = torch.full((batch_size, 1), self.config.bos_id, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_len - 1):
            causal_mask = self.generate_causal_mask(generated.size(1), device)
            
            logits = self.decode(generated, memory, causal_mask)
            # Apply copy mechanism if available
            if self.copy_mechanism is not None and self._source_detail is not None:
                copy_lp = self.copy_mechanism(
                    self._last_decoded, self._source_detail,
                    self._source_ids, logits, self._source_mask,
                )
                next_logits = copy_lp[:, -1, :]
            else:
                next_logits = logits[:, -1, :]
            
            # Block EOS before min_len
            if step < min_len:
                next_logits[:, self.config.eos_id] = float('-inf')
            
            # Apply repetition penalty to already-generated tokens
            if repetition_penalty != 1.0:
                for batch_idx in range(batch_size):
                    prev_tokens = generated[batch_idx].unique()
                    for token_id in prev_tokens:
                        if next_logits[batch_idx, token_id] > 0:
                            next_logits[batch_idx, token_id] /= repetition_penalty
                        else:
                            next_logits[batch_idx, token_id] *= repetition_penalty
            
            # Block repeated n-grams
            if no_repeat_ngram_size > 0:
                next_logits = self._block_repeated_ngrams(next_logits, generated, no_repeat_ngram_size)
            
            if greedy:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            next_token = torch.where(finished.unsqueeze(1), torch.tensor(self.config.pad_id, device=device), next_token)
            generated = torch.cat([generated, next_token], dim=1)
            
            finished = finished | (next_token.squeeze(1) == self.config.eos_id)
            if finished.all():
                break
        
        return generated

    @torch.no_grad()
    def _beam_search(
        self,
        memory: torch.Tensor,
        batch_size: int,
        device: torch.device,
        max_len: int,
        min_len: int,
        beam_size: int,
        length_penalty: float,
        no_repeat_ngram_size: int,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """
        Beam search decoding for significantly better summarization quality.
        
        Uses length-normalized log-probability scoring with n-gram blocking
        and repetition penalty. Processes one sample at a time to manage memory.
        
        Returns:
            best sequences: [batch, gen_len]
        """
        all_best = []
        
        for b in range(batch_size):
            # Memory for this sample: [1, mem_len, d_model] -> [beam_size, mem_len, d_model]
            mem_b = memory[b:b+1].expand(beam_size, -1, -1)
            # Expand source state for copy mechanism
            src_detail_b = src_ids_b = src_mask_b = None
            if self._source_detail is not None:
                src_detail_b = self._source_detail[b:b+1].expand(beam_size, -1, -1)
                src_ids_b = self._source_ids[b:b+1].expand(beam_size, -1)
                src_mask_b = self._source_mask[b:b+1].expand(beam_size, -1)
            
            # Initialize beams: each beam is [1, seq_len] (always keep 2D)
            beam_seqs = torch.full((beam_size, 1), self.config.bos_id, device=device, dtype=torch.long)
            beam_scores = torch.zeros(beam_size, device=device)
            beam_scores[1:] = -1e9  # Only first beam active initially
            beam_finished = torch.zeros(beam_size, dtype=torch.bool, device=device)
            
            # Store completed hypotheses: (score, sequence)
            completed = []
            
            for step in range(max_len - 1):
                if beam_finished.all():
                    break
                
                cur_len = beam_seqs.size(1)
                causal_mask = self.generate_causal_mask(cur_len, device)
                
                logits = self.decode(beam_seqs, mem_b, causal_mask)
                # Apply copy mechanism if available
                if self.copy_mechanism is not None and src_detail_b is not None:
                    copy_lp = self.copy_mechanism(
                        self._last_decoded, src_detail_b, src_ids_b, logits, src_mask_b,
                    )
                    next_logits = copy_lp[:, -1, :]  # already log-probs
                else:
                    next_logits = logits[:, -1, :]  # [beam_size, vocab]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for bi in range(beam_size):
                        if beam_finished[bi]:
                            continue
                        prev_tokens = beam_seqs[bi].unique()
                        for token_id in prev_tokens:
                            if next_logits[bi, token_id] > 0:
                                next_logits[bi, token_id] /= repetition_penalty
                            else:
                                next_logits[bi, token_id] *= repetition_penalty
                
                # Block repeated n-grams
                if no_repeat_ngram_size > 0:
                    next_logits = self._block_repeated_ngrams(next_logits, beam_seqs, no_repeat_ngram_size)
                
                # Block EOS before min_len
                if step < min_len:
                    next_logits[:, self.config.eos_id] = float('-inf')
                
                # Mask finished beams (only allow pad)
                for bi in range(beam_size):
                    if beam_finished[bi]:
                        next_logits[bi, :] = float('-inf')
                        next_logits[bi, self.config.pad_id] = 0.0
                
                # Compute log probabilities
                if self.copy_mechanism is not None and src_detail_b is not None:
                    log_probs = next_logits  # already log-probs from copy mechanism
                else:
                    log_probs = F.log_softmax(next_logits, dim=-1)  # [beam_size, vocab]
                
                # Add to beam scores
                vocab_size = log_probs.size(-1)
                next_scores = beam_scores.unsqueeze(1) + log_probs  # [beam_size, vocab]
                next_scores = next_scores.view(-1)  # [beam_size * vocab]
                
                # Select top-k candidates
                topk_scores, topk_indices = torch.topk(next_scores, beam_size * 2, dim=-1)
                
                # Convert flat indices to beam and token indices
                topk_beam_idx = topk_indices // vocab_size
                topk_token_idx = topk_indices % vocab_size
                
                # Build new beams
                new_seqs = []
                new_scores = []
                new_finished = []
                n_filled = 0
                
                for k in range(topk_scores.size(0)):
                    if n_filled >= beam_size:
                        break
                    
                    bi = topk_beam_idx[k].item()
                    ti = topk_token_idx[k].item()
                    score = topk_scores[k].item()
                    
                    # Build candidate sequence — keep [1, seq_len] shape
                    candidate = torch.cat([beam_seqs[bi:bi+1], torch.tensor([[ti]], device=device)], dim=-1)
                    
                    if ti == self.config.eos_id:
                        # Length-normalized score
                        seq_len = candidate.size(-1)
                        norm_score = score / (seq_len ** length_penalty)
                        completed.append((norm_score, candidate))
                        # Don't count as filled beam — let other candidates fill
                        continue
                    
                    new_seqs.append(candidate)
                    new_scores.append(score)
                    new_finished.append(beam_finished[bi].item())
                    n_filled += 1
                
                # If not enough new beams, pad with existing
                while n_filled < beam_size:
                    pad_seq = torch.full((1, beam_seqs.size(1) + 1), self.config.pad_id, device=device, dtype=torch.long)
                    new_seqs.append(pad_seq)
                    new_scores.append(-1e9)
                    new_finished.append(True)
                    n_filled += 1
                
                # Pad sequences to same length
                max_seq_len = max(s.size(-1) for s in new_seqs)
                padded_seqs = []
                for s in new_seqs:
                    if s.size(-1) < max_seq_len:
                        s = F.pad(s, (0, max_seq_len - s.size(-1)), value=self.config.pad_id)
                    padded_seqs.append(s)
                
                # Stack into [beam_size, seq_len]
                beam_seqs = torch.cat(padded_seqs, dim=0)[:beam_size]
                beam_scores = torch.tensor(new_scores[:beam_size], device=device)
                beam_finished = torch.tensor(new_finished[:beam_size], dtype=torch.bool, device=device)
                
                # Early stopping: if we have enough completed hypotheses
                if len(completed) >= beam_size:
                    break
            
            # Add remaining beams as completed
            for bi in range(beam_size):
                seq_len = beam_seqs[bi].size(-1)
                norm_score = beam_scores[bi].item() / (seq_len ** length_penalty)
                completed.append((norm_score, beam_seqs[bi:bi+1]))
            
            # Select best hypothesis
            if completed:
                completed.sort(key=lambda x: x[0], reverse=True)
                best_seq = completed[0][1]
            else:
                best_seq = beam_seqs[0:1]
            
            all_best.append(best_seq.squeeze(0))
        
        # Pad all sequences to same length
        max_out_len = max(s.size(0) for s in all_best)
        padded_out = []
        for s in all_best:
            if s.size(0) < max_out_len:
                s = F.pad(s, (0, max_out_len - s.size(0)), value=self.config.pad_id)
            padded_out.append(s.unsqueeze(0))
        
        return torch.cat(padded_out, dim=0)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config: MambaTransformerConfig) -> MambaTransformerModel:
    """Build model from config"""
    model = MambaTransformerModel(config)
    print(f"Model built with {count_parameters(model):,} trainable parameters")
    print(f"  - Mamba encoder layers: {config.n_mamba_layers}")
    print(f"  - Transformer decoder layers: {config.n_decoder_layers}")
    print(f"  - d_model: {config.d_model}")
    print(f"  - n_heads: {config.n_heads}")
    print(f"  - Chunk size: {config.chunk_size}, stride: {config.stride}")
    print(f"  - Memory tokens per chunk: {config.n_memory_tokens}")
    # V2 features
    v2_features = []
    if config.use_rmsnorm: v2_features.append('RMSNorm')
    if config.use_swiglu: v2_features.append('SwiGLU')
    if config.use_bidirectional_mamba: v2_features.append('BiMamba')
    if config.use_cross_chunk_attn: v2_features.append(f'CrossChunkAttn(×{config.n_cross_chunk_layers})')
    if config.use_gated_cross_attn: v2_features.append('GatedCrossAttn')
    if getattr(config, 'source_bypass_stride', 0) > 0: v2_features.append(f'SourceBypass(stride={config.source_bypass_stride})')
    if getattr(config, 'use_copy_mechanism', False): v2_features.append('CopyMechanism')
    if v2_features:
        print(f"  - V2 features: {', '.join(v2_features)}")
    else:
        print(f"  - V2 features: None (baseline mode)")
    # V3 features
    v3_features = []
    if getattr(config, 'source_bypass_stride', 0) > 0:
        v3_features.append(f'SourceBypass(stride={config.source_bypass_stride})')
    if getattr(config, 'use_copy_mechanism', False):
        v3_features.append('CopyMechanism(balanced)')
    if v3_features:
        print(f"  - V3 features: {', '.join(v3_features)}")
    return model
