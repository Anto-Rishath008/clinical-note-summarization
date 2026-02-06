"""
Mamba Encoder + Transformer Decoder Hybrid Model
=================================================

From-scratch implementation for clinical note summarization.
Uses Mamba for efficient long-sequence encoding with memory token compression,
and standard Transformer decoder with cross-attention to memory tokens.

Architecture:
1. Token Embeddings (SentencePiece vocab)
2. Chunk long input into fixed length (chunk_size=256, stride=192)
3. Mamba encoder stack per chunk
4. Compress each chunk to K memory tokens via learned queries + attention pooling
5. Concatenate memory tokens across all chunks
6. Transformer decoder with cross-attention to memory tokens
7. Output projection to vocab
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

    @classmethod
    def from_dict(cls, d: dict) -> 'MambaTransformerConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


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


class MambaEncoder(nn.Module):
    """Stack of Mamba blocks for encoding"""
    def __init__(self, config: MambaTransformerConfig):
        super().__init__()
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
        self.final_norm = nn.LayerNorm(config.d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class MemoryTokenCompressor(nn.Module):
    """
    Compress chunk representations to K memory tokens using learned queries and attention pooling.
    """
    def __init__(self, d_model: int, n_memory_tokens: int = 8, n_heads: int = 8, dropout: float = 0.1):
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
        self.norm = nn.LayerNorm(d_model)
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


class TransformerDecoderLayer(nn.Module):
    """Standard Transformer decoder layer with self-attention and cross-attention"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
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
        
        # Cross-attention to memory tokens
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask)
        x = self.dropout(x) + residual
        
        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.ff(x) + residual
        
        return x


class TransformerDecoder(nn.Module):
    """Stack of Transformer decoder layers"""
    def __init__(self, config: MambaTransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
            )
            for _ in range(config.n_decoder_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)
    
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


class MambaTransformerModel(nn.Module):
    """
    Mamba Encoder + Transformer Decoder Hybrid Model for Summarization
    
    Architecture:
    1. Embed input tokens
    2. Chunk input sequence (chunk_size=256, stride=192)
    3. Encode each chunk with Mamba
    4. Compress each chunk to K=8 memory tokens
    5. Concatenate all memory tokens
    6. Decode with Transformer decoder cross-attending to memory
    7. Project to vocabulary
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
        )
        
        # Decoder (Transformer)
        self.decoder = TransformerDecoder(config)
        
        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie embeddings
        self.output_proj.weight = self.tgt_embedding.weight
        
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
        Encode source sequence with chunking, Mamba encoding, and memory compression.
        
        Args:
            src: [batch, src_len]
            src_mask: [batch, src_len] - True for padded positions
        
        Returns:
            memory: [batch, n_chunks * n_memory_tokens, d_model]
        """
        # Chunk input
        chunks, chunk_masks = self._chunk_input(src, src_mask)
        
        all_memory = []
        for chunk, mask in zip(chunks, chunk_masks):
            # Embed chunk
            chunk_emb = self.src_embedding(chunk)
            chunk_emb = self.pos_encoding(chunk_emb)
            
            # Encode with Mamba
            chunk_encoded = self.encoder(chunk_emb)
            
            # Compress to memory tokens
            memory_tokens = self.memory_compressor(chunk_encoded, mask)
            all_memory.append(memory_tokens)
        
        # Concatenate all memory tokens
        memory = torch.cat(all_memory, dim=1)
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
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        max_len: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            src: [batch, src_len] source token ids
            src_mask: [batch, src_len] - True for padded positions
            max_len: maximum generation length
            temperature: sampling temperature
            top_k: top-k sampling
        
        Returns:
            generated: [batch, gen_len] generated token ids
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        if src_mask is None:
            src_mask = (src == self.config.pad_id)
        
        # Encode source
        memory = self.encode(src, src_mask)
        
        # Start with BOS token
        generated = torch.full((batch_size, 1), self.config.bos_id, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len - 1):
            causal_mask = self.generate_causal_mask(generated.size(1), device)
            
            logits = self.decode(generated, memory, causal_mask)
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Don't update finished sequences
            next_token = torch.where(finished.unsqueeze(1), torch.tensor(self.config.pad_id, device=device), next_token)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            finished = finished | (next_token.squeeze(1) == self.config.eos_id)
            if finished.all():
                break
        
        return generated


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
    return model
