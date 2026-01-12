"""
Clinical Summarization - Core Module
Consolidates all models and utilities into single file for Kaggle deployment

Contains:
- HierarchicalEncoder, ChunkedEncoder
- AdditiveAttention, CoverageAttention
- PointerGeneratorDecoder
- PointerGeneratorModel
- TokenizedDataset, collate_fn, get_dataloader
- RougeMetric
- BeamSearchNode, beam_search_decode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
import pyarrow.parquet as pq
from rouge_score import rouge_scorer
import numpy as np


# ============================================================================
# ATTENTION MECHANISMS
# ============================================================================
"""
Attention Mechanisms for Pointer-Generator Network
Implements additive (Bahdanau) attention with coverage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Additive (Bahdanau) attention mechanism.
    
    Computes attention weights using:
    score(h_t, h_s) = v^T * tanh(W_h * h_s + W_d * h_t + b)
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Attention layers
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)  # *2 for bidirectional encoder
        self.W_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: (batch, hidden_dim) - current decoder state
            encoder_outputs: (batch, seq_len, hidden_dim*2) - all encoder states
            mask: (batch, seq_len) - 1 for valid positions, 0 for padding
            
        Returns:
            context: (batch, hidden_dim*2) - attention-weighted encoder states
            attn_weights: (batch, seq_len) - attention distribution
        """
        batch_size, seq_len, enc_dim = encoder_outputs.size()
        
        # Expand decoder hidden to match encoder sequence length
        # (batch, hidden_dim) -> (batch, seq_len, hidden_dim)
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(batch_size, seq_len, self.hidden_dim)
        
        # Compute attention scores
        # (batch, seq_len, hidden_dim)
        energy = torch.tanh(self.W_h(encoder_outputs) + self.W_d(decoder_hidden_expanded))
        
        # (batch, seq_len, 1) -> (batch, seq_len)
        scores = self.v(energy).squeeze(2)
        
        # Apply mask (use -1e4 instead of -1e9 to avoid FP16 overflow)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        
        # Compute context vector (weighted sum of encoder outputs)
        # (batch, 1, seq_len) x (batch, seq_len, enc_dim) -> (batch, 1, enc_dim) -> (batch, enc_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attn_weights


class CoverageAttention(nn.Module):
    """
    Attention mechanism with coverage to prevent repetition.
    
    Tracks cumulative attention history and penalizes re-attending to same positions.
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Attention layers
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.W_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_c = nn.Linear(1, hidden_dim, bias=False)  # Coverage feature
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs, coverage, mask=None):
        """
        Args:
            decoder_hidden: (batch, hidden_dim)
            encoder_outputs: (batch, seq_len, hidden_dim*2)
            coverage: (batch, seq_len) - cumulative attention from previous steps
            mask: (batch, seq_len) - attention mask
            
        Returns:
            context: (batch, hidden_dim*2)
            attn_weights: (batch, seq_len)
            coverage: (batch, seq_len) - updated coverage
        """
        batch_size, seq_len, enc_dim = encoder_outputs.size()
        
        # Expand decoder hidden
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(batch_size, seq_len, self.hidden_dim)
        
        # Add coverage feature
        # (batch, seq_len) -> (batch, seq_len, 1)
        coverage_feature = coverage.unsqueeze(2)
        
        # Compute attention scores with coverage
        energy = torch.tanh(
            self.W_h(encoder_outputs) + 
            self.W_d(decoder_hidden_expanded) + 
            self.W_c(coverage_feature)
        )
        
        scores = self.v(energy).squeeze(2)
        
        # Apply mask (use -1e4 instead of -1e9 to avoid FP16 overflow)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=1)
        
        # Update coverage (accumulate attention)
        coverage = coverage + attn_weights
        
        # Context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attn_weights, coverage


# ============================================================================
# ENCODER
# ============================================================================

"""
Hierarchical Encoder for Clinical Notes
Processes long documents by chunking and using bidirectional LSTM
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder that processes long input in chunks.
    
    Architecture:
    1. Split input into chunks of fixed length
    2. Process all chunks with shared BiLSTM
    3. Return all hidden states for attention
    """
    
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=2, dropout=0.3, pad_id=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_id = pad_id
        
        # Word embeddings (learned from scratch)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        
        # Bidirectional LSTM (output dim = hidden_dim * 2)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings with small random values
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[pad_id] = 0  # Zero out padding
        
    def forward(self, input_ids, lengths):
        """
        Args:
            input_ids: (batch, seq_len) - token IDs (can be flattened chunks)
            lengths: (batch,) - actual lengths (excluding padding)
            
        Returns:
            outputs: (batch, seq_len, hidden_dim*2) - all hidden states
            hidden: tuple of (h_n, c_n) - final hidden states
        """
        batch_size, seq_len = input_ids.size()
        
        # Embed tokens
        embedded = self.embedding(input_ids)  # (batch, seq_len, emb_dim)
        embedded = self.dropout(embedded)
        
        # Pack for efficient LSTM processing (handles variable lengths)
        packed = pack_padded_sequence(
            embedded, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Process with LSTM
        packed_outputs, hidden = self.lstm(packed)
        
        # Unpack
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=seq_len)
        # outputs: (batch, seq_len, hidden_dim*2)
        
        return outputs, hidden
    
    def get_output_dim(self):
        """Return the output dimension (bidirectional doubles it)"""
        return self.hidden_dim * 2


class ChunkedEncoder(nn.Module):
    """
    Wrapper that handles chunking input into fixed-size segments.
    Useful for very long documents.
    """
    
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=2, dropout=0.3, pad_id=3,
                 chunk_len=256, num_chunks=8):
        super().__init__()
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.max_len = chunk_len * num_chunks
        
        self.encoder = HierarchicalEncoder(
            vocab_size, emb_dim, hidden_dim, num_layers, dropout, pad_id
        )
        
    def forward(self, input_ids, lengths):
        """
        Args:
            input_ids: (batch, max_len) - may contain multiple chunks
            lengths: (batch,) - actual lengths
            
        Returns:
            outputs: (batch, max_len, hidden_dim*2)
            hidden: final hidden states
        """
        # Truncate to max_len if needed
        if input_ids.size(1) > self.max_len:
            input_ids = input_ids[:, :self.max_len]
            lengths = torch.clamp(lengths, max=self.max_len)
        
        return self.encoder(input_ids, lengths)
    
    def get_output_dim(self):
        return self.encoder.get_output_dim()


# ============================================================================
# DECODER
# ============================================================================

"""
Pointer-Generator Decoder with Coverage
Combines generation from vocabulary and copying from input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.attention import AdditiveAttention, CoverageAttention


class PointerGeneratorDecoder(nn.Module):
    """
    LSTM decoder with pointer-generator mechanism and coverage.
    
    At each step, computes:
    - P_vocab: probability distribution over vocabulary
    - P_copy: probability distribution over input tokens
    - p_gen: generation probability (scalar)
    - Final dist = p_gen * P_vocab + (1-p_gen) * P_copy
    """
    
    def __init__(self, vocab_size, emb_dim, hidden_dim, encoder_dim, 
                 num_layers=2, dropout=0.3, pad_id=3, use_coverage=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim  # Hidden dim * 2 for bidirectional
        self.num_layers = num_layers
        self.pad_id = pad_id
        self.use_coverage = use_coverage
        
        # Embeddings (shared with encoder if needed, but we keep separate)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        
        # LSTM decoder
        # Input: embedding + context from previous step
        self.lstm = nn.LSTM(
            emb_dim + encoder_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention
        if use_coverage:
            self.attention = CoverageAttention(hidden_dim)
        else:
            self.attention = AdditiveAttention(hidden_dim)
        
        # Output projection to vocabulary
        self.out_proj = nn.Linear(hidden_dim + encoder_dim + emb_dim, vocab_size)
        
        # Pointer-generator switch
        self.p_gen_linear = nn.Linear(encoder_dim + hidden_dim + emb_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[pad_id] = 0
        
    def forward(self, input_token, last_hidden, encoder_outputs, encoder_mask, 
                context_vec, coverage=None, encoder_input_ids=None):
        """
        Single decoding step.
        
        Args:
            input_token: (batch, 1) - previous token
            last_hidden: tuple (h, c) - previous LSTM state
            encoder_outputs: (batch, src_len, encoder_dim)
            encoder_mask: (batch, src_len) - 1 for valid, 0 for pad
            context_vec: (batch, encoder_dim) - context from previous step
            coverage: (batch, src_len) - cumulative attention (if using coverage)
            encoder_input_ids: (batch, src_len) - source token IDs for copying
            
        Returns:
            final_dist: (batch, extended_vocab_size) - final probability distribution
            hidden: tuple (h, c) - updated LSTM state
            context: (batch, encoder_dim) - new context vector
            attn_weights: (batch, src_len) - attention weights
            p_gen: (batch, 1) - generation probability
            coverage: (batch, src_len) - updated coverage
        """
        batch_size = input_token.size(0)
        
        # Embed input token
        embedded = self.embedding(input_token)  # (batch, 1, emb_dim)
        embedded = self.dropout(embedded)
        
        # Concatenate embedding with previous context
        lstm_input = torch.cat([embedded, context_vec.unsqueeze(1)], dim=2)  # (batch, 1, emb_dim + encoder_dim)
        
        # LSTM step - ensure hidden state is contiguous
        if last_hidden is not None:
            last_hidden = (last_hidden[0].contiguous(), last_hidden[1].contiguous())
        lstm_output, hidden = self.lstm(lstm_input, last_hidden)  # lstm_output: (batch, 1, hidden_dim)
        lstm_output = lstm_output.squeeze(1)  # (batch, hidden_dim)
        
        # Attention
        if self.use_coverage and coverage is not None:
            context, attn_weights, coverage = self.attention(
                lstm_output, encoder_outputs, coverage, encoder_mask
            )
        else:
            context, attn_weights = self.attention(
                lstm_output, encoder_outputs, encoder_mask
            )
            if coverage is None:
                coverage = torch.zeros_like(attn_weights)
        
        # Compute vocabulary distribution
        # Concatenate: LSTM output + context + embedding
        out_input = torch.cat([lstm_output, context, embedded.squeeze(1)], dim=1)
        vocab_logits = self.out_proj(out_input)  # (batch, vocab_size)
        vocab_dist = F.softmax(vocab_logits, dim=1)
        
        # Compute pointer-generator switch
        p_gen_input = out_input  # Same features
        p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))  # (batch, 1)
        
        # Combine generation and copying
        # Final dist = p_gen * P_vocab + (1 - p_gen) * P_copy
        final_dist = p_gen * vocab_dist  # (batch, vocab_size)
        
        # Add copying distribution
        if encoder_input_ids is not None:
            # Scatter attention weights to vocabulary positions
            # Create extended vocabulary (vocab + source tokens)
            # For simplicity, we use scatter_add to accumulate attention weights
            
            # Copy distribution: (1 - p_gen) * attn_weights
            copy_dist = (1 - p_gen) * attn_weights  # (batch, src_len)
            
            # Scatter copy probabilities to vocab positions
            # Note: This is a simplified version. In practice, you'd extend vocab size
            # to include all source tokens, but for now we just add to existing vocab positions
            final_dist = final_dist.scatter_add(
                1, 
                encoder_input_ids, 
                copy_dist
            )
        
        return final_dist, hidden, context, attn_weights, p_gen, coverage
    
    def init_hidden(self, batch_size, device):
        """Initialize LSTM hidden state"""
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        )
    
    def init_context(self, batch_size, device):
        """Initialize context vector"""
        return torch.zeros(batch_size, self.encoder_dim, device=device)


# ============================================================================
# COMPLETE MODEL
# ============================================================================

"""
Complete Pointer-Generator Model
Combines encoder and decoder with training and inference methods
"""

import torch
import torch.nn as nn
# from models.encoder import ChunkedEncoder
# from models.decoder import PointerGeneratorDecoder


class PointerGeneratorModel(nn.Module):
    """
    Complete seq2seq model with pointer-generator mechanism.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Extract config
        vocab_size = config['data']['vocab_size']
        emb_dim = config['model']['emb_dim']
        hidden_dim = config['model']['hidden_dim']
        num_layers = config['model']['num_layers']
        dropout = config['model']['dropout']
        pad_id = config['data']['pad_id']
        chunk_len = config['model']['chunk_len']
        num_chunks = config['model']['num_chunks']
        use_coverage = config['model']['use_coverage']
        
        self.config = config
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = config['data']['bos_id']
        self.eos_id = config['data']['eos_id']
        
        # Encoder
        self.encoder = ChunkedEncoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=pad_id,
            chunk_len=chunk_len,
            num_chunks=num_chunks
        )
        
        encoder_dim = self.encoder.get_output_dim()
        
        # Decoder
        self.decoder = PointerGeneratorDecoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            encoder_dim=encoder_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=pad_id,
            use_coverage=use_coverage
        )
        
        # Share embeddings between encoder and decoder (optional)
        # self.decoder.embedding = self.encoder.encoder.embedding
        
    def forward(self, src_ids, src_lengths, tgt_ids, teacher_forcing_ratio=1.0):
        """
        Training forward pass with teacher forcing.
        
        Args:
            src_ids: (batch, src_len) - source token IDs
            src_lengths: (batch,) - source lengths
            tgt_ids: (batch, tgt_len) - target token IDs (includes BOS and EOS)
            teacher_forcing_ratio: probability of using teacher forcing
            
        Returns:
            outputs: (batch, tgt_len-1, vocab_size) - logits for each position
            attentions: (batch, tgt_len-1, src_len) - attention weights
            coverages: (batch, tgt_len-1, src_len) - coverage over time
            p_gens: (batch, tgt_len-1) - generation probabilities
        """
        batch_size, tgt_len = tgt_ids.size()
        device = src_ids.device
        
        # Encode
        encoder_outputs, encoder_hidden = self.encoder(src_ids, src_lengths)
        # encoder_outputs: (batch, src_len, hidden_dim*2)
        
        # Create encoder mask
        src_len = src_ids.size(1)
        encoder_mask = (src_ids != self.pad_id).float()  # (batch, src_len)
        
        # Initialize decoder state
        decoder_hidden = self._bridge_encoder_hidden(encoder_hidden, batch_size, device)
        context = self.decoder.init_context(batch_size, device)
        coverage = torch.zeros(batch_size, src_len, device=device)
        
        # Prepare outputs
        outputs = []
        attentions = []
        coverages = []
        p_gens = []
        
        # Start with BOS token
        input_token = tgt_ids[:, 0].unsqueeze(1)  # (batch, 1)
        
        # Decode step by step (teacher forcing)
        for t in range(1, tgt_len):
            # Decoder step
            final_dist, decoder_hidden, context, attn, p_gen, coverage = self.decoder(
                input_token=input_token,
                last_hidden=decoder_hidden,
                encoder_outputs=encoder_outputs,
                encoder_mask=encoder_mask,
                context_vec=context,
                coverage=coverage,
                encoder_input_ids=src_ids
            )
            
            outputs.append(final_dist)
            attentions.append(attn)
            coverages.append(coverage)
            p_gens.append(p_gen)
            
            # Teacher forcing: use ground truth as next input
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt_ids[:, t].unsqueeze(1)
            else:
                input_token = final_dist.argmax(dim=1).unsqueeze(1)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # (batch, tgt_len-1, vocab_size)
        attentions = torch.stack(attentions, dim=1)  # (batch, tgt_len-1, src_len)
        coverages = torch.stack(coverages, dim=1)  # (batch, tgt_len-1, src_len)
        p_gens = torch.stack(p_gens, dim=1).squeeze(2)  # (batch, tgt_len-1)
        
        return outputs, attentions, coverages, p_gens
    
    def _bridge_encoder_hidden(self, encoder_hidden, batch_size, device):
        """
        Convert encoder hidden state to decoder initial state.
        Encoder is bidirectional, decoder is unidirectional.
        """
        # encoder_hidden is tuple of (h, c)
        # h, c: (num_layers * 2, batch, hidden_dim) for bidirectional
        
        h, c = encoder_hidden
        num_layers = h.size(0) // 2
        
        # Combine forward and backward by concatenating and projecting
        # For simplicity, we just take the forward direction
        h = h[::2]  # Take every other layer (forward direction)
        c = c[::2]
        
        return (h, c)
    
    def compute_loss(self, outputs, targets, attentions, coverages, coverage_lambda=1.0, label_smoothing=0.1):
        """
        Compute loss with coverage penalty and label smoothing.
        
        Args:
            outputs: (batch, tgt_len-1, vocab_size) - predicted distributions
            targets: (batch, tgt_len) - target token IDs (includes BOS)
            attentions: (batch, tgt_len-1, src_len) - attention weights
            coverages: (batch, tgt_len-1, src_len) - accumulated coverage
            coverage_lambda: weight for coverage loss
            label_smoothing: label smoothing factor (default 0.1)
            
        Returns:
            total_loss: scalar
            nll_loss: scalar (negative log-likelihood)
            coverage_loss: scalar
        """
        # Remove BOS from targets to match outputs
        targets = targets[:, 1:]  # (batch, tgt_len-1)
        
        batch_size, tgt_len, vocab_size = outputs.size()
        
        # Negative log-likelihood loss
        # Flatten for CrossEntropyLoss
        outputs_flat = outputs.reshape(-1, vocab_size)  # (batch * tgt_len, vocab_size)
        targets_flat = targets.reshape(-1)  # (batch * tgt_len)
        
        # Ignore padding in loss with label smoothing
        nll_loss = nn.functional.cross_entropy(
            outputs_flat,
            targets_flat,
            ignore_index=self.pad_id,
            reduction='mean',
            label_smoothing=label_smoothing
        )
        
        # Coverage loss: penalize re-attending to same positions
        # coverage_loss = sum_t min(a_t, c_t) where c_t is coverage up to step t
        # We compute: sum over all positions of min(attention, coverage)
        if coverage_lambda > 0:
            # For each time step, compute min(attention, coverage-at-previous-step)
            # coverages[:, t] already includes attention at step t
            # So we need coverage before step t
            prev_coverage = torch.cat([
                torch.zeros(batch_size, 1, coverages.size(2), device=coverages.device),
                coverages[:, :-1]
            ], dim=1)
            
            coverage_loss = torch.min(attentions, prev_coverage).sum(dim=2).mean()
        else:
            coverage_loss = torch.tensor(0.0, device=outputs.device)
        
        total_loss = nll_loss + coverage_lambda * coverage_loss
        
        return total_loss, nll_loss, coverage_loss
    
    def generate(self, src_ids, src_lengths, beam_size=4, max_length=384, 
                 min_length=50, length_penalty=1.0, no_repeat_ngram=3):
        """
        Generate summary using beam search.
        
        Args:
            src_ids: (batch, src_len) - typically batch=1 for inference
            src_lengths: (batch,)
            beam_size: beam width
            max_length: maximum generation length
            min_length: minimum generation length
            length_penalty: length penalty factor
            no_repeat_ngram: block repeating n-grams
            
        Returns:
            generated_ids: (batch, seq_len) - generated token IDs
            scores: (batch,) - sequence scores
        """
        self.eval()
        with torch.no_grad():
            batch_size = src_ids.size(0)
            device = src_ids.device
            
            # Encode
            encoder_outputs, encoder_hidden = self.encoder(src_ids, src_lengths)
            encoder_mask = (src_ids != self.pad_id).float()
            
            # Initialize for beam search
            # We'll use a simple beam search implementation
            from utils.beam_search import beam_search_decode
            
            generated_ids, scores = beam_search_decode(
                model=self,
                encoder_outputs=encoder_outputs,
                encoder_hidden=encoder_hidden,
                encoder_mask=encoder_mask,
                src_ids=src_ids,
                beam_size=beam_size,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                no_repeat_ngram=no_repeat_ngram,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                pad_id=self.pad_id,
                device=device
            )
            
        return generated_ids, scores


# ============================================================================
# DATASET UTILITIES
# ============================================================================

"""
PyTorch Dataset Classes for Clinical Note Summarization
Handles streaming from tokenized parquet files
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pyarrow.parquet as pq


class TokenizedDataset(Dataset):
    """
    Dataset for tokenized data stored in Parquet.
    Assumes columns: note_id, src_ids (List[int]), tgt_ids (List[int])
    """
    
    def __init__(self, parquet_path, max_src_len=2048, max_tgt_len=384, pad_id=3):
        self.parquet_path = parquet_path
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_id = pad_id
        
        # Load dataset
        self.df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(self.df)} examples from {parquet_path}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        src_ids = row['src_ids']
        tgt_ids = row['tgt_ids']
        
        # Convert to lists if needed
        if not isinstance(src_ids, list):
            src_ids = list(src_ids)
        if not isinstance(tgt_ids, list):
            tgt_ids = list(tgt_ids)
        
        # Truncate to max lengths
        src_ids = src_ids[:self.max_src_len]
        tgt_ids = tgt_ids[:self.max_tgt_len]
        
        # Get actual lengths
        src_len = len(src_ids)
        tgt_len = len(tgt_ids)
        
        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
            'src_len': src_len,
            'tgt_len': tgt_len
        }


def collate_fn(batch, pad_id=3):
    """
    Collate function for DataLoader.
    Pads sequences to same length within batch.
    
    Args:
        batch: list of dicts with src_ids, tgt_ids, src_len, tgt_len
        pad_id: padding token ID
        
    Returns:
        dict with padded tensors
    """
    src_ids = [item['src_ids'] for item in batch]
    tgt_ids = [item['tgt_ids'] for item in batch]
    src_lens = torch.tensor([item['src_len'] for item in batch])
    tgt_lens = torch.tensor([item['tgt_len'] for item in batch])
    
    # Find max lengths in batch
    max_src_len = max(len(s) for s in src_ids)
    max_tgt_len = max(len(t) for t in tgt_ids)
    
    # Pad sequences
    src_ids_padded = []
    for src in src_ids:
        padded = src + [pad_id] * (max_src_len - len(src))
        src_ids_padded.append(padded)
    
    tgt_ids_padded = []
    for tgt in tgt_ids:
        padded = tgt + [pad_id] * (max_tgt_len - len(tgt))
        tgt_ids_padded.append(padded)
    
    # Convert to tensors
    src_ids_padded = torch.tensor(src_ids_padded, dtype=torch.long)
    tgt_ids_padded = torch.tensor(tgt_ids_padded, dtype=torch.long)
    
    return {
        'src_ids': src_ids_padded,
        'tgt_ids': tgt_ids_padded,
        'src_lens': src_lens,
        'tgt_lens': tgt_lens
    }


def get_dataloader(parquet_path, batch_size, shuffle=True, num_workers=0, 
                   max_src_len=2048, max_tgt_len=384, pad_id=3):
    """
    Create DataLoader for tokenized dataset.
    
    Args:
        parquet_path: path to parquet file
        batch_size: batch size
        shuffle: whether to shuffle data
        num_workers: number of worker processes (0 for Windows compatibility)
        max_src_len: max source length
        max_tgt_len: max target length
        pad_id: padding token ID
        
    Returns:
        DataLoader
    """
    dataset = TokenizedDataset(parquet_path, max_src_len, max_tgt_len, pad_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


class StreamingDataset(Dataset):
    """
    Memory-efficient streaming dataset for very large files.
    Only loads data on-demand using pyarrow.
    """
    
    def __init__(self, parquet_path, max_src_len=2048, max_tgt_len=384, pad_id=3):
        self.parquet_path = parquet_path
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_id = pad_id
        
        # Open parquet file
        self.parquet_file = pq.ParquetFile(parquet_path)
        self.num_rows = self.parquet_file.metadata.num_rows
        
        print(f"Streaming dataset: {self.num_rows} examples from {parquet_path}")
        
    def __len__(self):
        return self.num_rows
    
    def __getitem__(self, idx):
        # Read single row
        table = self.parquet_file.read_row_group(idx // 10000, columns=['src_ids', 'tgt_ids'])
        df = table.to_pandas()
        row_idx = idx % 10000
        
        if row_idx >= len(df):
            # Fallback: read from beginning
            row = df.iloc[0]
        else:
            row = df.iloc[row_idx]
        
        src_ids = list(row['src_ids'])[:self.max_src_len]
        tgt_ids = list(row['tgt_ids'])[:self.max_tgt_len]
        
        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
            'src_len': len(src_ids),
            'tgt_len': len(tgt_ids)
        }


# ============================================================================
# METRICS
# ============================================================================

"""
ROUGE Metrics Computation
Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
"""

from rouge_score import rouge_scorer
import numpy as np


class RougeMetric:
    """
    Wrapper for ROUGE metric computation.
    """
    
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
    def compute(self, predictions, references):
        """
        Compute ROUGE scores for a batch of predictions.
        
        Args:
            predictions: list of str - predicted summaries
            references: list of str - reference summaries
            
        Returns:
            dict with rouge1, rouge2, rougeL scores (precision, recall, fmeasure)
        """
        assert len(predictions) == len(references), "Mismatch in number of predictions and references"
        
        scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        for pred, ref in zip(predictions, references):
            score = self.scorer.score(ref, pred)
            scores['rouge1'].append(score['rouge1'].fmeasure)
            scores['rouge2'].append(score['rouge2'].fmeasure)
            scores['rougeL'].append(score['rougeL'].fmeasure)
        
        # Compute mean scores
        mean_scores = {
            'rouge1': np.mean(scores['rouge1']),
            'rouge2': np.mean(scores['rouge2']),
            'rougeL': np.mean(scores['rougeL'])
        }
        
        return mean_scores
    
    def compute_single(self, prediction, reference):
        """
        Compute ROUGE scores for a single prediction.
        
        Args:
            prediction: str - predicted summary
            reference: str - reference summary
            
        Returns:
            dict with rouge1, rouge2, rougeL scores
        """
        score = self.scorer.score(reference, prediction)
        return {
            'rouge1': score['rouge1'].fmeasure,
            'rouge2': score['rouge2'].fmeasure,
            'rougeL': score['rougeL'].fmeasure
        }


def compute_lead_k_baseline(texts, targets, tokenizer, k=384):
    """
    Compute ROUGE for lead-K baseline (first K tokens).
    
    Args:
        texts: list of str - input texts
        targets: list of str - target summaries
        tokenizer: SentencePiece tokenizer
        k: number of tokens to take
        
    Returns:
        dict with rouge scores
    """
    predictions = []
    
    for text in texts:
        # Tokenize
        tokens = tokenizer.encode(text, out_type=int)
        
        # Take first k tokens
        lead_tokens = tokens[:k]
        
        # Decode
        lead_text = tokenizer.decode(lead_tokens)
        predictions.append(lead_text)
    
    # Compute ROUGE
    metric = RougeMetric()
    scores = metric.compute(predictions, targets)
    
    return scores


def format_rouge_scores(scores):
    """
    Format ROUGE scores for printing.
    
    Args:
        scores: dict with rouge1, rouge2, rougeL
        
    Returns:
        formatted string
    """
    return (
        f"ROUGE-1: {scores['rouge1']:.4f} | "
        f"ROUGE-2: {scores['rouge2']:.4f} | "
        f"ROUGE-L: {scores['rougeL']:.4f}"
    )


# ============================================================================
# BEAM SEARCH
# ============================================================================

"""
Beam Search Decoder
Implements beam search for sequence generation
"""

import torch
import torch.nn.functional as F


class BeamSearchNode:
    """Node in beam search tree"""
    
    def __init__(self, hidden, context, coverage, token_id, log_prob, length):
        self.hidden = hidden
        self.context = context
        self.coverage = coverage
        self.token_id = token_id
        self.log_prob = log_prob
        self.length = length
        self.tokens = [token_id] if token_id is not None else []
        
    def eval_score(self, length_penalty=1.0):
        """Compute sequence score with length penalty"""
        return self.log_prob / (self.length ** length_penalty)


def beam_search_decode(model, encoder_outputs, encoder_hidden, encoder_mask, src_ids,
                       beam_size=4, max_length=384, min_length=50, length_penalty=1.0,
                       no_repeat_ngram=3, bos_id=1, eos_id=2, pad_id=3, device='cuda'):
    """
    Beam search decoding.
    
    Args:
        model: PointerGeneratorModel
        encoder_outputs: (batch, src_len, hidden_dim*2) - must be batch=1
        encoder_hidden: encoder final state
        encoder_mask: (batch, src_len)
        src_ids: (batch, src_len)
        beam_size: beam width
        max_length: maximum generation length
        min_length: minimum generation length before allowing EOS
        length_penalty: length penalty factor (higher = longer sequences)
        no_repeat_ngram: block repeating n-grams (0 = disabled)
        bos_id, eos_id, pad_id: special token IDs
        device: torch device
        
    Returns:
        best_sequence: (1, seq_len) - generated tokens
        best_score: scalar - sequence score
    """
    batch_size = encoder_outputs.size(0)
    assert batch_size == 1, "Beam search only supports batch_size=1"
    
    src_len = src_ids.size(1)
    
    # Initialize decoder hidden state
    decoder_hidden = model._bridge_encoder_hidden(encoder_hidden, batch_size, device)
    context = model.decoder.init_context(batch_size, device)
    coverage = torch.zeros(batch_size, src_len, device=device)
    
    # Expand encoder outputs for beam
    encoder_outputs = encoder_outputs.expand(beam_size, -1, -1)  # (beam, src_len, hidden_dim*2)
    encoder_mask = encoder_mask.expand(beam_size, -1)  # (beam, src_len)
    src_ids_expanded = src_ids.expand(beam_size, -1)  # (beam, src_len)
    
    # Initialize beams with BOS token
    beams = [BeamSearchNode(
        hidden=decoder_hidden,
        context=context,
        coverage=coverage,
        token_id=bos_id,
        log_prob=0.0,
        length=1
    )]
    
    completed_beams = []
    
    for step in range(max_length):
        if len(beams) == 0:
            break
        
        # Expand all beams
        all_candidates = []
        
        for beam in beams:
            if beam.token_id == eos_id:
                completed_beams.append(beam)
                continue
            
            # Prepare input
            input_token = torch.tensor([[beam.token_id]], device=device)  # (1, 1)
            
            # Expand hidden, context, coverage to batch size 1
            hidden = beam.hidden
            context = beam.context
            coverage = beam.coverage
            
            # Decoder step
            final_dist, new_hidden, new_context, attn, p_gen, new_coverage = model.decoder(
                input_token=input_token,
                last_hidden=hidden,
                encoder_outputs=encoder_outputs[:1],  # Take first beam
                encoder_mask=encoder_mask[:1],
                context_vec=context,
                coverage=coverage,
                encoder_input_ids=src_ids_expanded[:1]
            )
            
            # Log probabilities
            log_probs = torch.log(final_dist + 1e-10)  # (1, vocab_size)
            
            # Block short sequences from generating EOS
            if step < min_length:
                log_probs[0, eos_id] = -float('inf')
            
            # Block repeating n-grams
            if no_repeat_ngram > 0 and len(beam.tokens) >= no_repeat_ngram:
                # Check for repeating n-grams
                blocked_tokens = get_blocked_tokens(beam.tokens, no_repeat_ngram)
                for token_id in blocked_tokens:
                    log_probs[0, token_id] = -float('inf')
            
            # Get top-k tokens
            top_log_probs, top_indices = torch.topk(log_probs[0], beam_size)
            
            # Create new candidate beams
            for log_prob, token_id in zip(top_log_probs, top_indices):
                new_beam = BeamSearchNode(
                    hidden=new_hidden,
                    context=new_context,
                    coverage=new_coverage,
                    token_id=token_id.item(),
                    log_prob=beam.log_prob + log_prob.item(),
                    length=beam.length + 1
                )
                new_beam.tokens = beam.tokens + [token_id.item()]
                all_candidates.append(new_beam)
        
        # Select top beams
        beams = sorted(all_candidates, key=lambda x: x.eval_score(length_penalty), reverse=True)[:beam_size]
    
    # Add remaining beams to completed
    completed_beams.extend(beams)
    
    # Select best completed beam
    if len(completed_beams) == 0:
        # No beam completed, take best ongoing beam
        completed_beams = beams
    
    best_beam = max(completed_beams, key=lambda x: x.eval_score(length_penalty))
    
    # Convert to tensor
    best_sequence = torch.tensor([best_beam.tokens], device=device)  # (1, seq_len)
    best_score = best_beam.eval_score(length_penalty)
    
    return best_sequence, best_score


def get_blocked_tokens(tokens, n):
    """
    Get tokens that would create repeating n-grams.
    
    Args:
        tokens: list of token IDs
        n: n-gram size
        
    Returns:
        set of blocked token IDs
    """
    if len(tokens) < n - 1:
        return set()
    
    # Get last n-1 tokens
    prefix = tuple(tokens[-(n-1):])
    
    # Find all n-grams in sequence
    blocked = set()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        if ngram[:-1] == prefix:
            blocked.add(ngram[-1])
    
    return blocked
