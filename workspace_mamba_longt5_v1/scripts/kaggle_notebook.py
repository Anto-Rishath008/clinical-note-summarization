"""
Kaggle Notebook Setup - Mamba-Transformer Hybrid Model
========================================================

Copy this entire file content into a Kaggle notebook cell and run it.
This will train the Mamba-Transformer model from scratch.

Prerequisites:
1. Upload your dataset (mimic-iv-bhc.csv) to Kaggle datasets
2. Upload tokenizer files (spm.model, spm.vocab) to Kaggle datasets
3. Enable GPU accelerator (A100 or T4)
4. Enable internet for package installation
"""

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
# !pip install -q mamba-ssm causal-conv1d>=1.1.0 sentencepiece rouge-score pyyaml

# ============================================================
# CELL 2: Setup and Imports
# ============================================================
import os
import sys
import shutil
import time
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("Mamba SSM loaded successfully!")
except ImportError:
    MAMBA_AVAILABLE = False
    print("WARNING: mamba-ssm not available, using GRU fallback")

import sentencepiece as spm

# ============================================================
# CELL 3: GPU Check
# ============================================================
def check_gpu():
    print("=" * 60)
    print("GPU CHECK")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Enable GPU in Kaggle settings.")
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Test
    a = torch.randn(100, 100, device='cuda')
    b = torch.randn(100, 100, device='cuda')
    c = torch.matmul(a, b)
    print(f"CUDA test passed! Shape: {c.shape}")
    print("=" * 60)

check_gpu()

# ============================================================
# CELL 4: Configuration
# ============================================================
@dataclass
class Config:
    # Paths - UPDATE THESE FOR YOUR KAGGLE DATASET
    data_path: str = "/kaggle/input/mimic-iv-bhc/mimic-iv-bhc.csv"
    tokenizer_path: str = "/kaggle/input/mimic-iv-bhc/tokenizer/spm.model"
    output_dir: str = "/kaggle/working/outputs"
    checkpoint_dir: str = "/kaggle/working/checkpoints"
    
    # Model
    vocab_size: int = 32000
    d_model: int = 512
    n_mamba_layers: int = 6
    n_decoder_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    chunk_size: int = 256
    stride: int = 192
    n_memory_tokens: int = 8
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    
    # Data
    max_src_len: int = 4096
    max_tgt_len: int = 512
    pad_id: int = 3
    bos_id: int = 1
    eos_id: int = 2
    
    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 50000
    warmup_steps: int = 2000
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    log_steps: int = 50
    
    seed: int = 42

config = Config()

# Create directories
os.makedirs(config.output_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True)

# ============================================================
# CELL 5: Model Definition
# ============================================================
class MambaFallback(nn.Module):
    """Fallback when mamba-ssm not available"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.norm(out)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mamba = MambaFallback(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.mamba(self.norm(x))) + x


class MambaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(config.d_model, config.mamba_d_state, config.mamba_d_conv, 
                      config.mamba_expand, config.dropout)
            for _ in range(config.n_mamba_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MemoryCompressor(nn.Module):
    def __init__(self, d_model, n_memory=8, n_heads=8, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_memory, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        b = x.size(0)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=mask)
        return self.norm(self.dropout(out) + q)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # Self attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.dropout(x) + residual
        
        # Cross attention
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(x, memory, memory)
        x = self.dropout(x) + residual
        
        # FFN
        residual = x
        x = self.norm3(x)
        x = self.ff(x) + residual
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_decoder_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, tgt_key_padding_mask)
        return self.norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8192, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class MambaTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.src_emb = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.tgt_emb = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.pos_enc = PositionalEncoding(config.d_model)
        
        self.encoder = MambaEncoder(config)
        self.compressor = MemoryCompressor(config.d_model, config.n_memory_tokens, 
                                           config.n_heads, config.dropout)
        self.decoder = Decoder(config)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.tgt_emb.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _chunk(self, src, mask):
        B, L = src.shape
        chunks, masks = [], []
        start = 0
        while start < L:
            end = min(start + self.config.chunk_size, L)
            c = src[:, start:end]
            m = mask[:, start:end]
            if c.size(1) < self.config.chunk_size:
                pad = self.config.chunk_size - c.size(1)
                c = F.pad(c, (0, pad), value=self.config.pad_id)
                m = F.pad(m, (0, pad), value=True)
            chunks.append(c)
            masks.append(m)
            start += self.config.stride
            if end >= L:
                break
        return chunks, masks
    
    def encode(self, src, src_mask):
        chunks, masks = self._chunk(src, src_mask)
        all_mem = []
        for c, m in zip(chunks, masks):
            e = self.pos_enc(self.src_emb(c))
            e = self.encoder(e)
            mem = self.compressor(e, m)
            all_mem.append(mem)
        return torch.cat(all_mem, dim=1)
    
    @staticmethod
    def causal_mask(sz, device):
        return torch.triu(torch.ones(sz, sz, device=device), 1).masked_fill_(
            torch.triu(torch.ones(sz, sz, device=device), 1) == 1, float('-inf'))
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if src_mask is None:
            src_mask = (src == self.config.pad_id)
        if tgt_mask is None:
            tgt_mask = (tgt == self.config.pad_id)
        
        memory = self.encode(src, src_mask)
        causal = self.causal_mask(tgt.size(1), tgt.device)
        
        t = self.pos_enc(self.tgt_emb(tgt))
        out = self.decoder(t, memory, causal, tgt_mask)
        return self.output(out)
    
    @torch.no_grad()
    def generate(self, src, src_mask=None, max_len=256):
        self.eval()
        device = src.device
        B = src.size(0)
        
        if src_mask is None:
            src_mask = (src == self.config.pad_id)
        
        memory = self.encode(src, src_mask)
        gen = torch.full((B, 1), self.config.bos_id, device=device, dtype=torch.long)
        
        for _ in range(max_len - 1):
            causal = self.causal_mask(gen.size(1), device)
            t = self.pos_enc(self.tgt_emb(gen))
            out = self.decoder(t, memory, causal)
            logits = self.output(out[:, -1:])
            next_tok = logits.argmax(dim=-1)
            gen = torch.cat([gen, next_tok], dim=1)
            if (next_tok == self.config.eos_id).all():
                break
        
        return gen

# Build model
model = MambaTransformer(config).cuda()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")

# ============================================================
# CELL 6: Dataset
# ============================================================
import csv

class ClinicalDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_src=4096, max_tgt=512, 
                 split='train', train_ratio=0.95, max_samples=None):
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.pad_id = config.pad_id
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        
        csv.field_size_limit(10 * 1024 * 1024)
        samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break
                src = row.get('source_text', '').strip()
                tgt = row.get('target_text', '').strip()
                if src and tgt:
                    samples.append((src, tgt))
        
        import random
        random.seed(42)
        random.shuffle(samples)
        
        split_idx = int(len(samples) * train_ratio)
        self.samples = samples[:split_idx] if split == 'train' else samples[split_idx:]
        print(f"{split} samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        src_ids = self.tokenizer.EncodeAsIds(src)[:self.max_src]
        tgt_ids = self.tokenizer.EncodeAsIds(tgt)
        tgt_in = [self.bos_id] + tgt_ids[:self.max_tgt - 2]
        tgt_out = tgt_ids[:self.max_tgt - 2] + [self.eos_id]
        return {
            'src': torch.tensor(src_ids),
            'tgt_in': torch.tensor(tgt_in),
            'tgt_out': torch.tensor(tgt_out)
        }


def collate(batch):
    max_src = max(b['src'].size(0) for b in batch)
    max_tgt = max(b['tgt_in'].size(0) for b in batch)
    B = len(batch)
    
    src = torch.full((B, max_src), config.pad_id)
    tgt_in = torch.full((B, max_tgt), config.pad_id)
    tgt_out = torch.full((B, max_tgt), config.pad_id)
    src_mask = torch.ones(B, max_src, dtype=torch.bool)
    tgt_mask = torch.ones(B, max_tgt, dtype=torch.bool)
    
    for i, b in enumerate(batch):
        s, ti, to = b['src'], b['tgt_in'], b['tgt_out']
        src[i, :s.size(0)] = s
        tgt_in[i, :ti.size(0)] = ti
        tgt_out[i, :to.size(0)] = to
        src_mask[i, :s.size(0)] = False
        tgt_mask[i, :ti.size(0)] = False
    
    return {'src': src, 'tgt_in': tgt_in, 'tgt_out': tgt_out, 
            'src_mask': src_mask, 'tgt_mask': tgt_mask}

# Load tokenizer and create datasets
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(config.tokenizer_path)
print(f"Tokenizer vocab size: {tokenizer.GetPieceSize()}")

train_ds = ClinicalDataset(config.data_path, tokenizer, split='train')
val_ds = ClinicalDataset(config.data_path, tokenizer, split='val')

train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, 
                          collate_fn=collate, num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ============================================================
# CELL 7: Training Loop
# ============================================================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, ignore_idx=3):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_idx = ignore_idx
    
    def forward(self, logits, targets):
        V = logits.size(-1)
        logits = logits.reshape(-1, V)
        targets = targets.reshape(-1)
        mask = (targets != self.ignore_idx)
        
        log_probs = F.log_softmax(logits, dim=-1)
        smooth = torch.zeros_like(log_probs).fill_(self.smoothing / (V - 1))
        smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -(smooth * log_probs).sum(dim=-1)
        return (loss * mask.float()).sum() / mask.sum()


def compute_rouge(preds, refs):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        r1, r2, rl = [], [], []
        for p, r in zip(preds, refs):
            s = scorer.score(r, p)
            r1.append(s['rouge1'].fmeasure)
            r2.append(s['rouge2'].fmeasure)
            rl.append(s['rougeL'].fmeasure)
        return {'r1': sum(r1)/len(r1), 'r2': sum(r2)/len(r2), 'rl': sum(rl)/len(rl)}
    except:
        return {'r1': 0, 'r2': 0, 'rl': 0}


def save_atomic(path, data):
    tmp = path + '.tmp'
    torch.save(data, tmp)
    shutil.move(tmp, path)
    size = os.path.getsize(path) / 1e6
    print(f"Saved: {path} ({size:.1f} MB)")
    return size > 50


def get_scheduler(opt, warmup, total):
    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * prog)))
    return LambdaLR(opt, lr_fn)


# Setup training
optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = get_scheduler(optimizer, config.warmup_steps, config.max_steps)
criterion = LabelSmoothingLoss(config.label_smoothing, config.pad_id)
scaler = GradScaler()

step = 0
best_rl = 0.0
running_loss = 0.0
start = time.time()

print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

model.train()
optimizer.zero_grad()

while step < config.max_steps:
    for batch in train_loader:
        if step >= config.max_steps:
            break
        
        src = batch['src'].cuda()
        tgt_in = batch['tgt_in'].cuda()
        tgt_out = batch['tgt_out'].cuda()
        src_mask = batch['src_mask'].cuda()
        tgt_mask = batch['tgt_mask'].cuda()
        
        with autocast():
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = criterion(logits, tgt_out) / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        running_loss += loss.item() * config.gradient_accumulation_steps
        
        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        step += 1
        
        # Logging
        if step % config.log_steps == 0:
            avg = running_loss / config.log_steps
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start
            print(f"Step {step}/{config.max_steps} | Loss: {avg:.4f} | "
                  f"Grad: {grad_norm:.4f} | LR: {lr:.2e} | Time: {elapsed/60:.1f}m")
            running_loss = 0.0
        
        # Evaluation
        if step % config.eval_steps == 0:
            model.eval()
            preds, refs = [], []
            with torch.no_grad():
                for i, vb in enumerate(val_loader):
                    if i >= 25:  # Eval on 100 samples
                        break
                    gen = model.generate(vb['src'].cuda(), vb['src_mask'].cuda())
                    for j in range(gen.size(0)):
                        pred = tokenizer.DecodeIds([t for t in gen[j].tolist() if t not in [0,1,2,3]])
                        ref = tokenizer.DecodeIds([t for t in vb['tgt_out'][j].tolist() if t not in [0,1,2,3]])
                        preds.append(pred)
                        refs.append(ref)
            
            scores = compute_rouge(preds, refs)
            print(f"Eval | R1: {scores['r1']:.4f} | R2: {scores['r2']:.4f} | RL: {scores['rl']:.4f}")
            
            if scores['rl'] > best_rl:
                best_rl = scores['rl']
                save_atomic(f"{config.checkpoint_dir}/best_model.pt", {
                    'step': step, 'model': model.state_dict(), 'opt': optimizer.state_dict(),
                    'sched': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                    'best_rl': best_rl, 'config': asdict(config)
                })
            
            model.train()
        
        # Checkpoint
        if step % config.save_steps == 0:
            save_atomic(f"{config.checkpoint_dir}/ckpt_{step}.pt", {
                'step': step, 'model': model.state_dict(), 'opt': optimizer.state_dict(),
                'sched': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                'best_rl': best_rl, 'config': asdict(config)
            })

# Final save
save_atomic(f"{config.checkpoint_dir}/final_model.pt", {
    'step': step, 'model': model.state_dict(), 'opt': optimizer.state_dict(),
    'sched': scheduler.state_dict(), 'scaler': scaler.state_dict(),
    'best_rl': best_rl, 'config': asdict(config)
})

print("=" * 60)
print("TRAINING COMPLETE")
print(f"Best ROUGE-L: {best_rl:.4f}")
print(f"Checkpoints: {config.checkpoint_dir}")
print("=" * 60)

# Verify outputs
best_path = f"{config.checkpoint_dir}/best_model.pt"
if os.path.exists(best_path):
    print(f"Best model: {best_path}")
    print(f"Size: {os.path.getsize(best_path)/1e6:.1f} MB")
