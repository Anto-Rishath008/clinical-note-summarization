"""
Training Script for Mamba-Transformer Hybrid Model
====================================================

Full training pipeline with:
- Mixed precision (AMP fp16)
- Gradient clipping
- Cosine annealing with warmup
- ROUGE evaluation
- Checkpoint saving (atomic)
- Best model tracking by ROUGE-L
- Checkpoint resumption support
"""

import os
import sys
import time
import math
import yaml
import shutil
import logging
import argparse
import glob
import warnings
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Suppress harmless PyTorch warnings
warnings.filterwarnings('ignore', message='.*lr_scheduler.step.*before.*optimizer.step.*')
warnings.filterwarnings('ignore', message='.*mismatched key_padding_mask and attn_mask.*')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from model import MambaTransformerConfig, MambaTransformerModel, build_model
from data_loader import DataConfig, create_dataloaders, load_tokenizer

# Setup logging with FLUSH for live output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class TeeOutput:
    """Duplicate stdout/stderr writes to a log file AND the terminal simultaneously."""
    def __init__(self, filepath: str, original):
        self.original = original
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.logfile = open(filepath, 'a', encoding='utf-8', buffering=1)  # line-buffered

    def write(self, data: str):
        self.original.write(data)
        self.original.flush()
        self.logfile.write(data)
        self.logfile.flush()

    def flush(self):
        self.original.flush()
        self.logfile.flush()

    def isatty(self):
        return False

    def close(self):
        self.logfile.close()


def setup_file_logging(log_dir: str, name: str = 'train') -> str:
    """Redirect stdout and stderr to both terminal and log file."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{name}.log')
    err_path = os.path.join(log_dir, f'{name}_err.log')
    sys.stdout = TeeOutput(log_path, sys.__stdout__)
    sys.stderr = TeeOutput(err_path, sys.__stderr__)
    # Also add file handler to logger
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S'))
    logging.getLogger().addHandler(fh)
    return log_path


def flush_print(msg: str):
    """Print with immediate flush for live terminal output"""
    print(msg, flush=True)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Paths
    output_dir: str = './outputs'
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    
    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 50000
    warmup_steps: int = 2000
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    num_lr_restarts: int = 2
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    log_steps: int = 50
    
    # Mixed precision
    use_amp: bool = True
    
    # Reproducibility
    seed: int = 42
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TrainingConfig':
        training_cfg = d.get('training', {})
        return cls(**{k: v for k, v in training_cfg.items() if k in cls.__dataclass_fields__})


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_cuda():
    """Check CUDA availability and run test"""
    flush_print("=" * 60)
    flush_print("GPU AVAILABILITY CHECK")
    flush_print("=" * 60)
    
    if not torch.cuda.is_available():
        flush_print("ERROR: CUDA is NOT available!")
        flush_print("Training requires a GPU. Exiting.")
        sys.exit(1)
    
    flush_print(f"CUDA is available: {torch.cuda.is_available()}")
    flush_print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    flush_print(f"GPU Memory: {gpu_mem_gb:.2f} GB")
    flush_print(f"CUDA Version: {torch.version.cuda}")
    flush_print(f"PyTorch Version: {torch.__version__}")
    
    # Quick CUDA test
    flush_print("\nRunning CUDA matmul test...")
    try:
        a = torch.randn(100, 100, device='cuda')
        b = torch.randn(100, 100, device='cuda')
        c = torch.matmul(a, b)
        flush_print(f"CUDA matmul test passed! Result shape: {c.shape}")
        # Cleanup
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        flush_print(f"CUDA TEST FAILED: {e}")
        sys.exit(1)
    
    flush_print("=" * 60)
    return True


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    num_restarts: int = 2,
):
    """Cosine schedule with linear warmup and warm restarts.
    
    Divides training into `num_restarts + 1` cycles. Each cycle has a fresh
    cosine decay from peak LR, allowing the model to escape plateaus.
    The peak LR decreases by 0.7x each cycle for stability.
    min_lr_ratio sets the floor at 10% of base LR.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        post_warmup = current_step - num_warmup_steps
        total_post_warmup = num_training_steps - num_warmup_steps
        
        if num_restarts <= 0:
            # Standard cosine decay
            progress = float(post_warmup) / float(max(1, total_post_warmup))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        # Cosine with warm restarts
        cycle_length = total_post_warmup // (num_restarts + 1)
        cycle_idx = min(post_warmup // max(1, cycle_length), num_restarts)
        cycle_progress = (post_warmup - cycle_idx * cycle_length) / float(max(1, cycle_length))
        cycle_progress = min(cycle_progress, 1.0)
        
        # Each restart has a lower peak (0.7^cycle_idx)
        peak_ratio = 0.7 ** cycle_idx
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
        
        return max(min_lr_ratio, peak_ratio * cosine_val)
    
    return LambdaLR(optimizer, lr_lambda)


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing"""
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 3):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
        """
        vocab_size = logits.size(-1)
        
        # Flatten
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        
        # Create mask for non-padding tokens
        mask = (targets != self.ignore_index)
        
        # Compute smooth labels
        log_probs = F.log_softmax(logits, dim=-1)
        
        # One-hot with smoothing
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (vocab_size - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute loss
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        loss = (loss * mask.float()).sum() / mask.sum()
        
        return loss


def compute_rouge(predictions: list, references: list) -> Dict[str, float]:
    """Compute ROUGE scores"""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge_score not installed, returning dummy scores")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    r1_scores, r2_scores, rl_scores = [], [], []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_scores.append(scores['rouge1'].fmeasure)
        r2_scores.append(scores['rouge2'].fmeasure)
        rl_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': sum(r1_scores) / len(r1_scores) if r1_scores else 0.0,
        'rouge2': sum(r2_scores) / len(r2_scores) if r2_scores else 0.0,
        'rougeL': sum(rl_scores) / len(rl_scores) if rl_scores else 0.0,
    }


def save_checkpoint_atomic(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    step: int,
    best_rouge_l: float,
    config: dict,
    checkpoint_path: str,
    min_size_mb: float = 10.0,
):
    """Save checkpoint atomically (write temp then rename) with verification"""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_rouge_l': best_rouge_l,
        'config': config,
    }
    
    temp_path = checkpoint_path + '.tmp'
    torch.save(checkpoint, temp_path)
    
    # Atomic rename
    shutil.move(temp_path, checkpoint_path)
    
    # Verify file exists and has reasonable size
    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        abs_path = os.path.abspath(checkpoint_path)
        if size_mb >= min_size_mb:
            flush_print(f"[CHECKPOINT SAVED] {abs_path}")
            flush_print(f"  -> Size: {size_mb:.2f} MB | Step: {step} | Best ROUGE-L: {best_rouge_l:.4f}")
            return True
        else:
            flush_print(f"[WARNING] Checkpoint size too small: {size_mb:.2f} MB < {min_size_mb} MB expected")
            flush_print(f"  -> Path: {abs_path}")
            return False
    else:
        flush_print(f"[ERROR] Checkpoint NOT saved: {checkpoint_path}")
        return False


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the directory by step number"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_step_*.pt'))
    if not checkpoints:
        # Fallback: check for best_model.pt
        best = os.path.join(checkpoint_dir, 'best_model.pt')
        if os.path.exists(best):
            flush_print(f"  Found best_model.pt as fallback checkpoint")
            return best
        return None
    
    # Sort by step number
    def get_step(path):
        try:
            return int(os.path.basename(path).split('_')[-1].replace('.pt', ''))
        except:
            return 0
    
    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[GradScaler] = None,
    reset_optimizer: bool = False,
    boost_cross_gates: bool = True,
) -> Tuple[int, float]:
    """Load checkpoint and return step and best ROUGE-L.
    
    Args:
        reset_optimizer: If True, don't load optimizer/scheduler state (fresh LR restart)
        boost_cross_gates: If True, boost suppressed cross-attention gate values
    """
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    
    # Load model state (with flexibility for new/changed params)
    model_state = checkpoint['model_state_dict']
    current_state = model.state_dict()
    
    # Check for mismatched keys and handle gracefully
    missing = set(current_state.keys()) - set(model_state.keys())
    unexpected = set(model_state.keys()) - set(current_state.keys())
    if missing:
        flush_print(f"  New params (randomly initialized): {missing}")
    if unexpected:
        flush_print(f"  Removed params (ignored): {unexpected}")
    
    # Load compatible keys
    compatible_state = {k: v for k, v in model_state.items() if k in current_state and v.shape == current_state[k].shape}
    current_state.update(compatible_state)
    model.load_state_dict(current_state)
    
    # Boost cross-attention gates if they were suppressed
    if boost_cross_gates:
        with torch.no_grad():
            boosted = 0
            for name, param in model.named_parameters():
                if 'cross_gate' in name:
                    old_val = param.item()
                    old_sigmoid = torch.sigmoid(torch.tensor(old_val)).item()
                    # With new additive gating: (1 + sigmoid(gate))
                    # Set gate to 1.0 so (1 + sigmoid(1)) = 1.73x boost
                    new_val = 1.0
                    param.fill_(new_val)
                    new_sigmoid = torch.sigmoid(torch.tensor(new_val)).item()
                    flush_print(f"  Boosted {name}: {old_val:.4f} (old mult: {old_sigmoid:.3f}) -> {new_val:.4f} (new additive: 1+{new_sigmoid:.3f}={1+new_sigmoid:.3f}x)")
                    boosted += 1
            if boosted > 0:
                flush_print(f"  Boosted {boosted} cross-attention gates for stronger source conditioning")
    
    if not reset_optimizer:
        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                flush_print(f"  Could not load optimizer state: {e}. Starting fresh.")
        if scheduler is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                flush_print(f"  Could not load scheduler state: {e}. Starting fresh.")
        if scaler is not None:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception as e:
                flush_print(f"  Could not load scaler state: {e}. Starting fresh.")
    else:
        flush_print("  Optimizer/scheduler state RESET for fresh LR warm restart")
    
    return checkpoint.get('step', 0), checkpoint.get('best_rouge_l', 0.0)


@torch.no_grad()
def evaluate(
    model: MambaTransformerModel,
    val_loader,
    tokenizer,
    device: torch.device,
    max_samples: int = 200,
) -> Dict[str, float]:
    """Evaluate model and compute ROUGE scores.
    Uses greedy decoding for stable, fast metrics."""
    model.eval()
    
    predictions = []
    references = []
    total_loss = 0.0
    n_batches = 0
    
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx * val_loader.batch_size >= max_samples:
            break
        
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)
        
        # Compute loss (cross_entropy handles logits directly)
        with autocast('cuda', enabled=True):
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
                ignore_index=3,
            )
        
        total_loss += loss.item()
        n_batches += 1
        
        # Generate with greedy decoding (fast and stable for evaluation)
        generated = model.generate(
            src, src_mask,
            max_len=512,
            min_len=200,             # match reference lengths for better ROUGE recall
            greedy=True,             # greedy for stable metrics
            no_repeat_ngram_size=3,
            repetition_penalty=1.0,  # no penalty - let copy mechanism work freely
        )
        
        for i in range(src.size(0)):
            pred_ids = generated[i].cpu().tolist()
            ref_ids = tgt_output[i].cpu().tolist()
            
            # Decode
            pred_text = tokenizer.DecodeIds([t for t in pred_ids if t not in [0, 1, 2, 3]])
            ref_text = tokenizer.DecodeIds([t for t in ref_ids if t not in [0, 1, 2, 3]])
            
            predictions.append(pred_text)
            references.append(ref_text)
    
    # Compute ROUGE
    rouge_scores = compute_rouge(predictions, references)
    rouge_scores['val_loss'] = total_loss / max(n_batches, 1)
    
    # Log sample predictions for debugging ROUGE-L
    n_samples_to_show = min(3, len(predictions))
    if n_samples_to_show > 0:
        flush_print("\n--- Sample Predictions vs References ---")
        for i in range(n_samples_to_show):
            flush_print(f"  [Sample {i+1}]")
            flush_print(f"    Pred ({len(predictions[i].split())} words): {predictions[i][:300]}...")
            flush_print(f"    Ref  ({len(references[i].split())} words): {references[i][:300]}...")
        flush_print("--- End Samples ---\n")
    
    model.train()
    return rouge_scores


def train(
    model_config: MambaTransformerConfig,
    data_config: DataConfig,
    training_config: TrainingConfig,
    resume_from: Optional[str] = None,
):
    """Main training loop with checkpoint resumption support"""
    device = torch.device('cuda')
    
    # Create directories
    os.makedirs(training_config.output_dir, exist_ok=True)
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    os.makedirs(training_config.log_dir, exist_ok=True)
    
    # Set seed
    set_seed(training_config.seed)
    
    # Create model
    flush_print("Building model...")
    model = build_model(model_config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flush_print(f"Total parameters: {total_params:,}")
    flush_print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataloaders
    flush_print("Creating dataloaders...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        config=data_config,
        batch_size=training_config.batch_size,
        num_workers=0,  # For local compatibility
        max_train_samples=data_config.max_train_samples,
        max_val_samples=data_config.max_val_samples,
    )
    flush_print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    
    # Scheduler - cosine with warm restarts to escape plateaus
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=training_config.max_steps,
        num_restarts=getattr(training_config, 'num_lr_restarts', 2),
    )
    
    # Loss (NLL loss since model returns log_probs)
    pad_id = data_config.pad_id
    
    # Mixed precision
    scaler = GradScaler('cuda', enabled=training_config.use_amp)
    
    # Training state
    global_step = 0
    best_rouge_l = 0.0
    running_loss = 0.0
    
    # Auto-resume from latest checkpoint if exists
    if resume_from is None:
        resume_from = find_latest_checkpoint(training_config.checkpoint_dir)
    elif os.path.isdir(resume_from):
        resume_from = find_latest_checkpoint(resume_from)
    
    if resume_from and os.path.exists(resume_from):
        flush_print(f"\n{'='*60}")
        flush_print("RESUMING FROM CHECKPOINT")
        flush_print(f"{'='*60}")
        flush_print(f"Loading: {resume_from}")
        global_step, best_rouge_l = load_checkpoint(
            resume_from, model, optimizer, scheduler, scaler,
            reset_optimizer=True,    # Fresh LR for warm restart
            boost_cross_gates=False, # Additive gates already fixed
        )
        # Reset copy mechanism gate to balanced after loading
        if hasattr(model, 'copy_mechanism') and model.copy_mechanism is not None:
            nn.init.constant_(model.copy_mechanism.gate_linear.bias, 3.0)
            flush_print("  Copy mechanism gate set to mostly-generate (p_gen=0.953)")
        flush_print(f"Resumed at step {global_step} with best ROUGE-L: {best_rouge_l:.4f}")
        flush_print(f"Eval every: {training_config.eval_steps} steps")
        flush_print(f"{'='*60}\n")
    
    # Config for saving
    full_config = {
        'model': asdict(model_config),
        'data': asdict(data_config),
        'training': asdict(training_config),
    }
    
    flush_print("\n" + "=" * 60)
    flush_print("STARTING TRAINING")
    flush_print("=" * 60)
    flush_print(f"Max steps: {training_config.max_steps}")
    flush_print(f"Starting from step: {global_step}")
    flush_print(f"Batch size: {training_config.batch_size}")
    flush_print(f"Gradient accumulation: {training_config.gradient_accumulation_steps}")
    flush_print(f"Effective batch size: {training_config.batch_size * training_config.gradient_accumulation_steps}")
    flush_print(f"Log every: {training_config.log_steps} steps")
    flush_print(f"Eval every: {training_config.eval_steps} steps")
    flush_print(f"Save every: {training_config.save_steps} steps")
    flush_print("=" * 60 + "\n")
    
    model.train()
    optimizer.zero_grad()
    
    start_time = time.time()
    epoch = 0
    tokens_processed = 0
    
    # Calculate total batches needed for remaining steps
    remaining_steps = training_config.max_steps - global_step
    total_batches = remaining_steps * training_config.gradient_accumulation_steps
    
    # Create progress bar for batches
    pbar = tqdm(
        total=total_batches,
        desc=f"Training",
        unit="batch",
        dynamic_ncols=True,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Step {postfix}"
    )
    pbar.set_postfix_str(f"{global_step}/{training_config.max_steps}")
    
    batches_in_current_run = 0
    
    # Register state for emergency save on crash/signal
    if hasattr(__builtins__, '_emergency_state') or hasattr(getattr(__builtins__, '__class__', type(None)), '_emergency_state'):
        import builtins
        if hasattr(builtins, '_emergency_state'):
            builtins._emergency_state.update({
                'model': model, 'optimizer': optimizer, 'scheduler': scheduler,
                'scaler': scaler, 'step': global_step, 'best_rouge_l': best_rouge_l,
                'config': full_config, 'ckpt_dir': training_config.checkpoint_dir,
            })
    
    while global_step < training_config.max_steps:
        epoch += 1
        
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= training_config.max_steps:
                break
            
            # Move to device
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            src_mask = batch['src_mask'].to(device)
            tgt_mask = batch['tgt_mask'].to(device)
            
            # Track tokens for throughput
            batch_tokens = src.numel() + tgt_input.numel()
            tokens_processed += batch_tokens

            # Forward pass with AMP + OOM protection
            try:
                with autocast('cuda', enabled=training_config.use_amp):
                    logits = model(src, tgt_input, src_mask, tgt_mask)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        tgt_output.reshape(-1),
                        ignore_index=pad_id,
                    )
                    loss = loss / training_config.gradient_accumulation_steps
                
                # Check for NaN loss - skip batch if NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    flush_print(f"WARNING: NaN/Inf loss at batch {batch_idx}, skipping...")
                    optimizer.zero_grad()
                    continue
                
                # Backward pass
                scaler.scale(loss).backward()
            except RuntimeError as oom_err:
                if 'out of memory' in str(oom_err):
                    flush_print(f"WARNING: OOM at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise

            running_loss += loss.item()
            
            # Update progress bar for each batch
            batches_in_current_run += 1
            pbar.update(1)
            gpu_mem_current = torch.cuda.memory_allocated() / 1e9
            pbar.set_postfix_str(f"{global_step}/{training_config.max_steps} | GPU: {gpu_mem_current:.1f}GB")
            
            # Gradient accumulation
            if (batch_idx + 1) % training_config.gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                
                # Check for NaN gradients - skip update if NaN
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    flush_print(f"WARNING: NaN/Inf gradient detected at step {global_step}, skipping update...")
                    optimizer.zero_grad()
                    scaler.update()
                    continue
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Scheduler step AFTER optimizer step
                scheduler.step()
                
                global_step += 1
                # Update emergency state step counter
                import builtins
                if hasattr(builtins, '_emergency_state') and builtins._emergency_state.get('model'):
                    builtins._emergency_state['step'] = global_step
                    builtins._emergency_state['best_rouge_l'] = best_rouge_l
                pbar.set_postfix_str(f"{global_step}/{training_config.max_steps} | GPU: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
                
                # Logging with FLUSH
                if global_step % training_config.log_steps == 0:
                    # running_loss accumulates loss.item() (= criterion_loss / grad_accum) per mini-batch
                    # Per step: sum of grad_accum values of (loss/grad_accum) = average loss from that step
                    # Over log_steps: avg_loss = running_loss / log_steps = true per-token loss
                    avg_loss = running_loss / training_config.log_steps
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / max(elapsed, 1)
                    tokens_per_sec = tokens_processed / max(elapsed, 1)
                    gpu_mem = torch.cuda.max_memory_allocated() / 1e9
                    
                    flush_print(
                        f"Step {global_step:5d}/{training_config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"GradNorm: {grad_norm:.3f} | "
                        f"LR: {lr:.2e} | "
                        f"tok/s: {tokens_per_sec:.0f} | "
                        f"GPU: {gpu_mem:.1f}GB"
                    )
                    running_loss = 0.0
                
                # Check for manual save trigger file
                trigger_path = os.path.join(training_config.checkpoint_dir, '..', '..', 'SAVE_NOW')
                if os.path.exists(trigger_path):
                    flush_print(f"[MANUAL SAVE] Trigger file detected, saving checkpoint...")
                    manual_path = os.path.join(
                        training_config.checkpoint_dir,
                        f'manual_checkpoint_step_{global_step}.pt'
                    )
                    save_checkpoint_atomic(
                        model, optimizer, scheduler, scaler,
                        global_step, best_rouge_l, full_config, manual_path
                    )
                    os.remove(trigger_path)
                    flush_print(f"[MANUAL SAVE] Trigger file removed. Training continues.")
                
                # Evaluation
                if global_step % training_config.eval_steps == 0:
                    flush_print("\n" + "-" * 60)
                    flush_print(f"EVALUATION at Step {global_step}")
                    flush_print("-" * 60)
                    eval_results = evaluate(model, val_loader, tokenizer, device, max_samples=100)
                    
                    flush_print(
                        f"Val Loss: {eval_results['val_loss']:.4f} | "
                        f"ROUGE-1: {eval_results['rouge1']:.4f} | "
                        f"ROUGE-2: {eval_results['rouge2']:.4f} | "
                        f"ROUGE-L: {eval_results['rougeL']:.4f}"
                    )
                    
                    # Save best model
                    if eval_results['rougeL'] > best_rouge_l:
                        best_rouge_l = eval_results['rougeL']
                        best_model_path = os.path.join(training_config.checkpoint_dir, 'best_model.pt')
                        flush_print(f"[NEW BEST] ROUGE-L improved to {best_rouge_l:.4f}")
                        save_checkpoint_atomic(
                            model, optimizer, scheduler, scaler,
                            global_step, best_rouge_l, full_config, best_model_path
                        )
                    flush_print("-" * 60 + "\n")
                
                # Save checkpoint
                if global_step % training_config.save_steps == 0:
                    checkpoint_path = os.path.join(
                        training_config.checkpoint_dir,
                        f'checkpoint_step_{global_step}.pt'
                    )
                    save_checkpoint_atomic(
                        model, optimizer, scheduler, scaler,
                        global_step, best_rouge_l, full_config, checkpoint_path
                    )
    
    # Close progress bar
    pbar.close()
    
    # Final save
    flush_print("\n" + "=" * 60)
    flush_print("SAVING FINAL MODEL")
    flush_print("=" * 60)
    final_model_path = os.path.join(training_config.checkpoint_dir, 'final_model.pt')
    save_checkpoint_atomic(
        model, optimizer, scheduler, scaler,
        global_step, best_rouge_l, full_config, final_model_path
    )
    
    flush_print("\n" + "=" * 60)
    flush_print("TRAINING COMPLETE")
    flush_print("=" * 60)
    flush_print(f"Total steps completed: {global_step}")
    flush_print(f"Best ROUGE-L achieved: {best_rouge_l:.4f}")
    
    # Verify all saved files
    flush_print("\n--- SAVED FILES VERIFICATION ---")
    
    best_model_path = os.path.join(training_config.checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
        flush_print(f"[OK] Best model: {os.path.abspath(best_model_path)} ({size_mb:.2f} MB)")
    else:
        flush_print("[WARNING] Best model NOT found!")
    
    if os.path.exists(final_model_path):
        size_mb = os.path.getsize(final_model_path) / (1024 * 1024)
        flush_print(f"[OK] Final model: {os.path.abspath(final_model_path)} ({size_mb:.2f} MB)")
    else:
        flush_print("[WARNING] Final model NOT found!")
    
    # List all checkpoints
    checkpoints = glob.glob(os.path.join(training_config.checkpoint_dir, 'checkpoint_step_*.pt'))
    if checkpoints:
        flush_print(f"[OK] Found {len(checkpoints)} intermediate checkpoints")
        latest = find_latest_checkpoint(training_config.checkpoint_dir)
        if latest:
            size_mb = os.path.getsize(latest) / (1024 * 1024)
            flush_print(f"     Latest: {os.path.abspath(latest)} ({size_mb:.2f} MB)")
    
    flush_print("=" * 60)
    
    return best_rouge_l


def main():
    parser = argparse.ArgumentParser(description='Train Mamba-Transformer Hybrid Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume (auto-detects if not provided)')
    args = parser.parse_args()
    
    flush_print("\n" + "=" * 60)
    flush_print("MAMBA-TRANSFORMER CLINICAL SUMMARIZATION TRAINING")
    flush_print("=" * 60)
    
    # Check CUDA
    check_cuda()
    
    # Load config
    flush_print(f"\nLoading config from: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Create configs
    model_config = MambaTransformerConfig.from_dict(config.get('model', {}))
    data_config = DataConfig.from_dict(config)
    training_config = TrainingConfig.from_dict(config)
    
    flush_print(f"Output dir: {training_config.output_dir}")
    flush_print(f"Checkpoint dir: {training_config.checkpoint_dir}")
    flush_print(f"Log dir: {training_config.log_dir}")

    # Redirect stdout/stderr to log file
    log_path = setup_file_logging(training_config.log_dir)
    flush_print(f"\nLogging to: {os.path.abspath(log_path)}")

    # Train
    train(model_config, data_config, training_config, resume_from=args.resume)


if __name__ == '__main__':
    import signal, traceback, atexit
    
    crash_log = open('crash_report.log', 'w')
    
    # Global refs for emergency save (populated by train())
    _emergency_state = {}
    
    def emergency_save(reason='unknown'):
        """Save checkpoint on crash/signal/exit"""
        state = _emergency_state
        if not state.get('model'):
            crash_log.write(f'Emergency save skipped: no model loaded yet\n')
            return
        try:
            import torch, os, shutil
            ckpt_dir = state.get('ckpt_dir', 'checkpoints/v2_run')
            path = os.path.join(ckpt_dir, f'emergency_step_{state.get("step", 0)}.pt')
            torch.save({
                'step': state.get('step', 0),
                'model_state_dict': state['model'].state_dict(),
                'optimizer_state_dict': state.get('optimizer', {}).state_dict() if hasattr(state.get('optimizer', {}), 'state_dict') else {},
                'scheduler_state_dict': state.get('scheduler', {}).state_dict() if hasattr(state.get('scheduler', {}), 'state_dict') else {},
                'scaler_state_dict': state.get('scaler', {}).state_dict() if hasattr(state.get('scaler', {}), 'state_dict') else {},
                'best_rouge_l': state.get('best_rouge_l', 0.0),
                'config': state.get('config', {}),
            }, path)
            crash_log.write(f'EMERGENCY SAVE: {path} ({os.path.getsize(path)/1e6:.1f}MB) reason={reason}\n')
        except Exception as save_err:
            crash_log.write(f'EMERGENCY SAVE FAILED: {save_err}\n')
    
    def crash_handler(sig, frame):
        crash_log.write(f'SIGNAL {sig} received!\n')
        crash_log.write(traceback.format_stack(frame).__str__() + '\n')
        crash_log.flush()
        emergency_save(reason=f'signal_{sig}')
        crash_log.flush()
        crash_log.close()
        
    for sig in [signal.SIGTERM, signal.SIGABRT, signal.SIGBREAK]:
        try:
            signal.signal(sig, crash_handler)
        except (OSError, ValueError):
            pass
    
    def at_exit():
        crash_log.write('ATEXIT called\n')
        import torch
        if torch.cuda.is_available():
            crash_log.write(f'GPU mem at exit: {torch.cuda.memory_allocated()/1e9:.2f}GB\n')
        emergency_save(reason='atexit')
        crash_log.flush()
        crash_log.close()
    atexit.register(at_exit)
    
    # Make _emergency_state accessible to train()
    import builtins
    builtins._emergency_state = _emergency_state
    
    try:
        main()
    except Exception as e:
        crash_log.write(f'EXCEPTION: {type(e).__name__}: {e}\n')
        crash_log.write(traceback.format_exc() + '\n')
        crash_log.flush()
        emergency_save(reason=f'exception_{type(e).__name__}')
        crash_log.close()
        raise
