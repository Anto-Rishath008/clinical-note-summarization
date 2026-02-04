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
"""

import os
import sys
import time
import math
import yaml
import shutil
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from model import MambaTransformerConfig, MambaTransformerModel, build_model
from data_loader import DataConfig, create_dataloaders, load_tokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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
    label_smoothing: float = 0.1
    
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
    print("=" * 60)
    print("GPU AVAILABILITY CHECK")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is NOT available!")
        print("Training requires a GPU. Exiting.")
        sys.exit(1)
    
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Quick CUDA test
    print("\nRunning CUDA matmul test...")
    a = torch.randn(100, 100, device='cuda')
    b = torch.randn(100, 100, device='cuda')
    c = torch.matmul(a, b)
    print(f"CUDA matmul test passed! Result shape: {c.shape}")
    print("=" * 60)
    
    return True


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine schedule with linear warmup"""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
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
):
    """Save checkpoint atomically (write temp then rename)"""
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
        if size_mb > 50:
            logger.info(f"Checkpoint saved: {checkpoint_path} ({size_mb:.1f} MB)")
            return True
        else:
            logger.warning(f"Checkpoint size too small: {size_mb:.1f} MB")
            return False
    return False


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[GradScaler] = None,
) -> Tuple[int, float]:
    """Load checkpoint and return step and best ROUGE-L"""
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint.get('step', 0), checkpoint.get('best_rouge_l', 0.0)


@torch.no_grad()
def evaluate(
    model: MambaTransformerModel,
    val_loader,
    tokenizer,
    device: torch.device,
    max_samples: int = 100,
) -> Dict[str, float]:
    """Evaluate model and compute ROUGE scores"""
    model.eval()
    
    predictions = []
    references = []
    total_loss = 0.0
    n_batches = 0
    
    criterion = LabelSmoothingLoss(smoothing=0.1, ignore_index=3)
    
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx * val_loader.batch_size >= max_samples:
            break
        
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)
        
        # Compute loss
        with autocast(enabled=True):
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(logits, tgt_output)
        
        total_loss += loss.item()
        n_batches += 1
        
        # Generate predictions
        generated = model.generate(src, src_mask, max_len=256)
        
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
    
    model.train()
    return rouge_scores


def train(
    model_config: MambaTransformerConfig,
    data_config: DataConfig,
    training_config: TrainingConfig,
):
    """Main training loop"""
    device = torch.device('cuda')
    
    # Create directories
    os.makedirs(training_config.output_dir, exist_ok=True)
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    os.makedirs(training_config.log_dir, exist_ok=True)
    
    # Set seed
    set_seed(training_config.seed)
    
    # Create model
    logger.info("Building model...")
    model = build_model(model_config)
    model = model.to(device)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        config=data_config,
        batch_size=training_config.batch_size,
        num_workers=0,  # For Kaggle compatibility
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    
    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=training_config.max_steps,
    )
    
    # Loss
    criterion = LabelSmoothingLoss(
        smoothing=training_config.label_smoothing,
        ignore_index=data_config.pad_id,
    )
    
    # Mixed precision
    scaler = GradScaler(enabled=training_config.use_amp)
    
    # Training state
    global_step = 0
    best_rouge_l = 0.0
    running_loss = 0.0
    
    # Config for saving
    full_config = {
        'model': asdict(model_config),
        'data': asdict(data_config),
        'training': asdict(training_config),
    }
    
    logger.info("Starting training...")
    logger.info(f"Max steps: {training_config.max_steps}")
    logger.info(f"Batch size: {training_config.batch_size}")
    logger.info(f"Gradient accumulation: {training_config.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {training_config.batch_size * training_config.gradient_accumulation_steps}")
    
    model.train()
    optimizer.zero_grad()
    
    start_time = time.time()
    epoch = 0
    
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
            
            # Forward pass with AMP
            with autocast(enabled=training_config.use_amp):
                logits = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(logits, tgt_output)
                loss = loss / training_config.gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            running_loss += loss.item() * training_config.gradient_accumulation_steps
            
            # Gradient accumulation
            if (batch_idx + 1) % training_config.gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Logging
                if global_step % training_config.log_steps == 0:
                    avg_loss = running_loss / training_config.log_steps
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed
                    
                    logger.info(
                        f"Step {global_step}/{training_config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Grad Norm: {grad_norm:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Steps/s: {steps_per_sec:.2f}"
                    )
                    running_loss = 0.0
                
                # Evaluation
                if global_step % training_config.eval_steps == 0:
                    logger.info("Running evaluation...")
                    eval_results = evaluate(model, val_loader, tokenizer, device, max_samples=100)
                    
                    logger.info(
                        f"Eval Step {global_step} | "
                        f"Val Loss: {eval_results['val_loss']:.4f} | "
                        f"ROUGE-1: {eval_results['rouge1']:.4f} | "
                        f"ROUGE-2: {eval_results['rouge2']:.4f} | "
                        f"ROUGE-L: {eval_results['rougeL']:.4f}"
                    )
                    
                    # Save best model
                    if eval_results['rougeL'] > best_rouge_l:
                        best_rouge_l = eval_results['rougeL']
                        best_model_path = os.path.join(training_config.checkpoint_dir, 'best_model.pt')
                        save_checkpoint_atomic(
                            model, optimizer, scheduler, scaler,
                            global_step, best_rouge_l, full_config, best_model_path
                        )
                        logger.info(f"New best model saved with ROUGE-L: {best_rouge_l:.4f}")
                
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
    
    # Final save
    final_model_path = os.path.join(training_config.checkpoint_dir, 'final_model.pt')
    save_checkpoint_atomic(
        model, optimizer, scheduler, scaler,
        global_step, best_rouge_l, full_config, final_model_path
    )
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total steps: {global_step}")
    logger.info(f"Best ROUGE-L: {best_rouge_l:.4f}")
    
    # Verify best model exists
    best_model_path = os.path.join(training_config.checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
        logger.info(f"Best model path: {os.path.abspath(best_model_path)}")
        logger.info(f"Best model size: {size_mb:.1f} MB")
    else:
        logger.warning("Best model not found!")
    
    return best_rouge_l


def main():
    parser = argparse.ArgumentParser(description='Train Mamba-Transformer Hybrid Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    args = parser.parse_args()
    
    # Check CUDA
    check_cuda()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create configs
    model_config = MambaTransformerConfig.from_dict(config.get('model', {}))
    data_config = DataConfig.from_dict(config)
    training_config = TrainingConfig.from_dict(config)
    
    # Train
    train(model_config, data_config, training_config)


if __name__ == '__main__':
    main()
