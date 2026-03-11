"""
Training Script for Simplified LongT5 Model
============================================

This training script implements all Issue #1 requirements:

✅ LongT5-inspired architecture (from src.longt5_model)
✅ Streaming data loading (OOM fix)
✅ Device-agnostic GradScaler (CPU/GPU compatible)
✅ Dynamic num_workers for CPU utilization
✅ Multi-GPU support with DataParallel
✅ Restart-safe checkpointing with RNG states
✅ Comprehensive logging and metrics tracking

Usage:
    python train_longt5.py --config configs/longt5_config.yaml
    python train_longt5.py --config configs/longt5_config.yaml --resume checkpoints/latest.pt
"""

import os
import sys
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import json

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.longt5_model import SimplifiedLongT5, LongT5Config, create_model
from src.data_loader import (
    create_train_dataloader,
    create_val_dataloader,
    load_tokenizer,
    get_num_workers,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    return device


def get_rng_states() -> Dict[str, Any]:
    """
    Get all RNG states for restart-safe checkpointing.
    
    Solves Issue #1: Robust restart-safe checkpointing
    """
    rng_states = {
        'python_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        rng_states['cuda_rng'] = torch.cuda.get_rng_state_all()
    
    return rng_states


def set_rng_states(rng_states: Dict[str, Any]) -> None:
    """Restore RNG states from checkpoint"""
    random.setstate(rng_states['python_rng'])
    np.random.set_state(rng_states['numpy_rng'])
    torch.set_rng_state(rng_states['torch_rng'])
    
    if 'cuda_rng' in rng_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_states['cuda_rng'])


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[GradScaler],
    epoch: int,
    global_step: int,
    best_metric: float,
    config: dict,
    metrics_history: list,
) -> None:
    """
    Save checkpoint with all states for perfect resumption.
    
    Solves Issue #1: Restart-safe checkpointing with RNG states
    """
    # Handle DataParallel
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'global_step': global_step,
        'best_metric': best_metric,
        'config': config,
        'metrics_history': metrics_history,
        'rng_states': get_rng_states(),
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save atomically
    temp_path = path + '.tmp'
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)
    
    logger.info(f"Saved checkpoint to {path} (step {global_step}, epoch {epoch})")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[GradScaler] = None,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, Any]:
    """
    Load checkpoint and restore all states.
    
    Returns metadata about the checkpoint.
    """
    logger.info(f"Loading checkpoint from {path}")
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Handle DataParallel
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Restore RNG states
    if 'rng_states' in checkpoint:
        set_rng_states(checkpoint['rng_states'])
        logger.info("Restored RNG states for perfect resumption")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'best_metric': checkpoint.get('best_metric', float('inf')),
        'metrics_history': checkpoint.get('metrics_history', []),
    }


def setup_model(config: dict, device: torch.device) -> nn.Module:
    """
    Create and setup model with multi-GPU support.
    
    Solves Issue #1: Auto Multi-GPU support with DataParallel
    """
    model = create_model(config)
    model = model.to(device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model


def setup_optimizer(model: nn.Module, config: dict) -> AdamW:
    """Create optimizer with weight decay"""
    training_cfg = config.get('training', {})
    
    lr = training_cfg.get('learning_rate', 1e-4)
    weight_decay = training_cfg.get('weight_decay', 0.01)
    
    # Separate parameters with and without weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    return optimizer


def setup_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    num_training_steps: int,
) -> Any:
    """Create learning rate scheduler"""
    training_cfg = config.get('training', {})
    
    scheduler_type = training_cfg.get('scheduler', 'cosine')
    warmup_steps = training_cfg.get('warmup_steps', 1000)
    
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_training_steps // 4,
            T_mult=2,
        )
    elif scheduler_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=training_cfg.get('learning_rate', 1e-4),
            total_steps=num_training_steps,
            pct_start=warmup_steps / num_training_steps,
            anneal_strategy='cos',
        )
    else:
        # Linear warmup with cosine decay (default)
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
    
    return scheduler


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[GradScaler],
    device: torch.device,
    config: dict,
    epoch: int,
    global_step: int,
    metrics_writer,
) -> Tuple[float, int]:
    """
    Train for one epoch.
    
    Returns:
        tuple: (average loss, updated global step)
    """
    model.train()
    training_cfg = config.get('training', {})
    
    gradient_accumulation_steps = training_cfg.get('gradient_accumulation_steps', 1)
    max_grad_norm = training_cfg.get('max_grad_norm', 1.0)
    log_interval = training_cfg.get('log_interval', 100)
    use_fp16 = training_cfg.get('fp16', True) and device.type == 'cuda'
    
    total_loss = 0.0
    num_batches = 0
    accumulated_loss = 0.0
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if use_fp16 and scaler is not None:
            with autocast():
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    decoder_attention_mask=batch['decoder_attention_mask'],
                )
                loss = outputs['loss']
                
                # Handle DataParallel
                if hasattr(model, 'module'):
                    loss = loss.mean()
                
                loss = loss / gradient_accumulation_steps
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
        else:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                decoder_attention_mask=batch['decoder_attention_mask'],
            )
            loss = outputs['loss']
            
            if hasattr(model, 'module'):
                loss = loss.mean()
            
            loss = loss / gradient_accumulation_steps
            loss.backward()
        
        accumulated_loss += loss.item() * gradient_accumulation_steps
        
        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_fp16 and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            total_loss += accumulated_loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{accumulated_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            })
            
            # Log metrics
            if global_step % log_interval == 0:
                metrics_writer.log_step(global_step, {
                    'train_loss': accumulated_loss,
                    'learning_rate': scheduler.get_last_lr()[0],
                })
            
            accumulated_loss = 0.0
    
    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    device: torch.device,
    config: dict,
) -> Dict[str, float]:
    """
    Run validation and compute metrics.
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    training_cfg = config.get('training', {})
    use_fp16 = training_cfg.get('fp16', True) and device.type == 'cuda'
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if use_fp16:
            with autocast():
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    decoder_attention_mask=batch['decoder_attention_mask'],
                )
                loss = outputs['loss']
        else:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                decoder_attention_mask=batch['decoder_attention_mask'],
            )
            loss = outputs['loss']
        
        if hasattr(model, 'module'):
            loss = loss.mean()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(1, num_batches)
    perplexity = np.exp(avg_loss)
    
    return {
        'val_loss': avg_loss,
        'val_perplexity': perplexity,
    }


class MetricsWriter:
    """Simple metrics logger to CSV"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_path = self.log_dir / 'metrics.csv'
        self.header_written = self.metrics_path.exists()
    
    def log_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Log metrics for a step"""
        metrics['step'] = step
        metrics['timestamp'] = datetime.now().isoformat()
        
        with open(self.metrics_path, 'a') as f:
            if not self.header_written:
                f.write(','.join(metrics.keys()) + '\n')
                self.header_written = True
            f.write(','.join(str(v) for v in metrics.values()) + '\n')
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for an epoch"""
        metrics['epoch'] = epoch
        self.log_step(metrics.get('step', 0), metrics)


def main():
    parser = argparse.ArgumentParser(description='Train Simplified LongT5 Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-streaming', action='store_true', help='Disable streaming data loading')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    
    # Paths
    training_cfg = config.get('training', {})
    checkpoint_dir = Path(training_cfg.get('checkpoint_dir', 'artifacts/checkpoints/longt5'))
    log_dir = Path(training_cfg.get('log_dir', 'artifacts/logs/longt5'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(checkpoint_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Load tokenizer
    data_cfg = config.get('data', {})
    tokenizer_path = data_cfg.get('tokenizer_path', 'artifacts/tokenizer/spm.model')
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Update vocab size in config if needed
    if 'vocab_size' not in data_cfg:
        config['data']['vocab_size'] = tokenizer.get_piece_size()
    
    # Create data loaders
    logger.info("Creating data loaders...")
    use_streaming = not args.no_streaming
    train_loader = create_train_dataloader(config, tokenizer, streaming=use_streaming)
    val_loader = create_val_dataloader(config, tokenizer)
    
    # Create model
    logger.info("Creating model...")
    model = setup_model(config, device)
    
    # Create optimizer
    optimizer = setup_optimizer(model, config)
    
    # Estimate training steps
    num_epochs = training_cfg.get('num_epochs', 10)
    steps_per_epoch = len(train_loader) if hasattr(train_loader, '__len__') else training_cfg.get('steps_per_epoch', 1000)
    num_training_steps = num_epochs * steps_per_epoch
    
    # Create scheduler
    scheduler = setup_scheduler(optimizer, config, num_training_steps)
    
    # Create GradScaler (device-agnostic)
    # Solves Issue #1: Fix GradScaler hardcoded to CUDA
    use_fp16 = training_cfg.get('fp16', True) and device.type == 'cuda'
    scaler = GradScaler(enabled=use_fp16) if use_fp16 else None
    
    # Initialize training state
    start_epoch = 0
    global_step = 0
    best_metric = float('inf')
    metrics_history = []
    
    # Resume from checkpoint if specified
    if args.resume:
        resume_info = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )
        start_epoch = resume_info['epoch'] + 1
        global_step = resume_info['global_step']
        best_metric = resume_info['best_metric']
        metrics_history = resume_info['metrics_history']
        logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Metrics writer
    metrics_writer = MetricsWriter(str(log_dir))
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    save_interval = training_cfg.get('save_interval', 500)
    eval_interval = training_cfg.get('eval_interval', 1000)
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*50}")
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, config, epoch, global_step, metrics_writer
        )
        
        logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, device, config)
        logger.info(f"Epoch {epoch + 1} - Val Loss: {val_metrics['val_loss']:.4f}, "
                   f"Val Perplexity: {val_metrics['val_perplexity']:.2f}")
        
        # Log metrics
        metrics_writer.log_epoch(epoch, {
            'train_loss': train_loss,
            **val_metrics,
            'step': global_step,
        })
        
        metrics_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics,
        })
        
        # Save checkpoint
        save_checkpoint(
            str(checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'),
            model, optimizer, scheduler, scaler,
            epoch, global_step, best_metric, config, metrics_history
        )
        
        # Save best model
        if val_metrics['val_loss'] < best_metric:
            best_metric = val_metrics['val_loss']
            save_checkpoint(
                str(checkpoint_dir / 'best_model.pt'),
                model, optimizer, scheduler, scaler,
                epoch, global_step, best_metric, config, metrics_history
            )
            logger.info(f"New best model saved! Val Loss: {best_metric:.4f}")
        
        # Save latest checkpoint
        save_checkpoint(
            str(checkpoint_dir / 'latest.pt'),
            model, optimizer, scheduler, scaler,
            epoch, global_step, best_metric, config, metrics_history
        )
    
    logger.info("\nTraining complete!")
    logger.info(f"Best validation loss: {best_metric:.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
