"""
Training Script for Pointer-Generator Model
Implements custom training loop with FP16, gradient accumulation, checkpointing
Kaggle-ready: supports custom data paths and auto GPU detection
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import argparse
import yaml
from pathlib import Path
import time
import pandas as pd
from tqdm import tqdm
import sys
import os

# Force unbuffered output for real-time logging
os.environ['PYTHONUNBUFFERED'] = '1'

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core import PointerGeneratorModel, get_dataloader, RougeMetric
import sentencepiece as spm


def print_flush(msg):
    """Print with immediate flush for logging"""
    print(msg, flush=True)


def print_gpu_info():
    """Print GPU information for Kaggle debugging"""
    if torch.cuda.is_available():
        print(f"\n{'='*60}")
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"{'='*60}\n")
    else:
        print("\nWARNING: No GPU detected! Training will be slow.\n")


def load_config(config_path):
    """Load YAML config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, scaler, step, epoch, best_rouge, config, checkpoint_path):
    """Save training checkpoint"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_rouge': best_rouge,
        'config': config
    }, checkpoint_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scaler):
    """Load checkpoint and resume training"""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    step = checkpoint['step']
    epoch = checkpoint['epoch']
    best_rouge = checkpoint['best_rouge']
    
    print(f"Resumed from checkpoint: step={step}, epoch={epoch}, best_rouge_L={best_rouge:.4f}")
    
    return step, epoch, best_rouge


def get_lr_schedule(optimizer, step, warmup_steps, total_steps, initial_lr):
    """Linear warmup + linear decay"""
    if step < warmup_steps:
        # Linear warmup
        lr = initial_lr * (step / warmup_steps)
    else:
        # Linear decay
        decay_steps = total_steps - warmup_steps
        lr = initial_lr * (1 - (step - warmup_steps) / decay_steps)
        lr = max(lr, 0)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def evaluate_model(model, dataloader, tokenizer, device, max_eval_batches=100):
    """
    Evaluate model on validation set.
    
    Returns:
        rouge_scores: dict with rouge1, rouge2, rougeL
    """
    model.eval()
    
    metric = RougeMetric()
    predictions = []
    references = []
    
    print("Generating predictions for evaluation...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=min(len(dataloader), max_eval_batches))):
            if i >= max_eval_batches:
                break
            
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            src_lens = batch['src_lens'].to(device)
            
            # Beam search requires batch_size=1, so process examples one by one
            batch_size = src_ids.size(0)
            for b in range(batch_size):
                # Generate (beam search) for single example
                generated_ids, _ = model.generate(
                    src_ids[b:b+1], src_lens[b:b+1],
                    beam_size=4,
                    max_length=384,
                    min_length=50
                )
                
                # Decode prediction and reference
                gen_ids = generated_ids[0].cpu().tolist()
                ref_ids = tgt_ids[b].cpu().tolist()
                
                # Filter special tokens
                gen_ids = [t for t in gen_ids if t not in [model.bos_id, model.eos_id, model.pad_id]]
                ref_ids = [t for t in ref_ids if t not in [model.bos_id, model.eos_id, model.pad_id]]
                
                # Decode
                pred_text = tokenizer.decode(gen_ids)
                ref_text = tokenizer.decode(ref_ids)
                
                predictions.append(pred_text)
                references.append(ref_text)
    
    # Compute ROUGE
    rouge_scores = metric.compute(predictions, references)
    formatted = {k: f"{v:.4f}" for k, v in rouge_scores.items()}
    
    model.train()
    
    return rouge_scores


def train(config, tokenized_dir, run_name, resume_ckpt=None, max_steps_override=None, data_path=None):
    """
    Main training loop - Kaggle compatible
    
    Args:
        config: Configuration dict
        tokenized_dir: Path to tokenized data (or None if using data_path)
        run_name: Name for this run
        resume_ckpt: Optional checkpoint to resume from
        max_steps_override: Override max_steps from config
        data_path: If provided, expects raw CSV and will stream/tokenize on-the-fly
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print_gpu_info()
    
    # Override max_steps if provided
    if max_steps_override:
        config['training']['max_steps'] = max_steps_override
    
    # Set random seed
    torch.manual_seed(config['training']['seed'])
    
    # Load tokenizer
    tokenizer_path = Path(config['paths']['tokenizer_dir']) / 'spm.model'
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(tokenizer_path))
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_path = Path(tokenized_dir) / 'train.parquet'
    val_path = Path(tokenized_dir) / 'val.parquet'
    
    train_loader = get_dataloader(
        str(train_path),
        batch_size=config['training']['batch_size'],
        shuffle=True,
        max_src_len=config['model']['chunk_len'] * config['model']['num_chunks'],
        max_tgt_len=config['model']['max_target_len'],
        pad_id=config['data']['pad_id']
    )
    
    val_loader = get_dataloader(
        str(val_path),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        max_src_len=config['model']['chunk_len'] * config['model']['num_chunks'],
        max_tgt_len=config['model']['max_target_len'],
        pad_id=config['data']['pad_id']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = PointerGeneratorModel(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['optimizer']['betas'],
        eps=config['optimizer']['eps'],
        weight_decay=config['optimizer']['weight_decay']
    )
    
    # Mixed precision scaler
    scaler = GradScaler('cuda', enabled=config['training']['fp16'])
    
    # Resume from checkpoint if provided
    start_step = 0
    start_epoch = 0
    best_rouge = 0.0
    
    if resume_ckpt:
        print(f"Loading checkpoint from {resume_ckpt}...")
        start_step, start_epoch, best_rouge = load_checkpoint(resume_ckpt, model, optimizer, scaler)
    
    # Create output directories
    checkpoint_dir = Path(config['paths']['checkpoint_dir']) / run_name
    log_dir = Path(config['paths']['log_dir']) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    global_step = start_step
    epoch = start_epoch
    grad_accum_steps = config['training']['grad_accum']
    max_steps = config['training']['max_steps']
    
    # Metrics logging
    metrics_log = []
    
    # Training
    model.train()
    optimizer.zero_grad()
    
    patience_counter = 0
    
    # Loss tracking for stuck detection
    initial_loss = None
    loss_history = []
    stuck_check_interval = 500  # Check every 500 steps
    min_loss_reduction_pct = 5.0  # Expect at least 5% reduction
    
    # Expected random loss (log vocab size)
    import math
    expected_random_loss = math.log(config['data']['vocab_size'])
    print(f"Expected random loss (log vocab): {expected_random_loss:.4f}")
    
    while global_step < max_steps:
        epoch += 1
        print(f"\nEpoch {epoch}")
        
        epoch_loss = 0
        epoch_steps = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            src_lens = batch['src_lens'].to(device)
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=config['training']['fp16']):
                outputs, attentions, coverages, p_gens = model(src_ids, src_lens, tgt_ids)
                
                # Compute loss
                # NOTE: label_smoothing default changed to 0.0 because the manual
                # implementation for pointer-gen was broken. Only use label smoothing
                # when pointer_gen=False (where cross_entropy handles it correctly)
                total_loss, nll_loss, coverage_loss = model.compute_loss(
                    outputs, tgt_ids, attentions, coverages,
                    coverage_lambda=config['model'].get('coverage_lambda', 1.0),
                    label_smoothing=config['training'].get('label_smoothing', 0.0)
                )
                
                # Scale loss for gradient accumulation
                total_loss = total_loss / grad_accum_steps
            
            # Backward pass
            scaler.scale(total_loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Clip gradients
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad'])
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update learning rate
                lr = get_lr_schedule(
                    optimizer, global_step,
                    config['training']['warmup_steps'],
                    config['training']['max_steps'],
                    config['training']['learning_rate']
                )
                
                global_step += 1
                epoch_loss += total_loss.item() * grad_accum_steps
                epoch_steps += 1
                
                # Track loss for stuck detection
                current_loss = total_loss.item() * grad_accum_steps
                loss_history.append(current_loss)
                if initial_loss is None:
                    initial_loss = current_loss
                
                # Logging
                if global_step % config['training']['log_every'] == 0:
                    avg_loss = epoch_loss / epoch_steps
                    print_flush(f"Step {global_step}/{max_steps} | Loss: {avg_loss:.4f} | Grad: {grad_norm:.4f} | LR: {lr:.6f}")
                
                # Loss stuck detection
                if global_step == stuck_check_interval and initial_loss is not None:
                    recent_avg = sum(loss_history[-50:]) / min(50, len(loss_history))
                    loss_reduction_pct = (initial_loss - recent_avg) / initial_loss * 100
                    
                    print_flush(f"\n[LOSS CHECK at step {global_step}]")
                    print_flush(f"  Initial loss: {initial_loss:.4f}")
                    print_flush(f"  Recent avg loss: {recent_avg:.4f}")
                    print_flush(f"  Reduction: {loss_reduction_pct:.1f}%")
                    print_flush(f"  Expected random: {expected_random_loss:.4f}")
                    
                    if loss_reduction_pct < min_loss_reduction_pct:
                        print(f"\n[WARNING] Loss not decreasing sufficiently!")
                        print(f"  Expected at least {min_loss_reduction_pct}% reduction, got {loss_reduction_pct:.1f}%")
                        if recent_avg > expected_random_loss * 0.95:
                            print(f"  Loss is still near random - model may not be learning.")
                            print(f"  Continuing training, but watch for improvement...")
                    else:
                        print(f"  [OK] Loss is decreasing normally.")
                
                # Evaluation
                if global_step % config['training']['eval_every'] == 0:
                    print(f"\n{'='*60}")
                    print(f"Evaluation at step {global_step}")
                    print(f"{'='*60}")
                    
                    rouge_scores = evaluate_model(model, val_loader, tokenizer, device, max_eval_batches=50)
                    
                    print_flush(f"ROUGE scores: R1={rouge_scores['rouge1']:.4f} R2={rouge_scores['rouge2']:.4f} RL={rouge_scores['rougeL']:.4f}")
                    
                    # Log metrics
                    metrics_log.append({
                        'step': global_step,
                        'epoch': epoch,
                        'loss': avg_loss,
                        'rouge1': rouge_scores['rouge1'],
                        'rouge2': rouge_scores['rouge2'],
                        'rougeL': rouge_scores['rougeL'],
                        'lr': lr
                    })
                    
                    # Save metrics
                    metrics_df = pd.DataFrame(metrics_log)
                    metrics_df.to_csv(log_dir / 'metrics.csv', index=False)
                    
                    # Save best model
                    if rouge_scores['rougeL'] > best_rouge:
                        best_rouge = rouge_scores['rougeL']
                        best_path = checkpoint_dir / 'best_model.pt'
                        save_checkpoint(model, optimizer, scaler, global_step, epoch, best_rouge, config, best_path)
                        print_flush(f"New best model! ROUGE-L: {best_rouge:.4f}")
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        print_flush(f"No improvement. Patience: {patience_counter}/{config['training']['patience']}")
                    
                    print_flush(f"{'='*60}\n")
                
                # Save checkpoint
                if global_step % config['training']['save_every'] == 0:
                    ckpt_path = checkpoint_dir / f'checkpoint_step_{global_step}.pt'
                    save_checkpoint(model, optimizer, scaler, global_step, epoch, best_rouge, config, ckpt_path)
                
                # Check max steps
                if global_step >= max_steps:
                    break
                
                # Early stopping
                if patience_counter >= config['training']['patience']:
                    print(f"Early stopping: no improvement for {patience_counter} evaluations")
                    break
        
        if global_step >= max_steps or patience_counter >= config['training']['patience']:
            break
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best ROUGE-L: {best_rouge:.4f}")
    print(f"Total steps: {global_step}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train Pointer-Generator Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--tokenized_dir', type=str, default=None, help='Directory with tokenized data')
    parser.add_argument('--run_name', type=str, default='kaggle_run', help='Name for this training run')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--max_steps', type=int, default=None, help='Override max steps from config')
    parser.add_argument('--data_path', type=str, default=None, help='Path to raw CSV (for Kaggle)')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output directory (for Kaggle)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override output paths if specified (for Kaggle)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        config['paths']['checkpoint_dir'] = str(output_dir / 'checkpoints')
        config['paths']['log_dir'] = str(output_dir / 'logs')
        print(f"Output directory set to: {output_dir}")
    
    # Auto-detect tokenized_dir if not provided
    if not args.tokenized_dir:
        # Try common Kaggle paths
        kaggle_input = Path('/kaggle/input')
        if kaggle_input.exists():
            # Look for tokenized data in Kaggle input
            possible_dirs = list(kaggle_input.glob('**/tokenized'))
            if possible_dirs:
                args.tokenized_dir = str(possible_dirs[0])
                print(f"Auto-detected tokenized_dir: {args.tokenized_dir}")
            else:
                print("[ERROR] No tokenized_dir found. Please provide --tokenized_dir or --data_path")
                sys.exit(1)
        else:
            print("[ERROR] --tokenized_dir is required")
            sys.exit(1)
    
    # Train
    train(config, args.tokenized_dir, args.run_name, args.resume, args.max_steps, args.data_path)


if __name__ == '__main__':
    main()
