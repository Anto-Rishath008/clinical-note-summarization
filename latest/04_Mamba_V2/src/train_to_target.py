"""
Target-Based Training Script for Mamba-Transformer Hybrid Model
================================================================

Continues training from a checkpoint until ROUGE-L >= target (0.5).
Does NOT stop at max_steps. Training only ends when:
  1. ROUGE-L >= 0.5 (explicitly set)
  2. Validation loss is reasonable (< max_loss)
  3. Gradient norms are stable (< max_grad_norm_avg)
  4. Stability confirmed over patience_evals consecutive evaluations

Usage:
  python src/train_to_target.py --config configs/rouge_target_train.yaml \
      --resume checkpoints/main_run/best_model.pt
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

warnings.filterwarnings('ignore', message='.*lr_scheduler.step.*before.*optimizer.step.*')
warnings.filterwarnings('ignore', message='.*mismatched key_padding_mask and attn_mask.*')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).parent))

from model import MambaTransformerConfig, MambaTransformerModel, build_model
from data_loader import DataConfig, create_dataloaders, load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def flush_print(msg: str):
    print(msg, flush=True)


# ── Training Config ──────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    output_dir: str = './outputs'
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 50000
    warmup_steps: int = 500
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    eval_steps: int = 250
    save_steps: int = 1000
    log_steps: int = 50
    use_amp: bool = True
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> 'TrainingConfig':
        training_cfg = d.get('training', {})
        return cls(**{k: v for k, v in training_cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class TargetConfig:
    """Stopping criteria — training continues until ALL are met."""
    rouge_l: float = 0.5          # ROUGE-L target (EXPLICIT: 0.5)
    max_loss: float = 4.0         # Val loss must be below this
    max_grad_norm_avg: float = 2.0  # Average grad norm must be below this
    patience_evals: int = 3       # Must stay at target for N consecutive evals

    @classmethod
    def from_dict(cls, d: dict) -> 'TargetConfig':
        target_cfg = d.get('target', {})
        return cls(**{k: v for k, v in target_cfg.items() if k in cls.__dataclass_fields__})


# ── Utilities ────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_cuda():
    flush_print("=" * 60)
    flush_print("GPU AVAILABILITY CHECK")
    flush_print("=" * 60)
    if not torch.cuda.is_available():
        flush_print("ERROR: CUDA is NOT available! Exiting.")
        sys.exit(1)
    flush_print(f"GPU: {torch.cuda.get_device_name(0)}")
    flush_print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    flush_print(f"CUDA: {torch.version.cuda} | PyTorch: {torch.__version__}")
    flush_print("=" * 60)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # After max_steps, maintain minimum LR (don't go to 0)
        if current_step >= num_training_steps:
            return min_lr_ratio
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=3):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        mask = (targets != self.ignore_index)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (vocab_size - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        loss = (loss * mask.float()).sum() / mask.sum()
        return loss


def compute_rouge(predictions, references):
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge_score not installed, returning dummy scores")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores['rouge1'].fmeasure)
        r2.append(scores['rouge2'].fmeasure)
        rl.append(scores['rougeL'].fmeasure)
    return {
        'rouge1': sum(r1) / len(r1) if r1 else 0.0,
        'rouge2': sum(r2) / len(r2) if r2 else 0.0,
        'rougeL': sum(rl) / len(rl) if rl else 0.0,
    }


# ── Checkpoint I/O ───────────────────────────────────────────────────────────

def save_checkpoint_atomic(model, optimizer, scheduler, scaler, step, best_rouge_l, config, path, min_size_mb=10.0):
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_rouge_l': best_rouge_l,
        'config': config,
    }
    tmp = path + '.tmp'
    torch.save(checkpoint, tmp)
    shutil.move(tmp, path)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        flush_print(f"  [SAVED] {os.path.abspath(path)} ({size_mb:.1f} MB) step={step} best_rougeL={best_rouge_l:.4f}")


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    cks = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_step_*.pt'))
    if not cks:
        return None
    def step_of(p):
        try: return int(os.path.basename(p).split('_')[-1].replace('.pt', ''))
        except: return 0
    cks.sort(key=step_of, reverse=True)
    return cks[0]


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ck = torch.load(path, map_location='cuda', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ck:
        try:
            optimizer.load_state_dict(ck['optimizer_state_dict'])
        except Exception as e:
            flush_print(f"  [WARN] Could not load optimizer state: {e}. Using fresh optimizer.")
    if scheduler and 'scheduler_state_dict' in ck:
        try:
            scheduler.load_state_dict(ck['scheduler_state_dict'])
        except Exception:
            pass
    if scaler and 'scaler_state_dict' in ck:
        try:
            scaler.load_state_dict(ck['scaler_state_dict'])
        except Exception:
            pass
    return ck.get('step', 0), ck.get('best_rouge_l', 0.0)


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, tokenizer, device, max_samples=100):
    model.eval()
    predictions, references = [], []
    total_loss, n_batches = 0.0, 0
    criterion = LabelSmoothingLoss(smoothing=0.1, ignore_index=3)

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx * val_loader.batch_size >= max_samples:
            break
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)

        with autocast('cuda', enabled=True):
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(logits, tgt_output)
        total_loss += loss.item()
        n_batches += 1

        generated = model.generate(src, src_mask, max_len=256)
        for i in range(src.size(0)):
            pred_ids = generated[i].cpu().tolist()
            ref_ids = tgt_output[i].cpu().tolist()
            pred_text = tokenizer.DecodeIds([t for t in pred_ids if t not in [0, 1, 2, 3]])
            ref_text = tokenizer.DecodeIds([t for t in ref_ids if t not in [0, 1, 2, 3]])
            predictions.append(pred_text)
            references.append(ref_text)

    rouge_scores = compute_rouge(predictions, references)
    rouge_scores['val_loss'] = total_loss / max(n_batches, 1)
    model.train()
    return rouge_scores


# ── Main Training Loop ───────────────────────────────────────────────────────

def train_to_target(
    model_config: MambaTransformerConfig,
    data_config: DataConfig,
    training_config: TrainingConfig,
    target_config: TargetConfig,
    resume_from: str,
):
    """
    Train until ROUGE-L >= 0.5 (explicit target).
    Does NOT stop at max_steps — continues indefinitely until the target is met.
    """
    ROUGE_L_TARGET = 0.5  # ← EXPLICIT: training stops only when ROUGE-L >= 0.5

    device = torch.device('cuda')
    os.makedirs(training_config.output_dir, exist_ok=True)
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    os.makedirs(training_config.log_dir, exist_ok=True)
    set_seed(training_config.seed)

    # ── Build model ──────────────────────────────────────────────────────
    flush_print("Building model...")
    model = build_model(model_config)
    model = model.to(device)

    # ── Data ─────────────────────────────────────────────────────────────
    flush_print("Creating dataloaders...")
    train_loader, val_loader, tokenizer = create_dataloaders(
        config=data_config, batch_size=training_config.batch_size, num_workers=0,
    )
    flush_print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Optimizer / Scheduler / Scaler ───────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, training_config.warmup_steps, training_config.max_steps,
    )
    criterion = LabelSmoothingLoss(smoothing=training_config.label_smoothing, ignore_index=data_config.pad_id)
    scaler = GradScaler('cuda', enabled=training_config.use_amp)

    # ── Resume ───────────────────────────────────────────────────────────
    global_step = 0
    best_rouge_l = 0.0

    # First try the explicit resume path, then auto-detect latest in checkpoint_dir
    actual_resume = resume_from
    if actual_resume is None or not os.path.exists(actual_resume):
        actual_resume = find_latest_checkpoint(training_config.checkpoint_dir)

    if actual_resume and os.path.exists(actual_resume):
        flush_print(f"\n{'='*60}")
        flush_print(f"RESUMING FROM: {actual_resume}")
        flush_print(f"{'='*60}")
        global_step, best_rouge_l = load_checkpoint(actual_resume, model, optimizer, scheduler, scaler)
        flush_print(f"  Loaded step={global_step}, best_rouge_l={best_rouge_l:.4f}")
        flush_print(f"{'='*60}\n")
    else:
        flush_print("[ERROR] No checkpoint found to resume from!")
        sys.exit(1)

    full_config = {
        'model': asdict(model_config),
        'data': asdict(data_config),
        'training': asdict(training_config),
    }

    # ── Stopping state ───────────────────────────────────────────────────
    target_met_count = 0          # How many consecutive evals met the target
    recent_grad_norms = []        # Rolling window of grad norms
    target_reached = False

    flush_print("\n" + "=" * 60)
    flush_print("TARGET-BASED TRAINING")
    flush_print("=" * 60)
    flush_print(f"  ROUGE-L TARGET       : {ROUGE_L_TARGET}")
    flush_print(f"  Max loss threshold   : {target_config.max_loss}")
    flush_print(f"  Max avg grad norm    : {target_config.max_grad_norm_avg}")
    flush_print(f"  Patience evals       : {target_config.patience_evals}")
    flush_print(f"  Starting from step   : {global_step}")
    flush_print(f"  Current best ROUGE-L : {best_rouge_l:.4f}")
    flush_print(f"  LR                   : {training_config.learning_rate}")
    flush_print(f"  Scheduler max_steps  : {training_config.max_steps} (training continues past this)")
    flush_print(f"  Eval every           : {training_config.eval_steps} steps")
    flush_print("=" * 60 + "\n")

    # ── Setup log file ───────────────────────────────────────────────────
    log_file = os.path.join(training_config.log_dir, 'training_rouge_target.txt')
    log_fh = open(log_file, 'a', encoding='utf-8')
    def log(msg):
        flush_print(msg)
        log_fh.write(msg + '\n')
        log_fh.flush()

    # ── Training loop (NO hard step limit) ───────────────────────────────
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    start_time = time.time()
    tokens_processed = 0
    epoch = 0

    while not target_reached:
        epoch += 1
        log(f"\n--- Epoch {epoch} (global_step={global_step}) ---")

        for batch_idx, batch in enumerate(train_loader):
            if target_reached:
                break

            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            src_mask = batch['src_mask'].to(device)
            tgt_mask = batch['tgt_mask'].to(device)

            tokens_processed += src.numel() + tgt_input.numel()

            with autocast('cuda', enabled=training_config.use_amp):
                logits = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(logits, tgt_output) / training_config.gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            running_loss += loss.item() * training_config.gradient_accumulation_steps

            # Gradient accumulation step
            if (batch_idx + 1) % training_config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    optimizer.zero_grad()
                    scaler.update()
                    continue

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                # Track grad norms (rolling window of last 100)
                recent_grad_norms.append(grad_norm.item())
                if len(recent_grad_norms) > 100:
                    recent_grad_norms.pop(0)

                # ── Logging ──────────────────────────────────────────
                if global_step % training_config.log_steps == 0:
                    avg_loss = running_loss / training_config.log_steps
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    avg_gn = sum(recent_grad_norms) / len(recent_grad_norms) if recent_grad_norms else 0
                    gpu_mem = torch.cuda.max_memory_allocated() / 1e9
                    log(
                        f"Step {global_step:6d} | Loss: {avg_loss:.4f} | "
                        f"GradNorm: {grad_norm:.3f} (avg: {avg_gn:.3f}) | "
                        f"LR: {lr:.2e} | GPU: {gpu_mem:.1f}GB | "
                        f"Best ROUGE-L: {best_rouge_l:.4f} / Target: {ROUGE_L_TARGET}"
                    )
                    running_loss = 0.0

                # ── Evaluation ───────────────────────────────────────
                if global_step % training_config.eval_steps == 0:
                    log("\n" + "-" * 60)
                    log(f"EVALUATION at Step {global_step}")
                    log("-" * 60)
                    eval_results = evaluate(model, val_loader, tokenizer, device, max_samples=50)

                    avg_gn = sum(recent_grad_norms) / len(recent_grad_norms) if recent_grad_norms else 999

                    log(
                        f"  Val Loss : {eval_results['val_loss']:.4f}\n"
                        f"  ROUGE-1  : {eval_results['rouge1']:.4f}\n"
                        f"  ROUGE-2  : {eval_results['rouge2']:.4f}\n"
                        f"  ROUGE-L  : {eval_results['rougeL']:.4f}  (TARGET: {ROUGE_L_TARGET})\n"
                        f"  Avg Grad : {avg_gn:.3f}"
                    )

                    # Save best model
                    if eval_results['rougeL'] > best_rouge_l:
                        best_rouge_l = eval_results['rougeL']
                        best_path = os.path.join(training_config.checkpoint_dir, 'best_model.pt')
                        log(f"  [NEW BEST] ROUGE-L = {best_rouge_l:.4f}")
                        save_checkpoint_atomic(
                            model, optimizer, scheduler, scaler,
                            global_step, best_rouge_l, full_config, best_path,
                        )

                    # ── Check stopping criteria ──────────────────────
                    rouge_ok = eval_results['rougeL'] >= ROUGE_L_TARGET
                    loss_ok = eval_results['val_loss'] <= target_config.max_loss
                    grad_ok = avg_gn <= target_config.max_grad_norm_avg

                    if rouge_ok and loss_ok and grad_ok:
                        target_met_count += 1
                        log(
                            f"  ✓ TARGET MET ({target_met_count}/{target_config.patience_evals}) "
                            f"ROUGE-L={eval_results['rougeL']:.4f} >= {ROUGE_L_TARGET}, "
                            f"loss={eval_results['val_loss']:.4f} <= {target_config.max_loss}, "
                            f"grad={avg_gn:.3f} <= {target_config.max_grad_norm_avg}"
                        )
                    else:
                        target_met_count = 0
                        reasons = []
                        if not rouge_ok:
                            reasons.append(f"ROUGE-L {eval_results['rougeL']:.4f} < {ROUGE_L_TARGET}")
                        if not loss_ok:
                            reasons.append(f"loss {eval_results['val_loss']:.4f} > {target_config.max_loss}")
                        if not grad_ok:
                            reasons.append(f"grad {avg_gn:.3f} > {target_config.max_grad_norm_avg}")
                        log(f"  ✗ Target NOT met: {'; '.join(reasons)}")

                    if target_met_count >= target_config.patience_evals:
                        log(f"\n{'='*60}")
                        log(f"TARGET REACHED! ROUGE-L >= {ROUGE_L_TARGET} for "
                            f"{target_config.patience_evals} consecutive evaluations.")
                        log(f"{'='*60}")
                        target_reached = True
                        break

                    log("-" * 60 + "\n")

                # ── Periodic checkpoint ──────────────────────────────
                if global_step % training_config.save_steps == 0:
                    ck_path = os.path.join(
                        training_config.checkpoint_dir,
                        f'checkpoint_step_{global_step}.pt',
                    )
                    save_checkpoint_atomic(
                        model, optimizer, scheduler, scaler,
                        global_step, best_rouge_l, full_config, ck_path,
                    )

    # ── Final save ───────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("SAVING FINAL MODEL")
    log("=" * 60)
    final_path = os.path.join(training_config.checkpoint_dir, 'final_model.pt')
    save_checkpoint_atomic(
        model, optimizer, scheduler, scaler,
        global_step, best_rouge_l, full_config, final_path,
    )

    elapsed = time.time() - start_time
    log(f"\nTraining complete in {elapsed/3600:.1f} hours")
    log(f"Final step       : {global_step}")
    log(f"Best ROUGE-L     : {best_rouge_l:.4f}")
    log(f"Target ROUGE-L   : {ROUGE_L_TARGET}")
    log("=" * 60)

    log_fh.close()
    return best_rouge_l


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train until ROUGE-L target is reached')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, required=True,
                        help='Path to checkpoint to resume from (e.g. checkpoints/main_run/best_model.pt)')
    args = parser.parse_args()

    flush_print("\n" + "=" * 60)
    flush_print("MAMBA-TRANSFORMER — TARGET-BASED TRAINING")
    flush_print(f"Target: ROUGE-L >= 0.5")
    flush_print("=" * 60)

    check_cuda()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_config = MambaTransformerConfig.from_dict(config.get('model', {}))
    data_config = DataConfig.from_dict(config)
    training_config = TrainingConfig.from_dict(config)
    target_config = TargetConfig.from_dict(config)

    # Force ROUGE-L target to 0.5 regardless of config
    target_config.rouge_l = 0.5

    flush_print(f"Output dir     : {training_config.output_dir}")
    flush_print(f"Checkpoint dir : {training_config.checkpoint_dir}")
    flush_print(f"Resume from    : {args.resume}")
    flush_print(f"ROUGE-L target : {target_config.rouge_l}")

    train_to_target(model_config, data_config, training_config, target_config, resume_from=args.resume)


if __name__ == '__main__':
    main()
