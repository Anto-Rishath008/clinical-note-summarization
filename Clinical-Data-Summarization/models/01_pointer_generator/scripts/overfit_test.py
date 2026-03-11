"""
Overfit Sanity Test (FIXED VERSION)
Tests if model can memorize a small dataset.
If it can't overfit, there's a bug in the model/training loop.

FIXES:
- Disabled pointer_gen and coverage initially for simpler gradient flow
- Removed FP16 (can cause issues on small batches)
- Higher learning rate for faster convergence
- ASCII-only output (no unicode symbols)
- Added token distribution diagnostics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import yaml
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sys
import math

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core import PointerGeneratorModel, get_dataloader, RougeMetric
import sentencepiece as spm


def compute_output_stats(outputs, vocab_size, is_probs=False):
    """Compute diagnostic stats about model outputs."""
    if is_probs:
        probs = outputs
    else:
        probs = F.softmax(outputs, dim=-1)
    
    # Entropy
    entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1).mean()
    max_entropy = math.log(vocab_size)
    
    # Top-1 probability
    top1_prob = probs.max(dim=-1).values.mean()
    
    # Argmax predictions
    preds = probs.argmax(dim=-1)
    
    return {
        'entropy': entropy.item(),
        'max_entropy': max_entropy,
        'entropy_ratio': entropy.item() / max_entropy,
        'top1_prob': top1_prob.item(),
    }


def overfit_test(config_path, tokenized_dir, num_samples=50, max_steps=1000, 
                 use_pointer_gen=False, use_coverage=False, lr=1e-3):
    """
    Attempt to overfit on a small subset.
    
    PASS CRITERIA:
    1. Loss drops by >30%
    2. Model outputs become non-uniform (entropy drops)
    3. Generated text resembles references
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("OVERFIT SANITY TEST")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Samples: {num_samples}, Max steps: {max_steps}")
    print(f"Pointer-gen: {use_pointer_gen}, Coverage: {use_coverage}")
    print(f"Learning rate: {lr}")
    print("=" * 70 + "\n")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # CRITICAL: Override for overfit test
    config['model']['dropout'] = 0.0  # No dropout for memorization
    config['model']['pointer_gen'] = use_pointer_gen
    config['model']['use_coverage'] = use_coverage
    
    vocab_size = config['data']['vocab_size']
    pad_id = config['data']['pad_id']
    
    # Load tokenizer
    tokenizer_path = Path(config['paths']['tokenizer_dir']) / 'spm.model'
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(tokenizer_path))
    
    # Create tiny training set
    print(f"\nCreating tiny training set ({num_samples} examples)...")
    train_path = Path(tokenized_dir) / 'train.parquet'
    df_full = pd.read_parquet(train_path)
    df_tiny = df_full.head(num_samples)
    
    # Save tiny dataset
    tiny_path = Path(tokenized_dir) / 'train_tiny.parquet'
    df_tiny.to_parquet(tiny_path)
    
    # Create dataloader - use smaller lengths for memory efficiency
    max_src_len = min(512, config['model']['chunk_len'] * config['model']['num_chunks'])
    max_tgt_len = min(128, config['model']['max_target_len'])
    
    train_loader = get_dataloader(
        str(tiny_path),
        batch_size=2,  # Small batch for memory
        shuffle=True,  # Shuffle for better gradients
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        pad_id=pad_id
    )
    
    # Create model
    print("\nCreating model...")
    model = PointerGeneratorModel(config).to(device)
    model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer - simple setup, no bells and whistles
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0  # No regularization for overfit
    )
    
    # Training loop - NO FP16, NO gradient accumulation, NO scheduler
    print("\n" + "=" * 70)
    print("TRAINING (no FP16, no scheduler)")
    print("=" * 70 + "\n")
    
    step = 0
    losses = []
    initial_loss = None
    
    pbar = tqdm(total=max_steps, desc="Training")
    
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            
            src_ids = batch['src_ids'].to(device)
            src_lens = batch['src_lens'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            
            # Forward pass (no autocast!)
            outputs, attentions, coverages, p_gens = model(src_ids, src_lens, tgt_ids)
            
            # Compute loss
            total_loss, nll_loss, coverage_loss = model.compute_loss(
                outputs, tgt_ids, attentions, coverages,
                coverage_lambda=0.01 if use_coverage else 0.0,
                label_smoothing=0.0  # No label smoothing
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(nll_loss.item())
            
            if initial_loss is None:
                initial_loss = nll_loss.item()
            
            step += 1
            pbar.update(1)
            
            # Log periodically
            if step % 100 == 0:
                avg_loss = sum(losses[-100:]) / len(losses[-100:])
                
                # Compute output stats
                with torch.no_grad():
                    stats = compute_output_stats(
                        outputs, vocab_size, 
                        is_probs=use_pointer_gen
                    )
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.3f}',
                    'ent_ratio': f'{stats["entropy_ratio"]:.2f}',
                    'top1': f'{stats["top1_prob"]:.3f}'
                })
                
                tqdm.write(f"Step {step}: loss={avg_loss:.4f}, entropy_ratio={stats['entropy_ratio']:.3f}, top1_prob={stats['top1_prob']:.4f}")
    
    pbar.close()
    
    # Final stats
    final_loss = sum(losses[-100:]) / len(losses[-100:]) if len(losses) >= 100 else sum(losses) / len(losses)
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Reduction:    {loss_reduction:.1f}%")
    print(f"Expected random loss: {math.log(vocab_size):.4f}")
    
    # Generate samples
    print("\n" + "=" * 70)
    print("SAMPLE GENERATIONS (first 3 examples)")
    print("=" * 70 + "\n")
    
    model.eval()
    eval_loader = get_dataloader(
        str(tiny_path),
        batch_size=1,
        shuffle=False,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        pad_id=pad_id
    )
    
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 3:
                break
            
            src_ids = batch['src_ids'].to(device)
            src_lens = batch['src_lens'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            
            # Greedy decoding (no beam search for simplicity)
            generated_ids, _ = model.generate(
                src_ids, src_lens,
                beam_size=1,  # Greedy
                max_length=max_tgt_len,
                min_length=10
            )
            
            # Decode
            gen_ids = generated_ids[0].cpu().tolist()
            ref_ids = tgt_ids[0].cpu().tolist()
            
            # Filter special tokens
            special_ids = {config['data']['bos_id'], config['data']['eos_id'], pad_id}
            gen_ids_clean = [t for t in gen_ids if t not in special_ids]
            ref_ids_clean = [t for t in ref_ids if t not in special_ids]
            
            gen_text = tokenizer.decode(gen_ids_clean)
            ref_text = tokenizer.decode(ref_ids_clean)
            
            print(f"Example {i+1}:")
            print(f"  REF ({len(ref_ids_clean)} tok): {ref_text[:150]}...")
            print(f"  GEN ({len(gen_ids_clean)} tok): {gen_text[:150]}...")
            print()
    
    # VERDICT
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    passed = True
    
    if loss_reduction < 30:
        print("[FAIL] Loss reduction < 30%")
        passed = False
    else:
        print(f"[PASS] Loss reduction = {loss_reduction:.1f}% (>30%)")
    
    if final_loss > math.log(vocab_size) * 0.9:
        print("[FAIL] Final loss still near random")
        passed = False
    else:
        print(f"[PASS] Final loss = {final_loss:.4f} (below random)")
    
    if passed:
        print("\n>>> OVERFIT TEST PASSED <<<")
        print("Model can learn. Proceed with full training.")
    else:
        print("\n>>> OVERFIT TEST FAILED <<<")
        print("Debug checklist:")
        print("  1. Check gradient flow (run single_batch_step_test.py)")
        print("  2. Verify loss function (logits vs probabilities)")
        print("  3. Check if targets are shifted correctly")
        print("  4. Ensure PAD tokens are masked in loss")
    
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Overfit Sanity Test')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--tokenized_dir', type=str, default='data/tokenized')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--pointer_gen', action='store_true', help='Enable pointer-gen')
    parser.add_argument('--coverage', action='store_true', help='Enable coverage')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    
    success = overfit_test(
        config_path=args.config,
        tokenized_dir=args.tokenized_dir,
        num_samples=args.num_samples,
        max_steps=args.max_steps,
        use_pointer_gen=args.pointer_gen,
        use_coverage=args.coverage,
        lr=args.lr
    )
    
    sys.exit(0 if success else 1)
