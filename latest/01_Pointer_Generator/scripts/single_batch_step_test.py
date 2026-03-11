"""
STEP 0: Minimal Learning Proof
Tests if gradient flows and parameters update correctly on a single batch.

PASS CRITERION: loss_after < loss_before AND params changed.
If this fails, the graph/loss/optimizer is broken.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core import PointerGeneratorModel, get_dataloader
import sentencepiece as spm


def single_batch_step_test(config_path="configs/default.yaml", tokenized_dir="data/tokenized"):
    """
    Minimal test: one batch, one gradient step, verify loss decreases.
    """
    print("=" * 70)
    print("SINGLE BATCH STEP TEST - Minimal Learning Proof")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # CRITICAL: Disable all complexity for this test
    config['model']['use_coverage'] = False
    config['model']['pointer_gen'] = False  # Force logits output, not probabilities
    config['model']['dropout'] = 0.0  # No randomness
    
    print("\nTest Config:")
    print(f"  pointer_gen: {config['model']['pointer_gen']}")
    print(f"  use_coverage: {config['model']['use_coverage']}")
    print(f"  dropout: {config['model']['dropout']}")
    print(f"  vocab_size: {config['data']['vocab_size']}")
    print(f"  pad_id: {config['data']['pad_id']}")
    
    # Load tokenizer
    tokenizer_path = Path(config['paths']['tokenizer_dir']) / 'spm.model'
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(tokenizer_path))
    
    # Create dataloader with batch_size=2
    train_path = Path(tokenized_dir) / 'train.parquet'
    train_loader = get_dataloader(
        str(train_path),
        batch_size=2,
        shuffle=False,
        max_src_len=config['model']['chunk_len'] * config['model']['num_chunks'],
        max_tgt_len=config['model']['max_target_len'],
        pad_id=config['data']['pad_id']
    )
    
    # Get ONE batch
    batch = next(iter(train_loader))
    src_ids = batch['src_ids'].to(device)
    src_lens = batch['src_lens'].to(device)
    tgt_ids = batch['tgt_ids'].to(device)
    
    print(f"\nBatch shapes:")
    print(f"  src_ids: {src_ids.shape}")
    print(f"  tgt_ids: {tgt_ids.shape}")
    print(f"  src_lens: {src_lens.tolist()}")
    
    # Create model (no dropout for determinism)
    print("\nCreating model...")
    model = PointerGeneratorModel(config).to(device)
    model.train()
    
    # Identify key parameters to track
    params_to_track = {
        'encoder_embedding': model.encoder.encoder.embedding.weight,
        'decoder_lstm_weight': model.decoder.lstm.weight_ih_l0,
        'output_proj': model.decoder.out_proj.weight,
    }
    
    # Store initial param values (clone to avoid reference issues)
    initial_params = {k: v.clone().detach() for k, v in params_to_track.items()}
    
    # Simple optimizer - no warmup, no scheduler, no weight decay
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # High LR for visible changes
    
    print("\n" + "=" * 70)
    print("FORWARD PASS 1 (before optimization)")
    print("=" * 70)
    
    # Forward pass 1
    outputs, attentions, coverages, p_gens = model(src_ids, src_lens, tgt_ids)
    
    print(f"\nOutputs shape: {outputs.shape}")
    print(f"Outputs dtype: {outputs.dtype}")
    print(f"Outputs range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
    
    # Check if outputs are logits or probabilities
    # Logits can be negative, probabilities are in [0, 1]
    if outputs.min() >= 0 and outputs.max() <= 1:
        print("WARNING: Outputs look like probabilities, not logits!")
        outputs_type = "probabilities"
    else:
        print("OK: Outputs are logits (can be negative)")
        outputs_type = "logits"
    
    # Compute loss manually to verify
    targets = tgt_ids[:, 1:]  # Remove BOS
    batch_size, tgt_len, vocab_size = outputs.size()
    
    print(f"\nTargets shape: {targets.shape}")
    print(f"Target sample (first 10): {targets[0, :10].tolist()}")
    
    # Mask for non-padding targets
    pad_id = config['data']['pad_id']
    mask = (targets != pad_id).float()
    num_tokens = mask.sum().item()
    print(f"Non-pad tokens: {int(num_tokens)}")
    
    # Compute loss
    if outputs_type == "logits":
        # Use cross_entropy (correct for logits)
        outputs_flat = outputs.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        loss_before = F.cross_entropy(outputs_flat, targets_flat, ignore_index=pad_id)
    else:
        # Use NLL loss (for probabilities - convert to log)
        log_probs = torch.log(outputs.clamp(min=1e-10))
        log_probs_flat = log_probs.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        loss_before = F.nll_loss(log_probs_flat, targets_flat, ignore_index=pad_id)
    
    print(f"\nLoss BEFORE: {loss_before.item():.6f}")
    print(f"Expected random loss (log vocab): {torch.log(torch.tensor(vocab_size, dtype=torch.float)).item():.6f}")
    
    # Check output distribution
    with torch.no_grad():
        if outputs_type == "logits":
            probs = F.softmax(outputs, dim=-1)
        else:
            probs = outputs
        
        # Entropy
        entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float))
        
        # Top-1 probability
        top1_prob = probs.max(dim=-1).values.mean()
        
        # Check for UNK/PAD predictions
        pred_ids = probs.argmax(dim=-1)
        pct_unk = (pred_ids == config['data']['unk_id']).float().mean().item() * 100
        pct_pad = (pred_ids == pad_id).float().mean().item() * 100
        
        print(f"\nOutput Distribution Stats:")
        print(f"  Entropy: {entropy.item():.4f} (max: {max_entropy.item():.4f})")
        print(f"  Top-1 prob mean: {top1_prob.item():.6f}")
        print(f"  %UNK predicted: {pct_unk:.2f}%")
        print(f"  %PAD predicted: {pct_pad:.2f}%")
        
        if entropy > max_entropy * 0.95:
            print("  WARNING: Entropy near maximum -> model outputs near-uniform!")
    
    print("\n" + "=" * 70)
    print("BACKWARD + OPTIMIZER STEP")
    print("=" * 70)
    
    # Backward pass
    optimizer.zero_grad()
    loss_before.backward()
    
    # Check gradients
    print("\nGradient statistics:")
    for name, param in params_to_track.items():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_max = param.grad.abs().max().item()
            print(f"  {name}: norm={grad_norm:.6f}, max={grad_max:.6f}")
            if grad_norm < 1e-10:
                print(f"    WARNING: Near-zero gradient!")
        else:
            print(f"  {name}: NO GRADIENT!")
    
    # Optimizer step
    optimizer.step()
    
    print("\n" + "=" * 70)
    print("FORWARD PASS 2 (after optimization)")
    print("=" * 70)
    
    # Forward pass 2 (same batch)
    outputs2, _, _, _ = model(src_ids, src_lens, tgt_ids)
    
    # Compute loss again
    if outputs_type == "logits":
        outputs2_flat = outputs2.reshape(-1, vocab_size)
        loss_after = F.cross_entropy(outputs2_flat, targets_flat, ignore_index=pad_id)
    else:
        log_probs2 = torch.log(outputs2.clamp(min=1e-10))
        log_probs2_flat = log_probs2.reshape(-1, vocab_size)
        loss_after = F.nll_loss(log_probs2_flat, targets_flat, ignore_index=pad_id)
    
    print(f"\nLoss AFTER: {loss_after.item():.6f}")
    print(f"Loss delta: {loss_before.item() - loss_after.item():.6f}")
    
    print("\n" + "=" * 70)
    print("PARAMETER CHANGES")
    print("=" * 70)
    
    # Check parameter changes
    param_changes = {}
    for name, param in params_to_track.items():
        delta = (param - initial_params[name]).abs().max().item()
        param_changes[name] = delta
        print(f"  {name}: max|delta| = {delta:.8f}")
    
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    loss_decreased = loss_after.item() < loss_before.item()
    params_changed = all(d > 1e-10 for d in param_changes.values())
    
    print(f"\n  Loss decreased: {'YES' if loss_decreased else 'NO'}")
    print(f"  Params changed: {'YES' if params_changed else 'NO'}")
    
    if loss_decreased and params_changed:
        print("\n  [PASS] Model can learn! Graph/loss/optimizer working correctly.")
    else:
        print("\n  [FAIL] Learning is broken!")
        if not loss_decreased:
            print("    - Loss did not decrease. Check:")
            print("      * Loss function (cross_entropy vs nll_loss)")
            print("      * Output type (logits vs probabilities)")
            print("      * Gradient flow (are grads reaching all params?)")
        if not params_changed:
            print("    - Parameters did not change. Check:")
            print("      * Optimizer configuration")
            print("      * Gradient values (are they zero?)")
            print("      * requires_grad on parameters")
    
    print("\n" + "=" * 70)
    return loss_decreased and params_changed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--tokenized_dir', default='data/tokenized')
    args = parser.parse_args()
    
    success = single_batch_step_test(args.config, args.tokenized_dir)
    sys.exit(0 if success else 1)
