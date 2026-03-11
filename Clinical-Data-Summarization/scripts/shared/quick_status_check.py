"""
Quick Status Check - Check if training is complete and show final stats
"""

import sys
from pathlib import Path
import pandas as pd
import torch

def check_training_status():
    """Check training status and show summary"""
    
    print("\n" + "="*80)
    print("TRAINING STATUS CHECK")
    print("="*80 + "\n")
    
    # Check if checkpoints exist
    checkpoint_dir = Path('artifacts/checkpoints/full_training_restart')
    if not checkpoint_dir.exists():
        print("❌ No checkpoints found!")
        print("   Training may not have started yet.")
        return False
    
    # Check for best model
    best_model = checkpoint_dir / 'best_model.pt'
    if not best_model.exists():
        print("⚠  Best model not found yet.")
        print("   Training may still be in progress.")
        return False
    
    print("✅ Best model found!")
    
    # Load checkpoint info
    checkpoint = torch.load(best_model, map_location='cpu')
    print(f"\nBest Model Info:")
    print(f"  - Step: {checkpoint.get('step', 'N/A')}")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    if 'rouge_scores' in checkpoint:
        scores = checkpoint['rouge_scores']
        print(f"\n  ROUGE Scores:")
        print(f"    • ROUGE-1: {scores.get('rouge1', 0):.4f}")
        print(f"    • ROUGE-2: {scores.get('rouge2', 0):.4f}")
        print(f"    • ROUGE-L: {scores.get('rougeL', 0):.4f}")
    
    # Check all saved checkpoints
    checkpoints = list(checkpoint_dir.glob('checkpoint_step_*.pt'))
    if checkpoints:
        print(f"\n✅ Found {len(checkpoints)} checkpoint(s):")
        for cp in sorted(checkpoints):
            step = cp.stem.split('_')[-1]
            print(f"  - Step {step}")
    
    # Check metrics
    metrics_path = Path('artifacts/logs/full_training_restart/metrics.csv')
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        print(f"\n✅ Training metrics found: {len(df)} recorded steps")
        print(f"  - Latest step: {df['step'].max()}")
        print(f"  - Latest loss: {df['train_loss'].iloc[-1]:.4f}")
        
        if 'rougeL' in df.columns:
            best_rouge_idx = df['rougeL'].idxmax()
            print(f"\n  Best ROUGE-L: {df.loc[best_rouge_idx, 'rougeL']:.4f} (Step {df.loc[best_rouge_idx, 'step']})")
    
    # Training completion check
    max_steps = 10000
    current_step = checkpoint.get('step', 0)
    
    print(f"\n{'='*80}")
    if current_step >= max_steps:
        print("🎉 TRAINING COMPLETE!")
        print(f"{'='*80}\n")
        print("✅ Ready for post-training analysis!")
        print("   Run: .\\run_post_training_analysis.ps1")
        return True
    else:
        progress = (current_step / max_steps) * 100
        print(f"⏳ TRAINING IN PROGRESS: {progress:.1f}% ({current_step}/{max_steps} steps)")
        print(f"{'='*80}\n")
        print(f"⚠  Wait for training to complete before running analysis.")
        return False

if __name__ == '__main__':
    try:
        is_complete = check_training_status()
        sys.exit(0 if is_complete else 1)
    except Exception as e:
        print(f"\n❌ Error checking status: {e}")
        sys.exit(1)
