#!/usr/bin/env python3
"""
Kaggle Runner Script for Mamba-Transformer Hybrid Model
=========================================================

End-to-end training pipeline for Kaggle notebooks.
Run this single script to: setup -> preprocess -> train -> evaluate -> save

Usage in Kaggle:
    !python kaggle_runner.py --data_source /kaggle/input/your-dataset-path

Or with uploaded dataset:
    !python kaggle_runner.py --data_source /kaggle/input/mimic-iv-bhc
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def setup_environment():
    """Install required packages"""
    print("=" * 60)
    print("INSTALLING REQUIRED PACKAGES")
    print("=" * 60)
    
    packages = [
        'mamba-ssm',
        'causal-conv1d>=1.1.0',
        'sentencepiece',
        'rouge-score',
        'pyyaml',
    ]
    
    for pkg in packages:
        print(f"Installing {pkg}...")
        os.system(f'{sys.executable} -m pip install -q {pkg}')
    
    print("Package installation complete!")


def check_cuda():
    """Check CUDA availability and run test"""
    import torch
    
    print("=" * 60)
    print("GPU AVAILABILITY CHECK")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is NOT available!")
        print("Please ensure you're using a GPU runtime.")
        print("Go to: Runtime -> Change runtime type -> GPU")
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
    assert c.shape == (100, 100), "CUDA test failed!"
    print(f"CUDA matmul test passed! Result shape: {c.shape}")
    
    # Memory check
    print(f"\nGPU Memory Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
    print("=" * 60)


def setup_workspace(data_source: str, workspace_dir: str):
    """Setup workspace directories and copy data"""
    print("=" * 60)
    print("SETTING UP WORKSPACE")
    print("=" * 60)
    
    # Create directories
    dirs = ['data/tokenizer', 'src', 'configs', 'scripts', 'outputs', 'checkpoints', 'logs']
    for d in dirs:
        Path(workspace_dir, d).mkdir(parents=True, exist_ok=True)
        print(f"Created: {workspace_dir}/{d}")
    
    # Copy data if source provided
    if data_source:
        data_path = Path(data_source)
        
        # Look for CSV file
        csv_files = list(data_path.glob('*.csv')) + list(data_path.glob('**/*.csv'))
        if csv_files:
            src_csv = csv_files[0]
            dst_csv = Path(workspace_dir) / 'data' / 'mimic-iv-bhc.csv'
            if not dst_csv.exists():
                print(f"Copying {src_csv} -> {dst_csv}")
                shutil.copy(src_csv, dst_csv)
        
        # Look for tokenizer files
        for fname in ['spm.model', 'spm.vocab']:
            matches = list(data_path.glob(f'**/{fname}'))
            if matches:
                src = matches[0]
                dst = Path(workspace_dir) / 'data' / 'tokenizer' / fname
                if not dst.exists():
                    print(f"Copying {src} -> {dst}")
                    shutil.copy(src, dst)
    
    print("Workspace setup complete!")
    print("=" * 60)


def create_config(workspace_dir: str, max_steps: int = 50000):
    """Create training config"""
    config_content = f"""# Mamba-Transformer Hybrid Model Configuration
# Auto-generated for Kaggle training

model:
  vocab_size: 32000
  d_model: 512
  n_mamba_layers: 6
  n_decoder_layers: 6
  n_heads: 8
  d_ff: 2048
  dropout: 0.1
  max_src_len: 4096
  max_tgt_len: 512
  chunk_size: 256
  stride: 192
  n_memory_tokens: 8
  mamba_d_state: 16
  mamba_d_conv: 4
  mamba_expand: 2
  pad_id: 3
  bos_id: 1
  eos_id: 2
  label_smoothing: 0.1

data:
  csv_path: {workspace_dir}/data/mimic-iv-bhc.csv
  tokenizer_path: {workspace_dir}/data/tokenizer/spm.model
  max_src_len: 4096
  max_tgt_len: 512
  pad_id: 3
  bos_id: 1
  eos_id: 2
  source_col: source_text
  target_col: target_text

training:
  output_dir: {workspace_dir}/outputs
  checkpoint_dir: {workspace_dir}/checkpoints
  log_dir: {workspace_dir}/logs
  batch_size: 4
  gradient_accumulation_steps: 8
  max_steps: {max_steps}
  warmup_steps: 2000
  learning_rate: 5.0e-4
  weight_decay: 0.01
  max_grad_norm: 1.0
  label_smoothing: 0.1
  eval_steps: 500
  save_steps: 500
  log_steps: 50
  use_amp: true
  seed: 42
"""
    
    config_path = Path(workspace_dir) / 'configs' / 'kaggle_config.yaml'
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Config created: {config_path}")
    return str(config_path)


def run_training(workspace_dir: str, config_path: str):
    """Run the training script"""
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    # Add src to path
    src_dir = Path(workspace_dir) / 'src'
    sys.path.insert(0, str(src_dir))
    
    import yaml
    from train import (
        MambaTransformerConfig, 
        DataConfig, 
        TrainingConfig,
        train,
        check_cuda as train_check_cuda
    )
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create configs
    model_config = MambaTransformerConfig.from_dict(config.get('model', {}))
    data_config = DataConfig.from_dict(config)
    training_config = TrainingConfig.from_dict(config)
    
    # Train
    best_rouge_l = train(model_config, data_config, training_config)
    
    return best_rouge_l


def verify_outputs(workspace_dir: str):
    """Verify training outputs exist"""
    print("=" * 60)
    print("VERIFYING OUTPUTS")
    print("=" * 60)
    
    checkpoint_dir = Path(workspace_dir) / 'checkpoints'
    
    # Check for best model
    best_model = checkpoint_dir / 'best_model.pt'
    if best_model.exists():
        size_mb = best_model.stat().st_size / (1024 * 1024)
        print(f"✓ Best model found: {best_model}")
        print(f"  Size: {size_mb:.1f} MB")
        
        if size_mb < 50:
            print(f"  WARNING: Model size seems small!")
    else:
        print(f"✗ Best model NOT found at: {best_model}")
    
    # Check for final model
    final_model = checkpoint_dir / 'final_model.pt'
    if final_model.exists():
        size_mb = final_model.stat().st_size / (1024 * 1024)
        print(f"✓ Final model found: {final_model}")
        print(f"  Size: {size_mb:.1f} MB")
    
    # List all checkpoints
    checkpoints = list(checkpoint_dir.glob('checkpoint_step_*.pt'))
    print(f"\nTotal checkpoints: {len(checkpoints)}")
    for cp in sorted(checkpoints)[-5:]:  # Show last 5
        size_mb = cp.stat().st_size / (1024 * 1024)
        print(f"  - {cp.name} ({size_mb:.1f} MB)")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Kaggle Runner for Mamba-Transformer')
    parser.add_argument('--data_source', type=str, default='/kaggle/input/mimic-iv-bhc',
                        help='Path to input data on Kaggle')
    parser.add_argument('--workspace_dir', type=str, default='/kaggle/working/mamba_longt5',
                        help='Workspace directory')
    parser.add_argument('--max_steps', type=int, default=50000,
                        help='Maximum training steps')
    parser.add_argument('--skip_install', action='store_true',
                        help='Skip package installation')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MAMBA-TRANSFORMER HYBRID MODEL TRAINING")
    print("Kaggle GPU Runner")
    print("=" * 60)
    
    # Step 1: Install packages
    if not args.skip_install:
        setup_environment()
    
    # Step 2: Check CUDA
    check_cuda()
    
    # Step 3: Setup workspace
    setup_workspace(args.data_source, args.workspace_dir)
    
    # Step 4: Create config
    config_path = create_config(args.workspace_dir, args.max_steps)
    
    # Step 5: Run training
    best_rouge_l = run_training(args.workspace_dir, config_path)
    
    # Step 6: Verify outputs
    verify_outputs(args.workspace_dir)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Best ROUGE-L: {best_rouge_l:.4f}")
    print(f"Checkpoints saved to: {args.workspace_dir}/checkpoints/")
    print(f"Outputs saved to: {args.workspace_dir}/outputs/")
    print("=" * 60)


if __name__ == '__main__':
    main()
