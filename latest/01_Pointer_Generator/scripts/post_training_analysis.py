"""
Post-Training Analysis Script
Loads best model, evaluates on test/validation data, and creates visualizations
"""

import torch
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer

import sys
sys.path.append('.')
from src.core import PointerGeneratorModel, load_tokenizer


def load_model_and_config(checkpoint_path, config_path, tokenizer_path):
    """Load trained model, config, and tokenizer"""
    print(f"\n{'='*60}")
    print("Loading Model and Configuration")
    print(f"{'='*60}")
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded from: {config_path}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_piece_size()
    print(f"✓ Tokenizer loaded: vocab_size={vocab_size}")
    
    # Create model
    model = PointerGeneratorModel(
        vocab_size=vocab_size,
        emb_dim=config['model']['emb_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        use_pointer_gen=config['model']['use_pointer_gen'],
        use_coverage=config['model']['use_coverage']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"  - Step: {checkpoint.get('step', 'N/A')}")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    if 'rouge_scores' in checkpoint:
        scores = checkpoint['rouge_scores']
        print(f"  - ROUGE-1: {scores.get('rouge1', 0):.4f}")
        print(f"  - ROUGE-2: {scores.get('rouge2', 0):.4f}")
        print(f"  - ROUGE-L: {scores.get('rougeL', 0):.4f}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"✓ Model moved to: {device}")
    
    return model, config, tokenizer, device


def generate_summary(model, tokenizer, source_ids, source_mask, device, max_len=150, beam_size=4):
    """Generate summary using beam search"""
    with torch.no_grad():
        source_ids = source_ids.to(device)
        source_mask = source_mask.to(device)
        
        # Greedy decoding for now (beam search can be added)
        batch_size = source_ids.size(0)
        decoder_input = torch.full((batch_size, 1), tokenizer.bos_id(), dtype=torch.long, device=device)
        
        generated = []
        for _ in range(max_len):
            decoder_mask = torch.ones_like(decoder_input, dtype=torch.bool)
            outputs = model(source_ids, decoder_input, source_mask, decoder_mask)
            
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            generated.append(next_token.squeeze(-1))
            
            # Stop if all sequences have EOS
            if (next_token == tokenizer.eos_id()).all():
                break
        
        generated = torch.stack(generated, dim=1)
        return generated


def evaluate_on_dataset(model, tokenizer, data_path, device, num_samples=None):
    """Evaluate model on dataset and compute ROUGE scores"""
    print(f"\n{'='*60}")
    print(f"Evaluating on: {data_path}")
    print(f"{'='*60}")
    
    # Load data
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    if num_samples:
        data = data[:num_samples]
    
    print(f"Total samples: {len(data)}")
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    results = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for item in tqdm(data, desc="Generating summaries"):
        source_ids = torch.tensor([item['source_ids']], dtype=torch.long)
        source_mask = torch.tensor([item['source_mask']], dtype=torch.bool)
        target_ids = item['target_ids']
        
        # Generate summary
        generated = generate_summary(model, tokenizer, source_ids, source_mask, device)
        
        # Decode
        generated_text = tokenizer.decode(generated[0].cpu().tolist())
        target_text = tokenizer.decode(target_ids)
        source_text = tokenizer.decode(item['source_ids'])
        
        # Clean texts
        generated_text = generated_text.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
        target_text = target_text.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
        source_text = source_text.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
        
        # Compute ROUGE
        scores = scorer.score(target_text, generated_text)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        results.append({
            'source': source_text,
            'target': target_text,
            'generated': generated_text,
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        })
    
    # Compute average scores
    avg_scores = {
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL'])
    }
    
    print(f"\n{'='*60}")
    print("Average ROUGE Scores:")
    print(f"{'='*60}")
    print(f"ROUGE-1: {avg_scores['rouge1']:.4f} (±{np.std(rouge_scores['rouge1']):.4f})")
    print(f"ROUGE-2: {avg_scores['rouge2']:.4f} (±{np.std(rouge_scores['rouge2']):.4f})")
    print(f"ROUGE-L: {avg_scores['rougeL']:.4f} (±{np.std(rouge_scores['rougeL']):.4f})")
    
    return results, avg_scores, rouge_scores


def plot_training_metrics(metrics_csv_path, output_dir):
    """Plot training loss and ROUGE scores over time"""
    print(f"\n{'='*60}")
    print("Creating Training Visualizations")
    print(f"{'='*60}")
    
    df = pd.read_csv(metrics_csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Training Loss
    ax = axes[0, 0]
    ax.plot(df['step'], df['train_loss'], label='Train Loss', linewidth=2, alpha=0.8)
    if 'val_loss' in df.columns:
        ax.plot(df['step'], df['val_loss'], label='Val Loss', linewidth=2, alpha=0.8)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. ROUGE Scores
    ax = axes[0, 1]
    if 'rouge1' in df.columns:
        ax.plot(df['step'], df['rouge1'], label='ROUGE-1', marker='o', linewidth=2)
        ax.plot(df['step'], df['rouge2'], label='ROUGE-2', marker='s', linewidth=2)
        ax.plot(df['step'], df['rougeL'], label='ROUGE-L', marker='^', linewidth=2)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('ROUGE Score', fontsize=12)
        ax.set_title('ROUGE Scores Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Learning Rate
    ax = axes[1, 0]
    if 'learning_rate' in df.columns:
        ax.plot(df['step'], df['learning_rate'], color='green', linewidth=2)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 4. Gradient Norm
    ax = axes[1, 1]
    if 'grad_norm' in df.columns:
        ax.plot(df['step'], df['grad_norm'], color='orange', linewidth=2, alpha=0.7)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Gradient Norm', fontsize=12)
        ax.set_title('Gradient Norm', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'training_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training metrics plot saved: {output_path}")
    plt.close()


def plot_rouge_distribution(rouge_scores, output_dir, dataset_name):
    """Plot distribution of ROUGE scores"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'ROUGE Score Distribution - {dataset_name}', fontsize=16, fontweight='bold')
    
    metrics = ['rouge1', 'rouge2', 'rougeL']
    titles = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    
    for ax, metric, title in zip(axes, metrics, titles):
        scores = rouge_scores[metric]
        ax.hist(scores, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.4f}')
        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / f'{dataset_name}_rouge_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROUGE distribution plot saved: {output_path}")
    plt.close()


def plot_comparison(val_scores, test_scores, output_dir):
    """Plot comparison between validation and test scores"""
    metrics = ['rouge1', 'rouge2', 'rougeL']
    val_means = [val_scores[m] for m in metrics]
    test_means = [test_scores[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, val_means, width, label='Validation', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, test_means, width, label='Test', alpha=0.8, color='coral')
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Validation vs Test Performance', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['ROUGE-1', 'ROUGE-2', 'ROUGE-L'], fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'val_vs_test_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved: {output_path}")
    plt.close()


def save_sample_predictions(results, output_dir, num_samples=10):
    """Save sample predictions to text file"""
    output_path = output_dir / 'sample_predictions.txt'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SAMPLE PREDICTIONS\n")
        f.write("="*80 + "\n\n")
        
        for i, result in enumerate(results[:num_samples], 1):
            f.write(f"{'='*80}\n")
            f.write(f"EXAMPLE {i}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"SOURCE:\n{result['source']}\n\n")
            f.write(f"REFERENCE:\n{result['target']}\n\n")
            f.write(f"GENERATED:\n{result['generated']}\n\n")
            f.write(f"ROUGE SCORES:\n")
            f.write(f"  - ROUGE-1: {result['rouge1']:.4f}\n")
            f.write(f"  - ROUGE-2: {result['rouge2']:.4f}\n")
            f.write(f"  - ROUGE-L: {result['rougeL']:.4f}\n\n")
    
    print(f"✓ Sample predictions saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Post-Training Analysis')
    parser.add_argument('--checkpoint', type=str, 
                       default='artifacts/checkpoints/full_training_restart/best_model.pt',
                       help='Path to best model checkpoint')
    parser.add_argument('--config', type=str, 
                       default='configs/rtx4070_8gb.yaml',
                       help='Path to config file')
    parser.add_argument('--tokenizer', type=str,
                       default='artifacts/tokenizer/spm.model',
                       help='Path to tokenizer')
    parser.add_argument('--tokenized_dir', type=str,
                       default='data/tokenized',
                       help='Directory with tokenized data')
    parser.add_argument('--output_dir', type=str,
                       default='results',
                       help='Output directory for results and visualizations')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*80)
    print("POST-TRAINING ANALYSIS")
    print("="*80)
    
    # Load model
    model, config, tokenizer, device = load_model_and_config(
        args.checkpoint, args.config, args.tokenizer
    )
    
    # Evaluate on validation set
    val_path = Path(args.tokenized_dir) / 'val.jsonl'
    if val_path.exists():
        val_results, val_avg_scores, val_rouge_scores = evaluate_on_dataset(
            model, tokenizer, val_path, device, args.num_samples
        )
        
        # Save validation results
        pd.DataFrame(val_results).to_csv(output_dir / 'validation_results.csv', index=False)
        save_sample_predictions(val_results, output_dir, num_samples=10)
        plot_rouge_distribution(val_rouge_scores, output_dir, 'validation')
    else:
        print(f"⚠ Validation file not found: {val_path}")
        val_avg_scores = None
    
    # Evaluate on test set if available
    test_path = Path(args.tokenized_dir) / 'test.jsonl'
    if test_path.exists():
        test_results, test_avg_scores, test_rouge_scores = evaluate_on_dataset(
            model, tokenizer, test_path, device, args.num_samples
        )
        
        # Save test results
        pd.DataFrame(test_results).to_csv(output_dir / 'test_results.csv', index=False)
        plot_rouge_distribution(test_rouge_scores, output_dir, 'test')
        
        # Plot comparison if both exist
        if val_avg_scores:
            plot_comparison(val_avg_scores, test_avg_scores, output_dir)
    else:
        print(f"⚠ Test file not found: {test_path}")
    
    # Plot training metrics
    metrics_path = Path(args.checkpoint).parent.parent.parent / 'logs' / 'full_training_restart' / 'metrics.csv'
    if metrics_path.exists():
        plot_training_metrics(metrics_path, output_dir)
    else:
        print(f"⚠ Metrics file not found: {metrics_path}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir.absolute()}")
    print(f"  - validation_results.csv")
    print(f"  - test_results.csv (if available)")
    print(f"  - sample_predictions.txt")
    print(f"  - training_metrics.png")
    print(f"  - validation_rouge_distribution.png")
    print(f"  - test_rouge_distribution.png (if available)")
    print(f"  - val_vs_test_comparison.png (if available)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
