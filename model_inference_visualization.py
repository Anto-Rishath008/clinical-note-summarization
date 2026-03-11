"""
Model Inference and Visualization Script
=========================================
This script loads the trained Pointer-Generator model and performs comprehensive analysis including:
- Loading the best saved model checkpoint
- Generating summaries for test set inputs
- Displaying 10 Original vs Generated Summary comparisons
- Computing ROUGE scores
"""

import torch
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import warnings
import sys

warnings.filterwarnings('ignore')

# Import custom modules
sys.path.append('.')
from src.core import PointerGeneratorModel
import sentencepiece as spm
from rouge_score import rouge_scorer

# =============================================================================
# Configuration
# =============================================================================
CONFIG_PATH = 'configs/rtx4070_8gb.yaml'
CHECKPOINT_PATH = 'artifacts/checkpoints/full_training_restart/best_model.pt'
TOKENIZER_PATH = 'artifacts/tokenizer/spm.model'
NUM_SAMPLES = 10


def load_model_and_tokenizer():
    """Load configuration, tokenizer, and model checkpoint."""
    print("=" * 60)
    print("Loading Model and Tokenizer")
    print("=" * 60)
    
    # Load configuration
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Configuration loaded from {CONFIG_PATH}")
    print(f"\nModel Configuration:")
    print(f"  - Embedding dim: {config['model']['emb_dim']}")
    print(f"  - Hidden dim: {config['model']['hidden_dim']}")
    print(f"  - Num layers: {config['model']['num_layers']}")
    print(f"  - Pointer-Gen: {config['model']['pointer_gen']}")
    print(f"  - Coverage: {config['model']['use_coverage']}")
    
    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)
    vocab_size = tokenizer.get_piece_size()
    print(f"\n✓ Tokenizer loaded: {vocab_size} vocab size")
    
    # Add tokenizer info to config for model
    config['data'] = {
        'vocab_size': vocab_size,
        'pad_id': tokenizer.pad_id(),
        'bos_id': tokenizer.bos_id(),
        'eos_id': tokenizer.eos_id(),
    }
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    print(f"\n✓ Checkpoint loaded from {CHECKPOINT_PATH}")
    print(f"  - Training step: {checkpoint['step']}")
    print(f"  - Best ROUGE: {checkpoint.get('best_rouge', 'N/A')}")
    
    # Create model instance using config
    model = PointerGeneratorModel(config)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model loaded successfully!")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Device: {device}")
    
    return model, tokenizer, config, vocab_size, device


def load_test_data():
    """Load test dataset."""
    print("\n" + "=" * 60)
    print("Loading Test Dataset")
    print("=" * 60)
    
    test_data = pd.read_parquet('data/tokenized/test.parquet')
    
    print(f"✓ Test dataset loaded: {len(test_data)} samples")
    print(f"\nDataset columns: {list(test_data.columns)}")
    print(f"\nDataset Statistics:")
    print(f"  - Source text lengths: mean={test_data['src_ids'].apply(len).mean():.1f}")
    print(f"  - Target text lengths: mean={test_data['tgt_ids'].apply(len).mean():.1f}")
    
    return test_data


def generate_summary(model, src_ids, tokenizer, device, config, max_length=192):
    """
    Generate summary using beam search.
    
    Returns:
        generated_text: generated summary text
        gen_ids: generated token IDs
    """
    model.eval()
    
    # Truncate source to model's max input length
    chunk_len = config['model']['chunk_len']
    num_chunks = config['model']['num_chunks']
    max_src_len = chunk_len * num_chunks  # 128 * 6 = 768
    src_ids = src_ids[:max_src_len]
    
    # Prepare input
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    src_lengths = torch.tensor([len(src_ids)], dtype=torch.long).to(device)
    
    # Generate using model's generate method
    with torch.no_grad():
        generated_ids, scores = model.generate(
            src_ids=src_tensor,
            src_lengths=src_lengths,
            beam_size=4,
            max_length=max_length,
            min_length=30,
            length_penalty=1.0,
            no_repeat_ngram=3
        )
    
    # Decode
    gen_ids = generated_ids[0].cpu().tolist()
    
    # Remove padding and special tokens for decoding
    eos_id = tokenizer.eos_id()
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    
    gen_text = tokenizer.decode(gen_ids)
    
    return gen_text, gen_ids


def generate_predictions(model, tokenizer, test_data, device, config, num_samples=10):
    """Generate predictions on test samples."""
    print("\n" + "=" * 60)
    print(f"Generating Predictions for {num_samples} Samples")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    sample_indices = np.random.choice(len(test_data), num_samples, replace=False)
    results = []
    
    print(f"Generating summaries for {num_samples} samples...")
    for idx in tqdm(sample_indices):
        row = test_data.iloc[idx]
        src_ids = row['src_ids']
        tgt_ids = row['tgt_ids']
        
        # Convert to Python list of ints (important for SentencePiece)
        if hasattr(src_ids, 'tolist'):
            src_ids = src_ids.tolist()
        elif not isinstance(src_ids, list):
            src_ids = list(src_ids)
        src_ids = [int(x) for x in src_ids]
        
        if hasattr(tgt_ids, 'tolist'):
            tgt_ids = tgt_ids.tolist()
        elif not isinstance(tgt_ids, list):
            tgt_ids = list(tgt_ids)
        tgt_ids = [int(x) for x in tgt_ids]
        
        # Generate
        gen_text, gen_ids = generate_summary(model, src_ids, tokenizer, device, config, max_length=192)
        
        # Decode texts
        src_text = tokenizer.decode(src_ids)
        tgt_text = tokenizer.decode(tgt_ids)
        
        results.append({
            'index': idx,
            'source': src_text,
            'reference': tgt_text,
            'generated': gen_text,
            'src_ids': src_ids,
            'gen_ids': gen_ids,
            'tgt_ids': tgt_ids,
            'src_len': len(src_ids),
            'gen_len': len(gen_ids),
            'ref_len': len(tgt_ids)
        })
    
    print(f"✓ Generated {len(results)} summaries!")
    return results


def display_original_vs_summary(results, num_display=10):
    """Display Original vs Generated Summary comparisons."""
    print("\n" + "=" * 100)
    print("📋 ORIGINAL TEXT vs GENERATED SUMMARY - 10 Examples")
    print("=" * 100)
    
    for i, result in enumerate(results[:num_display]):
        print(f"\n{'─' * 100}")
        print(f"📌 EXAMPLE {i+1} (Sample Index: {result['index']})")
        print(f"{'─' * 100}")
        
        # Original text (truncated for readability)
        print(f"\n🔵 ORIGINAL TEXT ({result['src_len']} tokens):")
        print("-" * 50)
        original_text = result['source']
        if len(original_text) > 800:
            print(f"{original_text[:800]}...\n[Truncated - full length: {len(original_text)} chars]")
        else:
            print(original_text)
        
        # Generated summary
        print(f"\n🟢 GENERATED SUMMARY ({result['gen_len']} tokens):")
        print("-" * 50)
        print(result['generated'])
        
        # Reference summary for comparison
        print(f"\n🟡 REFERENCE SUMMARY ({result['ref_len']} tokens):")
        print("-" * 50)
        print(result['reference'])
        
        print()
    
    print("=" * 100)
    print(f"✅ Displayed {min(num_display, len(results))} Original vs Summary comparisons")
    print("=" * 100)


def compute_rouge_scores(results):
    """Compute ROUGE scores for all samples."""
    print("\n" + "=" * 60)
    print("Computing ROUGE Scores")
    print("=" * 60)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = []
    for result in results:
        scores = scorer.score(result['reference'], result['generated'])
        rouge_scores.append({
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        })
    
    rouge_df = pd.DataFrame(rouge_scores)
    
    print("✓ ROUGE scores computed for all samples")
    print(f"\n📊 ROUGE Score Statistics:")
    print(f"   ROUGE-1: {rouge_df['rouge1'].mean():.4f} (±{rouge_df['rouge1'].std():.4f})")
    print(f"   ROUGE-2: {rouge_df['rouge2'].mean():.4f} (±{rouge_df['rouge2'].std():.4f})")
    print(f"   ROUGE-L: {rouge_df['rougeL'].mean():.4f} (±{rouge_df['rougeL'].std():.4f})")
    
    # Show individual scores
    print(f"\n📊 Per-Sample ROUGE Scores:")
    print("-" * 60)
    for i, (result, scores) in enumerate(zip(results, rouge_scores)):
        print(f"  Sample {i+1} (idx {result['index']}): R1={scores['rouge1']:.4f}, R2={scores['rouge2']:.4f}, RL={scores['rougeL']:.4f}")
    
    return rouge_df


def analyze_lengths(results):
    """Analyze summary lengths."""
    print("\n" + "=" * 60)
    print("Summary Length Analysis")
    print("=" * 60)
    
    src_lengths = [r['src_len'] for r in results]
    ref_lengths = [r['ref_len'] for r in results]
    gen_lengths = [r['gen_len'] for r in results]
    compression_ratios = [g/s if s > 0 else 0 for g, s in zip(gen_lengths, src_lengths)]
    
    print(f"\n📏 Length Statistics:")
    print(f"  - Source: {np.mean(src_lengths):.1f} ± {np.std(src_lengths):.1f} tokens")
    print(f"  - Reference: {np.mean(ref_lengths):.1f} ± {np.std(ref_lengths):.1f} tokens")
    print(f"  - Generated: {np.mean(gen_lengths):.1f} ± {np.std(gen_lengths):.1f} tokens")
    print(f"  - Average compression: {np.mean(compression_ratios):.1%}")
    
    return src_lengths, ref_lengths, gen_lengths, compression_ratios


def export_results(results, rouge_df, output_dir='results/notebook_analysis'):
    """Export results to CSV files."""
    print("\n" + "=" * 60)
    print("Exporting Results")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    src_lengths = [r['src_len'] for r in results]
    ref_lengths = [r['ref_len'] for r in results]
    gen_lengths = [r['gen_len'] for r in results]
    compression_ratios = [g/s if s > 0 else 0 for g, s in zip(gen_lengths, src_lengths)]
    
    # Export detailed results to CSV
    export_df = pd.DataFrame({
        'sample_idx': [r['index'] for r in results],
        'source': [r['source'][:500] + '...' if len(r['source']) > 500 else r['source'] for r in results],
        'reference': [r['reference'] for r in results],
        'generated': [r['generated'] for r in results],
        'rouge1': rouge_df['rouge1'],
        'rouge2': rouge_df['rouge2'],
        'rougeL': rouge_df['rougeL'],
        'src_length': src_lengths,
        'ref_length': ref_lengths,
        'gen_length': gen_lengths,
        'compression_ratio': compression_ratios,
    })
    
    csv_path = output_dir / 'detailed_results.csv'
    export_df.to_csv(csv_path, index=False)
    print(f"✓ Results exported to {csv_path}")
    
    # Export summary statistics
    summary_stats = {
        'Metric': ['ROUGE-1 Mean', 'ROUGE-2 Mean', 'ROUGE-L Mean',
                   'ROUGE-1 Std', 'ROUGE-2 Std', 'ROUGE-L Std',
                   'Avg Compression', 'Total Samples'],
        'Value': [
            rouge_df['rouge1'].mean(),
            rouge_df['rouge2'].mean(),
            rouge_df['rougeL'].mean(),
            rouge_df['rouge1'].std(),
            rouge_df['rouge2'].std(),
            rouge_df['rougeL'].std(),
            np.mean(compression_ratios),
            len(results)
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_path = output_dir / 'summary_statistics.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary statistics exported to {summary_path}")
    
    print(f"\n✅ All results saved in {output_dir}")


def main():
    """Main function to run all analysis."""
    print("\n" + "=" * 80)
    print("   MODEL INFERENCE AND VISUALIZATION SCRIPT")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    Path('results/notebook_analysis').mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer, config, vocab_size, device = load_model_and_tokenizer()
    
    # Load test data
    test_data = load_test_data()
    
    # Generate predictions
    results = generate_predictions(model, tokenizer, test_data, device, config, num_samples=NUM_SAMPLES)
    
    # Display 10 Original vs Summary comparisons
    display_original_vs_summary(results, num_display=10)
    
    # Compute ROUGE scores
    rouge_df = compute_rouge_scores(results)
    
    # Analyze lengths
    analyze_lengths(results)
    
    # Export results
    export_results(results, rouge_df)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("   ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Final Summary:")
    print(f"   - Samples analyzed: {len(results)}")
    print(f"   - Average ROUGE-1: {rouge_df['rouge1'].mean():.4f}")
    print(f"   - Average ROUGE-2: {rouge_df['rouge2'].mean():.4f}")
    print(f"   - Average ROUGE-L: {rouge_df['rougeL'].mean():.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
