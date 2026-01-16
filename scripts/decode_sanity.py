"""
Decode Sanity Check
Analyzes generation quality: repetition, length, EOS patterns, copy ratio
"""

import torch
import argparse
import pandas as pd
import sentencepiece as spm
from pathlib import Path
from collections import Counter
import re
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core import PointerGeneratorModel, get_dataloader


def analyze_repetition(tokens):
    """Compute repetition metrics"""
    if len(tokens) < 3:
        return {'trigram_repeat_rate': 0.0, 'unique_token_ratio': 1.0}
    
    # Trigram repetition
    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    trigram_counts = Counter(trigrams)
    repeated_trigrams = sum(1 for count in trigram_counts.values() if count > 1)
    trigram_repeat_rate = repeated_trigrams / len(trigrams) if len(trigrams) > 0 else 0
    
    # Unique token ratio
    unique_tokens = len(set(tokens))
    unique_ratio = unique_tokens / len(tokens) if len(tokens) > 0 else 0
    
    return {
        'trigram_repeat_rate': trigram_repeat_rate,
        'unique_token_ratio': unique_ratio
    }


def analyze_copying(generated_ids, source_ids):
    """Compute how many tokens were copied from source"""
    gen_set = set(generated_ids)
    src_set = set(source_ids)
    copied = len(gen_set & src_set)
    copy_ratio = copied / len(gen_set) if len(gen_set) > 0 else 0
    return copy_ratio


def decode_sanity_check(checkpoint_path, tokenized_dir, num_examples=20):
    """Run decode sanity checks"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Load tokenizer
    tokenizer_path = Path(config['paths']['tokenizer_dir']) / 'spm.model'
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(tokenizer_path))
    
    # Load data
    data_path = Path(tokenized_dir) / 'val.parquet'
    print(f"Loading data from {data_path}...")
    
    dataloader = get_dataloader(
        str(data_path),
        batch_size=1,
        shuffle=False,
        max_src_len=config['model']['chunk_len'] * config['model']['num_chunks'],
        max_tgt_len=config['model']['max_target_len'],
        pad_id=config['data']['pad_id']
    )
    
    # Create model
    print("Creating model...")
    model = PointerGeneratorModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n{'='*80}")
    print(f"DECODE SANITY CHECK ({num_examples} examples)")
    print(f"{'='*80}\n")
    
    # Metrics
    lengths = []
    repetition_rates = []
    unique_ratios = []
    copy_ratios = []
    early_eos_count = 0
    
    examples = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_examples:
                break
            
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            src_lens = batch['src_lens'].to(device)
            
            # Generate
            generated_ids, scores = model.generate(
                src_ids, src_lens,
                beam_size=config['decoding']['beam_size'],
                max_length=config['decoding']['max_length'],
                min_length=config['decoding']['min_length'],
                length_penalty=config['decoding']['length_penalty'],
                no_repeat_ngram=config['decoding']['no_repeat_ngram']
            )
            
            # Extract tokens
            gen_ids = generated_ids[0].cpu().tolist()
            src_ids_list = src_ids[0].cpu().tolist()
            ref_ids = tgt_ids[0].cpu().tolist()
            
            # Filter special tokens
            gen_ids_clean = [t for t in gen_ids if t not in [config['data']['bos_id'], config['data']['eos_id'], config['data']['pad_id']]]
            src_ids_clean = [t for t in src_ids_list if t not in [config['data']['pad_id']]]
            ref_ids_clean = [t for t in ref_ids if t not in [config['data']['bos_id'], config['data']['eos_id'], config['data']['pad_id']]]
            
            # Decode
            gen_text = tokenizer.decode(gen_ids_clean)
            ref_text = tokenizer.decode(ref_ids_clean)
            
            # Metrics
            gen_len = len(gen_ids_clean)
            ref_len = len(ref_ids_clean)
            lengths.append(gen_len)
            
            # Check early EOS (generated less than min_length)
            if gen_len < config['decoding']['min_length']:
                early_eos_count += 1
            
            # Repetition
            rep_metrics = analyze_repetition(gen_ids_clean)
            repetition_rates.append(rep_metrics['trigram_repeat_rate'])
            unique_ratios.append(rep_metrics['unique_token_ratio'])
            
            # Copying
            copy_ratio = analyze_copying(gen_ids_clean, src_ids_clean)
            copy_ratios.append(copy_ratio)
            
            examples.append({
                'gen_text': gen_text,
                'ref_text': ref_text,
                'gen_len': gen_len,
                'ref_len': ref_len,
                'trigram_repeat': rep_metrics['trigram_repeat_rate'],
                'unique_ratio': rep_metrics['unique_token_ratio'],
                'copy_ratio': copy_ratio
            })
    
    # Print summary
    print(f"{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Average generation length: {np.mean(lengths):.1f} tokens (std: {np.std(lengths):.1f})")
    print(f"Average reference length: {np.mean([ex['ref_len'] for ex in examples]):.1f} tokens")
    print(f"Trigram repetition rate: {np.mean(repetition_rates):.4f} (std: {np.std(repetition_rates):.4f})")
    print(f"Unique token ratio: {np.mean(unique_ratios):.4f} (std: {np.std(unique_ratios):.4f})")
    print(f"Copy ratio from source: {np.mean(copy_ratios):.4f} (std: {np.std(copy_ratios):.4f})")
    print(f"Early EOS count (< min_length): {early_eos_count}/{num_examples} ({100*early_eos_count/num_examples:.1f}%)")
    
    # Print examples
    print(f"\n{'='*80}")
    print("SAMPLE GENERATIONS (first 5)")
    print(f"{'='*80}\n")
    
    for i, ex in enumerate(examples[:5]):
        print(f"Example {i+1}:")
        print(f"  Gen length: {ex['gen_len']}, Ref length: {ex['ref_len']}")
        print(f"  Trigram repeat: {ex['trigram_repeat']:.4f}, Unique ratio: {ex['unique_ratio']:.4f}, Copy ratio: {ex['copy_ratio']:.4f}")
        print(f"  Reference: {ex['ref_text'][:150]}...")
        print(f"  Generated: {ex['gen_text'][:150]}...")
        print()
    
    # Save results
    results_df = pd.DataFrame(examples)
    results_path = Path('experiments/decode_sanity_results.csv')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"Detailed results saved to {results_path}")
    
    # Summary verdict
    print(f"\n{'='*80}")
    print("DIAGNOSTIC VERDICT")
    print(f"{'='*80}")
    
    issues = []
    if np.mean(repetition_rates) > 0.3:
        issues.append("⚠️  HIGH REPETITION: Trigram repeat rate > 0.3 indicates model is repeating phrases")
    if np.mean(unique_ratios) < 0.5:
        issues.append("⚠️  LOW DIVERSITY: Unique token ratio < 0.5 indicates limited vocabulary usage")
    if early_eos_count / num_examples > 0.5:
        issues.append("⚠️  EARLY EOS: >50% of generations are shorter than min_length")
    if np.mean(lengths) < 30:
        issues.append("⚠️  TOO SHORT: Average generation is < 30 tokens")
    if np.mean(copy_ratios) < 0.3:
        issues.append("⚠️  LOW COPYING: Copy ratio < 0.3 may indicate pointer mechanism not working")
    
    if issues:
        print("Issues detected:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ No major issues detected")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Decode Sanity Check')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--tokenized_dir', type=str, required=True, help='Directory with tokenized data')
    parser.add_argument('--num_examples', type=int, default=20, help='Number of examples to analyze')
    
    args = parser.parse_args()
    
    decode_sanity_check(args.ckpt, args.tokenized_dir, args.num_examples)


if __name__ == '__main__':
    main()
