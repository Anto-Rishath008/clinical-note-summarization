"""
Comprehensive Evaluation Pipeline
Tracks ROUGE over training, compares to Lead-150 baseline, saves results and examples
"""

import torch
import argparse
import yaml
import pandas as pd
import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
import sys
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core import PointerGeneratorModel, get_dataloader


def compute_rouge(predictions, references):
    """Compute ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1, rouge2, rougeL = 0, 0, 0
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougeL += scores['rougeL'].fmeasure
    
    n = len(predictions)
    return {
        'rouge1': rouge1 / n,
        'rouge2': rouge2 / n,
        'rougeL': rougeL / n
    }


def lead_n_baseline(src_ids, n_tokens, pad_id):
    """Extract first N tokens from source"""
    src_clean = [t for t in src_ids if t != pad_id]
    return src_clean[:n_tokens]


def evaluate_checkpoint(checkpoint_path, tokenized_dir, tokenizer, config, 
                        max_eval_samples=500, lead_n=150, split='val',
                        save_examples=True, beam_size=4):
    """
    Evaluate a checkpoint and compare to Lead-N baseline.
    
    Returns:
        dict with model_scores, baseline_scores, and metadata
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    step = checkpoint.get('step', 0)
    epoch = checkpoint.get('epoch', 0)
    
    print(f"  Step: {step}, Epoch: {epoch}")
    
    # Load data
    data_path = Path(tokenized_dir) / f'{split}.parquet'
    print(f"Loading {split} data from {data_path}...")
    
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
    
    # Generate predictions
    print(f"Generating predictions (max {max_eval_samples} samples)...")
    
    model_predictions = []
    baseline_predictions = []
    references = []
    
    max_batches = min(max_eval_samples, len(dataloader))
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=max_batches, desc="Evaluating")):
            if i >= max_batches:
                break
            
            src_ids = batch['src_ids'].to(device)
            src_lens = batch['src_lens'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            
            # Model generation
            generated_ids, scores = model.generate(
                src_ids, src_lens,
                beam_size=beam_size,
                max_length=config['model']['max_target_len'],
                min_length=config['decoding']['min_length'],
                length_penalty=config['decoding']['length_penalty'],
                no_repeat_ngram=config['decoding'].get('no_repeat_ngram', 3)
            )
            
            # Extract tokens
            gen_ids = generated_ids[0].cpu().tolist()
            src_ids_list = src_ids[0].cpu().tolist()
            ref_ids = tgt_ids[0].cpu().tolist()
            
            # Filter special tokens
            gen_ids_clean = [t for t in gen_ids if t not in [config['data']['bos_id'], config['data']['eos_id'], config['data']['pad_id']]]
            ref_ids_clean = [t for t in ref_ids if t not in [config['data']['bos_id'], config['data']['eos_id'], config['data']['pad_id']]]
            
            # Baseline: Lead-N
            baseline_ids = lead_n_baseline(src_ids_list, lead_n, config['data']['pad_id'])
            
            # Decode
            model_text = tokenizer.decode(gen_ids_clean)
            baseline_text = tokenizer.decode(baseline_ids)
            ref_text = tokenizer.decode(ref_ids_clean)
            
            model_predictions.append(model_text)
            baseline_predictions.append(baseline_text)
            references.append(ref_text)
    
    # Compute ROUGE scores
    print("Computing ROUGE scores...")
    model_scores = compute_rouge(model_predictions, references)
    baseline_scores = compute_rouge(baseline_predictions, references)
    
    # Print results
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}\n")
    
    print(f"Model (step {step}):")
    print(f"  ROUGE-1: {model_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {model_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {model_scores['rougeL']:.4f}")
    print()
    
    print(f"Lead-{lead_n} Baseline:")
    print(f"  ROUGE-1: {baseline_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {baseline_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {baseline_scores['rougeL']:.4f}")
    print()
    
    # Delta
    delta_r1 = model_scores['rouge1'] - baseline_scores['rouge1']
    delta_r2 = model_scores['rouge2'] - baseline_scores['rouge2']
    delta_rL = model_scores['rougeL'] - baseline_scores['rougeL']
    
    print(f"Delta (Model - Baseline):")
    print(f"  ROUGE-1: {delta_r1:+.4f}")
    print(f"  ROUGE-2: {delta_r2:+.4f}")
    print(f"  ROUGE-L: {delta_rL:+.4f}")
    
    if delta_rL > 0:
        print(f"\n✅ Model BEATS baseline by {delta_rL:.4f} ROUGE-L!")
    else:
        print(f"\n❌ Model is {abs(delta_rL):.4f} ROUGE-L BELOW baseline")
    
    print()
    
    # Save results to CSV
    results_dir = Path('experiments')
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / 'stage1_fromscratch_results.csv'
    
    result_row = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': str(checkpoint_path),
        'step': step,
        'epoch': epoch,
        'split': split,
        'num_samples': len(model_predictions),
        'model_rouge1': model_scores['rouge1'],
        'model_rouge2': model_scores['rouge2'],
        'model_rougeL': model_scores['rougeL'],
        'baseline_rouge1': baseline_scores['rouge1'],
        'baseline_rouge2': baseline_scores['rouge2'],
        'baseline_rougeL': baseline_scores['rougeL'],
        'delta_rouge1': delta_r1,
        'delta_rouge2': delta_r2,
        'delta_rougeL': delta_rL,
        'beats_baseline': delta_rL > 0
    }
    
    if results_file.exists():
        results_df = pd.read_csv(results_file)
        results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        results_df = pd.DataFrame([result_row])
    
    results_df.to_csv(results_file, index=False)
    print(f"Results appended to {results_file}")
    
    # Save examples
    if save_examples:
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        
        examples_file = reports_dir / f'examples_step{step}.txt'
        
        with open(examples_file, 'w', encoding='utf-8') as f:
            f.write(f"Evaluation Examples - Step {step}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Model ROUGE-L: {model_scores['rougeL']:.4f}\n")
            f.write(f"Baseline ROUGE-L: {baseline_scores['rougeL']:.4f}\n")
            f.write(f"Delta: {delta_rL:+.4f}\n\n")
            f.write(f"{'='*80}\n\n")
            
            for i in range(min(20, len(model_predictions))):
                f.write(f"Example {i+1}\n")
                f.write("-"*80 + "\n\n")
                f.write(f"REFERENCE:\n{references[i]}\n\n")
                f.write(f"MODEL:\n{model_predictions[i]}\n\n")
                f.write(f"BASELINE (Lead-{lead_n}):\n{baseline_predictions[i]}\n\n")
                f.write("="*80 + "\n\n")
        
        print(f"Examples saved to {examples_file}")
    
    return {
        'model_scores': model_scores,
        'baseline_scores': baseline_scores,
        'delta': {'rouge1': delta_r1, 'rouge2': delta_r2, 'rougeL': delta_rL},
        'step': step,
        'epoch': epoch
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation Pipeline')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--tokenized_dir', type=str, default='data/tokenized',
                        help='Directory with tokenized data')
    parser.add_argument('--tokenizer_path', type=str, default='artifacts/tokenizer/spm.model',
                        help='Path to SentencePiece model')
    parser.add_argument('--max_samples', type=int, default=500,
                        help='Maximum samples to evaluate')
    parser.add_argument('--lead_n', type=int, default=150,
                        help='Number of lead tokens for baseline')
    parser.add_argument('--split', type=str, default='val',
                        help='Data split to evaluate')
    parser.add_argument('--beam_size', type=int, default=4,
                        help='Beam search size')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer_path)
    
    # Evaluate
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenized_dir=args.tokenized_dir,
        tokenizer=tokenizer,
        config=config,
        max_eval_samples=args.max_samples,
        lead_n=args.lead_n,
        split=args.split,
        save_examples=True,
        beam_size=args.beam_size
    )
