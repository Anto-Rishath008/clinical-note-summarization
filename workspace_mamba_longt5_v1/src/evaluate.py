"""
Evaluation Script for Mamba-Transformer Hybrid Model
======================================================

Load trained model and evaluate on test data.
Computes ROUGE-1/2/L metrics.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.cuda.amp import autocast

sys.path.insert(0, str(Path(__file__).parent))

from model import MambaTransformerConfig, MambaTransformerModel, build_model
from data_loader import DataConfig, create_dataloaders, load_tokenizer


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores"""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("ERROR: rouge_score not installed. Run: pip install rouge-score")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    r1_scores, r2_scores, rl_scores = [], [], []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_scores.append(scores['rouge1'].fmeasure)
        r2_scores.append(scores['rouge2'].fmeasure)
        rl_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': sum(r1_scores) / len(r1_scores) if r1_scores else 0.0,
        'rouge2': sum(r2_scores) / len(r2_scores) if r2_scores else 0.0,
        'rougeL': sum(rl_scores) / len(rl_scores) if rl_scores else 0.0,
    }


@torch.no_grad()
def evaluate_model(
    model: MambaTransformerModel,
    val_loader,
    tokenizer,
    device: torch.device,
    max_samples: int = None,
    output_file: str = None,
    max_gen_len: int = 512,
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        tokenizer: SentencePiece tokenizer
        device: Device to run on
        max_samples: Maximum samples to evaluate (None for all)
        output_file: Optional file to save predictions
        max_gen_len: Maximum generation length (default 256)
    
    Returns:
        Dictionary with ROUGE scores
    """
    model.eval()
    
    predictions = []
    references = []
    sources = []
    
    n_samples = 0
    
    for batch_idx, batch in enumerate(val_loader):
        src = batch['src'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        src_mask = batch['src_mask'].to(device)
        
        # Generate predictions using beam search for better quality
        with autocast(enabled=True):
            generated = model.generate(
                src, src_mask,
                max_len=max_gen_len,
                greedy=False,
                beam_size=4,
                length_penalty=0.8,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
            )
        
        for i in range(src.size(0)):
            if max_samples and n_samples >= max_samples:
                break
            
            pred_ids = generated[i].cpu().tolist()
            ref_ids = tgt_output[i].cpu().tolist()
            src_ids = src[i].cpu().tolist()
            
            # Decode (skip special tokens)
            pred_text = tokenizer.DecodeIds([t for t in pred_ids if t not in [0, 1, 2, 3]])
            ref_text = tokenizer.DecodeIds([t for t in ref_ids if t not in [0, 1, 2, 3]])
            src_text = tokenizer.DecodeIds([t for t in src_ids if t not in [0, 1, 2, 3]])
            
            predictions.append(pred_text)
            references.append(ref_text)
            sources.append(src_text)
            n_samples += 1
        
        if max_samples and n_samples >= max_samples:
            break
        
        if batch_idx % 10 == 0:
            print(f"Evaluated {n_samples} samples...")
    
    # Compute ROUGE
    print(f"\nComputing ROUGE on {len(predictions)} samples...")
    rouge_scores = compute_rouge(predictions, references)
    
    # Save predictions if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (src, pred, ref) in enumerate(zip(sources, predictions, references)):
                f.write(f"=== Sample {i+1} ===\n")
                f.write(f"SOURCE:\n{src[:500]}...\n\n")
                f.write(f"PREDICTION:\n{pred}\n\n")
                f.write(f"REFERENCE:\n{ref}\n\n")
                f.write("-" * 80 + "\n\n")
        print(f"Predictions saved to: {output_file}")
    
    return rouge_scores


def load_model(checkpoint_path: str, device: torch.device) -> MambaTransformerModel:
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model config
    model_config = MambaTransformerConfig.from_dict(config.get('model', {}))
    
    # Build model
    model = build_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from step {checkpoint.get('step', 'unknown')}")
    print(f"Best ROUGE-L at save: {checkpoint.get('best_rouge_l', 'unknown')}")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description='Evaluate Mamba-Transformer Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config (optional, uses checkpoint config)')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to evaluate')
    parser.add_argument('--max_gen_len', type=int, default=256, help='Maximum generation length (default: 256)')
    parser.add_argument('--output', type=str, default=None, help='Output file for predictions')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Override config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Create data config and loader
    data_config = DataConfig.from_dict(config)
    _, val_loader, tokenizer = create_dataloaders(
        config=data_config,
        batch_size=4,
        num_workers=0,
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"  Decoding: GREEDY (deterministic)")
    print(f"  Max generation length: {args.max_gen_len}")
    
    rouge_scores = evaluate_model(
        model=model,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        max_samples=args.max_samples,
        output_file=args.output,
        max_gen_len=args.max_gen_len,
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
