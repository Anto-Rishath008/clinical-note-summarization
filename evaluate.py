"""
Evaluation Script
Load checkpoint and compute ROUGE on validation or test set
"""

import torch
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sentencepiece as spm
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core import PointerGeneratorModel, get_dataloader, RougeMetric


def load_config_from_checkpoint(checkpoint_path):
    """Load config from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint['config']


def evaluate(checkpoint_path, tokenized_dir, split='val', beam_size=4, max_eval_examples=-1):
    """
    Evaluate model on validation or test set.
    
    Args:
        checkpoint_path: path to model checkpoint
        tokenized_dir: directory with tokenized data
        split: 'val' or 'test'
        beam_size: beam search width
        max_eval_examples: max examples to evaluate (-1 = all)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    print(f"Checkpoint info:")
    print(f"  Step: {checkpoint.get('step', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best ROUGE-L: {checkpoint.get('best_rouge', 0):.4f}")
    
    # Load tokenizer
    tokenizer_path = Path(config['paths']['tokenizer_dir']) / 'spm.model'
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(tokenizer_path))
    
    # Create dataloader
    print(f"Loading {split} data...")
    data_path = Path(tokenized_dir) / f'{split}.parquet'
    
    dataloader = get_dataloader(
        str(data_path),
        batch_size=1,  # Evaluate one at a time for beam search
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
    
    # Evaluate
    print(f"\n{'='*60}")
    print(f"Evaluating on {split} set")
    print(f"{'='*60}\n")
    
    metric = RougeMetric()
    predictions = []
    references = []
    
    max_batches = len(dataloader) if max_eval_examples == -1 else max_eval_examples
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=max_batches, desc="Generating")):
            if i >= max_batches:
                break
            
            src_ids = batch['src_ids'].to(device)
            tgt_ids = batch['tgt_ids'].to(device)
            src_lens = batch['src_lens'].to(device)
            
            # Generate
            generated_ids, scores = model.generate(
                src_ids, src_lens,
                beam_size=beam_size,
                max_length=config['model']['max_target_len'],
                min_length=config['decoding']['min_length'],
                length_penalty=config['decoding']['length_penalty'],
                no_repeat_ngram=config['decoding']['no_repeat_ngram']
            )
            
            # Decode
            gen_ids = generated_ids[0].cpu().tolist()
            ref_ids = tgt_ids[0].cpu().tolist()
            
            # Remove special tokens
            gen_ids = [t for t in gen_ids if t not in [model.bos_id, model.eos_id, model.pad_id]]
            ref_ids = [t for t in ref_ids if t not in [model.bos_id, model.eos_id, model.pad_id]]
            
            # Decode to text
            pred_text = tokenizer.decode(gen_ids)
            ref_text = tokenizer.decode(ref_ids)
            
            predictions.append(pred_text)
            references.append(ref_text)
    
    # Compute ROUGE
    rouge_scores = metric.compute(predictions, references)
    
    print(f"\n{'='*60}")
    print(f"Results on {split} set ({len(predictions)} examples)")
    print(f"{'='*60}")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")\n    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")\n    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"{'='*60}\n")
    
    # Save predictions
    predictions_dir = Path(config['paths']['predictions_dir']) / Path(checkpoint_path).parent.name
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_path = predictions_dir / f'{split}_predictions.csv'
    predictions_df = pd.DataFrame({
        'prediction': predictions,
        'reference': references
    })
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    # Print some examples
    print(f"\n{'='*60}")
    print("Sample Predictions")
    print(f"{'='*60}\n")
    
    for i in range(min(3, len(predictions))):
        print(f"Example {i+1}:")
        print(f"Reference: {references[i][:300]}...")
        print(f"Predicted: {predictions[i][:300]}...")
        print()
    
    return rouge_scores


def main():
    parser = argparse.ArgumentParser(description='Evaluate Pointer-Generator Model')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--tokenized_dir', type=str, required=True, help='Directory with tokenized data')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Split to evaluate')
    parser.add_argument('--beam_size', type=int, default=4, help='Beam search size')
    parser.add_argument('--max_examples', type=int, default=-1, help='Max examples to evaluate (-1 = all)')
    
    args = parser.parse_args()
    
    evaluate(args.ckpt, args.tokenized_dir, args.split, args.beam_size, args.max_examples)


if __name__ == '__main__':
    main()
