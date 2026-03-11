"""
Demo script to show model inference with examples and performance metrics.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import pandas as pd
import sentencepiece as spm
from src.core import PointerGeneratorModel, beam_search_decode
from rouge_score import rouge_scorer

def main():
    # Load config
    with open('configs/rtx4070_8gb.yaml') as f:
        config = yaml.safe_load(f)

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('artifacts/tokenizer/spm.model')
    vocab_size = tokenizer.get_piece_size()
    print(f'Vocab size: {vocab_size}')

    # Add data config for model initialization
    config['data'] = {
        'vocab_size': vocab_size,
        'pad_id': 3,
        'bos_id': 1,
        'eos_id': 2,
        'unk_id': 0
    }

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = PointerGeneratorModel(config)

    # Load best checkpoint
    checkpoint = torch.load('artifacts/checkpoints/full_training_final/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    step = checkpoint.get('global_step', 'unknown')
    best_rouge = checkpoint.get('best_rouge', 'unknown')
    print(f'Model loaded from step {step}')
    print(f'Best ROUGE-L: {best_rouge}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    print()

    # Load validation data
    val_df = pd.read_parquet('data/tokenized/val.parquet')
    print(f'Validation samples: {len(val_df)}')

    # ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Test on 5 examples
    print('='*80)
    print('INFERENCE EXAMPLES')
    print('='*80)

    all_r1, all_r2, all_rl = [], [], []

    for i in range(5):
        row = val_df.iloc[i]
        src_text = row['source']
        ref_summary = row['target']
        
        # Tokenize
        src_ids = tokenizer.encode(src_text, out_type=int)
        
        # Truncate to max length
        max_src_len = config['model']['chunk_len'] * config['model']['num_chunks']
        src_ids = src_ids[:max_src_len]
        
        # Convert to tensor
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_lens = torch.tensor([len(src_ids)], dtype=torch.long, device=device)
        
        # Generate
        with torch.no_grad():
            generated_ids = beam_search_decode(
                model, src_tensor, src_lens, tokenizer,
                max_len=150,
                beam_size=4,
                device=device
            )
        
        # Decode
        generated_text = tokenizer.decode(generated_ids[0])
        
        # Calculate ROUGE
        scores = scorer.score(ref_summary, generated_text)
        r1 = scores['rouge1'].fmeasure
        r2 = scores['rouge2'].fmeasure
        rl = scores['rougeL'].fmeasure
        all_r1.append(r1)
        all_r2.append(r2)
        all_rl.append(rl)
        
        print(f'\n{"="*80}')
        print(f'EXAMPLE {i+1}')
        print(f'{"="*80}')
        print(f'\n[SOURCE] (first 500 chars):')
        print(src_text[:500] + '...')
        print(f'\n[REFERENCE SUMMARY]:')
        print(ref_summary[:600])
        print(f'\n[GENERATED SUMMARY]:')
        print(generated_text)
        print(f'\n[SCORES] ROUGE-1: {r1:.4f} | ROUGE-2: {r2:.4f} | ROUGE-L: {rl:.4f}')

    # Average scores
    print('\n' + '='*80)
    print('AVERAGE SCORES (5 examples)')
    print('='*80)
    print(f'ROUGE-1: {sum(all_r1)/len(all_r1):.4f}')
    print(f'ROUGE-2: {sum(all_r2)/len(all_r2):.4f}')
    print(f'ROUGE-L: {sum(all_rl)/len(all_rl):.4f}')

if __name__ == '__main__':
    main()
