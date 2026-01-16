"""
Lead-K Baseline: Extract first K sentences or first N tokens from source
Simple but surprisingly strong baseline for summarization
"""

import argparse
import pandas as pd
from pathlib import Path
from rouge_score import rouge_scorer
import sentencepiece as spm
from tqdm import tqdm
import re


def compute_rouge(predictions, references):
    """Compute ROUGE scores"""
    from rouge_score import rouge_scorer
    
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


# Baseline 1: Lead-K (first K sentences)
def lead_k_baseline(src_text, k=3):
    """Extract first K sentences as summary"""
    sentences = src_text.split('.')[:k]
    return '. '.join(sentences) + '.'


print("Loading validation data...")
val_path = Path(tokenized_dir) / 'val.parquet'
val_df = pd.read_parquet(val_path)

print(f"Validation set size: {len(val_df)}")

# Sample subset for quick baseline
sample_size = min(500, len(val_df))
val_sample = val_df.sample(n=sample_size, random_state=42)

print(f"\nEvaluating on {len(val_sample)} examples...")

# Load tokenizer
tokenizer_path = Path(r"artifacts\tokenizer\spm.model")
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(str(tokenizer_path))

# Compute baselines
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Lead-3 baseline: take first 3 sentences
def compute_lead_k(k=3):
    predictions = []
    references = []
    
    for idx in range(min(len(val_df), 500)):  # Evaluate on subset for speed
        row = val_df.iloc[idx]
        src_text = tokenizer.decode(row['src_ids'][:1536])
        tgt_text = tokenizer.decode([t for t in row['tgt_ids'] if t not in [0, 1, 2, 3]])
        
        # Lead-K baseline: take first K sentences
        sentences = src_text.split('. ')
        lead_summary = '. '.join(sentences[:K]) + '.'
        
        predictions.append(pred_text)
        references.append(ref_text)
    
    return predictions, references

print("Not implemented yet - need to create proper baseline evaluation")
