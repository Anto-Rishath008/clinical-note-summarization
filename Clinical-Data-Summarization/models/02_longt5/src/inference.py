"""
Inference Script for Simplified LongT5 Model
=============================================

This script provides batched inference capabilities for the LongT5 model.
Solves Issue #1: Batched beam search inference

Features:
- Batched generation for efficiency
- Support for both greedy and beam search
- ROUGE evaluation metrics
- CSV output for results

Usage:
    python inference_longt5.py --checkpoint path/to/best_model.pt --input data/test.csv
    python inference_longt5.py --checkpoint path/to/best_model.pt --text "Clinical note here..."
"""

import os
import sys
import csv
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

import yaml
import torch
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try local imports first, then fallback to src folder
try:
    from model import SimplifiedLongT5, LongT5Config, create_model
except ImportError:
    from src.longt5_model import SimplifiedLongT5, LongT5Config, create_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional: ROUGE metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge_score not installed. Install with: pip install rouge-score")


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length: int = 512
    min_length: int = 50
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    early_stopping: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'GenerationConfig':
        gen_cfg = config_dict.get('generation', {})
        return cls(
            max_length=gen_cfg.get('max_length', 512),
            min_length=gen_cfg.get('min_length', 50),
            num_beams=gen_cfg.get('num_beams', 4),
            length_penalty=gen_cfg.get('length_penalty', 1.0),
            no_repeat_ngram_size=gen_cfg.get('no_repeat_ngram_size', 3),
            early_stopping=gen_cfg.get('early_stopping', True),
        )


class LongT5Summarizer:
    """High-level interface for clinical note summarization"""
    
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        logger.info(f"Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        
        # Determine tokenizer path
        if tokenizer_path is None:
            tokenizer_path = self.config.get('data', {}).get(
                'tokenizer_path', 'artifacts/tokenizer/spm.model'
            )
        
        # Load tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        logger.info(f"Loaded tokenizer with vocab size: {self.tokenizer.get_piece_size()}")
        
        # Load model
        self.model = create_model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
        
        # Generation config
        self.gen_config = GenerationConfig.from_dict(self.config)
        
        # Data config
        self.data_config = self.config.get('data', {})
        self.max_src_len = self.data_config.get('max_src_len', 4096)
        self.pad_id = self.data_config.get('pad_id', 3)
    
    @torch.no_grad()
    def summarize(
        self,
        texts: List[str],
        batch_size: int = 4,
        **generation_kwargs,
    ) -> List[str]:
        """
        Generate summaries for a batch of texts.
        
        Implements batched inference for efficiency (Issue #1).
        
        Args:
            texts: List of clinical notes to summarize
            batch_size: Number of texts to process at once
            **generation_kwargs: Override generation config
        
        Returns:
            List of generated summaries
        """
        summaries = []
        
        # Merge generation config with overrides
        gen_params = {
            'max_length': generation_kwargs.get('max_length', self.gen_config.max_length),
            'min_length': generation_kwargs.get('min_length', self.gen_config.min_length),
            'num_beams': generation_kwargs.get('num_beams', self.gen_config.num_beams),
            'length_penalty': generation_kwargs.get('length_penalty', self.gen_config.length_penalty),
            'no_repeat_ngram_size': generation_kwargs.get('no_repeat_ngram_size', self.gen_config.no_repeat_ngram_size),
            'early_stopping': generation_kwargs.get('early_stopping', self.gen_config.early_stopping),
        }
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
            batch_texts = texts[i:i + batch_size]
            batch_summaries = self._generate_batch(batch_texts, **gen_params)
            summaries.extend(batch_summaries)
        
        return summaries
    
    def _generate_batch(self, texts: List[str], **gen_params) -> List[str]:
        """Generate summaries for a single batch"""
        # Tokenize
        batch_ids = []
        for text in texts:
            ids = self.tokenizer.encode(text)
            if len(ids) > self.max_src_len:
                ids = ids[:self.max_src_len]
            batch_ids.append(ids)
        
        # Pad to same length
        max_len = max(len(ids) for ids in batch_ids)
        input_ids = torch.full(
            (len(texts), max_len),
            self.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            len(texts), max_len,
            dtype=torch.float,
            device=self.device,
        )
        
        for i, ids in enumerate(batch_ids):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1.0
        
        # Generate
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_params,
        )
        
        # Decode
        summaries = []
        for ids in output_ids:
            ids_list = ids.cpu().tolist()
            # Remove special tokens
            ids_list = [
                tok for tok in ids_list
                if tok not in [self.pad_id, 
                              self.model.config.bos_token_id,
                              self.model.config.eos_token_id]
            ]
            text = self.tokenizer.decode(ids_list)
            summaries.append(text)
        
        return summaries
    
    def summarize_single(self, text: str, **generation_kwargs) -> str:
        """Convenience method for single text summarization"""
        return self.summarize([text], batch_size=1, **generation_kwargs)[0]


def evaluate_summaries(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE metrics for generated summaries.
    
    Args:
        predictions: List of generated summaries
        references: List of reference summaries
    
    Returns:
        Dictionary of ROUGE scores
    """
    if not ROUGE_AVAILABLE:
        logger.warning("ROUGE evaluation not available")
        return {}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {
        'rouge1_precision': [],
        'rouge1_recall': [],
        'rouge1_fmeasure': [],
        'rouge2_precision': [],
        'rouge2_recall': [],
        'rouge2_fmeasure': [],
        'rougeL_precision': [],
        'rougeL_recall': [],
        'rougeL_fmeasure': [],
    }
    
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            scores[f'{metric}_precision'].append(result[metric].precision)
            scores[f'{metric}_recall'].append(result[metric].recall)
            scores[f'{metric}_fmeasure'].append(result[metric].fmeasure)
    
    # Compute averages
    avg_scores = {}
    for key, values in scores.items():
        avg_scores[key] = sum(values) / len(values) if values else 0.0
    
    return avg_scores


def main():
    parser = argparse.ArgumentParser(description='LongT5 Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--tokenizer', type=str, default=None, help='Path to tokenizer')
    parser.add_argument('--input', type=str, default=None, help='Input CSV file')
    parser.add_argument('--text', type=str, default=None, help='Single text to summarize')
    parser.add_argument('--output', type=str, default='results/summaries.csv', help='Output CSV file')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for generation')
    parser.add_argument('--source-col', type=str, default='source_text', help='Source column name')
    parser.add_argument('--target-col', type=str, default='target_text', help='Target column name')
    parser.add_argument('--num-beams', type=int, default=4, help='Number of beams')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum output length')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    # Initialize summarizer
    summarizer = LongT5Summarizer(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )
    
    if args.text:
        # Single text mode
        summary = summarizer.summarize_single(
            args.text,
            num_beams=args.num_beams,
            max_length=args.max_length,
        )
        print("\n" + "="*50)
        print("Generated Summary:")
        print("="*50)
        print(summary)
        return
    
    if args.input:
        # CSV file mode
        logger.info(f"Processing file: {args.input}")
        
        # Read input
        texts = []
        references = []
        with open(args.input, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if args.num_samples and i >= args.num_samples:
                    break
                texts.append(row[args.source_col])
                if args.target_col in row:
                    references.append(row[args.target_col])
        
        logger.info(f"Loaded {len(texts)} samples")
        
        # Generate summaries
        summaries = summarizer.summarize(
            texts,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            max_length=args.max_length,
        )
        
        # Evaluate if references available
        if references:
            metrics = evaluate_summaries(summaries, references)
            print("\n" + "="*50)
            print("ROUGE Scores:")
            print("="*50)
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['source', 'generated']
            if references:
                header.append('reference')
            writer.writerow(header)
            
            for i, summary in enumerate(summaries):
                row = [texts[i][:500] + '...', summary]
                if references:
                    row.append(references[i])
                writer.writerow(row)
        
        logger.info(f"Results saved to {output_path}")
        
        # Print sample results
        print("\n" + "="*50)
        print("Sample Results (first 3):")
        print("="*50)
        for i in range(min(3, len(summaries))):
            print(f"\n--- Sample {i+1} ---")
            print(f"Source (truncated): {texts[i][:200]}...")
            print(f"Generated: {summaries[i]}")
            if references:
                print(f"Reference: {references[i]}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
