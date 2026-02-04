"""
Data Loading for Mamba-Transformer Hybrid Model
================================================

Streaming data loader for clinical note summarization.
Handles tokenization, chunking, batching with dynamic padding.
"""

import os
import csv
import random
import logging
from typing import Optional, Tuple, List, Dict, Iterator
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import sentencepiece as spm

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading"""
    csv_path: str
    tokenizer_path: str
    max_src_len: int = 4096
    max_tgt_len: int = 512
    pad_id: int = 3
    bos_id: int = 1
    eos_id: int = 2
    source_col: str = 'source_text'
    target_col: str = 'target_text'
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'DataConfig':
        data_cfg = config_dict.get('data', {})
        return cls(
            csv_path=data_cfg.get('csv_path', ''),
            tokenizer_path=data_cfg.get('tokenizer_path', ''),
            max_src_len=data_cfg.get('max_src_len', 4096),
            max_tgt_len=data_cfg.get('max_tgt_len', 512),
            pad_id=data_cfg.get('pad_id', 3),
            bos_id=data_cfg.get('bos_id', 1),
            eos_id=data_cfg.get('eos_id', 2),
            source_col=data_cfg.get('source_col', 'source_text'),
            target_col=data_cfg.get('target_col', 'target_text'),
        )


def load_tokenizer(path: str) -> spm.SentencePieceProcessor:
    """Load SentencePiece tokenizer"""
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    logger.info(f"Loaded tokenizer with vocab size: {sp.GetPieceSize()}")
    return sp


class ClinicalDataset(Dataset):
    """
    In-memory dataset for clinical note summarization.
    For moderate-sized datasets that fit in memory.
    """
    
    def __init__(
        self,
        csv_path: str,
        tokenizer: spm.SentencePieceProcessor,
        max_src_len: int = 4096,
        max_tgt_len: int = 512,
        pad_id: int = 3,
        bos_id: int = 1,
        eos_id: int = 2,
        source_col: str = 'source_text',
        target_col: str = 'target_text',
        max_samples: Optional[int] = None,
        split: str = 'train',
        train_ratio: float = 0.95,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.source_col = source_col
        self.target_col = target_col
        
        # Load data
        self.samples = self._load_csv(csv_path, max_samples, split, train_ratio, seed)
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_csv(
        self,
        csv_path: str,
        max_samples: Optional[int],
        split: str,
        train_ratio: float,
        seed: int,
    ) -> List[Tuple[str, str]]:
        """Load CSV data and split into train/val"""
        samples = []
        
        # Increase CSV field size limit for large clinical notes
        csv.field_size_limit(10 * 1024 * 1024)  # 10MB
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break
                
                source = row.get(self.source_col, '').strip()
                target = row.get(self.target_col, '').strip()
                
                if source and target:
                    samples.append((source, target))
        
        # Shuffle and split
        random.seed(seed)
        random.shuffle(samples)
        
        split_idx = int(len(samples) * train_ratio)
        if split == 'train':
            samples = samples[:split_idx]
        else:
            samples = samples[split_idx:]
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source, target = self.samples[idx]
        
        # Tokenize
        src_ids = self.tokenizer.EncodeAsIds(source)[:self.max_src_len]
        tgt_ids = self.tokenizer.EncodeAsIds(target)
        
        # Add BOS/EOS to target
        tgt_input = [self.bos_id] + tgt_ids[:self.max_tgt_len - 2]
        tgt_output = tgt_ids[:self.max_tgt_len - 2] + [self.eos_id]
        
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt_input': torch.tensor(tgt_input, dtype=torch.long),
            'tgt_output': torch.tensor(tgt_output, dtype=torch.long),
            'src_text': source,
            'tgt_text': target,
        }


class StreamingClinicalDataset(IterableDataset):
    """
    Streaming dataset for very large datasets.
    Loads samples on-demand without keeping all in memory.
    """
    
    def __init__(
        self,
        csv_path: str,
        tokenizer: spm.SentencePieceProcessor,
        max_src_len: int = 4096,
        max_tgt_len: int = 512,
        pad_id: int = 3,
        bos_id: int = 1,
        eos_id: int = 2,
        source_col: str = 'source_text',
        target_col: str = 'target_text',
        shuffle: bool = True,
        seed: int = 42,
        skip_rows: int = 0,
        max_rows: Optional[int] = None,
    ):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.source_col = source_col
        self.target_col = target_col
        self.shuffle = shuffle
        self.seed = seed
        self.skip_rows = skip_rows
        self.max_rows = max_rows
        
        self._total_rows = None
    
    def _count_rows(self) -> int:
        if self._total_rows is None:
            csv.field_size_limit(10 * 1024 * 1024)
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                self._total_rows = sum(1 for _ in f) - 1  # Subtract header
        return self._total_rows
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        csv.field_size_limit(10 * 1024 * 1024)
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                if i < self.skip_rows:
                    continue
                if self.max_rows and i >= self.skip_rows + self.max_rows:
                    break
                
                source = row.get(self.source_col, '').strip()
                target = row.get(self.target_col, '').strip()
                
                if not source or not target:
                    continue
                
                # Tokenize
                src_ids = self.tokenizer.EncodeAsIds(source)[:self.max_src_len]
                tgt_ids = self.tokenizer.EncodeAsIds(target)
                
                # Add BOS/EOS
                tgt_input = [self.bos_id] + tgt_ids[:self.max_tgt_len - 2]
                tgt_output = tgt_ids[:self.max_tgt_len - 2] + [self.eos_id]
                
                yield {
                    'src': torch.tensor(src_ids, dtype=torch.long),
                    'tgt_input': torch.tensor(tgt_input, dtype=torch.long),
                    'tgt_output': torch.tensor(tgt_output, dtype=torch.long),
                    'src_text': source,
                    'tgt_text': target,
                }


def collate_fn(batch: List[Dict], pad_id: int = 3) -> Dict[str, torch.Tensor]:
    """
    Collate batch with dynamic padding.
    """
    # Find max lengths
    max_src_len = max(item['src'].size(0) for item in batch)
    max_tgt_len = max(item['tgt_input'].size(0) for item in batch)
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    src = torch.full((batch_size, max_src_len), pad_id, dtype=torch.long)
    tgt_input = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
    tgt_output = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
    src_mask = torch.ones((batch_size, max_src_len), dtype=torch.bool)
    tgt_mask = torch.ones((batch_size, max_tgt_len), dtype=torch.bool)
    
    # Fill in data
    for i, item in enumerate(batch):
        src_len = item['src'].size(0)
        tgt_len = item['tgt_input'].size(0)
        
        src[i, :src_len] = item['src']
        tgt_input[i, :tgt_len] = item['tgt_input']
        tgt_output[i, :tgt_len] = item['tgt_output']
        src_mask[i, :src_len] = False
        tgt_mask[i, :tgt_len] = False
    
    return {
        'src': src,
        'tgt_input': tgt_input,
        'tgt_output': tgt_output,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
    }


def create_dataloaders(
    config: DataConfig,
    batch_size: int = 4,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, spm.SentencePieceProcessor]:
    """
    Create train and validation dataloaders.
    """
    tokenizer = load_tokenizer(config.tokenizer_path)
    
    # Create datasets
    train_dataset = ClinicalDataset(
        csv_path=config.csv_path,
        tokenizer=tokenizer,
        max_src_len=config.max_src_len,
        max_tgt_len=config.max_tgt_len,
        pad_id=config.pad_id,
        bos_id=config.bos_id,
        eos_id=config.eos_id,
        source_col=config.source_col,
        target_col=config.target_col,
        max_samples=max_train_samples,
        split='train',
    )
    
    val_dataset = ClinicalDataset(
        csv_path=config.csv_path,
        tokenizer=tokenizer,
        max_src_len=config.max_src_len,
        max_tgt_len=config.max_tgt_len,
        pad_id=config.pad_id,
        bos_id=config.bos_id,
        eos_id=config.eos_id,
        source_col=config.source_col,
        target_col=config.target_col,
        max_samples=max_val_samples,
        split='val',
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, config.pad_id),
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, config.pad_id),
        pin_memory=True,
    )
    
    return train_loader, val_loader, tokenizer
