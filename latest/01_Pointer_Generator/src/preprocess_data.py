"""
Preprocess and tokenize MIMIC-IV-BHC clinical notes for Kaggle
Converts raw CSV to tokenized Parquet format
"""

import pandas as pd
import sentencepiece as spm
from pathlib import Path
import argparse
from tqdm import tqdm


def load_raw_data(data_path, data_format='csv'):
    """Load raw clinical data"""
    if data_format == 'csv':
        print(f"Loading CSV from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
        
        # Handle different column formats
        if 'input' in df.columns and 'target' in df.columns:
            df = df[['input', 'target']].rename(columns={'input': 'note', 'target': 'summary'})
            print("Detected MIMIC-IV-BHC format")
        elif 'text' in df.columns and 'summary' in df.columns:
            df = df[['text', 'summary']].rename(columns={'text': 'note'})
        elif 'note' in df.columns and 'summary' in df.columns:
            df = df[['note', 'summary']]
        else:
            raise ValueError(f"Invalid columns: {df.columns.tolist()}")
    
    elif data_format == 'json':
        df = pd.read_json(data_path)
    
    else:
        raise ValueError(f"Unsupported format: {data_format}")
    
    # Clean data
    print("Cleaning data...")
    df = df.dropna(subset=['note', 'summary'])
    df = df[df['note'].str.len() > 50]
    df = df[df['summary'].str.len() > 10]
    
    print(f"After cleaning: {len(df)} valid examples")
    return df


def tokenize_data(df, tokenizer_path, max_src_len=1024, max_tgt_len=384):
    """Tokenize notes and summaries"""
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))
    
    print(f"Tokenizing {len(df)} examples...")
    tokenized_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        src_ids = sp.encode(row['note'], out_type=int)[:max_src_len]
        tgt_ids = sp.encode(row['summary'], out_type=int)[:max_tgt_len]
        
        bos_id = sp.bos_id()
        eos_id = sp.eos_id()
        
        src_ids = [bos_id] + src_ids + [eos_id]
        tgt_ids = [bos_id] + tgt_ids + [eos_id]
        
        tokenized_data.append({
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
            'src_len': len(src_ids),
            'tgt_len': len(tgt_ids)
        })
    
    return pd.DataFrame(tokenized_data)


def split_dataset(df, train_ratio=0.8, val_ratio=0.1):
    """Split dataset into train/val/test"""
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    print(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Preprocess MIMIC-IV-BHC for Kaggle')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, default='artifacts/tokenizer/spm.model')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'json'])
    parser.add_argument('--max_src_len', type=int, default=1024)
    parser.add_argument('--max_tgt_len', type=int, default=384)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--subset', type=int, default=None, help='Use first N examples only')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("MIMIC-IV-BHC PREPROCESSING FOR KAGGLE")
    print(f"{'='*70}\n")
    
    # Load
    df = load_raw_data(args.input, args.format)
    
    # Subset if requested
    if args.subset:
        print(f"Using subset of {args.subset} examples")
        df = df.head(args.subset)
    
    # Tokenize
    print(f"\nTokenizing with: {args.tokenizer}")
    tokenized_df = tokenize_data(df, args.tokenizer, args.max_src_len, args.max_tgt_len)
    
    # Split
    train_df, val_df, test_df = split_dataset(tokenized_df, args.train_ratio, args.val_ratio)
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to: {output_dir}")
    train_df.to_parquet(output_dir / 'train.parquet', index=False)
    val_df.to_parquet(output_dir / 'val.parquet', index=False)
    test_df.to_parquet(output_dir / 'test.parquet', index=False)
    
    # Print sizes
    print(f"\nFile sizes:")
    for fname in ['train.parquet', 'val.parquet', 'test.parquet']:
        fpath = output_dir / fname
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  {fname}: {size_mb:.2f} MB")
    
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'='*70}")
    print("\nKAGGLE UPLOAD CHECKLIST:")
    print(f"  □ {output_dir}/train.parquet")
    print(f"  □ {output_dir}/val.parquet")
    print(f"  □ {output_dir}/test.parquet")
    print(f"  □ artifacts/tokenizer/spm.model")
    print(f"  □ artifacts/tokenizer/spm.vocab")
    print("\nKaggle training command:")
    print("  !python train.py --config configs/default.yaml \\")
    print("      --tokenized_dir /kaggle/input/your-dataset \\")
    print("      --run_name kaggle_run --max_steps 10000")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
