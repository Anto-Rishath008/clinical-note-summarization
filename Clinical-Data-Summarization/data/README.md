# Data Directory

This directory contains the MIMIC-IV Brief Hospital Course (BHC) dataset and preprocessing artifacts.

## Structure

```
data/
├── raw/
│   └── mimic-iv-bhc.csv          # Raw clinical notes dataset (270K+ samples)
│
├── tokenized/
│   ├── train.parquet              # Training set (256,500 samples)
│   ├── val.parquet                # Validation set (13,500 samples)
│   ├── test.parquet               # Test set
│   ├── train_tiny.parquet         # Tiny set for debugging
│   ├── train_text_5k.csv          # 5K samples in text format
│   └── val_text_500.csv           # 500 validation samples in text
│
├── tokenizer/
│   ├── spm.model                  # SentencePiece Unigram model (16K vocab)
│   ├── spm.vocab                  # Vocabulary file
│   ├── spm_mamba.model            # Mamba workspace tokenizer
│   └── spm_mamba.vocab            # Mamba workspace vocabulary
│
└── sample/
    └── sample_data.json           # Sample data for quick testing
```

## Dataset Statistics

| Property | Value |
|----------|-------|
| Total Samples | 270,000+ |
| Training Set | 256,500 (95%) |
| Validation Set | 13,500 (5%) |
| Source Length | 1,000–8,000 tokens |
| Target Length | 50–400 tokens |
| Compression Ratio | ~10–20× |

## Tokenizer

- **Type:** SentencePiece Unigram
- **Vocabulary Size:** 16,000
- **Character Coverage:** 0.9995
- **Special Tokens:** BOS=1, EOS=2, PAD=3

## Data Access

The MIMIC-IV dataset requires PhysioNet credentialing. See [PhysioNet](https://physionet.org/content/mimic-iv/) for access.
