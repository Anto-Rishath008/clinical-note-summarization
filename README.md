# Clinical Note Summarization

A PyTorch implementation of a **Pointer-Generator Network with Hierarchical Encoding** for automatic summarization of clinical notes into hospital course summaries.

## Overview

This project implements a neural sequence-to-sequence model that:
- Processes long clinical notes (up to 1,536 tokens) using hierarchical chunked encoding
- Generates concise summaries using a pointer-generator mechanism
- Handles medical terminology through a copy mechanism
- Prevents repetition with coverage attention

**⚠️ Important**: This repository contains **NO patient data or trained models**. The MIMIC-IV-BHC dataset requires credentialed access from PhysioNet.

## Project Structure

```
clinical-note-summarization/
├── src/
│   └── core.py                 # Complete model implementation (1,200+ lines)
│       ├── Attention mechanisms (Additive + Coverage)
│       ├── Hierarchical encoder (BiLSTM)
│       ├── Pointer-Generator decoder
│       ├── Full training pipeline
│       └── Beam search decoder
├── configs/
│   ├── default.yaml            # Standard configuration
│   ├── resume_config.yaml      # Resume training config
│   └── rtx4070_8gb.yaml        # Memory-optimized (8GB VRAM)
├── train.py                    # Training script with FP16, gradient accumulation
├── evaluate.py                 # Evaluation with ROUGE metrics
├── preprocess_data.py          # Data tokenization pipeline
├── inspect_dataset.py          # Dataset inspection utility
├── requirements.txt            # Python dependencies
├── data/
│   ├── README.md              # Data acquisition instructions
│   └── sample_data.json       # 2 non-sensitive examples
└── README.md                  # This file
```

## Setup

### Prerequisites
- **Python**: 3.10+ recommended
- **PyTorch**: 2.0+ with CUDA support (for GPU training)
- **Hardware**: 
  - Minimum: 8GB VRAM GPU
  - Recommended: 12GB+ VRAM GPU
  - CPU training possible but slow

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
torch>=2.0.0                # Deep learning framework
pandas>=2.0.0               # Data manipulation
pyarrow>=12.0.0             # Parquet file support
sentencepiece>=0.1.99       # Tokenization
rouge-score>=0.1.2          # Evaluation metrics
PyYAML>=6.0                 # Configuration files
tqdm>=4.65.0                # Progress bars
numpy>=1.24.0               # Numerical operations
```

## Data

### Obtaining the Dataset

This project uses the **MIMIC-IV-BHC** (MIMIC-IV Extension - Behavioral Health Conditions) dataset:

1. **Request Access**: https://physionet.org/content/mimic-iv-bhc/1.2.0/
   - Requires completion of CITI training
   - Sign data use agreement
   - **HIPAA compliance required**

2. **Download**: After approval, download `mimic-iv-bhc.csv` (2.6 GB)

3. **Place Dataset**: 
   ```
   data/raw/mimic-iv-bhc.csv
   ```

**⚠️ CRITICAL**: 
- Do NOT commit patient data to version control
- Do NOT share raw datasets publicly
- Follow all PhysioNet data use agreements

### Dataset Statistics
- **Total Examples**: 270,034 clinical note-summary pairs
- **Splits**: 80% train / 10% validation / 10% test
- **Format**: Input (clinical notes) → Target (hospital course summaries)

## Usage

### 1. Preprocess Data

Tokenize the raw dataset using SentencePiece:

```bash
python preprocess_data.py \
    --input data/raw/mimic-iv-bhc.csv \
    --output_dir data/tokenized/full \
    --tokenizer artifacts/tokenizer/spm.model \
    --max_src_len 1024 \
    --max_tgt_len 384
```

**Note**: You must first train a SentencePiece tokenizer or use an existing one.

### 2. Train Model

```bash
python train.py \
    --config configs/default.yaml \
    --tokenized_dir data/tokenized/full \
    --run_name my_training_run \
    --max_steps 50000
```

**Key Arguments**:
- `--config`: YAML configuration file
- `--tokenized_dir`: Path to tokenized parquet files
- `--run_name`: Experiment name for checkpoints/logs
- `--max_steps`: Training duration (50K = ~60-70 hours on RTX 4070)
- `--resume`: Resume from checkpoint path

**Training Features**:
- Mixed precision (FP16) for memory efficiency
- Gradient accumulation (effective batch size 16)
- Periodic evaluation with ROUGE metrics
- Automatic checkpointing
- Early stopping with patience

### 3. Resume Training

```bash
python train.py \
    --config configs/resume_config.yaml \
    --tokenized_dir data/tokenized/full \
    --run_name my_training_run \
    --resume artifacts/checkpoints/my_training_run/best_model.pt
```

### 4. Evaluate Model

```bash
python evaluate.py \
    --checkpoint artifacts/checkpoints/my_training_run/best_model.pt \
    --tokenized_dir data/tokenized/full \
    --split test \
    --beam_size 4
```

**Evaluation Metrics**:
- ROUGE-1 (unigram overlap)
- ROUGE-2 (bigram overlap)
- ROUGE-L (longest common subsequence)

### 5. Inspect Dataset

```bash
python inspect_dataset.py
```

## Model Architecture

### Hierarchical Pointer-Generator Network

**Encoder**:
- Bidirectional LSTM (1-2 layers, 384-512 hidden dim)
- Hierarchical chunking (6-8 chunks × 256 tokens)
- Handles documents up to 1,536 tokens

**Decoder**:
- LSTM with attention (1-2 layers)
- Additive (Bahdanau) attention mechanism
- Coverage attention to prevent repetition
- Pointer-generator switch (`p_gen`)

**Key Features**:
1. **Copy Mechanism**: Handles rare medical terms by copying from source
2. **Coverage Loss**: Reduces repetition in generated summaries
3. **Beam Search**: Decodes with beam size 4, length penalty, n-gram blocking

**Model Size**: ~32M parameters (configurable)

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
model:
  emb_dim: 192                # Embedding dimension
  hidden_dim: 384             # LSTM hidden size
  num_layers: 1               # LSTM layers
  chunk_len: 256              # Tokens per chunk
  num_chunks: 6               # Number of chunks (max input = 1536)
  max_target_len: 256         # Max summary length
  coverage_lambda: 1.0        # Coverage loss weight

training:
  batch_size: 2               # Per-GPU batch size
  grad_accum: 8               # Gradient accumulation steps
  learning_rate: 0.0003       # Initial learning rate
  max_steps: 50000            # Training steps
  fp16: true                  # Mixed precision
  eval_every: 500             # Evaluation frequency
  save_every: 300             # Checkpoint frequency

data:
  vocab_size: 16000           # SentencePiece vocab size
  pad_id: 0                   # Padding token
  bos_id: 2                   # Begin-of-sequence
  eos_id: 3                   # End-of-sequence
```

### GPU Memory Profiles

**RTX 4070 12GB** (`configs/default.yaml`):
- Batch size: 2, Gradient accumulation: 8
- Hidden dim: 384
- Works with full dataset

**RTX 4070 8GB** (`configs/rtx4070_8gb.yaml`):
- Batch size: 1, Gradient accumulation: 16
- Hidden dim: 256 (reduced)
- Memory-optimized

## Reproducibility

**Deterministic Training**:
- Fixed random seed (42) in configs
- Deterministic CUDA operations (where possible)
- Checkpoint entire optimizer state

**Checkpoints Include**:
- Model state dict
- Optimizer state
- Training step/epoch
- Best validation ROUGE score
- Full configuration

## Expected Performance

| Baseline | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| Lead-K   | ~30.0   | ~10.0   | ~18.0   |
| Random   | ~15.0   | ~3.0    | ~10.0   |

| **Our Model** | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** |
|---------------|-------------|-------------|-------------|
| Target        | 36-40       | 12-15       | 22-25       |

*Note: Actual performance depends on training duration and hyperparameters*

## Common Issues

### Out of Memory (OOM)

**Solution**: Reduce batch size or use memory-optimized config:
```bash
python train.py --config configs/rtx4070_8gb.yaml ...
```

### Import Errors

**Solution**: Ensure you're in the project root and virtual environment is activated:
```bash
cd /path/to/clinical-note-summarization
source .venv/bin/activate
python train.py ...
```

### Slow Training

**Solutions**:
- Enable mixed precision: `fp16: true` in config
- Increase gradient accumulation to reduce optimizer steps
- Use fewer evaluation batches: reduce `max_eval_batches` in training loop

## Development

### Code Quality

Run static checks:
```bash
# Check syntax
python -m compileall src/ *.py

# Format code (if using ruff/black)
ruff format src/ *.py
```

### Testing

```bash
# Verify training script
python train.py --help

# Verify evaluation script
python evaluate.py --help

# Test with small subset
python preprocess_data.py --subset 1000 ...
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make changes with clear commit messages
4. Run code quality checks
5. Submit a pull request

**Areas for Improvement**:
- Add unit tests
- Implement BLEU/BERTScore metrics
- Add pre-trained checkpoint downloads
- Create inference API
- Add Docker support

## Citation

**Dataset**:
```
Johnson, A., Pollard, T., Mark, R. (2024). MIMIC-IV-BHC. PhysioNet.
https://physionet.org/content/mimic-iv-bhc/1.2.0/
```

**Model Architecture Inspiration**:
```
See, A., Liu, P. J., & Manning, C. D. (2017).
Get To The Point: Summarization with Pointer-Generator Networks.
ACL 2017.
```

## License

This code is provided for research and educational purposes. The MIMIC-IV-BHC dataset has its own license and usage terms that must be followed separately.

## Contact

For questions or issues:
- Open a GitHub issue
- Repository: https://github.com/Anto-Rishath008/clinical-note-summarization

---

**⚠️ Data Privacy Reminder**: Never commit patient data, trained models, or any outputs derived from PHI-containing datasets to version control.
