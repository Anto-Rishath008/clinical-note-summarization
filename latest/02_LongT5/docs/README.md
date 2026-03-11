# Simplified LongT5 Model for Clinical Note Summarization

This folder contains the complete implementation of a **LongT5-inspired Transformer model** trained from scratch for clinical note summarization.

## ✅ Solves All Issue #1 Requirements

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| LongT5/LED Architecture | ✅ | `model.py` - Full Transformer encoder-decoder with Local-Global attention |
| Fix OOM (Streaming Data) | ✅ | `data_loader.py` - IterableDataset with lazy loading |
| Fix Hardcoded CUDA | ✅ | `train.py` - Device-agnostic GradScaler |
| CPU Utilization | ✅ | `data_loader.py` - Dynamic num_workers (80% of cores) |
| Batched Inference | ✅ | `model.py` - generate() with batched beam search |
| Multi-GPU Support | ✅ | `train.py` - Automatic DataParallel wrapper |
| Restart-Safe Checkpoints | ✅ | `train.py` - Full RNG state saving |
| Config Cleanup | ✅ | `config.yaml` - No LSTM/coverage parameters |

## Files

| File | Description |
|------|-------------|
| `model.py` | LongT5-inspired model architecture (from scratch) |
| `data_loader.py` | Streaming data loader with bucketing |
| `train.py` | Complete training script with all fixes |
| `inference.py` | Inference and evaluation script |
| `config.yaml` | Configuration file (RTX 4070 8GB optimized) |
| `issue1_resolution.md` | Detailed documentation of all fixes |

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers sentencepiece pyyaml tqdm rouge-score pandas
```

### 2. Prepare Data

Ensure you have the MIMIC-IV-BHC data at `data/mimic-iv-bhc.csv` with columns:
- `source_text`: Clinical notes
- `target_text`: Brief hospital course summaries

### 3. Train Model

```bash
cd longt5_model
python train.py --config config.yaml
```

### 4. Resume Training

```bash
python train.py --config config.yaml --resume artifacts/checkpoints/longt5/latest.pt
```

### 5. Run Inference

```bash
python inference.py --checkpoint artifacts/checkpoints/longt5/best_model.pt
```

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      INPUT: Clinical Note                            │
│                   (Up to 4096 tokens - no chunking!)                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ENCODER                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Token Embedding + Relative Position Bias                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Local-Global Attention (Efficient for Long Sequences)       │   │
│  │  - Local: Window of 255 tokens                               │   │
│  │  - Global: Every 16th token attends to all                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Feed-Forward Network (d_ff = 4 × d_model)                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                      × num_encoder_layers                           │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DECODER                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Causal Self-Attention (Masked)                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Cross-Attention (Attend to Encoder Outputs)                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Feed-Forward Network                                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                      × num_decoder_layers                           │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                        │
│                    Vocabulary Projection                             │
│                  (Tied with embeddings)                             │
└─────────────────────────────────────────────────────────────────────┘
```

## Configuration (8GB GPU Optimized)

```yaml
model:
  d_model: 256                    # Hidden dimension
  d_ff: 1024                      # Feed-forward dimension
  num_encoder_layers: 4           # Encoder depth
  num_decoder_layers: 4           # Decoder depth
  num_heads: 8                    # Attention heads
  max_position_embeddings: 4096   # Max sequence length
  local_radius: 127               # Local attention window

training:
  batch_size: 2                   # GPU batch size
  gradient_accumulation_steps: 8  # Effective batch = 16
  learning_rate: 5e-5
  fp16: true                      # Mixed precision
  streaming: true                 # Memory-efficient data loading
```

## Key Improvements Over Previous Model

| Feature | Old (PointerGenerator) | New (LongT5) |
|---------|------------------------|--------------|
| Max Input Length | 768 (chunked) | 4096 (native) |
| Architecture | BiLSTM + Attention | Transformer |
| Memory | OOM from coverage | Efficient streaming |
| Multi-GPU | No | Automatic DataParallel |
| Checkpointing | Basic | Full RNG state |
| Data Loading | All in RAM | Streaming |
| CPU Utilization | Single-threaded | 80% cores |

## Training Progress Monitoring

Training logs are saved to `artifacts/logs/longt5/`:
- `metrics.json` - Training/validation metrics per step
- Checkpoints saved to `artifacts/checkpoints/longt5/`

## License

This implementation is for educational and research purposes.
