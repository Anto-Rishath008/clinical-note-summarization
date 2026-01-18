# Issue #1 Resolution: LongT5 Architecture Migration

This document describes the complete resolution of GitHub Issue #1:
**"Refactor Training Pipeline & Upgrade Architecture to LongT5"**

## Overview

The original Pointer-Generator model with chunked BiLSTM encoder had several limitations:
- Maximum 768 tokens with chunking (needed complex tensor manipulation)
- OOM issues from coverage tensors
- Hardcoded CUDA in GradScaler
- No multi-GPU support
- Basic checkpointing (not restart-safe)

The new Simplified LongT5 architecture addresses ALL these issues.

## Issue #1 Requirements Checklist

### ✅ 1. Architecture Upgrade (LongT5-inspired)

**File:** `src/longt5_model.py`

- Implemented from scratch (no pretrained weights)
- Native support for 4096+ token sequences
- Local-Global attention mechanism inspired by LongT5's Transient Global attention
- T5-style relative position biases
- Standard Transformer encoder-decoder architecture

Key classes:
- `SimplifiedLongT5`: Main model class
- `LocalGlobalAttention`: Efficient attention for long sequences
- `RelativePositionBias`: T5-style position encoding
- `LongT5Config`: Configuration dataclass

### ✅ 2. Fix OOM: Streaming Data Loading

**File:** `src/data_loader.py`

Implemented two dataset types:
1. `StreamingClinicalDataset` (IterableDataset): Loads samples on-demand
2. `MapStyleClinicalDataset` (Dataset): For smaller validation sets

Features:
- Lazy loading prevents full dataset in memory
- On-the-fly tokenization
- Multi-worker support with proper work splitting
- Bucket batch sampling for minimal padding overhead

```python
# Usage
train_loader = create_train_dataloader(config, tokenizer, streaming=True)
```

### ✅ 3. Fix Hardcoded CUDA: Device-Agnostic GradScaler

**File:** `train_longt5.py`

```python
# OLD (hardcoded CUDA)
scaler = GradScaler()

# NEW (device-agnostic)
use_fp16 = training_cfg.get('fp16', True) and device.type == 'cuda'
scaler = GradScaler(enabled=use_fp16) if use_fp16 else None
```

The GradScaler is now:
- Automatically disabled on CPU
- Configurable via config file
- Properly handled in training loop

### ✅ 4. Maximize CPU Utilization

**File:** `src/data_loader.py`

```python
def get_num_workers() -> int:
    """Get optimal number of workers for DataLoader."""
    cpu_count = os.cpu_count() or 1
    num_workers = max(1, min(8, int(cpu_count * 0.8)))
    return num_workers
```

Features:
- Uses 80% of available CPUs
- Capped at 8 workers maximum
- Minimum of 1 worker
- Logged for debugging

### ✅ 5. Batched Beam Search Inference

**File:** `src/longt5_model.py` and `inference_longt5.py`

The `SimplifiedLongT5.generate()` method supports:
- Batched processing for efficiency
- Greedy decoding (num_beams=1)
- Beam search with configurable beams
- Length penalty and n-gram blocking
- Early stopping

```python
# Batched inference
summarizer = LongT5Summarizer(checkpoint_path='...')
summaries = summarizer.summarize(texts, batch_size=4, num_beams=4)
```

### ✅ 6. Multi-GPU Support (DataParallel)

**File:** `train_longt5.py`

```python
def setup_model(config: dict, device: torch.device) -> nn.Module:
    model = create_model(config)
    model = model.to(device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    return model
```

Checkpoint saving/loading handles DataParallel:
```python
# Save
model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

# Load
if hasattr(model, 'module'):
    model.module.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint['model_state_dict'])
```

### ✅ 7. Restart-Safe Checkpointing

**File:** `train_longt5.py`

Checkpoints now include:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- GradScaler state dict
- Epoch and global step
- Best metric
- Metrics history
- **All RNG states** (Python, NumPy, PyTorch, CUDA)

```python
def get_rng_states() -> Dict[str, Any]:
    """Get all RNG states for restart-safe checkpointing."""
    rng_states = {
        'python_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_states['cuda_rng'] = torch.cuda.get_rng_state_all()
    return rng_states
```

### ✅ 8. Config Cleanup

**File:** `configs/longt5_config.yaml`

Removed obsolete LSTM parameters:
- ~~emb_dim~~
- ~~hidden_dim~~
- ~~num_layers~~
- ~~chunk_len~~
- ~~num_chunks~~
- ~~pointer_gen~~
- ~~coverage_lambda~~
- ~~hierarchical~~
- ~~use_glove~~

New clean configuration with:
- Transformer dimensions (d_model, d_ff, num_heads)
- Encoder/decoder layers
- Long-context attention settings
- Generation parameters

## File Structure

```
src/
├── longt5_model.py      # NEW: LongT5-inspired model
├── data_loader.py       # NEW: Streaming data loading
└── core.py              # Legacy: Pointer-Generator (kept for compatibility)

configs/
├── longt5_config.yaml   # NEW: LongT5 configuration
└── *.yaml               # Legacy configs

train_longt5.py          # NEW: Training script for LongT5
inference_longt5.py      # NEW: Inference script for LongT5
```

## Usage

### Training

```bash
# Basic training
python train_longt5.py --config configs/longt5_config.yaml

# Resume from checkpoint
python train_longt5.py --config configs/longt5_config.yaml --resume artifacts/checkpoints/longt5/latest.pt

# Disable streaming (for small datasets)
python train_longt5.py --config configs/longt5_config.yaml --no-streaming
```

### Inference

```bash
# Single text
python inference_longt5.py --checkpoint best_model.pt --text "Clinical note..."

# CSV file with evaluation
python inference_longt5.py --checkpoint best_model.pt --input test.csv --output results.csv

# Custom generation settings
python inference_longt5.py --checkpoint best_model.pt --input test.csv --num-beams 8 --max-length 256
```

## Model Architecture Details

### Encoder
- Stack of Transformer layers with Local-Global attention
- Local attention: Each token attends within a configurable window (default: 127 tokens each side)
- Global attention: Every 16th token attends/is attended globally
- T5-style relative position biases

### Decoder
- Standard Transformer decoder layers
- Self-attention with causal masking
- Cross-attention to encoder outputs
- Relative position biases

### Attention Mechanism

```python
class LocalGlobalAttention:
    """
    For sequences > 1024 tokens:
    - Local attention band (±127 tokens)
    - Global tokens every 16 positions
    
    For sequences ≤ 1024 tokens:
    - Full attention (faster, no approximation)
    """
```

## Performance Comparison

| Metric | Pointer-Generator | LongT5 (Expected) |
|--------|-------------------|-------------------|
| Max Input | 768 tokens | 4096 tokens |
| Training Speed | Slow (coverage) | Fast (efficient attention) |
| Memory Usage | High (OOM issues) | Lower (no coverage) |
| Multi-GPU | ❌ | ✅ |
| Restart-Safe | ❌ | ✅ |

## Future Improvements

1. **DistributedDataParallel**: For even better multi-GPU scaling
2. **Flash Attention**: For faster attention computation
3. **Gradient Checkpointing**: For training larger models
4. **Model Parallelism**: For models that don't fit on single GPU

## References

- LongT5: https://arxiv.org/abs/2112.07916
- T5: https://arxiv.org/abs/1910.10683
- Pointer-Generator: https://arxiv.org/abs/1704.04368
