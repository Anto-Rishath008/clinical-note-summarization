# Clinical Note Summarization - From-Scratch Training

A PyTorch implementation of transformer-based models for automatic summarization of clinical notes into hospital course summaries.

## 🆕 NEW: Simplified LongT5 Architecture (Issue #1 Resolution)

The project now includes a **LongT5-inspired transformer model** that solves all Issue #1 problems:

### Architecture Comparison

| Feature | Pointer-Generator (Legacy) | Simplified LongT5 (New) |
|---------|---------------------------|-------------------------|
| Max Length | 768 tokens (chunked) | **4096 tokens (native)** |
| Encoder | BiLSTM with chunking | Transformer with Local-Global attention |
| Attention | Additive + Coverage | Multi-head with relative position bias |
| Memory | OOM issues with coverage | **Memory efficient** |
| Multi-GPU | Not supported | **DataParallel support** |
| Checkpointing | Basic | **Restart-safe with RNG states** |

### Quick Start with LongT5

```powershell
# Train with new LongT5 model
python train_longt5.py --config configs/longt5_config.yaml

# Resume training
python train_longt5.py --config configs/longt5_config.yaml --resume artifacts/checkpoints/longt5/latest.pt

# Run inference
python inference_longt5.py --checkpoint artifacts/checkpoints/longt5/best_model.pt --input data/test.csv
```

---

## ⚠️ IMPORTANT: From-Scratch Training Only

**This project trains models from random initialization (NO pretrained models, NO PEFT/LoRA).**

Goal: Beat Lead-150 baseline using disciplined from-scratch training with sanity checks.

**⚠️ Dataset**: This repository contains **NO patient data or trained models**. The MIMIC-IV-BHC dataset requires credentialed access from PhysioNet.

---

## 🚀 Quick Start (Windows PowerShell)

### Prerequisites
```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Prepare tokenized data (after obtaining MIMIC-IV-BHC dataset)
python preprocess_data.py
```

### Essential Commands

```powershell
# 1. Overfit test (MANDATORY - verifies model can learn)
python scripts\overfit_test.py --config configs\stage1_fromscratch.yaml

# 2. Establish baseline target
python scripts\lead_baseline.py --n_tokens 150

# 3. Train from scratch
python train.py --config configs\stage1_fromscratch.yaml

# 4. Evaluate trained model
python evaluate.py --checkpoint artifacts\checkpoints\best_model.pt --split test

# 5. Decode sanity check (verify generation quality)
python scripts\decode_sanity.py --checkpoint artifacts\checkpoints\best_model.pt
```

--- ## 📁 Repository Structure

```
clinical-note-summarization/
├── train.py                     # Main training script (FP16, gradient accumulation)
├── evaluate.py                  # Evaluation with ROUGE metrics
├── preprocess_data.py           # Data tokenization pipeline
├── inspect_dataset.py           # Dataset inspection utility
├── requirements.txt             # Python dependencies
│
├── src/
│   └── core.py                  # Model implementation (1,200+ lines)
│                                  - Hierarchical BiLSTM encoder
│                                  - Pointer-Generator decoder
│                                  - Attention mechanisms (Additive + Coverage)
│                                  - Beam search decoder
│
├── configs/
│   ├── stage1_fromscratch.yaml  # From-scratch training (RECOMMENDED)
│   ├── stage1_debug.yaml        # Quick debugging config
│   ├── default.yaml             # Standard configuration
│   ├── resume_config.yaml       # Resume training
│   ├── rtx4070_8gb.yaml         # Memory-optimized (8GB VRAM)
│   └── _deprecated/             # Pretrained model configs (not used)
│       ├── hf_baseline.yaml
│       └── hf_experiments/
│
├── scripts/
│   ├── overfit_test.py          # Overfit sanity check (200 samples)
│   ├── lead_baseline.py         # Lead-N baseline evaluation
│   ├── baseline_eval.py         # Baseline comparison
│   ├── decode_sanity.py         # Generation quality analysis
│   ├── eval_pipeline.py         # Comprehensive evaluation
│   ├── clean_repo.ps1           # Remove all artifacts (PowerShell)
│   ├── prepare_data.ps1         # Dataset preparation script
│   ├── train_sanity.ps1         # Quick sanity check training
│   └── _deprecated/             # Pretrained model scripts (not used)
│       ├── _DEPRECATED_hf_train.py
│       └── _DEPRECATED_hf_eval.py
│
├── experiments/
│   ├── results.csv              # Experiment tracking table
│   ├── baseline_results.csv     # Lead-K baseline results
│   └── decode_sanity_results.csv # Decode quality metrics
│
├── docs/
│   ├── implementation_summary.md # Implementation details
│   ├── rouge_optimization_plan.md # Optimization strategies
│   └── archive/                  # Archived process documentation
│       ├── CLEANUP_SUMMARY.md
│       ├── QUICK_REFERENCE.md
│       ├── STAGE1_*.md
│       └── ...
│
├── data/ (⚠️ Gitignored - PHI Protected)
│   ├── README.md                # Data acquisition instructions
│   ├── sample_data.json         # 2 non-sensitive examples
│   └── tokenized/               # Preprocessed data (generated)
│
├── artifacts/ (⚠️ Gitignored - Generated files)
│   ├── README.md                # Artifacts documentation
│   ├── checkpoints/             # Model weights (*.pt files)
│   ├── logs/                    # Training logs
│   └── tokenizer/               # SentencePiece models
│
└── reports/
    └── README.md                # Small result summaries
```

### Key Directories

| Directory | Purpose | Git Tracked |
|-----------|---------|-------------|
| `src/` | Source code | ✅ Yes |
| `configs/` | YAML configurations | ✅ Yes |
| `scripts/` | Utility scripts | ✅ Yes |
| `experiments/` | Small CSV results | ✅ Yes |
| `docs/` | Documentation | ✅ Yes |
| `data/` | Raw/processed datasets | ❌ **No** (PHI protected) |
| `artifacts/` | Model checkpoints, logs | ❌ **No** (10+ GB) |

**⚠️ Important**: Artifacts and data are **never committed** to version control. They are regenerated during training.

---

## 📋 Prerequisites

- **Python**: 3.10+ recommended
- **PyTorch**: 2.0+ with CUDA support (for GPU training)
- **Hardware**: 
  - Minimum: 8GB VRAM GPU (RTX 4070 or better)
  - Recommended: 12GB+ VRAM GPU
  - CPU training possible but very slow (~100x slower)
- **Storage**: 15-20 GB free space (datasets + artifacts)

---

## 📦 Model Architecture

This project implements a neural sequence-to-sequence model that:
- Processes long clinical notes (up to 2,048 tokens) using hierarchical chunked encoding
- Generates concise summaries using a pointer-generator mechanism
- Handles medical terminology through a copy mechanism
- Prevents repetition with coverage attention

See [docs/implementation_summary.md](docs/implementation_summary.md) for detailed architecture information.

---

## 📚 Documentation

- **Implementation Details**: [docs/implementation_summary.md](docs/implementation_summary.md)
- **Optimization Strategies**: [docs/rouge_optimization_plan.md](docs/rouge_optimization_plan.md)
- **Archived Process Docs**: [docs/archive/](docs/archive/)

---

## ⚠️ Deprecated Components

The following components are **NOT USED** in this from-scratch training project:

- `configs/_deprecated/hf_baseline.yaml` - HuggingFace BART fine-tuning configs
- `configs/_deprecated/hf_experiments/` - HuggingFace experiment configs
- `scripts/_deprecated/_DEPRECATED_hf_train.py` - Pretrained model training
- `scripts/_deprecated/_DEPRECATED_hf_eval.py` - Pretrained model evaluation

These files are kept for reference but are not part of the active codebase.

---
## 🔬 Training Workflow

1. **Sanity Check**: Run overfit test to verify model can learn (5-10 minutes)
2. **Baseline**: Establish Lead-150 baseline target
3. **Training**: Train from scratch (40-60 hours on RTX 4070)
4. **Evaluation**: Compare model performance against baseline

---

## 🧪 Development & Testing

```powershell
# Clean all generated artifacts
.\scripts\clean_repo.ps1

# Quick sanity check (5-10 min)
.\scripts\train_sanity.ps1

# Inspect preprocessed dataset
python inspect_dataset.py
```

---

## 📈 Results & Experiments

Training results and metrics are tracked in:
- `experiments/results.csv` - Main experiment tracking
- `experiments/baseline_results.csv` - Baseline comparisons
- `experiments/decode_sanity_results.csv` - Generation quality

---

## 🤝 Contributing

This is an academic project. For questions or collaboration, please open an issue.

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- MIMIC-IV-BHC dataset from PhysioNet
- Pointer-Generator architecture based on See et al. (2017)
- Hierarchical encoding inspired by clinical NLP literature
## 📊 Data

### Obtaining the Dataset

This project uses the **MIMIC-IV-BHC** dataset from PhysioNet:

1. **Request Access**: https://physionet.org/content/mimic-iv-bhc/1.2.0/
   - Requires CITI training completion
   - Sign data use agreement
   - **HIPAA compliance required**

2. **Download**: After approval, download `mimic-iv-bhc.csv` (2.6 GB)

3. **Preprocess**:
   ```powershell
   python preprocess_data.py
   ```

**⚠️ CRITICAL**: Do NOT commit patient data to version control. Follow all PhysioNet data use agreements.

### Dataset Statistics
- **Total Examples**: 270,034 clinical note-summary pairs
- **Splits**: 80% train / 10% validation / 10% test

---

## 🔧 Configuration

Edit YAML files in `configs/` to customize training:

**Key Parameters**:
- `model.emb_dim` - Embedding dimension (192)
- `model.hidden_dim` - LSTM hidden size (384-512)
- `model.num_layers` - LSTM layers (1-2)
- `training.batch_size` - Per-GPU batch size (1-2)
- `training.grad_accum` - Gradient accumulation steps (8-16)
- `training.learning_rate` - Initial learning rate (0.0003)
- `training.max_steps` - Training steps (50000)

See `configs/default.yaml` for full configuration options.

---

## 📈 Expected Performance

**Baselines (Lead-K extractive)**:
- Lead-100: ROUGE-L 0.1186
- Lead-150: ROUGE-L 0.1510 (target to beat)

**From-Scratch Training Goals**:
- ✅ Beat Lead-150 baseline (ROUGE-L > 0.15)
- ✅ Achieve ROUGE-L > 0.25 (acceptable performance)
- 🎯 Target ROUGE-L > 0.30 (strong performance)

Training time: ~40-60 hours on RTX 4070 12GB

---

## 🧹 Cleanup

Remove generated artifacts to reclaim disk space:

```powershell
# Preview what will be deleted
.\scripts\clean_repo.ps1 -DryRun

# Clean all artifacts (keeps .venv)
.\scripts\clean_repo.ps1
```

All artifacts can be regenerated by rerunning preprocessing and training.

---

## 🤝 Contributing

This is an academic research project. For questions or collaboration, open a GitHub issue.

**Areas for Improvement**:
- Add unit tests
- Implement additional metrics (BLEU, BERTScore)
- Add Docker support
- Create inference API

---

## 📄 License

MIT License - See LICENSE file for details

**⚠️ Data Privacy**: Never commit patient data or trained models to version control.

---

## 🙏 Acknowledgments

- MIMIC-IV-BHC dataset from PhysioNet
- Pointer-Generator architecture based on See et al. (2017)
- Hierarchical encoding inspired by clinical NLP literature

---

## 📧 Contact

For questions or issues:
- Open a GitHub issue
- Repository: https://github.com/Anto-Rishath008/clinical-note-summarization
