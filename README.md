# Clinical Note Summarization with Pointer-Generator Network

A hierarchical sequence-to-sequence model for summarizing clinical narratives from the MIMIC-IV-BHC dataset.

## 🚀 Quick Start on Kaggle

### 1. Upload Dataset
Upload the tokenized dataset as a Kaggle dataset. Your dataset should contain:
- `train.parquet`
- `val.parquet`
- `test.parquet`

### 2. Create Kaggle Notebook

```python
# Install dependencies
!pip install -q sentencepiece rouge-score pyyaml

# Clone or copy this repo to /kaggle/working/
import shutil
import os

# Assuming you've uploaded this as a dataset
!cp -r /kaggle/input/clinical-summarization-code/* /kaggle/working/
os.chdir('/kaggle/working')
```

### 3. Run Training

```python
!python train.py \
    --config configs/default.yaml \
    --tokenized_dir /kaggle/input/your-tokenized-dataset \
    --run_name kaggle_run \
    --output_dir /kaggle/working/artifacts \
    --max_steps 5000
```

**Key Arguments:**
- `--config`: Configuration file (use `configs/default.yaml`)
- `--tokenized_dir`: Path to your tokenized dataset on Kaggle
- `--run_name`: Name for this training run
- `--output_dir`: Where to save checkpoints (default: `/kaggle/working/artifacts`)
- `--max_steps`: Training steps (adjust based on your GPU quota)
- `--resume`: Path to checkpoint to resume training

### 4. Resume from Checkpoint

```python
# Resume training from the best saved checkpoint
!python train.py \
    --config configs/default.yaml \
    --tokenized_dir /kaggle/input/your-tokenized-dataset \
    --run_name kaggle_run \
    --output_dir /kaggle/working/artifacts \
    --resume /kaggle/working/artifacts/checkpoints/kaggle_run/best_model.pt \
    --max_steps 10000
```

### 5. Evaluate Model

```python
!python evaluate.py \
    --checkpoint /kaggle/working/artifacts/checkpoints/kaggle_run/best_model.pt \
    --tokenized_dir /kaggle/input/your-tokenized-dataset \
    --split val \
    --beam_size 4
```

### 6. Run Baselines (Optional)

```python
!python baselines.py \
    --tokenized_dir /kaggle/input/your-tokenized-dataset \
    --tokenizer_dir /kaggle/input/clinical-summarization-code/artifacts/tokenizer \
    --output_dir /kaggle/working/artifacts/baselines
```

---

## 📦 Project Structure

```
clinical-summarization/
├── Clinical_Summarization_EndToEnd.ipynb  # Complete training & evaluation notebook
├── train.py                               # Training script (Kaggle-ready)
├── evaluate.py                            # Evaluation script
├── baselines.py                           # Baseline models
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
├── configs/
│   └── default.yaml                       # Model & training configuration
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py                       # Pointer-Generator Model
│   │   ├── encoder.py                     # Hierarchical Encoder
│   │   ├── decoder.py                     # Decoder with attention
│   │   └── attention.py                   # Attention mechanisms
│   └── utils/
│       ├── __init__.py
│       ├── dataset.py                     # Data loading utilities
│       ├── metrics.py                     # ROUGE evaluation
│       └── beam_search.py                 # Beam search decoder
├── artifacts/
│   ├── tokenizer/
│   │   ├── spm.model                      # SentencePiece model
│   │   └── spm.vocab                      # Vocabulary
│   └── checkpoints/
│       └── final_check/
│           └── best_model.pt              # Pre-trained checkpoint
└── data/
    └── .gitkeep                           # Placeholder (Kaggle provides data)
```

---

## 🔧 Local Training (Non-Kaggle)

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training Locally
```bash
python train.py \
    --config configs/default.yaml \
    --tokenized_dir path/to/tokenized/data \
    --run_name local_run \
    --max_steps 10000
```

### Evaluation
```bash
python evaluate.py \
    --checkpoint artifacts/checkpoints/local_run/best_model.pt \
    --tokenized_dir path/to/tokenized/data \
    --split test
```

---

## 📊 Model Architecture

**Pointer-Generator Network with Hierarchical Encoder:**
- **Encoder**: Hierarchical Transformer with 4 chunk-level layers + 2 document-level layers
- **Decoder**: 4-layer Transformer with coverage mechanism
- **Attention**: Bahdanau-style attention + pointer network for copy mechanism
- **Vocab Size**: 8,000 SentencePiece tokens
- **Parameters**: ~42M trainable parameters

**Key Features:**
- Coverage mechanism to reduce repetition
- Copy mechanism to handle medical terminology
- Label smoothing (0.1) for better generalization
- Mixed precision (FP16) training
- Gradient accumulation for large effective batch sizes

---

## 📈 Expected Performance

| Method          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-----------------|---------|---------|---------|
| Lead-K Baseline | ~30.0   | ~10.0   | ~18.0   |
| Random Baseline | ~15.0   | ~3.0    | ~10.0   |
| **Our Model**   | **36-40**| **12-15**| **22-25**|

---

## ⚙️ Configuration

Edit `configs/default.yaml` to customize:

```yaml
model:
  vocab_size: 8000
  d_model: 256
  num_encoder_layers_chunk: 4
  num_encoder_layers_doc: 2
  num_decoder_layers: 4
  nhead: 8
  coverage_lambda: 1.0

training:
  batch_size: 4
  grad_accum: 4          # Effective batch = 16
  learning_rate: 0.0001
  max_steps: 50000
  warmup_steps: 1000
  fp16: true
  clip_grad: 1.0
  eval_every: 500
  save_every: 1000
```

---

## 🛠️ Troubleshooting

### Out of Memory on Kaggle
Reduce batch size or gradient accumulation:
```yaml
training:
  batch_size: 2
  grad_accum: 2
```

### Training Too Slow
Enable mixed precision and reduce evaluation frequency:
```yaml
training:
  fp16: true
  eval_every: 1000
```

### Resume Training
Always save checkpoints regularly. Resume with:
```bash
--resume /path/to/checkpoint.pt
```

---

## 📝 Citation

Dataset: MIMIC-IV-BHC (Behavioral Health Conditions)
- https://physionet.org/content/mimic-iv-bhc/1.0/

Model inspired by:
- See et al. (2017): "Get To The Point: Summarization with Pointer-Generator Networks"

---

## 📄 License

This code is provided for research and educational purposes. The MIMIC-IV-BHC dataset requires credentialed access from PhysioNet.

---

## 🤝 Contributing

For bugs or feature requests, please open an issue on GitHub.

---

**Happy Training! 🎉**
