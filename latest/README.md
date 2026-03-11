# Clinical Note Summarization — Organized Project

> All 4 models consolidated from 9 Git branches + local workspace into a single, navigable folder.

---

## Directory Structure

```
latest/
├── 01_Pointer_Generator/       # Model 1: BiLSTM + Attention + Copy Mechanism
│   ├── src/                    # core.py (model), train.py, evaluate.py, preprocess, inference
│   ├── configs/                # 5 YAML configs (default, full_training, resume, rtx4070, stage1)
│   ├── scripts/                # Evaluation & analysis scripts
│   ├── checkpoints/            # 19 .pt files (best_model + step checkpoints)
│   ├── logs/                   # Training logs + metrics.csv
│   ├── visualizations/         # Architecture .dot files
│   ├── results/                # ROUGE plots, length analysis, detailed results
│   └── testing/                # Inference visualization, example outputs
│
├── 02_LongT5/                  # Model 2: Local-Global Attention Transformer
│   ├── src/                    # model.py, train.py, inference.py, data_loader.py, visualize
│   ├── configs/                # config.yaml, longt5_config.yaml
│   ├── scripts/                # test_longt5.py
│   ├── docs/                   # README, architecture.md, issue1_resolution.md
│   └── visualizations/         # 14 files: .dot, .png, graphviz renders
│
├── 03_Mamba_V1/                # Model 3: SSM Encoder + Transformer Decoder (V1)
│   ├── src/                    # model.py, train.py, data_loader.py, evaluate.py
│   ├── configs/                # default.yaml, local_train.yaml, main_train.yaml
│   ├── scripts/                # Kaggle scripts (notebook runner, push_to_kaggle)
│   ├── checkpoints/            # 2 .pt files (best_model, bart_step_13000)
│   ├── logs/                   # train.log
│   ├── notebooks/              # Kaggle training notebook (.ipynb) + metadata
│   └── debug/                  # debug_nan.py
│
├── 04_Mamba_V2/                # Model 4: BiMamba + Gated Cross-Attn + SwiGLU (V2)
│   ├── src/                    # model.py, train.py, data_loader.py, evaluate.py, clinical_eval.py, train_to_target.py
│   ├── configs/                # 6 YAML configs (full_train, exp_memory32, optimized, resume, rouge_target, default)
│   ├── scripts/                # app.py, fast_eval, gated_eval_v2, monitor, comprehensive_eval, batch_eval
│   ├── checkpoints/            # 49 .pt files (best, final, emergency, safety, manual + step checkpoints)
│   ├── logs/                   # 10 files (train logs, crash report, status)
│   ├── docs/                   # 6 docs (ARCHITECTURE, CHANGELOG_V3, CLINICAL_METRICS, MODEL_ARCHITECTURE, MODEL_LEARNING, MODEL_MATHEMATICS)
│   ├── webapp/                 # Web demo (server.py, launch.bat, templates/index.html)
│   ├── patches/                # 8 hotfix scripts (patch1-3, patch_oom, patch_train, fix_clean/dup/try)
│   └── utilities/              # 8 utility scripts (check_env, debug_check, launch/run/restart train, verify)
│
└── shared/                     # Resources shared across all models
    ├── data/
    │   ├── raw/                # mimic-iv-bhc.csv (2.6 GB)
    │   ├── tokenized/          # train/val/test .parquet + .csv splits
    │   ├── tokenizer/          # SentencePiece models (PG + Mamba variants)
    │   └── sample/             # sample_data.json
    ├── docs/                   # 5 shared docs (implementation, architecture, optimization)
    ├── experiments/             # 3 result CSVs (baseline, decode_sanity, results)
    ├── presentation/           # LaTeX Beamer (45 slides) + 27 figures
    ├── results/                # Notebook analysis + example outputs
    └── requirements.txt        # Python dependencies
```

---

## Model Summary

| # | Model | Architecture | Parameters | Branch Origin |
|---|-------|-------------|-----------|---------------|
| 1 | Pointer Generator | BiLSTM + Bahdanau Attention + Copy | ~22.5M | `main` |
| 2 | LongT5 | Local-Global Attention Transformer | ~12.5M | `simplified_longt5` |
| 3 | Mamba V1 | SSM Encoder + Transformer Decoder | ~52.1M | `feature/mamba-longt5-hybrid-v1`, `training-v2` |
| 4 | Mamba V2 | BiMamba + Gated Cross-Attn + SwiGLU | ~79.4M | `latest-mamba-enhanced-version`, `v3-source-bypass`, `memory32`, `rouge-v3` |

---

## Source Branches Consolidated

| Branch | Content | Destination |
|--------|---------|-------------|
| `main` | PG model, shared infra, docs, experiments | `01_Pointer_Generator/`, `shared/` |
| `simplified_longt5` | LongT5 model core | `02_LongT5/` |
| `feature/mamba-longt5-hybrid-v1` | V1 Mamba + Kaggle notebooks | `03_Mamba_V1/` |
| `feature/mamba-longt5-training-v2` | V1 training + debug_nan | `03_Mamba_V1/` |
| `feature/mamba-memory32-experiment` | V2 memory expansion experiments | `04_Mamba_V2/` |
| `feature/rouge-target-training-v3` | V2 ROUGE-target configs + train_to_target | `04_Mamba_V2/` |
| `feature/v3-source-bypass-training-improvements` | V2 final training improvements | `04_Mamba_V2/` |
| `latest-mamba-enhanced-version` | V2 enhanced architecture | `04_Mamba_V2/` |
| `organized/clinical-summarization-models` | Presentation, extra viz files | `shared/`, `02_LongT5/visualizations/` |

---

## Dataset

**MIMIC-IV BHC** (Brief Hospital Course) — 270K+ discharge summaries from PhysioNet.

Access requires PhysioNet credentialed access: https://physionet.org/content/mimic-iv-note/

---

## Quick Start

```bash
# Navigate to any model
cd latest/01_Pointer_Generator/

# View configs
cat configs/default.yaml

# Train (example for PG)
python src/train.py --config configs/default.yaml

# Evaluate
python src/evaluate.py --checkpoint checkpoints/best_model.pt
```
