# Reports Directory

Small text-based reports and metrics summaries.

## Contents

- `metrics.json` - Final model metrics (ROUGE scores, perplexity)
- `examples.txt` - Sample predictions for qualitative analysis
- `error_analysis.txt` - Common error patterns
- `timing_benchmarks.txt` - Inference speed measurements

## Guidelines

- **Keep files small** (< 1 MB)
- Text-based formats preferred (JSON, TXT, CSV)
- Commit representative results only
- Large prediction files go in `artifacts/predictions/` (gitignored)

## Generating Reports

```powershell
# Generate metrics report
python evaluate.py --checkpoint artifacts/checkpoints/best_model.pt --output reports/metrics.json

# Create example predictions
python scripts/generate_examples.py --output reports/examples.txt
```
