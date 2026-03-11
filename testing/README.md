# Testing & Model Inference

This folder contains scripts and example outputs for testing the trained Pointer-Generator model.

## Files

- `model_inference_visualization.py` - Main inference script that:
  - Loads the trained model checkpoint
  - Generates summaries for test samples
  - Displays Original vs Generated Summary comparisons
  - Computes ROUGE scores
  - Exports results to CSV

## Example Outputs

The `example_outputs/` folder contains sample results from running the inference script:

- `detailed_results.csv` - Per-sample results including:
  - Source text (truncated)
  - Reference summary
  - Generated summary
  - ROUGE-1, ROUGE-2, ROUGE-L scores
  - Length statistics and compression ratios

- `summary_statistics.csv` - Aggregated metrics:
  - Average ROUGE scores
  - Standard deviations
  - Compression ratios

## Usage

```bash
# Run inference visualization
cd <project_root>
python testing/model_inference_visualization.py
```

## Results Summary (10 samples)

| Metric | Mean | Std Dev |
|--------|------|---------|
| ROUGE-1 | 0.3214 | ±0.0561 |
| ROUGE-2 | 0.0977 | ±0.0477 |
| ROUGE-L | 0.1894 | ±0.0401 |

### Length Statistics
- Source: ~1010 tokens
- Reference: ~319 tokens
- Generated: ~139 tokens
- Compression: 13.6%
