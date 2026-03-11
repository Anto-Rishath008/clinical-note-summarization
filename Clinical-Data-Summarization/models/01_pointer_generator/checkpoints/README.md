# Checkpoints

Model checkpoint files (.pt) are excluded from Git due to size constraints.

## Files (available locally)

| File | Description |
|------|-------------|
| `best_model.pt` | Best model by validation loss |
| `checkpoint_step_*.pt` | Step-wise checkpoints (500–9000) |

## Reproduction

Train from scratch:
```bash
python src/train.py --config configs/default.yaml
```
