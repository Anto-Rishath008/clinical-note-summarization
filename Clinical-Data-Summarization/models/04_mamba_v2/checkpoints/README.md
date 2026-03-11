# Checkpoints

Model checkpoint files (.pt) are excluded from Git due to size constraints.

## Files (available locally)

| File | Description |
|------|-------------|
| `best_model.pt` | ★ Best model by ROUGE-L |
| `final_model.pt` | Final model at step 150K |
| `safety_backup_best_model.pt` | Safety backup |
| `emergency_step_150000.pt` | Emergency save |
| `checkpoint_step_50000.pt` | 50K milestone |
| `checkpoint_step_75000.pt` | 75K milestone |
| `checkpoint_step_100000.pt` | 100K milestone |
| `checkpoint_step_125000.pt` | 125K milestone |
| `checkpoint_step_130000.pt` | 130K (best ROUGE) |
| `checkpoint_step_150000.pt` | 150K (final) |

## Reproduction

```bash
python src/train.py --config configs/full_train.yaml
```
