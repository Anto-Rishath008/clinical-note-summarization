"""
Batch Evaluation Script
========================
Re-evaluate ALL main_run checkpoints with deterministic greedy decoding.
Also does length sweep on the best checkpoint found.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.cuda.amp import autocast

from model import MambaTransformerConfig, MambaTransformerModel, build_model
from data_loader import DataConfig, create_dataloaders, load_tokenizer
from evaluate import compute_rouge, evaluate_model


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model_config = MambaTransformerConfig.from_dict(config.get('model', {}))
    model = build_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    step = checkpoint.get('step', 'unknown')
    return model, config, step


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use workspace root for all paths
    ws = Path(__file__).resolve().parent.parent
    os.chdir(ws.parent)  # cd to parent so workspace_mamba_longt5_v1/ paths resolve

    # ── Phase 1.2: Checkpoint re-evaluation ─────────────────────
    checkpoint_dir = ws / "checkpoints" / "main_run"
    results_file = ws / "outputs" / "phase1_diagnostics" / "checkpoint_reeval.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"),
                         key=lambda p: int(p.stem.split("_")[-1]))
    # Also include best_model and final_model
    for name in ["best_model.pt", "final_model.pt"]:
        p = checkpoint_dir / name
        if p.exists() and p not in checkpoints:
            checkpoints.append(p)

    print(f"\n{'='*70}")
    print(f"PHASE 1.2: CHECKPOINT RE-EVALUATION (greedy, max_len=256, 50 samples)")
    print(f"{'='*70}")
    print(f"Found {len(checkpoints)} checkpoints to evaluate\n")

    # Load data once
    first_model, first_config, _ = load_checkpoint(str(checkpoints[0]), device)
    data_config = DataConfig.from_dict(first_config)
    _, val_loader, tokenizer = create_dataloaders(
        config=data_config, batch_size=4, num_workers=0
    )
    del first_model
    torch.cuda.empty_cache()

    all_results = {}
    best_rouge_l = 0.0
    best_ckpt = None

    for ckpt_path in checkpoints:
        name = ckpt_path.stem
        print(f"\n--- Evaluating: {name} ---")
        t0 = time.time()

        model, config, step = load_checkpoint(str(ckpt_path), device)
        scores = evaluate_model(
            model=model,
            val_loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            max_samples=50,
            max_gen_len=256,
        )
        elapsed = time.time() - t0
        print(f"  Step {step} | R1={scores['rouge1']:.4f}  R2={scores['rouge2']:.4f}  RL={scores['rougeL']:.4f}  ({elapsed:.1f}s)")

        all_results[name] = {
            "step": step,
            "rouge1": scores['rouge1'],
            "rouge2": scores['rouge2'],
            "rougeL": scores['rougeL'],
            "time_s": round(elapsed, 1),
        }

        if scores['rougeL'] > best_rouge_l:
            best_rouge_l = scores['rougeL']
            best_ckpt = str(ckpt_path)

        del model
        torch.cuda.empty_cache()

    # Save results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"CHECKPOINT RE-EVAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Checkpoint':<30} {'Step':>6} {'R-1':>8} {'R-2':>8} {'R-L':>8}")
    print(f"{'-'*70}")
    for name, r in sorted(all_results.items(), key=lambda x: x[1].get('step', 0)):
        marker = " ← BEST" if r['rougeL'] == best_rouge_l else ""
        print(f"{name:<30} {r['step']:>6} {r['rouge1']:>8.4f} {r['rouge2']:>8.4f} {r['rougeL']:>8.4f}{marker}")
    print(f"\nBest checkpoint: {best_ckpt} (ROUGE-L={best_rouge_l:.4f})")
    print(f"Results saved to: {results_file}")

    # ── Phase 1.1: Length sweep on best checkpoint ──────────────
    print(f"\n{'='*70}")
    print(f"PHASE 1.1: LENGTH SWEEP on best checkpoint")
    print(f"{'='*70}")

    model, config, step = load_checkpoint(best_ckpt, device)
    length_results = {}

    for max_len in [64, 100, 128, 150, 180, 200, 256, 350]:
        print(f"\n--- max_gen_len={max_len} ---")
        t0 = time.time()
        scores = evaluate_model(
            model=model,
            val_loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            max_samples=50,
            max_gen_len=max_len,
        )
        elapsed = time.time() - t0
        print(f"  R1={scores['rouge1']:.4f}  R2={scores['rouge2']:.4f}  RL={scores['rougeL']:.4f}  ({elapsed:.1f}s)")
        length_results[max_len] = {
            "rouge1": scores['rouge1'],
            "rouge2": scores['rouge2'],
            "rougeL": scores['rougeL'],
        }

    # Find optimal length
    best_len = max(length_results, key=lambda k: length_results[k]['rougeL'])

    print(f"\n{'='*70}")
    print(f"LENGTH SWEEP SUMMARY (checkpoint: {Path(best_ckpt).stem})")
    print(f"{'='*70}")
    print(f"{'max_len':>8} {'R-1':>8} {'R-2':>8} {'R-L':>8}")
    print(f"{'-'*40}")
    for ml in sorted(length_results.keys()):
        r = length_results[ml]
        marker = " ← BEST" if ml == best_len else ""
        print(f"{ml:>8} {r['rouge1']:>8.4f} {r['rouge2']:>8.4f} {r['rougeL']:>8.4f}{marker}")
    print(f"\nOptimal max_gen_len: {best_len}")

    # ── Phase 1.3: Save inspection samples with best settings ──
    print(f"\n{'='*70}")
    print(f"PHASE 1.3: SAVING INSPECTION SAMPLES")
    print(f"{'='*70}")
    output_file = str(ws / "outputs" / "phase1_diagnostics" / "best_samples.txt")
    scores = evaluate_model(
        model=model,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        max_samples=30,
        output_file=output_file,
        max_gen_len=best_len,
    )
    print(f"Best config scores: R1={scores['rouge1']:.4f}  R2={scores['rouge2']:.4f}  RL={scores['rougeL']:.4f}")
    print(f"Samples saved to: {output_file}")

    # Save all results
    all_diagnostics = {
        "checkpoint_reeval": all_results,
        "best_checkpoint": best_ckpt,
        "best_checkpoint_rougeL": best_rouge_l,
        "length_sweep": {str(k): v for k, v in length_results.items()},
        "optimal_max_gen_len": best_len,
    }
    with open(str(ws / "outputs" / "phase1_diagnostics" / "all_diagnostics.json"), 'w') as f:
        json.dump(all_diagnostics, f, indent=2)

    print(f"\n{'='*70}")
    print(f"ALL DIAGNOSTICS COMPLETE")
    print(f"{'='*70}")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Best ROUGE-L: {best_rouge_l:.4f}")
    print(f"Optimal length: {best_len}")


if __name__ == '__main__':
    main()
