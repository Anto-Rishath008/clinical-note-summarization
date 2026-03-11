"""
Fast Gated Clinical Evaluation - Greedy-first approach
Uses the model's built-in generate() for speed, 
then beam search on a small subset for comparison.
"""
import os, sys, re, json, time, argparse, warnings
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from src.model import MambaTransformerConfig, MambaTransformerModel, build_model
from src.data_loader import DataConfig, create_dataloaders, load_tokenizer

warnings.filterwarnings("ignore")

def flush(msg):
    print(msg, flush=True)

# ── Import safety/quality functions from clinical_eval ──
from src.clinical_eval import (
    compute_rouge, check_negation_consistency,
    check_entity_preservation, check_factual_consistency_proxy,
    check_phi_leakage, check_repetition, check_length_stats,
)

# ── Beam search (simplified, max 10 samples) ──
@torch.no_grad()
def generate_beam_simple(model, src, src_mask, beam_size=4, max_len=128, 
                         no_repeat_ngram=3, min_len=15):
    """Beam search on single sample (batch=1 only)."""
    device = src.device
    memory = model.encode(src, src_mask)  # [1, mem_len, d]
    
    beams = [(0.0, [model.config.bos_id])]
    completed = []
    
    for step in range(max_len - 1):
        candidates = []
        for log_prob, seq in beams:
            if seq[-1] == model.config.eos_id:
                completed.append((log_prob, seq))
                continue
            
            inp = torch.tensor([seq], device=device, dtype=torch.long)
            causal = model.generate_causal_mask(inp.size(1), device)
            logits = model.decode(inp, memory, causal)
            next_logits = logits[0, -1, :]
            
            # Block repeated n-grams
            if no_repeat_ngram > 0 and len(seq) >= no_repeat_ngram - 1:
                ngrams_seen = set()
                for i in range(len(seq) - no_repeat_ngram + 1):
                    ngrams_seen.add(tuple(seq[i:i + no_repeat_ngram]))
                partial = tuple(seq[-(no_repeat_ngram - 1):])
                for ng in ngrams_seen:
                    if ng[:-1] == partial:
                        next_logits[ng[-1]] = float('-inf')
            
            if step < min_len:
                next_logits[model.config.eos_id] = float('-inf')
            
            log_probs = F.log_softmax(next_logits, dim=-1)
            topk_lp, topk_ids = log_probs.topk(beam_size)
            
            for k in range(beam_size):
                candidates.append((log_prob + topk_lp[k].item(), seq + [topk_ids[k].item()]))
        
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0] / (len(x[1]) ** 0.6), reverse=True)
        beams = candidates[:beam_size]
    
    all_seqs = completed + beams
    if not all_seqs:
        return [model.config.bos_id]
    all_seqs.sort(key=lambda x: x[0] / (len(x[1]) ** 0.6), reverse=True)
    return all_seqs[0][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--beam_samples", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    flush(f"Loading checkpoint: {args.checkpoint}")
    cp = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = cp["config"]
    model_cfg = MambaTransformerConfig.from_dict(cfg.get("model", {}))
    model = build_model(model_cfg)
    model.load_state_dict(cp["model_state_dict"])
    model = model.to(device).eval()
    step = cp.get("step", 0)
    flush(f"  Step: {step}  |  Best ROUGE-L: {cp.get('best_rouge_l', 0):.4f}")
    
    # Load data
    with open(args.config) as f:
        ycfg = yaml.safe_load(f)
    data_cfg = DataConfig.from_dict(ycfg)
    _, val_loader, tokenizer = create_dataloaders(config=data_cfg, batch_size=4, num_workers=0)
    
    decode = lambda ids: tokenizer.DecodeIds([t for t in ids if t not in [0, 1, 2, 3]])
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: GREEDY generation (fast, uses model.generate)
    # ═══════════════════════════════════════════════════════════════
    flush("\n" + "="*60)
    flush("  PHASE 1: GREEDY DECODING")
    flush("="*60)
    
    sources, preds_greedy, refs = [], [], []
    src_tensors, mask_tensors = [], []
    n = 0
    t0 = time.time()
    
    for batch in val_loader:
        if n >= args.max_samples:
            break
        
        src = batch["src"].to(device)
        tgt_out = batch["tgt_output"].to(device)
        src_mask = batch["src_mask"].to(device)
        
        with torch.no_grad():
            gen = model.generate(src, src_mask, max_len=256, greedy=True,
                                 no_repeat_ngram_size=3, repetition_penalty=1.2)
        
        for i in range(src.size(0)):
            if n >= args.max_samples:
                break
            sources.append(decode(src[i].cpu().tolist()))
            refs.append(decode(tgt_out[i].cpu().tolist()))
            preds_greedy.append(decode(gen[i].cpu().tolist()))
            
            if n < args.beam_samples:
                src_tensors.append(src[i:i+1])
                mask_tensors.append(src_mask[i:i+1])
            
            n += 1
        
        if n % 10 == 0:
            flush(f"  Greedy: {n}/{args.max_samples} ({time.time()-t0:.0f}s)")
    
    greedy_time = time.time() - t0
    flush(f"  Greedy done: {n} samples in {greedy_time:.1f}s ({greedy_time/n:.1f}s/sample)")
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: BEAM SEARCH on small subset
    # ═══════════════════════════════════════════════════════════════
    flush("\n" + "="*60)
    flush(f"  PHASE 2: BEAM SEARCH (beam=4, {args.beam_samples} samples)")
    flush("="*60)
    
    preds_beam = []
    t1 = time.time()
    
    for i in range(min(args.beam_samples, len(src_tensors))):
        beam_ids = generate_beam_simple(model, src_tensors[i], mask_tensors[i],
                                        beam_size=4, max_len=256, 
                                        no_repeat_ngram=3, min_len=20)
        preds_beam.append(decode(beam_ids))
        elapsed = time.time() - t1
        flush(f"  Beam sample {i+1}/{args.beam_samples} ({elapsed:.0f}s)")
    
    beam_time = time.time() - t1
    flush(f"  Beam done: {len(preds_beam)} samples in {beam_time:.1f}s ({beam_time/max(len(preds_beam),1):.1f}s/sample)")
    
    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: GATED EVALUATION
    # ═══════════════════════════════════════════════════════════════
    flush("\n" + "="*60)
    flush("  GATED CLINICAL EVALUATION")
    flush("="*60)
    
    preds = preds_greedy  # Primary evaluation uses greedy
    
    # ── Gate 1: Safety ──
    flush("\n-- Gate 1: SAFETY --")
    
    neg_rate, neg_fails = check_negation_consistency(sources, preds, refs)
    flush(f"  Negation error rate  : {neg_rate:.4f} ({len(neg_fails)} failures)")
    
    ent_metrics, ent_fails = check_entity_preservation(sources, preds, refs)
    flush(f"  Entity F1            : {ent_metrics['entity_f1']:.4f}")
    flush(f"  Entity recall        : {ent_metrics['entity_recall']:.4f}")
    flush(f"  Entity precision     : {ent_metrics['entity_precision']:.4f}")
    
    fact_metrics, fact_fails = check_factual_consistency_proxy(sources, preds)
    flush(f"  Hallucination rate   : {fact_metrics['hallucination_rate']:.4f}")
    flush(f"  Numeric conflict     : {fact_metrics['numeric_conflict_rate']:.4f}")
    flush(f"  Contradiction rate   : {fact_metrics['contradiction_rate']:.4f}")
    
    phi_rate, phi_fails = check_phi_leakage(preds)
    flush(f"  PHI leakage rate     : {phi_rate:.4f}")
    
    # Gating
    failure_reasons = []
    if phi_rate > 0:
        failure_reasons.append(f"PHI_LEAKAGE={phi_rate:.4f}")
    if neg_rate > 0.02:
        failure_reasons.append(f"NEGATION_ERROR={neg_rate:.4f}>0.02")
    if fact_metrics["contradiction_rate"] > 0.15:
        failure_reasons.append(f"CONTRADICTION={fact_metrics['contradiction_rate']:.4f}>0.15")
    
    safety_pass = len(failure_reasons) == 0
    flush(f"\n  SAFETY GATE: {'PASS' if safety_pass else 'FAIL'}")
    if failure_reasons:
        for r in failure_reasons:
            flush(f"    - {r}")
    
    # ── Gate 2: Quality ──
    flush("\n-- Gate 2: QUALITY --")
    rep = check_repetition(preds)
    lens = check_length_stats(preds, refs)
    flush(f"  Dup bigram rate      : {rep['dup_bigram_rate']:.4f}")
    flush(f"  Dup trigram rate     : {rep['dup_trigram_rate']:.4f}")
    flush(f"  Avg pred len (words) : {lens['avg_pred_len']:.1f}")
    flush(f"  Avg ref  len (words) : {lens['avg_ref_len']:.1f}")
    flush(f"  Length ratio         : {lens['len_ratio']:.2f}")
    
    # ── Gate 3: Similarity ──
    flush("\n-- Gate 3: SIMILARITY --")
    
    rouge_greedy, per_sample = compute_rouge(preds_greedy, refs)
    flush(f"  ROUGE-1 (greedy)     : {rouge_greedy['rouge1']:.4f}")
    flush(f"  ROUGE-2 (greedy)     : {rouge_greedy['rouge2']:.4f}")
    flush(f"  ROUGE-L (greedy)     : {rouge_greedy['rougeL']:.4f}")
    
    # Beam comparison on subset
    if preds_beam:
        beam_refs = refs[:len(preds_beam)]
        greedy_subset = preds_greedy[:len(preds_beam)]
        
        rouge_beam, _ = compute_rouge(preds_beam, beam_refs)
        rouge_greedy_sub, _ = compute_rouge(greedy_subset, beam_refs)
        
        flush(f"\n  -- BEAM vs GREEDY (same {len(preds_beam)} samples) --")
        flush(f"  ROUGE-L greedy       : {rouge_greedy_sub['rougeL']:.4f}")
        flush(f"  ROUGE-L beam         : {rouge_beam['rougeL']:.4f}")
        delta = rouge_beam['rougeL'] - rouge_greedy_sub['rougeL']
        flush(f"  Delta (beam-greedy)  : {delta:+.4f}")
        flush(f"  ROUGE-1 beam         : {rouge_beam['rouge1']:.4f}")
        flush(f"  ROUGE-2 beam         : {rouge_beam['rouge2']:.4f}")
    
    # BERTScore (optional)
    bertscore_f1 = 0.0
    try:
        from bert_score import score as bscore
        P, R, F = bscore(preds[:20], refs[:20], model_type="bert-base-uncased", verbose=False)
        bertscore_f1 = F.mean().item()
        flush(f"\n  BERTScore-F1 (20 samples): {bertscore_f1:.4f}")
    except Exception as e:
        flush(f"\n  BERTScore: skipped ({e})")
    
    # ═══════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════
    verdict = "PASS" if safety_pass else "FAIL"
    
    flush("\n" + "="*60)
    flush(f"  OVERALL VERDICT: {verdict}")
    flush("="*60)
    
    if verdict == "FAIL":
        flush("  Failure reasons:")
        for r in failure_reasons:
            flush(f"    - {r}")
    else:
        flush("  All safety gates passed.")
    
    # ── Top failure examples ──
    all_failures = neg_fails + ent_fails[:10] + fact_fails[:10] + phi_fails
    if all_failures:
        flush(f"\n-- Top {min(10, len(all_failures))} Failure Examples --")
        for i, f in enumerate(all_failures[:10]):
            flush(f"\n  [{i+1}] Type: {f['type']}")
            if "flipped_entities" in f:
                flush(f"      Flipped: {f['flipped_entities']}")
            if "missed_entities" in f:
                flush(f"      Missed:  {f['missed_entities']}")
            if "hallucinated_entities" in f:
                flush(f"      Halluc:  {f['hallucinated_entities']}")
            if "leaked_items" in f:
                flush(f"      PHI:     {f['leaked_items']}")
    
    # ── Save JSON ──
    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "checkpoint_step": step,
        "n_samples": n,
        "n_beam_samples": len(preds_beam),
        "greedy_time_s": round(greedy_time, 1),
        "beam_time_s": round(beam_time, 1),
        "safety": {
            "negation_error_rate": neg_rate,
            "entity_f1": ent_metrics["entity_f1"],
            "entity_recall": ent_metrics["entity_recall"],
            "entity_precision": ent_metrics["entity_precision"],
            "hallucination_rate": fact_metrics["hallucination_rate"],
            "numeric_conflict_rate": fact_metrics["numeric_conflict_rate"],
            "contradiction_rate": fact_metrics["contradiction_rate"],
            "phi_leakage_rate": phi_rate,
            "safety_pass": safety_pass,
            "failure_reasons": failure_reasons,
        },
        "quality": {
            "dup_bigram_rate": rep["dup_bigram_rate"],
            "dup_trigram_rate": rep["dup_trigram_rate"],
            "avg_pred_len": lens["avg_pred_len"],
            "avg_ref_len": lens["avg_ref_len"],
            "len_ratio": lens["len_ratio"],
        },
        "similarity": {
            "greedy_rouge1": rouge_greedy["rouge1"],
            "greedy_rouge2": rouge_greedy["rouge2"],
            "greedy_rougeL": rouge_greedy["rougeL"],
            "bertscore_f1": bertscore_f1,
        },
        "beam_vs_greedy": {
            "beam_rougeL": rouge_beam["rougeL"] if preds_beam else 0,
            "greedy_rougeL_subset": rouge_greedy_sub["rougeL"] if preds_beam else 0,
            "delta": delta if preds_beam else 0,
            "beam_rouge1": rouge_beam["rouge1"] if preds_beam else 0,
            "beam_rouge2": rouge_beam["rouge2"] if preds_beam else 0,
        },
        "verdict": verdict,
    }
    
    # Sample outputs
    sample_outputs = []
    for i in range(min(5, n)):
        entry = {
            "source": sources[i][:500],
            "reference": refs[i][:500],
            "greedy": preds_greedy[i][:500],
        }
        if i < len(preds_beam):
            entry["beam"] = preds_beam[i][:500]
        sample_outputs.append(entry)
    
    out = {"report": report, "failures": all_failures[:30], "sample_outputs": sample_outputs}
    
    json_path = os.path.join(args.output_dir, f"eval_step_{step}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    flush(f"\nSaved -> {json_path}")


if __name__ == "__main__":
    main()
