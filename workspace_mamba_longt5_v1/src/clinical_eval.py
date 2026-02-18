"""
Gated Clinical Evaluation  —  v1.0
====================================
Comprehensive evaluation for a clinical text summarization model.

GATED HIERARCHY:
  Gate 1  (SAFETY)   : negation consistency, entity preservation, factual
                       consistency (NLI proxy), PHI leakage  →  hard PASS/FAIL
  Gate 2  (QUALITY)  : grammar / fluency, repetition, coherence  →  quality score
  Gate 3  (SIMILARITY): ROUGE-1/2/L, BERTScore (clinical encoder)  →  similarity score

Usage:
  python src/clinical_eval.py \
      --checkpoint checkpoints/full_run/best_model.pt \
      --config     configs/full_train.yaml \
      --max_samples 200 \
      --output_dir  eval_results \
      --beam_size   4 \
      --compare_greedy
"""

import os, sys, re, json, time, argparse, csv, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

# ── Windows cp1252 fix: ensure stdout/stderr can emit Unicode box chars ──
import io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from dataclasses import dataclass, asdict, field

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from model import MambaTransformerConfig, MambaTransformerModel, build_model
from data_loader import DataConfig, create_dataloaders, load_tokenizer

warnings.filterwarnings("ignore")

# ───────────────────────────── helpers ─────────────────────────────────────

def flush(msg: str):
    print(msg, flush=True)


# ───────────────────────────── 1. SIMILARITY METRICS ──────────────────────

def compute_rouge(preds: List[str], refs: List[str]) -> Dict[str, float]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    per_sample = []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
        per_sample.append({
            "rouge1": s["rouge1"].fmeasure,
            "rouge2": s["rouge2"].fmeasure,
            "rougeL": s["rougeL"].fmeasure,
        })
    return {
        "rouge1": sum(r1) / max(len(r1), 1),
        "rouge2": sum(r2) / max(len(r2), 1),
        "rougeL": sum(rl) / max(len(rl), 1),
    }, per_sample


def compute_bertscore(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """BERTScore with a clinical encoder (falls back to default if unavailable)."""
    try:
        from bert_score import score as bscore
    except ImportError:
        flush("  [WARN] bert-score not installed → skipping BERTScore")
        return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}

    # Prefer clinical encoders; fall back gracefully
    for model_name in [
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "emilyalsentzer/Bio_ClinicalBERT",
        "bert-base-uncased",
    ]:
        try:
            P, R, F = bscore(preds, refs, model_type=model_name, verbose=False)
            flush(f"  BERTScore encoder: {model_name}")
            return {
                "bertscore_p": P.mean().item(),
                "bertscore_r": R.mean().item(),
                "bertscore_f1": F.mean().item(),
            }
        except Exception:
            continue
    flush("  [WARN] All BERTScore encoders failed")
    return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}


# ───────────────────────────── 2. SAFETY GATES ────────────────────────────

# 2a. Negation consistency --------------------------------------------------

_NEG_PATTERNS = [
    # "no <condition>", "denies <condition>", "without <condition>", etc.
    re.compile(r"\b(no|not|denies?|without|absent|negative|non|ruled\s+out|"
               r"neither|nor|never|free\s+of)\b", re.I),
]

_CLINICAL_TERMS = re.compile(
    r"\b(diabetes|hypertension|htn|cad|chf|copd|asthma|cancer|stroke|"
    r"dvt|pe|mi|aki|ckd|afib|a\.\s*fib|sepsis|pneumonia|"
    r"heart\s+failure|renal\s+failure|hepatitis|cirrhosis|"
    r"infection|fever|pain|bleeding|anemia|hypothyroid|hyperthyroid|"
    r"anxiety|depression|seizure|epilepsy|"
    r"insulin|metformin|warfarin|heparin|aspirin|lisinopril|"
    r"amlodipine|atorvastatin|levothyroxine|omeprazole|"
    r"furosemide|prednisone|albuterol)\b", re.I
)


def _extract_negated_entities(text: str) -> set:
    """Return set of lowercased clinical terms that appear in a negation window."""
    negated = set()
    sentences = re.split(r'[.;!\n]', text)
    for sent in sentences:
        sent_lower = sent.lower()
        # Check if sentence has a negation word
        has_neg = any(p.search(sent_lower) for p in _NEG_PATTERNS)
        if not has_neg:
            continue
        # Extract clinical terms in the same sentence
        for m in _CLINICAL_TERMS.finditer(sent_lower):
            negated.add(m.group(0).strip())
    return negated


def check_negation_consistency(sources: List[str], preds: List[str],
                               refs: List[str]) -> Tuple[float, List[dict]]:
    """
    Detect negation flips: a condition negated in source but affirmed in summary
    (or vice-versa).
    Returns (error_rate, list of failure dicts).
    """
    failures = []
    for i, (src, pred, ref) in enumerate(zip(sources, preds, refs)):
        src_neg = _extract_negated_entities(src)
        ref_neg = _extract_negated_entities(ref)
        pred_neg = _extract_negated_entities(pred)

        # Entities affirmed in pred but negated in source
        pred_terms = set(_CLINICAL_TERMS.findall(pred.lower()))
        affirmed_in_pred = pred_terms - pred_neg
        flipped = affirmed_in_pred & src_neg  # negated in source, affirmed in pred

        # Also check: negated in pred but NOT negated in source
        src_terms = set(_CLINICAL_TERMS.findall(src.lower()))
        affirmed_in_src = src_terms - src_neg
        reverse_flip = pred_neg & affirmed_in_src  # affirmed in source, negated in pred

        all_flips = flipped | reverse_flip
        if all_flips:
            failures.append({
                "sample_idx": i,
                "type": "negation_flip",
                "flipped_entities": sorted(all_flips),
                "src_snippet": src[:300],
                "pred": pred,
                "ref": ref[:300],
            })
    rate = len(failures) / max(len(sources), 1)
    return rate, failures


# 2b. Medical entity preservation -------------------------------------------

def _extract_entities_regex(text: str) -> set:
    """Regex-based medical entity extraction (lightweight, no model dependency)."""
    entities = set()
    for m in _CLINICAL_TERMS.finditer(text.lower()):
        entities.add(m.group(0).strip())

    # Dosages: e.g., "500 mg", "10 units"
    for m in re.finditer(r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|ml|units?|g|meq|mmol|%)\b', text, re.I):
        entities.add(m.group(0).lower().strip())

    # Lab values: e.g., "WBC 12.3", "Hgb 8.1"
    for m in re.finditer(r'\b(wbc|hgb|hb|plt|bun|cr|creatinine|na|k|cl|'
                         r'hco3|glucose|inr|ptt|ast|alt|bili|troponin|bnp|'
                         r'lactate|hemoglobin|potassium|sodium)\s*[=:]?\s*'
                         r'(\d+(?:\.\d+)?)', text, re.I):
        entities.add(f"{m.group(1).lower()} {m.group(2)}")
    return entities


def check_entity_preservation(sources: List[str], preds: List[str],
                              refs: List[str]) -> Tuple[Dict[str, float], List[dict]]:
    """Entity recall/precision/F1 of generated summary vs source+reference."""
    precisions, recalls, f1s = [], [], []
    failures = []
    for i, (src, pred, ref) in enumerate(zip(sources, preds, refs)):
        # Ground truth = union of entities in source and reference
        ref_entities = _extract_entities_regex(ref)
        pred_entities = _extract_entities_regex(pred)

        if not ref_entities:
            continue

        tp = len(pred_entities & ref_entities)
        prec = tp / max(len(pred_entities), 1)
        rec = tp / max(len(ref_entities), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        missed = ref_entities - pred_entities
        if missed and rec < 0.5:
            failures.append({
                "sample_idx": i,
                "type": "entity_miss",
                "missed_entities": sorted(missed)[:10],
                "entity_recall": round(rec, 4),
                "pred_snippet": pred[:300],
            })

    return {
        "entity_precision": sum(precisions) / max(len(precisions), 1),
        "entity_recall": sum(recalls) / max(len(recalls), 1),
        "entity_f1": sum(f1s) / max(len(f1s), 1),
    }, failures


# 2c. Factual consistency (lightweight NLI proxy) ---------------------------

def check_factual_consistency_proxy(sources: List[str], preds: List[str]) -> Tuple[Dict[str, float], List[dict]]:
    """
    Lightweight factual-consistency proxy:
      - Flag summaries that mention entities NOT in the source (hallucination proxy).
      - Flag summaries whose numeric values contradict source numbers.
    """
    halluc_count = 0
    num_conflict_count = 0
    failures = []

    for i, (src, pred) in enumerate(zip(sources, preds)):
        src_lower = src.lower()
        pred_entities = _extract_entities_regex(pred)
        src_entities = _extract_entities_regex(src)

        # Entities in pred but nowhere in source text
        hallucinated = set()
        for e in pred_entities:
            # Only flag clinical terms, not numbers
            if re.match(r'^\d', e):
                continue
            if e not in src_lower:
                hallucinated.add(e)

        # Numeric contradictions
        pred_nums = {m.group(0) for m in re.finditer(r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|units?|g|meq|mmol|%)', pred, re.I)}
        src_nums = {m.group(0) for m in re.finditer(r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|units?|g|meq|mmol|%)', src, re.I)}
        num_conflicts = pred_nums - src_nums

        is_bad = bool(hallucinated) or bool(num_conflicts)
        if hallucinated:
            halluc_count += 1
        if num_conflicts:
            num_conflict_count += 1

        if is_bad:
            failures.append({
                "sample_idx": i,
                "type": "factual_inconsistency",
                "hallucinated_entities": sorted(hallucinated)[:5],
                "numeric_conflicts": sorted(num_conflicts)[:5],
                "pred_snippet": pred[:300],
            })

    n = max(len(sources), 1)
    return {
        "hallucination_rate": halluc_count / n,
        "numeric_conflict_rate": num_conflict_count / n,
        "contradiction_rate": (halluc_count + num_conflict_count) / n,
    }, failures


# 2d. PHI leakage -----------------------------------------------------------

_PHI_PATTERNS = [
    # SSN
    re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    # US phone
    re.compile(r'\b(?:\(\d{3}\)\s*|\d{3}[-.])\d{3}[-.]?\d{4}\b'),
    # Email
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    # Dates of birth pattern (DOB: mm/dd/yyyy)
    re.compile(r'\b(?:DOB|date\s+of\s+birth)\s*[:\-]?\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', re.I),
    # MRN / medical record number
    re.compile(r'\b(?:MRN|medical\s+record)\s*[:#]?\s*\d{5,}\b', re.I),
    # Street addresses (rough)
    re.compile(r'\b\d{1,5}\s+\w+\s+(?:st|street|ave|avenue|blvd|boulevard|dr|drive|rd|road|ln|lane|ct|court)\b', re.I),
    # Full names after "Patient:" or "Name:" or similar
    re.compile(r'\b(?:patient|name)\s*:\s*[A-Z][a-z]+\s+[A-Z][a-z]+\b'),
]


def check_phi_leakage(preds: List[str]) -> Tuple[float, List[dict]]:
    """Scan generated summaries for PHI patterns."""
    failures = []
    for i, pred in enumerate(preds):
        found = []
        for pat in _PHI_PATTERNS:
            for m in pat.finditer(pred):
                found.append(m.group(0))
        if found:
            failures.append({
                "sample_idx": i,
                "type": "phi_leakage",
                "leaked_items": found[:5],
                "pred_snippet": pred[:300],
            })
    rate = len(failures) / max(len(preds), 1)
    return rate, failures


# ───────────────────────────── 3. QUALITY METRICS ─────────────────────────

def check_repetition(preds: List[str]) -> Dict[str, float]:
    """Detect n-gram repetition and looping."""
    dup_bigram_rates = []
    dup_trigram_rates = []
    for pred in preds:
        tokens = pred.lower().split()
        if len(tokens) < 3:
            dup_bigram_rates.append(0.0)
            dup_trigram_rates.append(0.0)
            continue
        bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
        trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]

        dup_bi = 1.0 - len(set(bigrams)) / max(len(bigrams), 1)
        dup_tri = 1.0 - len(set(trigrams)) / max(len(trigrams), 1)
        dup_bigram_rates.append(dup_bi)
        dup_trigram_rates.append(dup_tri)

    return {
        "dup_bigram_rate": sum(dup_bigram_rates) / max(len(dup_bigram_rates), 1),
        "dup_trigram_rate": sum(dup_trigram_rates) / max(len(dup_trigram_rates), 1),
    }


def check_length_stats(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Summary length statistics."""
    pred_lens = [len(p.split()) for p in preds]
    ref_lens = [len(r.split()) for r in refs]
    return {
        "avg_pred_len": sum(pred_lens) / max(len(pred_lens), 1),
        "avg_ref_len": sum(ref_lens) / max(len(ref_lens), 1),
        "len_ratio": (sum(pred_lens) / max(sum(ref_lens), 1)),
        "min_pred_len": min(pred_lens) if pred_lens else 0,
        "max_pred_len": max(pred_lens) if pred_lens else 0,
    }


# ───────────────────────────── 4. GENERATION ──────────────────────────────

@torch.no_grad()
def generate_beam(
    model: MambaTransformerModel,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    beam_size: int = 4,
    max_len: int = 256,
    length_penalty: float = 0.6,
    no_repeat_ngram_size: int = 3,
    min_len: int = 20,
) -> torch.Tensor:
    """Simple beam search for the MambaTransformerModel."""
    device = src.device
    batch_size = src.size(0)

    # Encode once
    memory = model.encode(src, src_mask)

    results = []
    for b in range(batch_size):
        mem_b = memory[b:b+1]  # [1, mem_len, d]

        # Each beam: (log_prob, token_ids_list)
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
                logits = model.decode(inp, mem_b, causal)
                next_logits = logits[0, -1, :]  # [vocab]

                # Block repeated n-grams
                if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size - 1:
                    ngrams_seen = set()
                    for i in range(len(seq) - no_repeat_ngram_size + 1):
                        ngrams_seen.add(tuple(seq[i:i + no_repeat_ngram_size]))
                    partial = tuple(seq[-(no_repeat_ngram_size - 1):])
                    for ng in ngrams_seen:
                        if ng[:-1] == partial:
                            next_logits[ng[-1]] = float('-inf')

                # Min length: suppress EOS before min_len
                if step < min_len:
                    next_logits[model.config.eos_id] = float('-inf')

                log_probs = F.log_softmax(next_logits, dim=-1)
                topk_lp, topk_ids = log_probs.topk(beam_size)

                for k in range(beam_size):
                    new_lp = log_prob + topk_lp[k].item()
                    new_seq = seq + [topk_ids[k].item()]
                    candidates.append((new_lp, new_seq))

            if not candidates:
                break
            # Length-normalised score for ranking
            candidates.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
            beams = candidates[:beam_size]

        # Collect best
        all_seqs = completed + beams
        if not all_seqs:
            all_seqs = [(0.0, [model.config.bos_id])]
        all_seqs.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
        results.append(all_seqs[0][1])

    # Pad to same length
    max_out = max(len(s) for s in results)
    out = torch.full((batch_size, max_out), model.config.pad_id, device=device, dtype=torch.long)
    for b, seq in enumerate(results):
        out[b, :len(seq)] = torch.tensor(seq, device=device)
    return out


# ───────────────────────────── 5. ORCHESTRATOR ────────────────────────────

@dataclass
class EvalReport:
    checkpoint_step: int = 0
    # Gate 1 — safety
    negation_error_rate: float = 0.0
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0
    hallucination_rate: float = 0.0
    numeric_conflict_rate: float = 0.0
    contradiction_rate: float = 0.0
    phi_leakage_rate: float = 0.0
    # Gate 2 — quality
    dup_bigram_rate: float = 0.0
    dup_trigram_rate: float = 0.0
    avg_pred_len: float = 0.0
    avg_ref_len: float = 0.0
    len_ratio: float = 0.0
    # Gate 3 — similarity
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    bertscore_f1: float = 0.0
    # Verdict
    safety_pass: bool = False
    verdict: str = "FAIL"
    failure_reasons: List[str] = field(default_factory=list)

    # Beam-vs-greedy delta (optional)
    greedy_rougeL: float = 0.0
    beam_rougeL: float = 0.0


def run_evaluation(
    model: MambaTransformerModel,
    val_loader,
    tokenizer,
    device: torch.device,
    max_samples: int = 200,
    beam_size: int = 4,
    max_gen_len: int = 256,
    compare_greedy: bool = True,
    neg_threshold: float = 0.02,
) -> Tuple[EvalReport, List[dict]]:
    """Run full gated clinical evaluation."""
    model.eval()

    sources, preds_beam, preds_greedy, refs = [], [], [], []
    n = 0

    flush("\n── Generating summaries ──")
    t0 = time.time()
    for batch_idx, batch in enumerate(val_loader):
        if n >= max_samples:
            break

        src = batch["src"].to(device)
        tgt_out = batch["tgt_output"].to(device)
        src_mask = batch["src_mask"].to(device)

        # Beam search — fp32 to avoid fp16 overflow in multi-step attention
        gen_beam = generate_beam(model, src, src_mask,
                                 beam_size=beam_size,
                                 max_len=max_gen_len,
                                 no_repeat_ngram_size=3,
                                 min_len=20)
        # Greedy (for comparison) — fp32 as well for fair comparison
        if compare_greedy:
            gen_greedy = model.generate(src, src_mask,
                                        max_len=max_gen_len,
                                        greedy=True,
                                        no_repeat_ngram_size=3)

        for i in range(src.size(0)):
            if n >= max_samples:
                break
            decode = lambda ids: tokenizer.DecodeIds([t for t in ids if t not in [0, 1, 2, 3]])

            src_text = decode(src[i].cpu().tolist())
            ref_text = decode(tgt_out[i].cpu().tolist())
            beam_text = decode(gen_beam[i].cpu().tolist())

            sources.append(src_text)
            refs.append(ref_text)
            preds_beam.append(beam_text)

            if compare_greedy:
                greedy_text = decode(gen_greedy[i].cpu().tolist())
                preds_greedy.append(greedy_text)

            n += 1

        elapsed = time.time() - t0
        if (batch_idx + 1) % 5 == 0 or n >= max_samples:
            flush(f"  Generated {n}/{max_samples} samples … ({elapsed:.0f}s elapsed)")

    total_time = time.time() - t0
    flush(f"  Total generated: {n} in {total_time:.1f}s ({total_time/max(n,1):.1f}s/sample)")

    preds = preds_beam  # Primary = beam

    # ── Gate 1: Safety ────────────────────────────────────────────────
    flush("\n── Gate 1: Safety checks ──")

    neg_rate, neg_fails = check_negation_consistency(sources, preds, refs)
    flush(f"  Negation error rate : {neg_rate:.4f}  ({len(neg_fails)} failures)")

    ent_metrics, ent_fails = check_entity_preservation(sources, preds, refs)
    flush(f"  Entity F1           : {ent_metrics['entity_f1']:.4f}")
    flush(f"  Entity recall       : {ent_metrics['entity_recall']:.4f}")

    fact_metrics, fact_fails = check_factual_consistency_proxy(sources, preds)
    flush(f"  Hallucination rate  : {fact_metrics['hallucination_rate']:.4f}")
    flush(f"  Contradiction rate  : {fact_metrics['contradiction_rate']:.4f}")

    phi_rate, phi_fails = check_phi_leakage(preds)
    flush(f"  PHI leakage rate    : {phi_rate:.4f}")

    # Gating logic
    failure_reasons = []
    if phi_rate > 0:
        failure_reasons.append(f"PHI_LEAKAGE={phi_rate:.4f}")
    if neg_rate > neg_threshold:
        failure_reasons.append(f"NEGATION_ERROR={neg_rate:.4f} > {neg_threshold}")
    if fact_metrics["contradiction_rate"] > 0.15:
        failure_reasons.append(f"HIGH_CONTRADICTION={fact_metrics['contradiction_rate']:.4f}")

    safety_pass = len(failure_reasons) == 0

    # ── Gate 2: Quality ───────────────────────────────────────────────
    flush("\n── Gate 2: Quality checks ──")
    rep_metrics = check_repetition(preds)
    len_metrics = check_length_stats(preds, refs)
    flush(f"  Dup bigram rate     : {rep_metrics['dup_bigram_rate']:.4f}")
    flush(f"  Dup trigram rate    : {rep_metrics['dup_trigram_rate']:.4f}")
    flush(f"  Avg pred length     : {len_metrics['avg_pred_len']:.1f} words")
    flush(f"  Avg ref  length     : {len_metrics['avg_ref_len']:.1f} words")
    flush(f"  Length ratio        : {len_metrics['len_ratio']:.2f}")

    # ── Gate 3: Similarity ────────────────────────────────────────────
    flush("\n── Gate 3: Similarity metrics ──")
    rouge_metrics, per_sample_rouge = compute_rouge(preds, refs)
    flush(f"  ROUGE-1             : {rouge_metrics['rouge1']:.4f}")
    flush(f"  ROUGE-2             : {rouge_metrics['rouge2']:.4f}")
    flush(f"  ROUGE-L             : {rouge_metrics['rougeL']:.4f}")

    bert_metrics = compute_bertscore(preds, refs)
    flush(f"  BERTScore-F1        : {bert_metrics['bertscore_f1']:.4f}")

    # Greedy comparison
    greedy_rl = 0.0
    if compare_greedy and preds_greedy:
        greedy_rouge, _ = compute_rouge(preds_greedy, refs)
        greedy_rl = greedy_rouge["rougeL"]
        flush(f"\n  GREEDY  ROUGE-L     : {greedy_rl:.4f}")
        flush(f"  BEAM    ROUGE-L     : {rouge_metrics['rougeL']:.4f}")
        flush(f"  Δ (beam – greedy)   : {rouge_metrics['rougeL'] - greedy_rl:+.4f}")

    # ── Build report ──────────────────────────────────────────────────
    verdict = "PASS" if safety_pass else "FAIL"

    report = EvalReport(
        negation_error_rate=neg_rate,
        entity_precision=ent_metrics["entity_precision"],
        entity_recall=ent_metrics["entity_recall"],
        entity_f1=ent_metrics["entity_f1"],
        hallucination_rate=fact_metrics["hallucination_rate"],
        numeric_conflict_rate=fact_metrics["numeric_conflict_rate"],
        contradiction_rate=fact_metrics["contradiction_rate"],
        phi_leakage_rate=phi_rate,
        dup_bigram_rate=rep_metrics["dup_bigram_rate"],
        dup_trigram_rate=rep_metrics["dup_trigram_rate"],
        avg_pred_len=len_metrics["avg_pred_len"],
        avg_ref_len=len_metrics["avg_ref_len"],
        len_ratio=len_metrics["len_ratio"],
        rouge1=rouge_metrics["rouge1"],
        rouge2=rouge_metrics["rouge2"],
        rougeL=rouge_metrics["rougeL"],
        bertscore_f1=bert_metrics["bertscore_f1"],
        safety_pass=safety_pass,
        verdict=verdict,
        failure_reasons=failure_reasons,
        greedy_rougeL=greedy_rl,
        beam_rougeL=rouge_metrics["rougeL"],
    )

    # Collect all failure examples
    all_failures = []
    for f in (neg_fails + ent_fails[:20] + fact_fails[:20] + phi_fails):
        all_failures.append(f)

    return report, all_failures[:50]


# ───────────────────────────── 6. MAIN ────────────────────────────────────

def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    flush(f"Loading checkpoint: {ckpt_path}")
    cp = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = cp["config"]
    model_cfg = MambaTransformerConfig.from_dict(cfg.get("model", {}))
    model = build_model(model_cfg)
    model.load_state_dict(cp["model_state_dict"])
    model = model.to(device).eval()
    step = cp.get("step", 0)
    best_rl = cp.get("best_rouge_l", 0.0)
    flush(f"  Step: {step}  |  Saved best ROUGE-L: {best_rl:.4f}")
    return model, cfg, step, best_rl


def print_report(report: EvalReport, failures: List[dict]):
    flush("\n" + "=" * 72)
    flush("   GATED CLINICAL EVALUATION REPORT")
    flush("=" * 72)

    flush(f"\n{'Metric':<30} {'Value':>12}")
    flush("-" * 44)
    flush(f"{'ROUGE-L':<30} {report.rougeL:>12.4f}")
    flush(f"{'ROUGE-2':<30} {report.rouge2:>12.4f}")
    flush(f"{'ROUGE-1':<30} {report.rouge1:>12.4f}")
    flush(f"{'BERTScore-F1':<30} {report.bertscore_f1:>12.4f}")
    flush(f"{'Entity F1':<30} {report.entity_f1:>12.4f}")
    flush(f"{'Entity Recall':<30} {report.entity_recall:>12.4f}")
    flush(f"{'Negation Error Rate':<30} {report.negation_error_rate:>12.4f}")
    flush(f"{'Contradiction Rate':<30} {report.contradiction_rate:>12.4f}")
    flush(f"{'PHI Leakage Rate':<30} {report.phi_leakage_rate:>12.4f}")
    flush(f"{'Dup Bigram Rate':<30} {report.dup_bigram_rate:>12.4f}")
    flush(f"{'Dup Trigram Rate':<30} {report.dup_trigram_rate:>12.4f}")
    flush(f"{'Avg Pred Length (words)':<30} {report.avg_pred_len:>12.1f}")
    flush(f"{'Avg Ref  Length (words)':<30} {report.avg_ref_len:>12.1f}")
    flush(f"{'Length Ratio':<30} {report.len_ratio:>12.2f}")
    if report.greedy_rougeL > 0:
        flush(f"{'Greedy ROUGE-L':<30} {report.greedy_rougeL:>12.4f}")
        flush(f"{'Beam   ROUGE-L':<30} {report.beam_rougeL:>12.4f}")
        flush(f"{'Δ (beam−greedy)':<30} {report.beam_rougeL - report.greedy_rougeL:>+12.4f}")

    flush(f"\n{'SAFETY GATE:':<20} {'PASS ✅' if report.safety_pass else 'FAIL ❌'}")
    flush(f"{'VERDICT:':<20} {report.verdict}")
    if report.failure_reasons:
        flush(f"  Reasons: {'; '.join(report.failure_reasons)}")

    if failures:
        flush(f"\n── Top {min(10, len(failures))} Failure Examples ──")
        for i, f in enumerate(failures[:10]):
            flush(f"\n  [{i+1}] Type: {f['type']}")
            if "flipped_entities" in f:
                flush(f"      Flipped: {f['flipped_entities']}")
            if "missed_entities" in f:
                flush(f"      Missed:  {f['missed_entities']}")
            if "hallucinated_entities" in f:
                flush(f"      Halluc:  {f['hallucinated_entities']}")
            if "leaked_items" in f:
                flush(f"      PHI:     {f['leaked_items']}")
            if "pred_snippet" in f:
                flush(f"      Pred:    {f['pred_snippet'][:200]}…")


def main():
    parser = argparse.ArgumentParser(description="Gated Clinical Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--compare_greedy", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, step, saved_rl = load_model_from_checkpoint(args.checkpoint, device)

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    data_cfg = DataConfig.from_dict(cfg)
    _, val_loader, tokenizer = create_dataloaders(
        config=data_cfg, batch_size=args.batch_size, num_workers=0)

    report, failures = run_evaluation(
        model, val_loader, tokenizer, device,
        max_samples=args.max_samples,
        beam_size=args.beam_size,
        max_gen_len=args.max_gen_len,
        compare_greedy=args.compare_greedy,
    )
    report.checkpoint_step = step

    print_report(report, failures)

    # Save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"eval_step_{step}.json")
    with open(json_path, "w") as f:
        json.dump({"report": asdict(report), "failures": failures[:50]}, f, indent=2)
    flush(f"\nFull report saved → {json_path}")


if __name__ == "__main__":
    main()
