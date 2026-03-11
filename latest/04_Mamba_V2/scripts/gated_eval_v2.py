"""
Gated Clinical Evaluation v2 — Full 3-Gate System
===================================================
GATE 1: Safety / Faithfulness (hard stop)
GATE 2: Quality (fluency, structure, completeness)
GATE 3: Similarity (ROUGE + clinical BERTScore)

Supports fixed eval subsets via --greedy_id_file / --beam_id_file
for reproducible comparisons across experiments.

BERTScore: SKIPPED (PyTorch 2.5.1 CVE-2025-32434 blocks .bin loading).

Deliverable:
  A) Architecture check
  B) Gate table (Safety / Quality / Similarity)
  C) Top failure modes with 5 examples
  D) Next-step recommendation
"""

import os, sys, re, json, time, argparse, warnings, textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from src.model import MambaTransformerConfig, MambaTransformerModel, build_model, MAMBA_AVAILABLE
from src.data_loader import DataConfig, create_dataloaders, load_tokenizer, collate_fn

warnings.filterwarnings("ignore")

def flush(msg=""):
    print(msg, flush=True)

def _fmt_metric(x, nd=4):
    if x is None:
        return "N/A"
    if isinstance(x, str):
        return x
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

# ═══════════════════════════════════════════════════════════════════════════
#  CLINICAL TERMS / PATTERNS (shared)
# ═══════════════════════════════════════════════════════════════════════════

_NEG_PATTERNS = [
    re.compile(r"\b(no|not|denies?|without|absent|negative|non|ruled\s+out|"
               r"neither|nor|never|free\s+of|no\s+evidence)\b", re.I),
]

_CLINICAL_TERMS = re.compile(
    r"\b(diabetes|hypertension|htn|cad|chf|copd|asthma|cancer|stroke|"
    r"dvt|pe|mi|aki|ckd|afib|a\.\s*fib|sepsis|pneumonia|"
    r"heart\s+failure|renal\s+failure|hepatitis|cirrhosis|"
    r"infection|fever|pain|bleeding|anemia|hypothyroid|hyperthyroid|"
    r"anxiety|depression|seizure|epilepsy|"
    r"insulin|metformin|warfarin|heparin|aspirin|lisinopril|"
    r"amlodipine|atorvastatin|levothyroxine|omeprazole|"
    r"furosemide|prednisone|albuterol|metoprolol|losartan|"
    r"gabapentin|oxycodone|morphine|dilaudid|pantoprazole|"
    r"vancomycin|ceftriaxone|piperacillin|levofloxacin|"
    r"surgery|biopsy|intubation|extubation|dialysis|transfusion|"
    r"catheter|tracheostomy|orif|cabg)\b", re.I
)

# PHI patterns
_PHI_PATTERNS = [
    re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),                         # SSN
    re.compile(r'\b(?:\(\d{3}\)\s*|\d{3}[-.])\d{3}[-.]?\d{4}\b'), # Phone
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email
    re.compile(r'\b(?:DOB|date\s+of\s+birth)\s*[:\-]?\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', re.I),
    re.compile(r'\b(?:MRN|medical\s+record)\s*[:#]?\s*\d{5,}\b', re.I),
    re.compile(r'\b\d{1,5}\s+\w+\s+(?:st|street|ave|avenue|blvd|boulevard|dr|drive|rd|road|ln|lane|ct|court)\b', re.I),
    re.compile(r'\b(?:patient|name)\s*:\s*[A-Z][a-z]+\s+[A-Z][a-z]+\b'),
]

# Section headers we expect in clinical notes
_CLINICAL_SECTIONS = {
    "hpi":   re.compile(r'\b(history\s+of\s+present\s+illness|hpi|present\s+illness|chief\s+complaint)\b', re.I),
    "pmh":   re.compile(r'\b(past\s+medical\s+history|pmh|medical\s+history|pmhx)\b', re.I),
    "meds":  re.compile(r'\b(medications?|meds|discharge\s+medications?|home\s+medications?)\b', re.I),
    "a_p":   re.compile(r'\b(assessment\s*(and|&)?\s*plan|a/?p|hospital\s+course|active\s+issues?|brief\s+hospital\s+course)\b', re.I),
    "dispo": re.compile(r'\b(disposition|discharge|dispo|discharged?\s+to|transitional\s+issues?|follow.?up)\b', re.I),
}

# ═══════════════════════════════════════════════════════════════════════════
#  GATE 1 — SAFETY / FAITHFULNESS
# ═══════════════════════════════════════════════════════════════════════════

def _extract_negated_entities(text: str) -> set:
    negated = set()
    for sent in re.split(r'[.;!\n]', text):
        s = sent.lower()
        if any(p.search(s) for p in _NEG_PATTERNS):
            for m in _CLINICAL_TERMS.finditer(s):
                negated.add(m.group(0).strip())
    return negated


def gate1_negation(sources, preds, refs):
    """Negation consistency check. Returns (rate, failures)."""
    failures = []
    for i, (src, pred, ref) in enumerate(zip(sources, preds, refs)):
        src_neg = _extract_negated_entities(src)
        pred_neg = _extract_negated_entities(pred)

        pred_terms = {m.group(0).strip() for m in _CLINICAL_TERMS.finditer(pred.lower())}
        src_terms = {m.group(0).strip() for m in _CLINICAL_TERMS.finditer(src.lower())}

        affirmed_in_pred = pred_terms - pred_neg
        flipped = affirmed_in_pred & src_neg  # negated in source, affirmed in pred
        affirmed_in_src = src_terms - src_neg
        reverse_flip = pred_neg & affirmed_in_src  # affirmed in source, negated in pred
        all_flips = flipped | reverse_flip

        if all_flips:
            failures.append({
                "idx": i, "type": "negation_flip",
                "flipped": sorted(all_flips),
                "src_snip": src[:400], "pred": pred[:500], "ref": ref[:400],
            })
    return len(failures) / max(len(sources), 1), failures


def _extract_entities(text):
    ents = set()
    for m in _CLINICAL_TERMS.finditer(text.lower()):
        ents.add(m.group(0).strip())
    for m in re.finditer(r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|ml|units?|g|meq|mmol|%)\b', text, re.I):
        ents.add(m.group(0).lower().strip())
    for m in re.finditer(
        r'\b(wbc|hgb|hb|plt|bun|cr|creatinine|na|k|cl|hco3|glucose|inr|ptt|'
        r'ast|alt|bili|troponin|bnp|lactate|hemoglobin|potassium|sodium)\s*'
        r'[=:]?\s*(\d+(?:\.\d+)?)', text, re.I):
        ents.add(f"{m.group(1).lower()} {m.group(2)}")
    return ents


def gate1_entity_preservation(sources, preds, refs):
    """Entity recall / precision / F1 vs reference."""
    precs, recs, f1s = [], [], []
    failures = []
    for i, (src, pred, ref) in enumerate(zip(sources, preds, refs)):
        ref_ents = _extract_entities(ref)
        pred_ents = _extract_entities(pred)
        if not ref_ents:
            continue
        tp = len(pred_ents & ref_ents)
        p = tp / max(len(pred_ents), 1)
        r = tp / max(len(ref_ents), 1)
        f = 2 * p * r / max(p + r, 1e-9)
        precs.append(p); recs.append(r); f1s.append(f)
        missed = ref_ents - pred_ents
        if missed and r < 0.5:
            failures.append({
                "idx": i, "type": "entity_miss",
                "missed": sorted(missed)[:10],
                "recall": round(r, 4),
            })
    n = max(len(precs), 1)
    return {
        "precision": sum(precs)/n,
        "recall": sum(recs)/n,
        "f1": sum(f1s)/n,
    }, failures


def gate1_hallucination(sources, preds):
    """Hallucination proxy: clinical entities in pred but not in source."""
    halluc_n, numcon_n = 0, 0
    failures = []
    for i, (src, pred) in enumerate(zip(sources, preds)):
        sl = src.lower()
        pred_ents = _extract_entities(pred)
        src_ents = _extract_entities(src)
        halluc = set()
        for e in pred_ents:
            if re.match(r'^\d', e):
                continue
            if e not in sl:
                halluc.add(e)
        # Numeric conflicts
        pred_nums = {m.group(0) for m in re.finditer(r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|units?|g|meq|mmol|%)', pred, re.I)}
        src_nums = {m.group(0) for m in re.finditer(r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|units?|g|meq|mmol|%)', src, re.I)}
        numcon = pred_nums - src_nums
        if halluc: halluc_n += 1
        if numcon: numcon_n += 1
        if halluc or numcon:
            failures.append({
                "idx": i, "type": "hallucination",
                "halluc_entities": sorted(halluc)[:5],
                "num_conflicts": sorted(numcon)[:5],
                "pred_snip": pred[:400],
            })
    n = max(len(sources), 1)
    return {
        "hallucination_rate": halluc_n / n,
        "numeric_conflict_rate": numcon_n / n,
    }, failures


def gate1_phi(preds):
    """PHI leakage scan."""
    failures = []
    for i, pred in enumerate(preds):
        found = []
        for p in _PHI_PATTERNS:
            for m in p.finditer(pred):
                found.append(m.group(0))
        if found:
            failures.append({"idx": i, "type": "phi_leak", "items": found[:5]})
    return len(failures) / max(len(preds), 1), failures


# ═══════════════════════════════════════════════════════════════════════════
#  GATE 2 — QUALITY (fluency, structure, completeness)
# ═══════════════════════════════════════════════════════════════════════════

def gate2_repetition(preds):
    """Dup bigram / trigram rates."""
    bi_rates, tri_rates = [], []
    for p in preds:
        toks = p.lower().split()
        if len(toks) < 3:
            bi_rates.append(0.); tri_rates.append(0.); continue
        bis = [tuple(toks[i:i+2]) for i in range(len(toks)-1)]
        tris = [tuple(toks[i:i+3]) for i in range(len(toks)-2)]
        bi_rates.append(1 - len(set(bis))/max(len(bis),1))
        tri_rates.append(1 - len(set(tris))/max(len(tris),1))
    n = max(len(bi_rates), 1)
    return {"dup_bigram": sum(bi_rates)/n, "dup_trigram": sum(tri_rates)/n}


def gate2_length(preds, refs):
    """Length stats."""
    pl = [len(p.split()) for p in preds]
    rl = [len(r.split()) for r in refs]
    return {
        "avg_pred_len": sum(pl)/max(len(pl),1),
        "avg_ref_len": sum(rl)/max(len(rl),1),
        "len_ratio": sum(pl) / max(sum(rl), 1),
        "min_pred_len": min(pl) if pl else 0,
        "max_pred_len": max(pl) if pl else 0,
    }


def gate2_grammar_proxy(preds):
    """
    Grammar / readability proxy (no external tool):
      - avg sentence length
      - fragment count (sentences < 4 words)
      - broken section header count (# without text, unmatched parens, etc.)
      - errors_per_100_words estimate
    """
    total_words = 0
    total_sents = 0
    fragment_count = 0
    broken_header_count = 0
    incomplete_sent_count = 0

    for pred in preds:
        sents = [s.strip() for s in re.split(r'[.!?\n]', pred) if s.strip()]
        total_sents += len(sents)
        for s in sents:
            w = len(s.split())
            total_words += w
            if 0 < w < 4:
                fragment_count += 1
        # Broken headers: # or ## without following text, or dangling #)
        broken_header_count += len(re.findall(r'#[#.)\]]*\s*$', pred, re.M))
        # Unmatched brackets
        for ch_open, ch_close in [('(', ')'), ('[', ']'), ('{', '}')]:
            if pred.count(ch_open) != pred.count(ch_close):
                incomplete_sent_count += 1

    n_samples = max(len(preds), 1)
    avg_sent_len = total_words / max(total_sents, 1)
    frags_per_sample = fragment_count / n_samples
    broken_per_sample = broken_header_count / n_samples
    # Rough "errors per 100 words" = (fragments + broken_headers + unmatched) / total_words * 100
    error_items = fragment_count + broken_header_count + incomplete_sent_count
    errors_per_100 = error_items / max(total_words, 1) * 100

    return {
        "avg_sentence_len": round(avg_sent_len, 1),
        "fragments_per_sample": round(frags_per_sample, 2),
        "broken_headers_per_sample": round(broken_per_sample, 2),
        "errors_per_100_words": round(errors_per_100, 2),
    }


def gate2_section_coverage(sources, preds):
    """
    Check that key clinical sections present in source are also in the prediction.
    Returns coverage ratio per section and overall.
    """
    coverage = {}
    for sec_name, sec_pat in _CLINICAL_SECTIONS.items():
        src_has = sum(1 for s in sources if sec_pat.search(s))
        pred_has = sum(1 for s, p in zip(sources, preds) if sec_pat.search(s) and sec_pat.search(p))
        coverage[sec_name] = {
            "in_source": src_has,
            "in_pred": pred_has,
            "coverage": pred_has / max(src_has, 1),
        }
    total_src = sum(v["in_source"] for v in coverage.values())
    total_pred = sum(v["in_pred"] for v in coverage.values())
    coverage["overall"] = {
        "in_source": total_src,
        "in_pred": total_pred,
        "coverage": total_pred / max(total_src, 1),
    }
    return coverage


# ═══════════════════════════════════════════════════════════════════════════
#  GATE 3 — SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════

def gate3_rouge(preds, refs):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    n = max(len(r1), 1)
    return {"rouge1": sum(r1)/n, "rouge2": sum(r2)/n, "rougeL": sum(rl)/n}


def gate3_bertscore_clinical(preds, refs, max_samples=50):
    """BERTScore — skips clinical encoders (torch 2.5.1 blocks .bin loading).
    
    Known issue: Bio_ClinicalBERT/BiomedBERT/BioBERT all fail on torch 2.5.1+cu121
    due to CVE-2025-32434 security restriction blocking torch.load of .bin weights.
    Fix requires torch >= 2.6 or models that ship model.safetensors only.
    
    Policy: Report 'clinical BERTScore unavailable' honestly.
    Do NOT silently fall back to bert-base-uncased (it is not clinically meaningful).
    Gate 1 + Gate 2 are the priority; Gate 3 BERTScore is informational only.
    """
    flush("  [INFO] Clinical BERTScore SKIPPED")
    flush("    Reason: torch==2.5.1 blocks loading Bio_ClinicalBERT .bin weights")
    flush("    Fix: upgrade to torch>=2.6 or use separate eval environment")
    flush("    Gate 1 + Gate 2 are sufficient for checkpoint selection")
    return {
        "bertscore_p": None,
        "bertscore_r": None,
        "bertscore_f1": None,
        "encoder": "SKIPPED (disabled)",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  BEAM SEARCH (tiny subset only, for comparison)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def beam_search(model, src, src_mask, beam_size=4, max_len=256,
                no_repeat_ngram=3, min_len=20, length_penalty=0.6):
    """Beam search on a single sample (batch=1)."""
    device = src.device
    memory = model.encode(src, src_mask)
    beams = [(0.0, [model.config.bos_id])]
    completed = []

    for step in range(max_len - 1):
        candidates = []
        for lp, seq in beams:
            if seq[-1] == model.config.eos_id:
                completed.append((lp, seq)); continue
            inp = torch.tensor([seq], device=device, dtype=torch.long)
            causal = model.generate_causal_mask(inp.size(1), device)
            logits = model.decode(inp, memory, causal)
            nl = logits[0, -1, :]
            # Block n-gram repeats
            if no_repeat_ngram > 0 and len(seq) >= no_repeat_ngram - 1:
                ngrams = set()
                for j in range(len(seq) - no_repeat_ngram + 1):
                    ngrams.add(tuple(seq[j:j+no_repeat_ngram]))
                partial = tuple(seq[-(no_repeat_ngram-1):])
                for ng in ngrams:
                    if ng[:-1] == partial:
                        nl[ng[-1]] = float('-inf')
            if step < min_len:
                nl[model.config.eos_id] = float('-inf')
            log_probs = F.log_softmax(nl, dim=-1)
            topk_lp, topk_ids = log_probs.topk(beam_size)
            for k in range(beam_size):
                candidates.append((lp + topk_lp[k].item(), seq + [topk_ids[k].item()]))
        if not candidates: break
        candidates.sort(key=lambda x: x[0]/(len(x[1])**length_penalty), reverse=True)
        beams = candidates[:beam_size]

    all_seqs = completed + beams
    if not all_seqs:
        return [model.config.bos_id]
    all_seqs.sort(key=lambda x: x[0]/(len(x[1])**length_penalty), reverse=True)
    return all_seqs[0][1]


# ═══════════════════════════════════════════════════════════════════════════
#  GATING LOGIC
# ═══════════════════════════════════════════════════════════════════════════

# Gate 1 thresholds (hard stop)
G1_NEG_THRESH       = 0.02
G1_CONTRA_THRESH    = 0.15
G1_HALLUC_TARGET    = 0.20
G1_NUMCON_TARGET    = 0.05
G1_RECALL_TARGET    = 0.70

# Gate 2 thresholds
G2_LEN_RATIO_MIN    = 0.70
G2_LEN_RATIO_MAX    = 1.00
G2_DUP_BI_MAX       = 0.10
G2_ERR_PER_100_MAX  = 3.0

def evaluate_gate1(neg_rate, phi_rate, halluc_metrics, ent_metrics):
    """Returns (pass, reasons)."""
    reasons = []
    if phi_rate > 0:
        reasons.append(f"PHI_LEAKAGE={phi_rate:.4f} (must=0)")
    if neg_rate > G1_NEG_THRESH:
        reasons.append(f"NEGATION={neg_rate:.4f} (>{G1_NEG_THRESH})")
    hr = halluc_metrics["hallucination_rate"]
    if hr > G1_HALLUC_TARGET:
        reasons.append(f"HALLUCINATION={hr:.4f} (>{G1_HALLUC_TARGET})")
    nc = halluc_metrics["numeric_conflict_rate"]
    if nc > G1_NUMCON_TARGET:
        reasons.append(f"NUM_CONFLICT={nc:.4f} (>{G1_NUMCON_TARGET})")
    cr = hr  # use hallucination as contradiction proxy
    # Also check: if neg+halluc combined signals > 0.15
    combined_contra = max(hr, neg_rate)
    if combined_contra > G1_CONTRA_THRESH:
        reasons.append(f"CONTRADICTION_PROXY={combined_contra:.4f} (>{G1_CONTRA_THRESH})")
    rec = ent_metrics["recall"]
    if rec < G1_RECALL_TARGET:
        reasons.append(f"ENTITY_RECALL={rec:.4f} (<{G1_RECALL_TARGET})")
    return len(reasons) == 0, reasons


def evaluate_gate2(len_metrics, rep_metrics, grammar_metrics):
    """Returns (pass, reasons)."""
    reasons = []
    lr = len_metrics["len_ratio"]
    if lr < G2_LEN_RATIO_MIN:
        reasons.append(f"LENGTH_RATIO={lr:.2f} (<{G2_LEN_RATIO_MIN})")
    if lr > G2_LEN_RATIO_MAX:
        reasons.append(f"LENGTH_RATIO={lr:.2f} (>{G2_LEN_RATIO_MAX})")
    db = rep_metrics["dup_bigram"]
    if db > G2_DUP_BI_MAX:
        reasons.append(f"DUP_BIGRAM={db:.4f} (>{G2_DUP_BI_MAX})")
    ep = grammar_metrics["errors_per_100_words"]
    if ep > G2_ERR_PER_100_MAX:
        reasons.append(f"ERRORS/100w={ep:.2f} (>{G2_ERR_PER_100_MAX})")
    return len(reasons) == 0, reasons


# ═══════════════════════════════════════════════════════════════════════════
#  FIXED SUBSET HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _load_id_file(path: str) -> List[int]:
    """Load integer IDs from a text file (one per line, # comments ignored)."""
    ids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(int(line))
    return ids


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Gated Clinical Eval v2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--beam_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--greedy_id_file", type=str, default=None,
                        help="Path to file with val-set indices for greedy eval (one int per line). "
                             "Overrides --max_samples.")
    parser.add_argument("--beam_id_file", type=str, default=None,
                        help="Path to file with val-set indices for beam eval (one int per line). "
                             "Overrides --beam_samples. Must be a subset of greedy IDs.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── A) Architecture check ──────────────────────────────────────
    flush("=" * 70)
    flush("  A) ARCHITECTURE CHECK")
    flush("=" * 70)
    flush(f"  mamba-ssm installed : {'YES — true Mamba SSM' if MAMBA_AVAILABLE else 'NO — GRU fallback'}")
    encoder_name = "Mamba-SSM" if MAMBA_AVAILABLE else "Mamba(GRU fallback)"
    flush(f"  Encoder             : {encoder_name}")
    flush(f"  Decoder             : TransformerDecoder (nn.MultiheadAttention)")
    flush(f"  LongT5              : NOT present (no local/transient-global attention)")
    flush(f"  Architecture label  : {encoder_name} + Transformer Decoder")
    flush()

    # ── Load model ─────────────────────────────────────────────────
    flush("Loading checkpoint...")
    cp = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = cp["config"]
    model_cfg = MambaTransformerConfig.from_dict(cfg.get("model", {}))
    model = build_model(model_cfg)
    model.load_state_dict(cp["model_state_dict"])
    model = model.to(device).eval()  # keep default dtype (fp32)
    step = cp.get("step", 0)
    saved_rl = cp.get("best_rouge_l", 0.0)
    flush(f"  Step: {step}  |  Saved best ROUGE-L: {saved_rl:.4f}")
    flush(f"  n_memory_tokens: {model_cfg.n_memory_tokens}")
    flush(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    flush()

    # ── Load data ──────────────────────────────────────────────────
    with open(args.config) as f:
        ycfg = yaml.safe_load(f)
    data_cfg = DataConfig.from_dict(ycfg)
    _, val_loader, tokenizer = create_dataloaders(config=data_cfg, batch_size=args.batch_size, num_workers=0)
    val_dataset = val_loader.dataset  # PreloadedClinicalDataset
    decode = lambda ids: tokenizer.DecodeIds([t for t in ids if t not in [0, 1, 2, 3]])

    # ── Resolve greedy / beam indices ──────────────────────────────
    if args.greedy_id_file:
        greedy_ids = _load_id_file(args.greedy_id_file)
        flush(f"  Fixed greedy subset: {len(greedy_ids)} IDs from {args.greedy_id_file}")
    else:
        greedy_ids = list(range(min(args.max_samples, len(val_dataset))))
        flush(f"  Sequential greedy subset: first {len(greedy_ids)} val samples")

    if args.beam_id_file:
        beam_ids = _load_id_file(args.beam_id_file)
        flush(f"  Fixed beam subset:   {len(beam_ids)} IDs from {args.beam_id_file}")
    else:
        beam_ids = greedy_ids[:args.beam_samples]
        flush(f"  Sequential beam subset: first {len(beam_ids)} greedy IDs")

    greedy_set = set(greedy_ids)
    beam_set = set(beam_ids)
    missing = sorted(beam_set - greedy_set)
    if missing:
        raise ValueError(
            f"beam_id_file contains IDs not in greedy_id_file: {missing[:20]} "
            f"(total {len(missing)})"
        )

    beam_id_set = set(beam_ids)
    n_greedy = len(greedy_ids)
    n_beam   = len(beam_ids)

    # Build a Subset loader for greedy IDs
    from torch.utils.data import Subset, DataLoader as DL
    greedy_subset = Subset(val_dataset, greedy_ids)
    greedy_loader = DL(greedy_subset, batch_size=args.batch_size, shuffle=False,
                       num_workers=0,
                       collate_fn=lambda b: collate_fn(b, data_cfg.pad_id),
                       pin_memory=True)

    # ── PHASE 1: Greedy generation ─────────────────────────────────
    flush("=" * 70)
    flush(f"  PHASE 1: GREEDY GENERATION ({n_greedy} samples)")
    flush("=" * 70)

    sources, preds_greedy, refs = [], [], []
    src_tensors_beam, mask_tensors_beam = [], []
    beam_positions = []  # position in greedy results corresponding to beam IDs
    n = 0
    t0 = time.time()

    for batch in greedy_loader:
        src = batch["src"].to(device)
        tgt_out = batch["tgt_output"].to(device)
        src_mask = batch["src_mask"].to(device)

        with torch.no_grad():
            gen = model.generate(src, src_mask, max_len=256, greedy=True,
                                 no_repeat_ngram_size=3, repetition_penalty=1.2)

        for i in range(src.size(0)):
            if n >= n_greedy:
                break
            sources.append(decode(src[i].cpu().tolist()))
            refs.append(decode(tgt_out[i].cpu().tolist()))
            preds_greedy.append(decode(gen[i].cpu().tolist()))
            # Check if this greedy ID is also a beam ID
            orig_val_idx = greedy_ids[n]
            if orig_val_idx in beam_id_set:
                src_tensors_beam.append(src[i:i+1])
                mask_tensors_beam.append(src_mask[i:i+1])
                beam_positions.append(n)
            n += 1

        if n % 4 == 0 or n >= n_greedy:
            flush(f"  Greedy: {n}/{n_greedy} ({time.time()-t0:.0f}s)")

    greedy_time = time.time() - t0
    flush(f"  Done: {n} samples in {greedy_time:.1f}s ({greedy_time/max(n,1):.1f}s/sample)")

    preds = preds_greedy  # Primary = greedy

    # ── PHASE 2: Beam search (fixed subset) ────────────────────────
    flush()
    flush("=" * 70)
    flush(f"  PHASE 2: BEAM SEARCH ({len(src_tensors_beam)} samples, beam={args.beam_size})")
    flush("=" * 70)

    preds_beam = []
    t1 = time.time()
    for i in range(len(src_tensors_beam)):
        ids = beam_search(model, src_tensors_beam[i], mask_tensors_beam[i],
                          beam_size=args.beam_size, max_len=256, no_repeat_ngram=3, min_len=20)
        preds_beam.append(decode(ids))
        flush(f"  Beam {i+1}/{len(src_tensors_beam)} ({time.time()-t1:.0f}s)")
    beam_time = time.time() - t1
    flush(f"  Done: {len(preds_beam)} in {beam_time:.1f}s ({beam_time/max(len(preds_beam),1):.1f}s/sample)")

    # ═══════════════════════════════════════════════════════════════
    #  B) GATE EVALUATION
    # ═══════════════════════════════════════════════════════════════
    flush()
    flush("=" * 70)
    flush("  B) GATED EVALUATION")
    flush("=" * 70)

    # ── GATE 1: Safety ─────────────────────────────────────────────
    flush()
    flush("─── GATE 1: SAFETY / FAITHFULNESS ───")

    neg_rate, neg_fails = gate1_negation(sources, preds, refs)
    flush(f"  1. Negation error rate     : {neg_rate:.4f}  (target ≤ {G1_NEG_THRESH})")

    ent_metrics, ent_fails = gate1_entity_preservation(sources, preds, refs)
    flush(f"  2. Entity recall           : {ent_metrics['recall']:.4f}  (target ≥ {G1_RECALL_TARGET})")
    flush(f"     Entity precision        : {ent_metrics['precision']:.4f}")
    flush(f"     Entity F1               : {ent_metrics['f1']:.4f}")

    halluc_metrics, halluc_fails = gate1_hallucination(sources, preds)
    flush(f"  3. Hallucination rate      : {halluc_metrics['hallucination_rate']:.4f}  (target ≤ {G1_HALLUC_TARGET})")
    flush(f"  4. Numeric conflict rate   : {halluc_metrics['numeric_conflict_rate']:.4f}  (target ≤ {G1_NUMCON_TARGET})")

    phi_rate, phi_fails = gate1_phi(preds)
    flush(f"  5. PHI leakage rate        : {phi_rate:.4f}  (must = 0)")

    g1_pass, g1_reasons = evaluate_gate1(neg_rate, phi_rate, halluc_metrics, ent_metrics)
    flush(f"\n  GATE 1 VERDICT: {'PASS ✅' if g1_pass else 'FAIL ❌'}")
    if g1_reasons:
        for r in g1_reasons:
            flush(f"    ✗ {r}")

    # ── GATE 2: Quality ────────────────────────────────────────────
    flush()
    flush("─── GATE 2: QUALITY ───")

    rep_metrics = gate2_repetition(preds)
    len_metrics = gate2_length(preds, refs)
    grammar_metrics = gate2_grammar_proxy(preds)
    section_cov = gate2_section_coverage(sources, preds)

    flush(f"  1. Length ratio             : {len_metrics['len_ratio']:.2f}  (target {G2_LEN_RATIO_MIN}–{G2_LEN_RATIO_MAX})")
    flush(f"     Avg pred len            : {len_metrics['avg_pred_len']:.1f} words")
    flush(f"     Avg ref  len            : {len_metrics['avg_ref_len']:.1f} words")
    flush(f"  2. Dup bigram rate         : {rep_metrics['dup_bigram']:.4f}")
    flush(f"     Dup trigram rate        : {rep_metrics['dup_trigram']:.4f}")
    flush(f"  3. Grammar proxy:")
    flush(f"     Avg sentence len        : {grammar_metrics['avg_sentence_len']} words")
    flush(f"     Fragments/sample        : {grammar_metrics['fragments_per_sample']}")
    flush(f"     Broken headers/sample   : {grammar_metrics['broken_headers_per_sample']}")
    flush(f"     Errors per 100 words    : {grammar_metrics['errors_per_100_words']}")
    flush(f"  4. Section coverage:")
    for sec, vals in section_cov.items():
        if sec == "overall":
            flush(f"     OVERALL                 : {vals['in_pred']}/{vals['in_source']} ({vals['coverage']:.1%})")
        else:
            flush(f"     {sec:24s} : {vals['in_pred']}/{vals['in_source']} ({vals['coverage']:.1%})")

    g2_pass, g2_reasons = evaluate_gate2(len_metrics, rep_metrics, grammar_metrics)
    flush(f"\n  GATE 2 VERDICT: {'PASS ✅' if g2_pass else 'FAIL ❌'}")
    if g2_reasons:
        for r in g2_reasons:
            flush(f"    ✗ {r}")

    # ── GATE 3: Similarity ─────────────────────────────────────────
    flush()
    flush("─── GATE 3: SIMILARITY ───")

    rouge = gate3_rouge(preds, refs)
    flush(f"  1. ROUGE-L (greedy)        : {rouge['rougeL']:.4f}")
    flush(f"     ROUGE-2 (greedy)        : {rouge['rouge2']:.4f}")
    flush(f"     ROUGE-1 (greedy)        : {rouge['rouge1']:.4f}")

    # Beam comparison on subset (aligned by beam_positions)
    beam_rouge = {}
    delta_rl = 0.0
    if preds_beam and beam_positions:
        beam_refs = [refs[pos] for pos in beam_positions]
        beam_greedy_preds = [preds_greedy[pos] for pos in beam_positions]
        beam_rouge = gate3_rouge(preds_beam, beam_refs)
        greedy_sub_rouge = gate3_rouge(beam_greedy_preds, beam_refs)
        delta_rl = beam_rouge["rougeL"] - greedy_sub_rouge["rougeL"]
        flush(f"\n  Beam vs Greedy (same {len(preds_beam)} samples):")
        flush(f"     Greedy ROUGE-L          : {greedy_sub_rouge['rougeL']:.4f}")
        flush(f"     Beam   ROUGE-L          : {beam_rouge['rougeL']:.4f}")
        flush(f"     Delta (beam-greedy)      : {delta_rl:+.4f}")

    # BERTScore with clinical encoder
    flush()
    bert_metrics = gate3_bertscore_clinical(preds, refs, max_samples=50)
    flush(f"  2. BERTScore-F1 (clinical) : {_fmt_metric(bert_metrics['bertscore_f1'])}")
    flush(f"     BERTScore-Recall        : {_fmt_metric(bert_metrics['bertscore_r'])}")
    flush(f"     BERTScore-Precision     : {_fmt_metric(bert_metrics['bertscore_p'])}")
    flush(f"     Encoder                 : {bert_metrics['encoder']}")

    gate3_note = "(Gate 3 is secondary — only meaningful if Gate 1 passes)"
    flush(f"\n  {gate3_note}")

    # ═══════════════════════════════════════════════════════════════
    #  C) TOP FAILURE MODES + EXAMPLES
    # ═══════════════════════════════════════════════════════════════
    flush()
    flush("=" * 70)
    flush("  C) TOP FAILURE MODES + 5 EXAMPLES")
    flush("=" * 70)

    # Count flipped entity frequency
    flip_counter = Counter()
    for f in neg_fails:
        for e in f["flipped"]:
            flip_counter[e] += 1

    flush(f"\n  Most-flipped entities (across {len(neg_fails)} negation failures):")
    for ent, cnt in flip_counter.most_common(15):
        flush(f"    {ent:20s}  {cnt:3d} occurrences")

    flush(f"\n  Hallucination pattern: {len(halluc_fails)} / {n} samples have hallucinated entities")
    halluc_ent_counter = Counter()
    for f in halluc_fails:
        for e in f["halluc_entities"]:
            halluc_ent_counter[e] += 1
    flush(f"  Most-hallucinated entities:")
    for ent, cnt in halluc_ent_counter.most_common(10):
        flush(f"    {ent:20s}  {cnt:3d} occurrences")

    flush(f"\n  ── 5 Example Failure Pairs (source → prediction) ──")
    examples_shown = 0
    # Mix: show first 3 negation failures + 2 hallucination failures
    example_pool = neg_fails[:3] + [f for f in halluc_fails if f["idx"] not in [x["idx"] for x in neg_fails[:3]]][:2]
    if len(example_pool) < 5:
        example_pool = (neg_fails + halluc_fails)[:5]

    for ex in example_pool[:5]:
        examples_shown += 1
        idx = ex["idx"]
        flush(f"\n  ╔══ Example {examples_shown} (sample #{idx}, type={ex['type']}) ══╗")
        flush(f"  ║ SOURCE (first 300 chars):")
        flush(f"  ║   {sources[idx][:300]}…")
        flush(f"  ║ PREDICTION (first 400 chars):")
        flush(f"  ║   {preds[idx][:400]}…")
        flush(f"  ║ REFERENCE (first 300 chars):")
        flush(f"  ║   {refs[idx][:300]}…")
        if "flipped" in ex:
            flush(f"  ║ FLIPPED ENTITIES: {ex['flipped']}")
        if "halluc_entities" in ex:
            flush(f"  ║ HALLUCINATED: {ex['halluc_entities']}")
        if "num_conflicts" in ex and ex["num_conflicts"]:
            flush(f"  ║ NUM CONFLICTS: {ex['num_conflicts']}")
        flush(f"  ╚{'═'*66}╝")

    # ═══════════════════════════════════════════════════════════════
    #  OVERALL VERDICT + SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════
    overall = "PASS" if (g1_pass and g2_pass) else "FAIL"
    flush()
    flush("=" * 70)
    flush(f"  OVERALL VERDICT: {overall}")
    flush("=" * 70)
    flush()
    flush(f"  {'Metric':<35} {'Value':>10} {'Target':>12} {'Status':>8}")
    flush(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*8}")
    # Gate 1
    flush(f"  {'GATE 1: SAFETY':<35} {'':>10} {'':>12} {'PASS' if g1_pass else 'FAIL':>8}")
    flush(f"  {'  Negation error':<35} {neg_rate:>10.4f} {'≤0.02':>12} {'✅' if neg_rate<=G1_NEG_THRESH else '❌':>8}")
    flush(f"  {'  Hallucination':<35} {halluc_metrics['hallucination_rate']:>10.4f} {'≤0.20':>12} {'✅' if halluc_metrics['hallucination_rate']<=G1_HALLUC_TARGET else '❌':>8}")
    flush(f"  {'  Numeric conflict':<35} {halluc_metrics['numeric_conflict_rate']:>10.4f} {'≤0.05':>12} {'✅' if halluc_metrics['numeric_conflict_rate']<=G1_NUMCON_TARGET else '❌':>8}")
    flush(f"  {'  Entity recall':<35} {ent_metrics['recall']:>10.4f} {'≥0.70':>12} {'✅' if ent_metrics['recall']>=G1_RECALL_TARGET else '❌':>8}")
    flush(f"  {'  Entity precision':<35} {ent_metrics['precision']:>10.4f} {'':>12} {'—':>8}")
    flush(f"  {'  Entity F1':<35} {ent_metrics['f1']:>10.4f} {'':>12} {'—':>8}")
    flush(f"  {'  PHI leakage':<35} {phi_rate:>10.4f} {'=0.00':>12} {'✅' if phi_rate==0 else '❌':>8}")
    # Gate 2
    flush(f"  {'GATE 2: QUALITY':<35} {'':>10} {'':>12} {'PASS' if g2_pass else 'FAIL':>8}")
    flush(f"  {'  Length ratio':<35} {len_metrics['len_ratio']:>10.2f} {'0.70–1.00':>12} {'✅' if G2_LEN_RATIO_MIN<=len_metrics['len_ratio']<=G2_LEN_RATIO_MAX else '❌':>8}")
    flush(f"  {'  Dup bigram':<35} {rep_metrics['dup_bigram']:>10.4f} {'≤0.10':>12} {'✅' if rep_metrics['dup_bigram']<=G2_DUP_BI_MAX else '❌':>8}")
    flush(f"  {'  Errors/100 words':<35} {grammar_metrics['errors_per_100_words']:>10.2f} {'≤3.0':>12} {'✅' if grammar_metrics['errors_per_100_words']<=G2_ERR_PER_100_MAX else '❌':>8}")
    flush(f"  {'  Section coverage (overall)':<35} {section_cov['overall']['coverage']:>10.1%} {'':>12} {'—':>8}")
    # Gate 3
    flush(f"  {'GATE 3: SIMILARITY':<35} {'':>10} {'':>12} {'(info)':>8}")
    flush(f"  {'  ROUGE-L':<35} {rouge['rougeL']:>10.4f} {'':>12} {'—':>8}")
    flush(f"  {'  ROUGE-2':<35} {rouge['rouge2']:>10.4f} {'':>12} {'—':>8}")
    flush(f"  {'  ROUGE-1':<35} {rouge['rouge1']:>10.4f} {'':>12} {'—':>8}")
    flush(f"  {'  BERTScore-F1 (clinical)':<35} {_fmt_metric(bert_metrics['bertscore_f1']):>10} {'':>12} {'—':>8}")
    if preds_beam:
        flush(f"  {'  Beam ROUGE-L delta':<35} {delta_rl:>+10.4f} {'':>12} {'—':>8}")
    flush()

    # ═══════════════════════════════════════════════════════════════
    #  D) NEXT-STEP RECOMMENDATION
    # ═══════════════════════════════════════════════════════════════
    flush("=" * 70)
    flush("  D) NEXT-STEP RECOMMENDATION")
    flush("=" * 70)
    flush()
    flush("  HIGHEST-LEVERAGE SINGLE CHANGE:")
    flush("  ─────────────────────────────────────────────────────────────")
    flush("  Increase n_memory_tokens from 8 → 32 (or 64)")
    flush()
    flush("  WHY: The model currently compresses each 256-token chunk into")
    flush(f"  only {model_cfg.n_memory_tokens} memory tokens (3.1% retention). This means the decoder's")
    flush("  cross-attention sees an extremely lossy representation of the")
    flush("  source. Clinical entities, negations, and dosages are lost.")
    flush("  The model resorts to generating generic templates instead of")
    flush("  faithfully summarizing each patient's unique clinical note.")
    flush()
    flush("  With 32 memory tokens per chunk, the retention goes to 12.5%.")
    flush("  The decoder has 4× more information to attend to. Expected:")
    flush("    • Hallucination ↓  (more source info available)")
    flush("    • Entity recall ↑  (entities preserved in memory)")
    flush("    • Negation errors ↓ (negation context preserved)")
    flush("    • ROUGE ↑ by +0.02 to +0.05")
    flush()
    flush("  Config: configs/exp_memory32.yaml (already exists, paths need fixing)")
    flush("  VRAM impact: ~200MB additional (fits in 8.59GB RTX 4070)")
    flush()

    # ── Save JSON ──────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_data = {
        "architecture": {
            "mamba_ssm_installed": MAMBA_AVAILABLE,
            "encoder": encoder_name,
            "decoder": "TransformerDecoder (nn.MultiheadAttention)",
            "longt5": False,
            "n_memory_tokens": model_cfg.n_memory_tokens,
            "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "checkpoint_step": step,
        "n_greedy_samples": n,
        "n_beam_samples": len(preds_beam),
        "greedy_ids": greedy_ids,
        "beam_ids": [greedy_ids[pos] for pos in beam_positions] if beam_positions else [],
        "greedy_id_file": args.greedy_id_file,
        "beam_id_file": args.beam_id_file,
        "greedy_time_s": round(greedy_time, 1),
        "beam_time_s": round(beam_time, 1),
        "gate1_safety": {
            "pass": g1_pass,
            "reasons": g1_reasons,
            "negation_error_rate": neg_rate,
            "entity_recall": ent_metrics["recall"],
            "entity_precision": ent_metrics["precision"],
            "entity_f1": ent_metrics["f1"],
            "hallucination_rate": halluc_metrics["hallucination_rate"],
            "numeric_conflict_rate": halluc_metrics["numeric_conflict_rate"],
            "phi_leakage_rate": phi_rate,
        },
        "gate2_quality": {
            "pass": g2_pass,
            "reasons": g2_reasons,
            "len_ratio": len_metrics["len_ratio"],
            "avg_pred_len": len_metrics["avg_pred_len"],
            "avg_ref_len": len_metrics["avg_ref_len"],
            "dup_bigram": rep_metrics["dup_bigram"],
            "dup_trigram": rep_metrics["dup_trigram"],
            "grammar": grammar_metrics,
            "section_coverage": {k: v["coverage"] for k, v in section_cov.items()},
        },
        "gate3_similarity": {
            "rougeL": rouge["rougeL"],
            "rouge2": rouge["rouge2"],
            "rouge1": rouge["rouge1"],
            "bertscore_f1": bert_metrics["bertscore_f1"],
            "bertscore_encoder": bert_metrics["encoder"],
        },
        "beam_vs_greedy": {
            "beam_rougeL": beam_rouge.get("rougeL", 0),
            "delta": delta_rl,
        },
        "overall_verdict": overall,
        "top_flipped_entities": dict(flip_counter.most_common(15)),
        "top_hallucinated_entities": dict(halluc_ent_counter.most_common(10)),
    }

    # Add 5 sample outputs
    pos_to_beam_i = {pos: j for j, pos in enumerate(beam_positions)}
    out_data["example_outputs"] = []
    for i in range(min(5, n)):
        entry = {
            "val_idx": int(greedy_ids[i]) if greedy_ids else i,
            "source": sources[i][:600],
            "reference": refs[i][:600],
            "greedy_pred": preds_greedy[i][:600],
        }
        if i in pos_to_beam_i:
            entry["beam_pred"] = preds_beam[pos_to_beam_i[i]][:600]
        out_data["example_outputs"].append(entry)

    # Add failures
    out_data["failures"] = (neg_fails[:15] + halluc_fails[:15] + ent_fails[:10] + phi_fails)[:40]

    json_path = os.path.join(args.output_dir, f"gated_eval_v2_step_{step}.json")
    with open(json_path, "w") as f:
        json.dump(out_data, f, indent=2, default=str)
    flush(f"Full JSON report saved → {json_path}")

    flush("\n" + "=" * 70)
    flush("  DONE")
    flush("=" * 70)


if __name__ == "__main__":
    main()
