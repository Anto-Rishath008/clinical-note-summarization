#!/usr/bin/env python3
"""
Comprehensive NLP Evaluation Suite
====================================
Runs ALL NLP evaluations and metrics on the best model checkpoint.
Executes entirely on CPU to avoid interfering with GPU training.

Metrics computed:
  1. ROUGE-1 / ROUGE-2 / ROUGE-L (F1, Precision, Recall)
  2. BLEU (1-4 gram)
  3. METEOR
  4. BERTScore (clinical encoder)
  5. Perplexity (on validation set)
  6. Safety Gates: Negation consistency, Entity preservation, Hallucination, PHI leakage
  7. Quality Gates: Repetition, Grammar proxy, Length analysis, Section coverage
  8. Qualitative examples (best, worst, median by ROUGE-L)
  9. Compression ratio analysis
  10. Novel n-gram analysis (abstractiveness)
  11. Vocabulary usage / diversity

Usage:
    python comprehensive_eval.py --checkpoint checkpoints/v2_run/best_model.pt --config configs/full_train.yaml
"""

import os
import sys
import json
import re
import math
import time
import argparse
import logging
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

# Force CPU unless --use_gpu is passed 
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # commented out - we allow GPU to speed up eval

# ─── Logging ───
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("comprehensive_eval")

# ─── Add project root to path ───
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.model import MambaTransformerConfig, build_model
from src.data_loader import DataConfig, ClinicalDataset, load_tokenizer


# ════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════

def flush(msg: str):
    print(msg, flush=True)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model_cpu(checkpoint_path: str, config: dict, device="cpu"):
    """Load model on specified device."""
    flush(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Config yaml has nested 'model:' key
    model_dict = config.get("model", config)
    model_cfg = MambaTransformerConfig.from_dict(model_dict)
    model = build_model(model_cfg)
    
    # Handle state dict key variations
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    if hasattr(state_dict, "keys"):
        # Remove 'module.' prefix if present (from DataParallel)
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace("module.", "")] = v
        model.load_state_dict(cleaned, strict=False)
    
    model = model.to(device)
    model.eval()
    step = ckpt.get("global_step", ckpt.get("step", "unknown"))
    best_rouge = ckpt.get("best_rouge_l", "unknown")
    flush(f"  Model loaded on {device} | Step: {step} | Best ROUGE-L: {best_rouge}")
    flush(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, model_cfg, step


def load_val_data(config: dict, tokenizer, max_samples=200):
    """Load validation samples."""
    data_cfg = DataConfig.from_dict(config)
    # With 95/5 train/val split, we need ~20x desired val samples in total
    # e.g. for 200 val samples, load ~4000 total
    total_to_load = max(max_samples * 25, 5000)
    dataset = ClinicalDataset(
        csv_path=data_cfg.csv_path,
        tokenizer=tokenizer,
        max_src_len=data_cfg.max_src_len,
        max_tgt_len=data_cfg.max_tgt_len,
        pad_id=data_cfg.pad_id,
        bos_id=data_cfg.bos_id,
        eos_id=data_cfg.eos_id,
        source_col=data_cfg.source_col,
        target_col=data_cfg.target_col,
        max_samples=total_to_load,
        split="val",
    )
    return dataset


def decode_tokens(tokenizer, ids):
    """Decode token IDs to text, filtering special tokens."""
    filtered = [int(t) for t in ids if int(t) not in (0, 1, 2, 3)]
    return tokenizer.DecodeIds(filtered)


# ════════════════════════════════════════════════════════════════════════
#  GENERATION
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_greedy(model, src, src_mask, max_len=512, min_len=30,
                    no_repeat_ngram=3, repetition_penalty=1.2):
    """Greedy decoding on CPU."""
    device = src.device
    memory = model.encode(src, src_mask)
    generated = torch.full((1, 1), model.config.bos_id, device=device, dtype=torch.long)
    
    for step in range(max_len - 1):
        causal = model.generate_causal_mask(generated.size(1), device)
        logits = model.decode(generated, memory, causal)
        next_logits = logits[:, -1, :]
        
        # Min length
        if step < min_len:
            next_logits[:, model.config.eos_id] = float('-inf')
        
        # Repetition penalty
        if repetition_penalty != 1.0:
            prev_tokens = generated[0].unique()
            for token_id in prev_tokens:
                if next_logits[0, token_id] > 0:
                    next_logits[0, token_id] /= repetition_penalty
                else:
                    next_logits[0, token_id] *= repetition_penalty
        
        # N-gram blocking
        if no_repeat_ngram > 0 and generated.size(1) >= no_repeat_ngram:
            seq = generated[0].tolist()
            ngrams = set()
            for j in range(len(seq) - no_repeat_ngram + 1):
                ngrams.add(tuple(seq[j:j + no_repeat_ngram]))
            partial = tuple(seq[-(no_repeat_ngram - 1):])
            for ng in ngrams:
                if ng[:-1] == partial:
                    next_logits[0, ng[-1]] = float('-inf')
        
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        
        if next_token.item() == model.config.eos_id:
            break
    
    return generated[0].tolist()


@torch.no_grad()
def generate_beam(model, src, src_mask, beam_size=4, max_len=512, min_len=30,
                  no_repeat_ngram=3, length_penalty=0.8, repetition_penalty=1.2):
    """Beam search on CPU (single sample)."""
    device = src.device
    memory = model.encode(src, src_mask)
    beams = [(0.0, [model.config.bos_id])]
    completed = []
    
    for step in range(max_len - 1):
        candidates = []
        for lp, seq in beams:
            if seq[-1] == model.config.eos_id:
                completed.append((lp, seq))
                continue
            inp = torch.tensor([seq], device=device, dtype=torch.long)
            causal = model.generate_causal_mask(inp.size(1), device)
            logits = model.decode(inp, memory, causal)
            nl = logits[0, -1, :].clone()
            
            # Min length
            if step < min_len:
                nl[model.config.eos_id] = float('-inf')
            
            # N-gram blocking
            if no_repeat_ngram > 0 and len(seq) >= no_repeat_ngram:
                ngrams_set = set()
                for j in range(len(seq) - no_repeat_ngram + 1):
                    ngrams_set.add(tuple(seq[j:j + no_repeat_ngram]))
                partial = tuple(seq[-(no_repeat_ngram - 1):])
                for ng in ngrams_set:
                    if ng[:-1] == partial:
                        nl[ng[-1]] = float('-inf')
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(seq):
                    if nl[token_id] > 0:
                        nl[token_id] /= repetition_penalty
                    else:
                        nl[token_id] *= repetition_penalty
            
            log_probs = F.log_softmax(nl, dim=-1)
            topk_lp, topk_ids = log_probs.topk(beam_size)
            for k in range(beam_size):
                candidates.append((lp + topk_lp[k].item(), seq + [topk_ids[k].item()]))
        
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
        beams = candidates[:beam_size]
    
    all_seqs = completed + beams
    if not all_seqs:
        return [model.config.bos_id]
    all_seqs.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
    return all_seqs[0][1]


# ════════════════════════════════════════════════════════════════════════
#  METRIC 1: ROUGE
# ════════════════════════════════════════════════════════════════════════

def compute_rouge_detailed(predictions: List[str], references: List[str]) -> dict:
    """Compute ROUGE-1/2/L with F1, Precision, Recall."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    r1_f, r1_p, r1_r = [], [], []
    r2_f, r2_p, r2_r = [], [], []
    rl_f, rl_p, rl_r = [], [], []
    per_sample_rl = []
    
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1_f.append(s["rouge1"].fmeasure); r1_p.append(s["rouge1"].precision); r1_r.append(s["rouge1"].recall)
        r2_f.append(s["rouge2"].fmeasure); r2_p.append(s["rouge2"].precision); r2_r.append(s["rouge2"].recall)
        rl_f.append(s["rougeL"].fmeasure); rl_p.append(s["rougeL"].precision); rl_r.append(s["rougeL"].recall)
        per_sample_rl.append(s["rougeL"].fmeasure)
    
    n = max(len(r1_f), 1)
    return {
        "rouge1": {"f1": sum(r1_f)/n, "precision": sum(r1_p)/n, "recall": sum(r1_r)/n},
        "rouge2": {"f1": sum(r2_f)/n, "precision": sum(r2_p)/n, "recall": sum(r2_r)/n},
        "rougeL": {"f1": sum(rl_f)/n, "precision": sum(rl_p)/n, "recall": sum(rl_r)/n},
        "per_sample_rougeL": per_sample_rl,
    }


# ════════════════════════════════════════════════════════════════════════
#  METRIC 2: BLEU
# ════════════════════════════════════════════════════════════════════════

def compute_bleu(predictions: List[str], references: List[str]) -> dict:
    """Compute BLEU-1 through BLEU-4."""
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def modified_precision(pred_tokens, ref_tokens, n):
        pred_ngrams = Counter(get_ngrams(pred_tokens, n))
        ref_ngrams = Counter(get_ngrams(ref_tokens, n))
        clipped = {ng: min(count, ref_ngrams.get(ng, 0)) for ng, count in pred_ngrams.items()}
        return sum(clipped.values()), max(sum(pred_ngrams.values()), 1)
    
    bleu_scores = {}
    for n in range(1, 5):
        total_clipped = 0
        total_pred = 0
        for pred, ref in zip(predictions, references):
            pt = pred.lower().split()
            rt = ref.lower().split()
            c, t = modified_precision(pt, rt, n)
            total_clipped += c
            total_pred += t
        bleu_scores[f"bleu_{n}"] = total_clipped / max(total_pred, 1)
    
    # Brevity penalty
    pred_len = sum(len(p.split()) for p in predictions)
    ref_len = sum(len(r.split()) for r in references)
    bp = math.exp(min(0, 1 - ref_len / max(pred_len, 1)))
    
    # Combined BLEU-4
    log_avg = sum(math.log(max(bleu_scores[f"bleu_{n}"], 1e-10)) for n in range(1, 5)) / 4
    bleu_scores["bleu_4_combined"] = bp * math.exp(log_avg)
    bleu_scores["brevity_penalty"] = bp
    
    return bleu_scores


# ════════════════════════════════════════════════════════════════════════
#  METRIC 3: METEOR (simplified)
# ════════════════════════════════════════════════════════════════════════

def compute_meteor(predictions: List[str], references: List[str]) -> dict:
    """Simplified METEOR: unigram precision/recall with harmonic mean, penalty for chunking."""
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        
        # Unigram matches
        pred_set = Counter(pred_tokens)
        ref_set = Counter(ref_tokens)
        matches = sum((pred_set & ref_set).values())
        
        precision = matches / max(len(pred_tokens), 1)
        recall = matches / max(len(ref_tokens), 1)
        
        if precision + recall == 0:
            scores.append(0.0)
            continue
        
        # Harmonic mean with recall weight (alpha=0.9)
        alpha = 0.9
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        
        # Chunk penalty (count consecutive matching chunks)
        chunks = 0
        in_chunk = False
        ref_remaining = list(ref_tokens)
        for t in pred_tokens:
            if t in ref_remaining:
                if not in_chunk:
                    chunks += 1
                    in_chunk = True
                ref_remaining.remove(t)
            else:
                in_chunk = False
        
        penalty = 0.5 * (chunks / max(matches, 1)) ** 3 if matches > 0 else 0
        scores.append(fmean * (1 - penalty))
    
    return {"meteor": sum(scores) / max(len(scores), 1)}


# ════════════════════════════════════════════════════════════════════════
#  METRIC 4: BERTScore
# ════════════════════════════════════════════════════════════════════════

def compute_bertscore(predictions: List[str], references: List[str], max_samples=100) -> dict:
    """BERTScore with clinical encoder fallback chain."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        flush("  [WARN] bert-score not installed, skipping BERTScore")
        return {"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None, "encoder": "NOT_INSTALLED"}
    
    preds = predictions[:max_samples]
    refs = references[:max_samples]
    
    # Try clinical encoders in order
    encoders = [
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "emilyalsentzer/Bio_ClinicalBERT",
        "bert-base-uncased",
    ]
    
    for enc in encoders:
        try:
            flush(f"  Trying BERTScore with {enc}...")
            P, R, F1 = bert_score_fn(
                preds, refs,
                model_type=enc,
                device="cpu",
                batch_size=8,
                verbose=False,
            )
            return {
                "bertscore_p": P.mean().item(),
                "bertscore_r": R.mean().item(),
                "bertscore_f1": F1.mean().item(),
                "encoder": enc,
            }
        except Exception as e:
            flush(f"  [WARN] {enc} failed: {e}")
            continue
    
    flush("  [WARN] All BERTScore encoders failed")
    return {"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None, "encoder": "ALL_FAILED"}


# ════════════════════════════════════════════════════════════════════════
#  METRIC 5: Perplexity
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_perplexity(model, dataset, tokenizer, max_samples=100) -> dict:
    """Compute teacher-forced perplexity on validation data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_samples = min(len(dataset), max_samples)
    
    for i in range(n_samples):
        item = dataset[i]
        src_ids = item['src']
        tgt_input = item['tgt_input']
        tgt_output = item['tgt_output']
        src = src_ids.unsqueeze(0).to(next(model.parameters()).device)
        src_mask = (src == model.config.pad_id)
        
        memory = model.encode(src, src_mask)
        tgt_in = tgt_input.unsqueeze(0).to(src.device)
        tgt_labels = tgt_output.unsqueeze(0).to(src.device)
        
        causal = model.generate_causal_mask(tgt_in.size(1), src.device)
        logits = model.decode(tgt_in, memory, causal)
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_labels.reshape(-1),
            ignore_index=model.config.pad_id,
            reduction="sum",
        )
        
        non_pad = (tgt_labels != model.config.pad_id).sum().item()
        total_loss += loss.item()
        total_tokens += non_pad
        
        if (i + 1) % 20 == 0:
            flush(f"    Perplexity: {i+1}/{n_samples} samples processed")
    
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 100))  # cap to avoid overflow
    
    return {
        "perplexity": ppl,
        "avg_ce_loss": avg_loss,
        "total_tokens": total_tokens,
        "n_samples": n_samples,
    }


# ════════════════════════════════════════════════════════════════════════
#  SAFETY GATES (Gate 1)
# ════════════════════════════════════════════════════════════════════════

# PHI patterns
_PHI_PATTERNS = [
    re.compile(r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b'),        # SSN
    re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),        # Phone
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email
    re.compile(r'\b(?:DOB|Date of Birth)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', re.I),  # DOB
    re.compile(r'\bMRN[:\s#]*\d{4,}\b', re.I),           # MRN
]

# Clinical entity patterns
_ENTITY_PATTERNS = [
    re.compile(r'\b\d+\.?\d*\s*(?:mg|mcg|ml|mL|g|units?|mEq)\b', re.I),  # Dosages
    re.compile(r'\b(?:aspirin|metoprolol|lisinopril|insulin|heparin|vancomycin|metformin|'
               r'omeprazole|atorvastatin|furosemide|warfarin|prednisone|acetaminophen|'
               r'amoxicillin|clopidogrel|digoxin|levothyroxine|amlodipine)\b', re.I),  # Common meds
    re.compile(r'\b(?:WBC|RBC|Hgb|Hct|PLT|BUN|Cr|Na|K|Cl|CO2|glucose|troponin|'
               r'lactate|INR|PTT|AST|ALT|bilirubin|albumin|calcium|magnesium|'
               r'creatinine|hemoglobin|potassium|sodium)\s*(?:of\s+)?(?:=|:|\s)\s*\d', re.I),  # Labs
    re.compile(r'\b(?:diabetes|hypertension|CHF|COPD|pneumonia|sepsis|MI|CVA|stroke|'
               r'atrial fibrillation|heart failure|renal failure|cirrhosis|'
               r'pulmonary embolism|DVT|UTI|cellulitis)\b', re.I),  # Diagnoses
]

# Negation patterns
_NEGATION_PATTERNS = [
    re.compile(r'\b(?:no|not|without|denies?|denied|negative|absence|absent|'
               r'ruled out|unlikely|never|neither|nor)\b', re.I),
]

# Clinical sections
_CLINICAL_SECTIONS = {
    "history_of_present_illness": re.compile(r'\b(?:HPI|history of present illness|presenting complaint)\b', re.I),
    "medications": re.compile(r'\b(?:medications?|meds|home med|discharge med)\b', re.I),
    "assessment_plan": re.compile(r'\b(?:assessment|plan|A&P|A/P|impression)\b', re.I),
    "discharge_diagnosis": re.compile(r'\b(?:discharge diagnos[ie]s|principal diagnos[ie]s|final diagnos[ie]s)\b', re.I),
    "procedures": re.compile(r'\b(?:procedures?|surgeries|operations?|interventions?)\b', re.I),
    "labs_vitals": re.compile(r'\b(?:labs?|vitals?|laboratory|vital signs?)\b', re.I),
}


def safety_negation_consistency(sources: List[str], predictions: List[str]) -> dict:
    """Check if negations in source are preserved in predictions."""
    conflicts = 0
    total = 0
    details = []
    
    for i, (src, pred) in enumerate(zip(sources, predictions)):
        src_lower = src.lower()
        pred_lower = pred.lower()
        
        for pat in _NEGATION_PATTERNS:
            src_negs = [(m.start(), m.group(0)) for m in pat.finditer(src_lower)]
            for pos, neg_word in src_negs:
                # Get surrounding context (20 chars after negation)
                context = src_lower[pos:pos+50].split()[:5]
                context_str = " ".join(context)
                total += 1
                
                # Check if the negated concept appears WITHOUT negation in prediction
                if len(context) >= 2:
                    concept = context[-1] if len(context) > 1 else context[0]
                    if concept in pred_lower:
                        # Check if it's also negated in prediction
                        concept_pos = pred_lower.find(concept)
                        nearby = pred_lower[max(0, concept_pos-30):concept_pos]
                        has_neg_in_pred = any(p.search(nearby) for p in _NEGATION_PATTERNS)
                        if not has_neg_in_pred:
                            conflicts += 1
                            if len(details) < 10:
                                details.append({
                                    "idx": i, "source_context": context_str,
                                    "concept": concept, "pred_snippet": pred[max(0,concept_pos-20):concept_pos+30]
                                })
    
    rate = conflicts / max(total, 1)
    return {"negation_conflict_rate": rate, "conflicts": conflicts, "total_checked": total, "details": details[:5]}


def safety_entity_preservation(sources: List[str], predictions: List[str]) -> dict:
    """Check entity preservation between source and prediction."""
    all_precision, all_recall, all_f1 = [], [], []
    
    for src, pred in zip(sources, predictions):
        src_entities = set()
        pred_entities = set()
        
        for pat in _ENTITY_PATTERNS:
            for m in pat.finditer(src):
                src_entities.add(m.group(0).lower().strip())
            for m in pat.finditer(pred):
                pred_entities.add(m.group(0).lower().strip())
        
        if not src_entities:
            continue
        
        if not pred_entities:
            all_precision.append(0.0)
            all_recall.append(0.0)
            all_f1.append(0.0)
            continue
        
        # Fuzzy match: check if entity substring appears
        matches = 0
        for se in src_entities:
            for pe in pred_entities:
                if se in pe or pe in se:
                    matches += 1
                    break
        
        recall = matches / max(len(src_entities), 1)
        precision = min(matches, len(pred_entities)) / max(len(pred_entities), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
    
    n = max(len(all_f1), 1)
    return {
        "entity_precision": sum(all_precision) / n,
        "entity_recall": sum(all_recall) / n,
        "entity_f1": sum(all_f1) / n,
        "n_samples_with_entities": len(all_f1),
    }


def safety_hallucination(sources: List[str], predictions: List[str]) -> dict:
    """Detect potential hallucinated entities (in prediction but not in source)."""
    halluc_counts = []
    examples = []
    
    for i, (src, pred) in enumerate(zip(sources, predictions)):
        src_lower = src.lower()
        pred_entities = set()
        
        for pat in _ENTITY_PATTERNS:
            for m in pat.finditer(pred):
                ent = m.group(0).lower().strip()
                # Check if this entity appears in source
                if ent not in src_lower:
                    pred_entities.add(ent)
        
        halluc_counts.append(len(pred_entities))
        if pred_entities and len(examples) < 5:
            examples.append({"idx": i, "hallucinated": list(pred_entities)[:5]})
    
    n = max(len(halluc_counts), 1)
    has_halluc = sum(1 for h in halluc_counts if h > 0)
    return {
        "hallucination_rate": has_halluc / n,
        "avg_halluc_per_sample": sum(halluc_counts) / n,
        "examples": examples,
    }


def safety_phi_leakage(predictions: List[str]) -> dict:
    """Scan for PHI patterns in predictions."""
    leaks = 0
    examples = []
    
    for i, pred in enumerate(predictions):
        found = []
        for p in _PHI_PATTERNS:
            for m in p.finditer(pred):
                found.append(m.group(0))
        if found:
            leaks += 1
            if len(examples) < 5:
                examples.append({"idx": i, "phi_matches": found[:3]})
    
    return {
        "phi_leak_rate": leaks / max(len(predictions), 1),
        "total_leaks": leaks,
        "examples": examples,
    }


# ════════════════════════════════════════════════════════════════════════
#  QUALITY GATES (Gate 2)
# ════════════════════════════════════════════════════════════════════════

def quality_repetition(predictions: List[str]) -> dict:
    """Measure repetition via duplicate n-gram rates."""
    bi_rates, tri_rates, four_rates = [], [], []
    
    for pred in predictions:
        toks = pred.lower().split()
        if len(toks) < 4:
            bi_rates.append(0.); tri_rates.append(0.); four_rates.append(0.)
            continue
        
        bis = [tuple(toks[i:i+2]) for i in range(len(toks)-1)]
        tris = [tuple(toks[i:i+3]) for i in range(len(toks)-2)]
        fours = [tuple(toks[i:i+4]) for i in range(len(toks)-3)]
        
        bi_rates.append(1 - len(set(bis))/max(len(bis), 1))
        tri_rates.append(1 - len(set(tris))/max(len(tris), 1))
        four_rates.append(1 - len(set(fours))/max(len(fours), 1))
    
    n = max(len(bi_rates), 1)
    return {
        "dup_bigram_rate": sum(bi_rates) / n,
        "dup_trigram_rate": sum(tri_rates) / n,
        "dup_4gram_rate": sum(four_rates) / n,
    }


def quality_length(predictions: List[str], references: List[str]) -> dict:
    """Length analysis."""
    pl = [len(p.split()) for p in predictions]
    rl = [len(r.split()) for r in references]
    
    return {
        "avg_pred_len": sum(pl) / max(len(pl), 1),
        "avg_ref_len": sum(rl) / max(len(rl), 1),
        "median_pred_len": sorted(pl)[len(pl)//2] if pl else 0,
        "median_ref_len": sorted(rl)[len(rl)//2] if rl else 0,
        "min_pred_len": min(pl) if pl else 0,
        "max_pred_len": max(pl) if pl else 0,
        "len_ratio": sum(pl) / max(sum(rl), 1),
        "std_pred_len": float(np.std(pl)) if pl else 0,
    }


def quality_grammar_proxy(predictions: List[str]) -> dict:
    """Grammar/readability proxy."""
    total_words = 0
    total_sents = 0
    fragment_count = 0
    broken_headers = 0
    
    for pred in predictions:
        sents = [s.strip() for s in re.split(r'[.!?\n]', pred) if s.strip()]
        total_sents += len(sents)
        for s in sents:
            w = len(s.split())
            total_words += w
            if 0 < w < 4:
                fragment_count += 1
        broken_headers += len(re.findall(r'#[#.)\]]*\s*$', pred, re.M))
    
    n = max(len(predictions), 1)
    return {
        "avg_sentence_len": total_words / max(total_sents, 1),
        "fragments_per_sample": fragment_count / n,
        "broken_headers_per_sample": broken_headers / n,
        "errors_per_100_words": (fragment_count + broken_headers) / max(total_words, 1) * 100,
    }


def quality_section_coverage(sources: List[str], predictions: List[str]) -> dict:
    """Check clinical section coverage."""
    coverage = {}
    for sec_name, sec_pat in _CLINICAL_SECTIONS.items():
        src_has = sum(1 for s in sources if sec_pat.search(s))
        pred_has = sum(1 for s, p in zip(sources, predictions) if sec_pat.search(s) and sec_pat.search(p))
        coverage[sec_name] = {
            "in_source": src_has,
            "in_pred": pred_has,
            "coverage_pct": 100 * pred_has / max(src_has, 1),
        }
    
    total_src = sum(v["in_source"] for v in coverage.values())
    total_pred = sum(v["in_pred"] for v in coverage.values())
    coverage["overall_coverage_pct"] = 100 * total_pred / max(total_src, 1)
    return coverage


# ════════════════════════════════════════════════════════════════════════
#  ADDITIONAL METRICS
# ════════════════════════════════════════════════════════════════════════

def compute_compression_ratio(sources: List[str], predictions: List[str]) -> dict:
    """Compression ratio analysis."""
    ratios = []
    for src, pred in zip(sources, predictions):
        sl = len(src.split())
        pl = len(pred.split())
        if sl > 0:
            ratios.append(pl / sl)
    
    return {
        "avg_compression_ratio": sum(ratios) / max(len(ratios), 1),
        "min_compression_ratio": min(ratios) if ratios else 0,
        "max_compression_ratio": max(ratios) if ratios else 0,
        "median_compression_ratio": sorted(ratios)[len(ratios)//2] if ratios else 0,
    }


def compute_novel_ngrams(predictions: List[str], references: List[str]) -> dict:
    """Measure abstractiveness: what fraction of n-grams in prediction are novel (not in reference)?"""
    results = {}
    
    for n in [1, 2, 3, 4]:
        novel_count = 0
        total_count = 0
        
        for pred, ref in zip(predictions, references):
            pred_toks = pred.lower().split()
            ref_toks = ref.lower().split()
            
            pred_ngrams = [tuple(pred_toks[i:i+n]) for i in range(len(pred_toks)-n+1)]
            ref_ngrams = set(tuple(ref_toks[i:i+n]) for i in range(len(ref_toks)-n+1))
            
            for ng in pred_ngrams:
                total_count += 1
                if ng not in ref_ngrams:
                    novel_count += 1
        
        results[f"novel_{n}gram_pct"] = 100 * novel_count / max(total_count, 1)
    
    return results


def compute_vocabulary_diversity(predictions: List[str]) -> dict:
    """Vocabulary usage and diversity stats."""
    all_tokens = []
    unique_per_sample = []
    
    for pred in predictions:
        toks = pred.lower().split()
        all_tokens.extend(toks)
        unique_per_sample.append(len(set(toks)) / max(len(toks), 1))
    
    vocab = Counter(all_tokens)
    
    return {
        "total_tokens": len(all_tokens),
        "unique_tokens": len(vocab),
        "type_token_ratio": len(vocab) / max(len(all_tokens), 1),
        "avg_unique_ratio_per_sample": sum(unique_per_sample) / max(len(unique_per_sample), 1),
        "top_20_tokens": vocab.most_common(20),
    }


# ════════════════════════════════════════════════════════════════════════
#  QUALITATIVE EXAMPLES
# ════════════════════════════════════════════════════════════════════════

def get_qualitative_examples(sources, predictions, references, per_sample_rl, n_examples=3):
    """Get best, worst, and median examples by ROUGE-L."""
    indexed = list(enumerate(per_sample_rl))
    indexed.sort(key=lambda x: x[1])
    
    examples = {}
    
    # Worst
    worst_indices = [idx for idx, _ in indexed[:n_examples]]
    examples["worst"] = []
    for idx in worst_indices:
        examples["worst"].append({
            "index": idx,
            "rougeL": per_sample_rl[idx],
            "source_preview": sources[idx][:300] + "...",
            "reference": references[idx][:500],
            "prediction": predictions[idx][:500],
        })
    
    # Best
    best_indices = [idx for idx, _ in indexed[-n_examples:]]
    examples["best"] = []
    for idx in reversed(best_indices):
        examples["best"].append({
            "index": idx,
            "rougeL": per_sample_rl[idx],
            "source_preview": sources[idx][:300] + "...",
            "reference": references[idx][:500],
            "prediction": predictions[idx][:500],
        })
    
    # Median
    mid = len(indexed) // 2
    median_indices = [idx for idx, _ in indexed[mid-1:mid+2]]
    examples["median"] = []
    for idx in median_indices:
        examples["median"].append({
            "index": idx,
            "rougeL": per_sample_rl[idx],
            "source_preview": sources[idx][:300] + "...",
            "reference": references[idx][:500],
            "prediction": predictions[idx][:500],
        })
    
    return examples


# ════════════════════════════════════════════════════════════════════════
#  MAIN EVALUATION PIPELINE
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Comprehensive NLP Evaluation Suite")
    parser.add_argument("--checkpoint", default="checkpoints/v2_run/best_model.pt")
    parser.add_argument("--config", default="configs/full_train.yaml")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Number of validation samples for generation evaluation")
    parser.add_argument("--ppl_samples", type=int, default=100,
                        help="Number of samples for perplexity computation")
    parser.add_argument("--beam_samples", type=int, default=20,
                        help="Number of samples for beam search (slower but better)")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--skip_bertscore", action="store_true",
                        help="Skip BERTScore (saves time)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, or cuda")
    parser.add_argument("--output_dir", default="eval_results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()
    
    flush("=" * 80)
    flush("  COMPREHENSIVE NLP EVALUATION SUITE")
    flush("=" * 80)
    
    # Choose device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    flush(f"  Device: {device}")
    if device.type == "cuda":
        flush(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        flush(f"  GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # ─── Load model & data ───
    flush("\n[1/10] Loading model and data...")
    config = load_config(args.config)
    tokenizer = load_tokenizer(config["data"]["tokenizer_path"])
    model, model_cfg, step = load_model_cpu(args.checkpoint, config, device=device)
    
    dataset = load_val_data(config, tokenizer, max_samples=max(args.max_samples, args.ppl_samples))
    flush(f"  Validation samples loaded: {len(dataset)}")
    
    # ─── Generate predictions (greedy) ───
    flush(f"\n[2/10] Generating predictions (greedy, {args.max_samples} samples)...")
    sources_text = []
    references_text = []
    predictions_text = []
    
    gen_samples = min(args.max_samples, len(dataset))
    gen_start = time.time()
    
    for i in range(gen_samples):
        item = dataset[i]
        src_ids = item['src']
        tgt_ids = item['tgt_output']
        src = src_ids.unsqueeze(0).to(device)
        src_mask = (src == model.config.pad_id)
        
        pred_ids = generate_greedy(model, src, src_mask, max_len=args.max_gen_len)
        
        src_text = item.get('src_text', decode_tokens(tokenizer, src_ids.tolist()))
        ref_text = item.get('tgt_text', decode_tokens(tokenizer, tgt_ids.tolist()))
        pred_text = decode_tokens(tokenizer, pred_ids)
        
        sources_text.append(src_text)
        references_text.append(ref_text)
        predictions_text.append(pred_text)
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - gen_start
            rate = (i + 1) / elapsed
            eta = (gen_samples - i - 1) / max(rate, 0.001)
            flush(f"  Generated {i+1}/{gen_samples} | {rate:.1f} samples/s | ETA: {eta:.0f}s")
    
    gen_time = time.time() - gen_start
    flush(f"  Greedy generation done in {gen_time:.1f}s ({gen_samples/gen_time:.1f} samples/s)")
    
    # ─── Generate beam search predictions (small subset) ───
    flush(f"\n[3/10] Generating beam search predictions ({args.beam_samples} samples, beam={args.beam_size})...")
    beam_predictions = []
    beam_references = []
    beam_sources = []
    
    beam_n = min(args.beam_samples, len(dataset))
    beam_start = time.time()
    
    for i in range(beam_n):
        item = dataset[i]
        src_ids = item['src']
        src = src_ids.unsqueeze(0).to(device)
        src_mask = (src == model.config.pad_id)
        
        pred_ids = generate_beam(model, src, src_mask, beam_size=args.beam_size, max_len=args.max_gen_len)
        
        beam_sources.append(sources_text[i] if i < len(sources_text) else item.get('src_text', ''))
        beam_references.append(references_text[i] if i < len(references_text) else item.get('tgt_text', ''))
        beam_predictions.append(decode_tokens(tokenizer, pred_ids))
        
        if (i + 1) % 5 == 0:
            elapsed = time.time() - beam_start
            rate = (i + 1) / elapsed
            eta = (beam_n - i - 1) / max(rate, 0.001)
            flush(f"  Beam {i+1}/{beam_n} | {rate:.2f} samples/s | ETA: {eta:.0f}s")
    
    beam_time = time.time() - beam_start
    flush(f"  Beam search done in {beam_time:.1f}s")
    
    # ─── ROUGE ───
    flush("\n[4/10] Computing ROUGE scores...")
    rouge_greedy = compute_rouge_detailed(predictions_text, references_text)
    rouge_beam = compute_rouge_detailed(beam_predictions, beam_references)
    per_sample_rl = rouge_greedy["per_sample_rougeL"]
    
    flush(f"  ROUGE-1 (greedy): {rouge_greedy['rouge1']['f1']:.4f}")
    flush(f"  ROUGE-2 (greedy): {rouge_greedy['rouge2']['f1']:.4f}")
    flush(f"  ROUGE-L (greedy): {rouge_greedy['rougeL']['f1']:.4f}")
    flush(f"  ROUGE-1 (beam):   {rouge_beam['rouge1']['f1']:.4f}")
    flush(f"  ROUGE-2 (beam):   {rouge_beam['rouge2']['f1']:.4f}")
    flush(f"  ROUGE-L (beam):   {rouge_beam['rougeL']['f1']:.4f}")
    
    # ─── BLEU ───
    flush("\n[5/10] Computing BLEU scores...")
    bleu_greedy = compute_bleu(predictions_text, references_text)
    bleu_beam = compute_bleu(beam_predictions, beam_references)
    
    flush(f"  BLEU-1 (greedy): {bleu_greedy['bleu_1']:.4f}")
    flush(f"  BLEU-2 (greedy): {bleu_greedy['bleu_2']:.4f}")
    flush(f"  BLEU-4 (greedy): {bleu_greedy['bleu_4_combined']:.4f}")
    flush(f"  BLEU-4 (beam):   {bleu_beam['bleu_4_combined']:.4f}")
    
    # ─── METEOR ───
    flush("\n[6/10] Computing METEOR scores...")
    meteor_greedy = compute_meteor(predictions_text, references_text)
    meteor_beam = compute_meteor(beam_predictions, beam_references)
    
    flush(f"  METEOR (greedy): {meteor_greedy['meteor']:.4f}")
    flush(f"  METEOR (beam):   {meteor_beam['meteor']:.4f}")
    
    # ─── BERTScore ───
    if not args.skip_bertscore:
        flush("\n[7/10] Computing BERTScore...")
        bertscore_results = compute_bertscore(predictions_text, references_text)
        flush(f"  BERTScore F1: {bertscore_results.get('bertscore_f1', 'N/A')}")
        flush(f"  Encoder: {bertscore_results.get('encoder', 'N/A')}")
    else:
        flush("\n[7/10] BERTScore SKIPPED (--skip_bertscore)")
        bertscore_results = {"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None, "encoder": "SKIPPED"}
    
    # ─── Perplexity ───
    flush(f"\n[8/10] Computing perplexity ({args.ppl_samples} samples)...")
    ppl_results = compute_perplexity(model, dataset, tokenizer, max_samples=args.ppl_samples)
    flush(f"  Perplexity: {ppl_results['perplexity']:.2f}")
    flush(f"  Avg CE Loss: {ppl_results['avg_ce_loss']:.4f}")
    
    # ─── Safety Gates ───
    flush("\n[9/10] Running Safety & Quality Gates...")
    
    # Gate 1: Safety
    flush("  Gate 1 (Safety):")
    neg_results = safety_negation_consistency(sources_text, predictions_text)
    flush(f"    Negation conflict rate: {neg_results['negation_conflict_rate']:.4f}")
    
    entity_results = safety_entity_preservation(sources_text, predictions_text)
    flush(f"    Entity precision: {entity_results['entity_precision']:.4f}")
    flush(f"    Entity recall: {entity_results['entity_recall']:.4f}")
    flush(f"    Entity F1: {entity_results['entity_f1']:.4f}")
    
    halluc_results = safety_hallucination(sources_text, predictions_text)
    flush(f"    Hallucination rate: {halluc_results['hallucination_rate']:.4f}")
    
    phi_results = safety_phi_leakage(predictions_text)
    flush(f"    PHI leak rate: {phi_results['phi_leak_rate']:.4f}")
    
    # Gate 2: Quality
    flush("  Gate 2 (Quality):")
    rep_results = quality_repetition(predictions_text)
    flush(f"    Dup bigram rate: {rep_results['dup_bigram_rate']:.4f}")
    flush(f"    Dup trigram rate: {rep_results['dup_trigram_rate']:.4f}")
    
    len_results = quality_length(predictions_text, references_text)
    flush(f"    Avg pred len: {len_results['avg_pred_len']:.1f} words")
    flush(f"    Avg ref len: {len_results['avg_ref_len']:.1f} words")
    flush(f"    Length ratio: {len_results['len_ratio']:.3f}")
    
    grammar_results = quality_grammar_proxy(predictions_text)
    flush(f"    Avg sentence len: {grammar_results['avg_sentence_len']:.1f} words")
    flush(f"    Errors/100 words: {grammar_results['errors_per_100_words']:.2f}")
    
    section_results = quality_section_coverage(sources_text, predictions_text)
    flush(f"    Overall section coverage: {section_results.get('overall_coverage_pct', 0):.1f}%")
    
    # ─── Additional metrics ───
    flush("\n[10/10] Computing additional metrics...")
    
    compression = compute_compression_ratio(sources_text, predictions_text)
    flush(f"  Avg compression ratio: {compression['avg_compression_ratio']:.4f}")
    
    novel = compute_novel_ngrams(predictions_text, references_text)
    flush(f"  Novel unigram %: {novel['novel_1gram_pct']:.1f}%")
    flush(f"  Novel bigram %: {novel['novel_2gram_pct']:.1f}%")
    flush(f"  Novel trigram %: {novel['novel_3gram_pct']:.1f}%")
    
    vocab_div = compute_vocabulary_diversity(predictions_text)
    flush(f"  Unique tokens: {vocab_div['unique_tokens']}")
    flush(f"  Type-token ratio: {vocab_div['type_token_ratio']:.4f}")
    
    # ─── Qualitative examples ───
    examples = get_qualitative_examples(sources_text, predictions_text, references_text, per_sample_rl)
    
    total_time = time.time() - start_time
    
    # ════════════════════════════════════════════════════════════════════
    #  PRINT COMPREHENSIVE SUMMARY
    # ════════════════════════════════════════════════════════════════════
    
    flush("\n" + "=" * 80)
    flush("  COMPREHENSIVE EVALUATION RESULTS")
    flush("=" * 80)
    
    flush(f"\n  Model: Mamba-Transformer Hybrid V2")
    flush(f"  Checkpoint: {args.checkpoint} (step {step})")
    flush(f"  Eval samples: {gen_samples} greedy + {beam_n} beam")
    flush(f"  Total time: {total_time:.0f}s")
    
    flush("\n┌─────────────────────────────────────────────────────────────┐")
    flush("│  AUTOMATIC METRICS                                         │")
    flush("├─────────────────────────┬──────────┬────────────────────────┤")
    flush("│  Metric                 │  Greedy  │  Beam (size={:d})       │".format(args.beam_size))
    flush("├─────────────────────────┼──────────┼────────────────────────┤")
    flush("│  ROUGE-1 F1             │  {:.4f}  │  {:.4f}                │".format(rouge_greedy['rouge1']['f1'], rouge_beam['rouge1']['f1']))
    flush("│  ROUGE-2 F1             │  {:.4f}  │  {:.4f}                │".format(rouge_greedy['rouge2']['f1'], rouge_beam['rouge2']['f1']))
    flush("│  ROUGE-L F1             │  {:.4f}  │  {:.4f}                │".format(rouge_greedy['rougeL']['f1'], rouge_beam['rougeL']['f1']))
    flush("│  BLEU-1                 │  {:.4f}  │  {:.4f}                │".format(bleu_greedy['bleu_1'], bleu_beam['bleu_1']))
    flush("│  BLEU-2                 │  {:.4f}  │  {:.4f}                │".format(bleu_greedy['bleu_2'], bleu_beam['bleu_2']))
    flush("│  BLEU-4                 │  {:.4f}  │  {:.4f}                │".format(bleu_greedy['bleu_4_combined'], bleu_beam['bleu_4_combined']))
    flush("│  METEOR                 │  {:.4f}  │  {:.4f}                │".format(meteor_greedy['meteor'], meteor_beam['meteor']))
    bs_f1 = bertscore_results.get('bertscore_f1')
    flush("│  BERTScore F1           │  {}  │  —                     │".format(f"{bs_f1:.4f}" if bs_f1 else "  N/A "))
    flush("│  Perplexity             │  {:.2f}  │  —                     │".format(ppl_results['perplexity']))
    flush("└─────────────────────────┴──────────┴────────────────────────┘")
    
    flush("\n┌─────────────────────────────────────────────────────────────┐")
    flush("│  SAFETY GATE (Gate 1)                                      │")
    flush("├─────────────────────────┬──────────┬────────────────────────┤")
    flush("│  Negation conflict rate │  {:.4f}  │  Target: < 0.02        │".format(neg_results['negation_conflict_rate']))
    flush("│  Entity precision       │  {:.4f}  │                        │".format(entity_results['entity_precision']))
    flush("│  Entity recall          │  {:.4f}  │  Target: > 0.70        │".format(entity_results['entity_recall']))
    flush("│  Entity F1              │  {:.4f}  │                        │".format(entity_results['entity_f1']))
    flush("│  Hallucination rate     │  {:.4f}  │  Target: < 0.20        │".format(halluc_results['hallucination_rate']))
    flush("│  PHI leak rate          │  {:.4f}  │  Target: 0.00          │".format(phi_results['phi_leak_rate']))
    flush("└─────────────────────────┴──────────┴────────────────────────┘")
    
    flush("\n┌─────────────────────────────────────────────────────────────┐")
    flush("│  QUALITY GATE (Gate 2)                                     │")
    flush("├─────────────────────────┬──────────┬────────────────────────┤")
    flush("│  Dup bigram rate        │  {:.4f}  │  Target: < 0.10        │".format(rep_results['dup_bigram_rate']))
    flush("│  Dup trigram rate       │  {:.4f}  │                        │".format(rep_results['dup_trigram_rate']))
    flush("│  Length ratio           │  {:.3f}   │  Target: 0.70–1.00     │".format(len_results['len_ratio']))
    flush("│  Avg pred len           │  {:.0f}     │                        │".format(len_results['avg_pred_len']))
    flush("│  Avg ref len            │  {:.0f}     │                        │".format(len_results['avg_ref_len']))
    flush("│  Errors/100 words       │  {:.2f}   │  Target: < 3.0         │".format(grammar_results['errors_per_100_words']))
    flush("│  Section coverage       │  {:.1f}%   │                        │".format(section_results.get('overall_coverage_pct', 0)))
    flush("└─────────────────────────┴──────────┴────────────────────────┘")
    
    flush("\n┌─────────────────────────────────────────────────────────────┐")
    flush("│  ABSTRACTIVENESS & DIVERSITY                               │")
    flush("├─────────────────────────┬──────────────────────────────────┤")
    flush("│  Novel unigrams         │  {:.1f}%                          │".format(novel['novel_1gram_pct']))
    flush("│  Novel bigrams          │  {:.1f}%                          │".format(novel['novel_2gram_pct']))
    flush("│  Novel trigrams         │  {:.1f}%                          │".format(novel['novel_3gram_pct']))
    flush("│  Novel 4-grams          │  {:.1f}%                          │".format(novel['novel_4gram_pct']))
    flush("│  Type-token ratio       │  {:.4f}                          │".format(vocab_div['type_token_ratio']))
    flush("│  Compression ratio      │  {:.4f}                          │".format(compression['avg_compression_ratio']))
    flush("└─────────────────────────┴──────────────────────────────────┘")
    
    # ─── Print qualitative examples ───
    flush("\n" + "=" * 80)
    flush("  QUALITATIVE EXAMPLES")
    flush("=" * 80)
    
    for category in ["best", "worst", "median"]:
        flush(f"\n--- {category.upper()} examples (by ROUGE-L) ---")
        for ex in examples[category][:2]:
            flush(f"\n  [Sample {ex['index']}] ROUGE-L = {ex['rougeL']:.4f}")
            flush(f"  SOURCE: {ex['source_preview'][:200]}")
            flush(f"  REFERENCE: {ex['reference'][:300]}")
            flush(f"  PREDICTION: {ex['prediction'][:300]}")
    
    # ─── Save full report ───
    report = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "step": step,
            "n_greedy_samples": gen_samples,
            "n_beam_samples": beam_n,
            "beam_size": args.beam_size,
            "total_time_s": total_time,
            "gen_time_s": gen_time,
            "beam_time_s": beam_time,
        },
        "rouge_greedy": {k: v for k, v in rouge_greedy.items() if k != "per_sample_rougeL"},
        "rouge_beam": {k: v for k, v in rouge_beam.items() if k != "per_sample_rougeL"},
        "bleu_greedy": bleu_greedy,
        "bleu_beam": bleu_beam,
        "meteor_greedy": meteor_greedy,
        "meteor_beam": meteor_beam,
        "bertscore": bertscore_results,
        "perplexity": ppl_results,
        "safety": {
            "negation": neg_results,
            "entity_preservation": entity_results,
            "hallucination": halluc_results,
            "phi_leakage": phi_results,
        },
        "quality": {
            "repetition": rep_results,
            "length": len_results,
            "grammar": grammar_results,
            "section_coverage": section_results,
        },
        "abstractiveness": novel,
        "vocabulary_diversity": {k: v for k, v in vocab_div.items() if k != "top_20_tokens"},
        "compression": compression,
        "qualitative_examples": examples,
        "per_sample_rougeL": per_sample_rl,
    }
    
    report_path = os.path.join(args.output_dir, f"comprehensive_eval_step{step}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    flush(f"\n  Full report saved to: {report_path}")
    
    # Also save a summary text file
    summary_path = os.path.join(args.output_dir, f"eval_summary_step{step}.txt")
    with open(summary_path, "w") as f:
        f.write("COMPREHENSIVE EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint} (step {step})\n")
        f.write(f"Samples: {gen_samples} greedy + {beam_n} beam\n\n")
        f.write("AUTOMATIC METRICS (Greedy / Beam):\n")
        f.write(f"  ROUGE-1: {rouge_greedy['rouge1']['f1']:.4f} / {rouge_beam['rouge1']['f1']:.4f}\n")
        f.write(f"  ROUGE-2: {rouge_greedy['rouge2']['f1']:.4f} / {rouge_beam['rouge2']['f1']:.4f}\n")
        f.write(f"  ROUGE-L: {rouge_greedy['rougeL']['f1']:.4f} / {rouge_beam['rougeL']['f1']:.4f}\n")
        f.write(f"  BLEU-4:  {bleu_greedy['bleu_4_combined']:.4f} / {bleu_beam['bleu_4_combined']:.4f}\n")
        f.write(f"  METEOR:  {meteor_greedy['meteor']:.4f} / {meteor_beam['meteor']:.4f}\n")
        f.write(f"  BERTScore F1: {bs_f1 if bs_f1 else 'N/A'}\n")
        f.write(f"  Perplexity: {ppl_results['perplexity']:.2f}\n\n")
        f.write("SAFETY:\n")
        f.write(f"  Negation conflicts: {neg_results['negation_conflict_rate']:.4f}\n")
        f.write(f"  Entity F1: {entity_results['entity_f1']:.4f}\n")
        f.write(f"  Hallucination rate: {halluc_results['hallucination_rate']:.4f}\n")
        f.write(f"  PHI leak rate: {phi_results['phi_leak_rate']:.4f}\n\n")
        f.write("QUALITY:\n")
        f.write(f"  Dup bigram: {rep_results['dup_bigram_rate']:.4f}\n")
        f.write(f"  Length ratio: {len_results['len_ratio']:.3f}\n")
        f.write(f"  Errors/100w: {grammar_results['errors_per_100_words']:.2f}\n")
        f.write(f"  Section coverage: {section_results.get('overall_coverage_pct', 0):.1f}%\n")
    
    flush(f"  Summary saved to: {summary_path}")
    flush(f"\n  Total evaluation time: {total_time:.0f}s ({total_time/60:.1f} min)")
    flush("=" * 80)
    flush("  EVALUATION COMPLETE")
    flush("=" * 80)


if __name__ == "__main__":
    main()
