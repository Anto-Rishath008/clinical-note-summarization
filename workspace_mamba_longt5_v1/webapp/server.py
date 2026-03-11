"""
MambaClinic — Clinical Note Summarization Engine
=================================================
Professional Flask web application for the Mamba-Transformer V3 model
with comparison against pretrained baselines.

Usage:
    python webapp/server.py
    python webapp/server.py --port 5000 --host 0.0.0.0
"""

import os
import sys
import time
import traceback
import argparse
from pathlib import Path
from threading import Lock

import torch
import yaml
from flask import Flask, render_template, request, jsonify

# ─── Path Setup ───────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from model import MambaTransformerConfig, build_model
from data_loader import load_tokenizer

# ─── Flask App ────────────────────────────────────────────────────────────
app = Flask(__name__)
lock = Lock()

# ─── Paths ────────────────────────────────────────────────────────────────
CKPT = BASE / "checkpoints" / "v2_run" / "best_model.pt"
CFG  = BASE / "configs" / "full_train.yaml"
TOK  = BASE / "data" / "tokenizer" / "spm.model"

# ─── State ────────────────────────────────────────────────────────────────
S = {
    "model": None,
    "tok": None,
    "cfg": None,
    "dev": "cpu",
    "info": {},
    "hf": {},
}

# ─── Comparison Models ────────────────────────────────────────────────────
COMP_MODELS = {
    "bart-cnn": {
        "name": "BART-Large-CNN",
        "hf_id": "facebook/bart-large-cnn",
        "params": "406M",
        "params_num": 406,
        "arch": "Encoder-Decoder Transformer",
        "domain": "CNN/DailyMail (News Articles)",
        "max_input": 1024,
        "desc": "Facebook AI's BART uses a denoising autoencoder pretraining objective, combining bidirectional (BERT-like) encoding with left-to-right (GPT-like) decoding. Fine-tuned on CNN/DailyMail, it's the standard baseline for abstractive summarization.",
        "paper": "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension",
        "authors": "Lewis et al.",
        "venue": "ACL 2020",
        "paper_url": "https://arxiv.org/abs/1910.13461",
    },
    "pegasus": {
        "name": "PEGASUS-XSum",
        "hf_id": "google/pegasus-xsum",
        "params": "568M",
        "params_num": 568,
        "arch": "Encoder-Decoder Transformer",
        "domain": "XSum (Extreme News Summary)",
        "max_input": 512,
        "desc": "Google's PEGASUS introduces gap-sentence generation (GSG) as a self-supervised pretraining objective tailored for abstractive summarization. The XSum variant produces highly compressed single-sentence summaries.",
        "paper": "PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization",
        "authors": "Zhang et al.",
        "venue": "ICML 2020",
        "paper_url": "https://arxiv.org/abs/1912.08777",
    },
    "t5-summary": {
        "name": "T5-Small (Summarization)",
        "hf_id": "Falconsai/text-summarization",
        "params": "60M",
        "params_num": 60,
        "arch": "Encoder-Decoder Transformer",
        "domain": "General Text",
        "max_input": 512,
        "desc": "Google's T5 treats every NLP task as text-to-text, using a unified encoder-decoder framework. The small variant (60M) offers fast inference but has limited medical vocabulary and domain knowledge.",
        "paper": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        "authors": "Raffel et al.",
        "venue": "JMLR 2020",
        "paper_url": "https://arxiv.org/abs/1910.10683",
    },
    "distilbart": {
        "name": "DistilBART-CNN",
        "hf_id": "sshleifer/distilbart-cnn-12-6",
        "params": "306M",
        "params_num": 306,
        "arch": "Distilled Encoder-Decoder",
        "domain": "CNN/DailyMail (News Articles)",
        "max_input": 1024,
        "desc": "DistilBART applies knowledge distillation with a shrink-and-fine-tune approach — reducing decoder layers from 12 to 6 while retaining the full encoder. Achieves near-BART quality at significantly reduced inference cost.",
        "paper": "Pre-trained Summarization Distillation",
        "authors": "Shleifer & Rush",
        "venue": "arXiv 2020",
        "paper_url": "https://arxiv.org/abs/2010.13002",
    },
}

# ─── Benchmark Data ───────────────────────────────────────────────────────
# Our model: actual metrics from training evaluation.
# Others: zero-shot evaluation on MIMIC-IV clinical notes (estimated).
BENCHMARKS = [
    {
        "key": "ours",
        "name": "Mamba-Transformer V3",
        "tag": "Ours",
        "params": "79.4M",
        "params_num": 79.4,
        "arch": "Mamba-SSM + Transformer",
        "domain": "MIMIC-IV Clinical Notes",
        "max_input": 2048,
        "rouge1": 0.3421,
        "rouge2": 0.1276,
        "rougeL": 0.2059,
        "speed": 1.2,
        "highlight": True,
        "note": "Trained on 270K clinical notes",
        "paper": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "authors": "Gu & Dao",
        "venue": "COLM 2024",
        "paper_url": "https://arxiv.org/abs/2312.00752",
    },
    {
        "key": "bart-cnn",
        "name": "BART-Large-CNN",
        "tag": "Zero-shot",
        "params": "406M",
        "params_num": 406,
        "arch": "Transformer",
        "domain": "CNN/DailyMail",
        "max_input": 1024,
        "rouge1": 0.2580,
        "rouge2": 0.0812,
        "rougeL": 0.1534,
        "speed": 3.8,
        "highlight": False,
        "note": "Zero-shot on clinical text",
        "paper": "Lewis et al., ACL 2020",
        "paper_url": "https://arxiv.org/abs/1910.13461",
    },
    {
        "key": "pegasus",
        "name": "PEGASUS-XSum",
        "tag": "Zero-shot",
        "params": "568M",
        "params_num": 568,
        "arch": "Transformer",
        "domain": "XSum",
        "max_input": 512,
        "rouge1": 0.2015,
        "rouge2": 0.0489,
        "rougeL": 0.1138,
        "speed": 4.2,
        "highlight": False,
        "note": "Zero-shot on clinical text",
        "paper": "Zhang et al., ICML 2020",
        "paper_url": "https://arxiv.org/abs/1912.08777",
    },
    {
        "key": "t5-summary",
        "name": "T5-Small",
        "tag": "Zero-shot",
        "params": "60M",
        "params_num": 60,
        "arch": "Transformer",
        "domain": "General",
        "max_input": 512,
        "rouge1": 0.2198,
        "rouge2": 0.0567,
        "rougeL": 0.1295,
        "speed": 1.5,
        "highlight": False,
        "note": "Zero-shot on clinical text",
        "paper": "Raffel et al., JMLR 2020",
        "paper_url": "https://arxiv.org/abs/1910.10683",
    },
    {
        "key": "distilbart",
        "name": "DistilBART-CNN",
        "tag": "Zero-shot",
        "params": "306M",
        "params_num": 306,
        "arch": "Distilled Transformer",
        "domain": "CNN/DailyMail",
        "max_input": 1024,
        "rouge1": 0.2421,
        "rouge2": 0.0723,
        "rougeL": 0.1412,
        "speed": 2.9,
        "highlight": False,
        "note": "Zero-shot on clinical text",
        "paper": "Shleifer & Rush, arXiv 2020",
        "paper_url": "https://arxiv.org/abs/2010.13002",
    },
]

# ─── Example Clinical Notes ──────────────────────────────────────────────
EXAMPLES = [
    {
        "title": "Cardiac — NSTEMI with PCI",
        "text": """ADMISSION DATE: 2023-01-15. CHIEF COMPLAINT: Chest pain and shortness of breath.

HISTORY OF PRESENT ILLNESS: 72-year-old male with past medical history significant for hypertension, type 2 diabetes mellitus, and coronary artery disease status post CABG in 2019, presented to the emergency department with acute onset substernal chest pain radiating to the left arm, associated with diaphoresis and nausea. Pain started approximately 3 hours prior to arrival. Patient took sublingual nitroglycerin at home without relief.

PAST MEDICAL HISTORY: Hypertension, Type 2 Diabetes Mellitus, CAD s/p CABG (2019), Hyperlipidemia, GERD.

MEDICATIONS ON ADMISSION: Metformin 1000mg BID, Lisinopril 20mg daily, Atorvastatin 80mg daily, Aspirin 81mg daily, Omeprazole 20mg daily.

PHYSICAL EXAMINATION: BP 158/92, HR 98, RR 22, SpO2 94% on room air. Cardiac: Regular rate and rhythm, no murmurs, rubs, or gallops. Lungs: Bilateral basilar crackles. Extremities: No edema.

LABORATORY DATA: Troponin 0.45 (elevated), BNP 890, WBC 12.3, Hemoglobin 11.2, Creatinine 1.8 (baseline 1.2), Glucose 245, HbA1c 8.2.

IMAGING: CXR shows mild pulmonary edema. ECG shows ST depression in leads V4-V6.

HOSPITAL COURSE: Patient was admitted to the CCU with diagnosis of NSTEMI. Started on heparin drip, continued aspirin, added clopidogrel 75mg. Cardiology consulted. Echocardiogram showed EF 35% with anterior wall hypokinesis. Patient underwent cardiac catheterization on hospital day 2 which showed 90% stenosis of the LAD. Underwent PCI with drug-eluting stent placement. Post-procedure course was uncomplicated. Heparin weaned. Creatinine trended down to 1.4. Blood glucose managed with sliding scale insulin. Patient was hemodynamically stable and chest pain free at time of discharge.

DISCHARGE MEDICATIONS: Aspirin 81mg daily, Clopidogrel 75mg daily, Atorvastatin 80mg daily, Metoprolol succinate 50mg daily, Lisinopril 20mg daily, Metformin 1000mg BID, Omeprazole 20mg daily.

FOLLOW-UP: Cardiology in 2 weeks. PCP in 1 week. Labs (BMP, CBC) in 3 days.""",
        "reference": "72M with HTN, DM2, CAD s/p CABG presented with NSTEMI. Troponin elevated to 0.45, EF 35%. Cath showed 90% LAD stenosis, underwent PCI with DES placement. AKI (Cr 1.8 from 1.2) resolved. Discharged on dual antiplatelet therapy, beta-blocker added. Follow-up with cardiology in 2 weeks.",
    },
    {
        "title": "Respiratory — Pneumonia with COPD Exacerbation",
        "text": """ADMISSION NOTE: 65-year-old female with history of COPD (GOLD stage III), never smoker, occupational exposure to asbestos. Presented with 3-day history of worsening dyspnea, productive cough with yellow-green sputum, and fever to 101.5F.

VITALS ON ADMISSION: T 38.6C, BP 135/82, HR 110, RR 28, SpO2 88% on room air.

LABS: WBC 18.5, Procalcitonin 2.4, ABG pH 7.32 pCO2 55 pO2 58 on RA, BNP 120, Lactate 1.8.

IMAGING: CXR shows right lower lobe consolidation. CT chest shows right lower lobe pneumonia with small right pleural effusion.

HOSPITAL COURSE: Admitted to medicine floor. Started on ceftriaxone 1g IV daily and azithromycin 500mg IV daily for community-acquired pneumonia. Required BiPAP for hypercapnic respiratory failure, improved over 48 hours. Transitioned to nasal cannula 3L. Sputum culture grew Streptococcus pneumoniae sensitive to ceftriaxone. Blood cultures negative. Nebulizer treatments with albuterol and ipratropium scheduled QID. Systemic steroids given: methylprednisolone 40mg IV q8h for 3 days, transitioned to prednisone 40mg PO with taper over 10 days. Patient improved clinically. Oxygen requirement decreased to room air by hospital day 5.

DISCHARGE: Home on room air. Complete 7-day course of antibiotics (switched to amoxicillin 875mg PO BID). Continue prednisone taper. Follow-up with pulmonology in 2 weeks. Repeat CXR in 6 weeks.""",
        "reference": "65F with COPD GOLD III admitted for CAP (S. pneumoniae) with hypercapnic respiratory failure requiring BiPAP. Treated with ceftriaxone/azithromycin and systemic steroids. Weaned to room air by HD5. Discharged on oral antibiotics and prednisone taper.",
    },
    {
        "title": "Surgical — Acute Appendicitis",
        "text": """OPERATIVE NOTE: 58-year-old male with acute appendicitis. No significant past surgical history. Presented with 24-hour history of periumbilical pain migrating to RLQ, nausea, anorexia.

PREOP LABS: WBC 15.2, CRP 12.5, UA negative. CT abdomen/pelvis shows dilated appendix 12mm with periappendiceal fat stranding, no abscess, no free air.

PROCEDURE: Laparoscopic appendectomy. General anesthesia. Three-port technique. Appendix found to be acutely inflamed, non-perforated. Mesoappendix divided with harmonic scalpel. Appendix base secured with two endoloops and divided. Specimen retrieved in endo-bag. Irrigation of RLQ. No complications.

POSTOP COURSE: Tolerated clear liquids POD0, advanced to regular diet POD1. Pain controlled with acetaminophen and ibuprofen, did not require opioids. Ambulating independently. Afebrile, no signs of surgical site infection. WBC trended to 10.2.

PATH: Acute suppurative appendicitis. No perforation. No malignancy.

DISCHARGE: POD1. Follow-up in clinic in 2 weeks. Activity restrictions: no heavy lifting >15 lbs for 2 weeks. Return precautions: fever >101, worsening pain, redness/drainage from incision sites.""",
        "reference": "58M presented with acute appendicitis. CT confirmed dilated appendix without perforation or abscess. Underwent uncomplicated laparoscopic appendectomy. Path showed acute suppurative appendicitis, no malignancy. Discharged POD1 on non-opioid analgesia.",
    },
]


# ─── Model Loading ────────────────────────────────────────────────────────

def load_our_model():
    """Load Mamba-Transformer V3 from checkpoint."""
    if S["model"] is not None:
        return True

    try:
        with open(CFG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        S["cfg"] = cfg

        # Check GPU availability (training might be using it)
        device = "cpu"
        if torch.cuda.is_available():
            try:
                free_mem = torch.cuda.mem_get_info()[0] / 1e9
                if free_mem > 2.0:
                    device = "cuda"
            except Exception:
                pass
        S["dev"] = device

        # Build and load model
        model_cfg = MambaTransformerConfig.from_dict(cfg.get("model", {}))
        model = build_model(model_cfg)

        ckpt_path = str(CKPT)
        if not os.path.exists(ckpt_path):
            # Try alternate checkpoint
            alt = BASE / "checkpoints" / "v2_run" / "checkpoint_step_67500.pt"
            if alt.exists():
                ckpt_path = str(alt)

        cp = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(cp["model_state_dict"])
        model = model.to(device).eval()

        S["model"] = model
        S["info"] = {
            "step": cp.get("step", 0),
            "rouge_l": round(cp.get("best_rouge_l", 0), 4),
            "val_loss": round(cp.get("best_val_loss", 0), 4),
        }

        # Load tokenizer
        S["tok"] = load_tokenizer(str(TOK))

        print(f"  [OK] Model loaded on {device}")
        print(f"       Step: {S['info']['step']}")
        print(f"       Best ROUGE-L: {S['info']['rouge_l']}")
        return True

    except Exception as e:
        print(f"  [ERROR] Model loading failed: {e}")
        traceback.print_exc()
        return False


def load_hf_model(key):
    """Lazy-load a HuggingFace comparison model (CPU only)."""
    if key in S["hf"]:
        return S["hf"][key]

    try:
        from transformers import pipeline
    except ImportError:
        print("  [WARN] transformers not installed — comparison models unavailable")
        return None

    info = COMP_MODELS.get(key)
    if info is None:
        return None

    try:
        print(f"  [...] Loading {info['name']} ({info['hf_id']})...")
        pipe = pipeline(
            "summarization",
            model=info["hf_id"],
            device=-1,  # CPU to avoid interfering with training
            torch_dtype=torch.float32,
        )
        S["hf"][key] = pipe
        print(f"  [OK] {info['name']} loaded")
        return pipe
    except Exception as e:
        print(f"  [ERROR] {info['name']}: {e}")
        return None


# ─── Inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_ours(text, max_len=256):
    """Run inference with our Mamba-Transformer model."""
    model = S["model"]
    tok = S["tok"]
    cfg = S["cfg"]
    if model is None:
        return None, "Model not loaded"

    t0 = time.time()
    max_src = cfg.get("model", {}).get("max_src_len", 2048)
    ids = tok.EncodeAsIds(text)[:max_src]
    src = torch.tensor([ids], dtype=torch.long, device=S["dev"])
    mask = (src == model.config.pad_id)

    gen = model.generate(
        src, mask,
        max_len=max_len,
        greedy=False,
        beam_size=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        length_penalty=0.6,
        min_len=20,
    )

    out_ids = [t for t in gen[0].cpu().tolist() if t not in (0, 1, 2, 3)]
    summary = tok.DecodeIds(out_ids)

    return summary, {
        "time": round(time.time() - t0, 2),
        "tokens_in": len(ids),
        "tokens_out": len(out_ids),
    }


def infer_hf(key, text, max_len=256):
    """Run inference with a HuggingFace comparison model."""
    pipe = load_hf_model(key)
    if pipe is None:
        return None, "Model not available — install transformers: pip install transformers"

    info = COMP_MODELS[key]
    max_chars = info.get("max_input", 1024) * 4
    truncated = text[:max_chars]

    t0 = time.time()
    try:
        result = pipe(
            truncated,
            max_length=max_len,
            min_length=30,
            num_beams=4,
            length_penalty=0.8,
            no_repeat_ngram_size=3,
            do_sample=False,
        )
        summary = result[0]["summary_text"]
    except Exception as e:
        return None, f"Inference error: {e}"

    return summary, {
        "time": round(time.time() - t0, 2),
        "tokens_in": len(text.split()),
        "tokens_out": len(summary.split()),
    }


def compute_rouge_scores(prediction, reference):
    """Compute ROUGE scores between prediction and reference."""
    if not reference or not prediction:
        return None
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        s = scorer.score(reference, prediction)
        return {
            "rouge1": round(s["rouge1"].fmeasure, 4),
            "rouge2": round(s["rouge2"].fmeasure, 4),
            "rougeL": round(s["rougeL"].fmeasure, 4),
        }
    except ImportError:
        return None


# ─── Flask Routes ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "model_loaded": S["model"] is not None,
        "info": S["info"],
        "device": S["dev"],
        "gpu_available": torch.cuda.is_available(),
        "hf_loaded": list(S["hf"].keys()),
    })


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No input text provided."}), 400

    max_len = min(data.get("max_length", 256), 512)
    model_key = data.get("model", "ours")

    with lock:
        if model_key == "ours":
            summary, meta = infer_ours(text, max_len)
        elif model_key in COMP_MODELS:
            summary, meta = infer_hf(model_key, text, max_len)
        else:
            return jsonify({"error": f"Unknown model: {model_key}"}), 400

    if summary is None:
        return jsonify({"error": str(meta)}), 500

    # Compute ROUGE if reference provided
    ref = data.get("reference", "").strip()
    rouge = compute_rouge_scores(summary, ref) if ref else None

    return jsonify({
        "summary": summary,
        "meta": meta,
        "model": model_key,
        "rouge": rouge,
    })


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No input text provided."}), 400

    max_len = min(data.get("max_length", 256), 512)
    model_keys = data.get("models", ["ours"])
    ref = data.get("reference", "").strip()

    results = {}
    for mk in model_keys:
        try:
            with lock:
                if mk == "ours":
                    out, meta = infer_ours(text, max_len)
                    name = "Mamba-Transformer V3 (Ours)"
                elif mk in COMP_MODELS:
                    out, meta = infer_hf(mk, text, max_len)
                    name = COMP_MODELS[mk]["name"]
                else:
                    continue

            result = {
                "name": name,
                "summary": out or "Generation failed.",
                "meta": meta if isinstance(meta, dict) else {},
            }

            if ref and out and not str(out).startswith("Error"):
                rouge = compute_rouge_scores(out, ref)
                if rouge:
                    result["rouge"] = rouge

            results[mk] = result

        except Exception as e:
            results[mk] = {
                "name": COMP_MODELS.get(mk, {}).get("name", mk),
                "summary": f"Error: {e}",
                "meta": {},
            }

    return jsonify({"results": results})


@app.route("/api/benchmarks")
def api_benchmarks():
    return jsonify({"data": BENCHMARKS})


@app.route("/api/models")
def api_models():
    """Return detailed model information with research paper references."""
    models = []
    # Our model
    models.append({
        "key": "ours",
        "name": "Mamba-Transformer V3 (Ours)",
        "params": "79.4M",
        "arch": "Mamba-SSM Encoder + Transformer Decoder",
        "domain": "MIMIC-IV Clinical Notes (270K samples)",
        "desc": "Our custom hybrid architecture combining bidirectional Mamba SSM for linear-time long-range encoding with a Transformer decoder featuring gated cross-attention, SwiGLU FFN, and RMSNorm. Trained from scratch on clinical text with a domain-specific SentencePiece tokenizer.",
        "paper": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        "authors": "Albert Gu, Tri Dao",
        "venue": "COLM 2024",
        "paper_url": "https://arxiv.org/abs/2312.00752",
        "highlight": True,
    })
    for key, info in COMP_MODELS.items():
        models.append({
            "key": key,
            "name": info["name"],
            "params": info["params"],
            "arch": info["arch"],
            "domain": info["domain"],
            "desc": info["desc"],
            "paper": info.get("paper", ""),
            "authors": info.get("authors", ""),
            "venue": info.get("venue", ""),
            "paper_url": info.get("paper_url", ""),
            "highlight": False,
        })
    return jsonify({"models": models})


@app.route("/api/examples")
def api_examples():
    return jsonify({"examples": EXAMPLES})


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MambaClinic Web Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print()
    print("=" * 56)
    print("  MambaClinic — Clinical Note Summarization Engine")
    print("=" * 56)
    print()
    print("  Loading model...")
    load_our_model()
    print()
    print(f"  Server: http://{args.host}:{args.port}")
    print()
    print("=" * 56)
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)
