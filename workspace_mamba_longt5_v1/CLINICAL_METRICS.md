# Clinical Text Summarization Metrics — Beyond ROUGE

> **Context:** Evaluating clinical note summarization models like our Mamba-Transformer Hybrid V2  
> **Problem:** ROUGE metrics have significant limitations for clinical text — they measure surface-level word overlap but miss medical accuracy, safety, and clinical utility  
> **This document:** Comprehensive guide to better evaluation metrics for clinical summarization

---

## Table of Contents

1. [Why ROUGE Is Insufficient for Clinical Text](#1-why-rouge-is-insufficient-for-clinical-text)
2. [Metric Categories](#2-metric-categories)
3. [Semantic Similarity Metrics](#3-semantic-similarity-metrics)
4. [Factual Consistency Metrics](#4-factual-consistency-metrics)
5. [Clinical-Specific Metrics](#5-clinical-specific-metrics)
6. [Safety Metrics](#6-safety-metrics)
7. [Readability & Fluency Metrics](#7-readability--fluency-metrics)
8. [Comprehensive Evaluation Framework](#8-comprehensive-evaluation-framework)
9. [Implementation Guide](#9-implementation-guide)
10. [Recommended Metric Suite](#10-recommended-metric-suite)
11. [References](#11-references)

---

## 1. Why ROUGE Is Insufficient for Clinical Text

### What ROUGE Measures

| Metric | What It Computes |
|---|---|
| **ROUGE-1** | Unigram overlap between generated and reference summary |
| **ROUGE-2** | Bigram overlap |
| **ROUGE-L** | Longest Common Subsequence (LCS) |

### Critical Limitations for Clinical Text

| Problem | Example |
|---|---|
| **Synonym blindness** | "HTN" vs "hypertension" vs "elevated blood pressure" → ROUGE sees these as different words |
| **Negation insensitivity** | "Patient **has** diabetes" vs "Patient **denies** diabetes" → high ROUGE overlap, opposite meaning |
| **Number insensitivity** | "Troponin **0.45**" vs "Troponin **4.50**" → nearly identical ROUGE, clinically catastrophic |
| **Multiple valid summaries** | Clinical notes can be correctly summarized in many different phrasings — ROUGE penalizes valid alternatives |
| **Safety blindness** | ROUGE cannot detect hallucinated medications, wrong dosages, or PHI leakage |
| **No factual grounding** | ROUGE doesn't check if the summary is **faithful** to the source document |

### Real-World Failure Case

```
Reference: "Patient denies chest pain. Started on metoprolol 25mg daily."
Generated: "Patient has chest pain. Started on metoprolol 250mg daily."
                     ↑ negation flip               ↑ 10x dosage error

ROUGE-1: 0.85  (very high!)
ROUGE-2: 0.71  (very high!)
ROUGE-L: 0.78  (very high!)

Clinical Safety: DANGEROUS — Two potentially fatal errors
```

**ROUGE gives a high score to a clinically dangerous summary.** This is why additional metrics are essential.

---

## 2. Metric Categories

```
┌─────────────────────────────────────────────────────────────┐
│              CLINICAL SUMMARIZATION METRICS                 │
├─────────────────────────┬───────────────────────────────────┤
│  SURFACE-LEVEL (Basic)  │  SEMANTIC (Better)                │
│  ├── ROUGE-1/2/L        │  ├── BERTScore                   │
│  ├── BLEU               │  ├── MoverScore                  │
│  └── METEOR             │  ├── BARTScore                   │
│                         │  └── MEDCON                      │
├─────────────────────────┼───────────────────────────────────┤
│  FACTUAL CONSISTENCY    │  CLINICAL-SPECIFIC               │
│  ├── FactCC             │  ├── Entity Recall/F1            │
│  ├── SummaC             │  ├── UMLS Concept Overlap        │
│  ├── QuestEval          │  ├── Negation Consistency         │
│  ├── AlignScore         │  ├── Dosage Accuracy             │
│  └── SumFactScore       │  ├── ICD Code Coverage           │
│                         │  └── Temporal Ordering           │
├─────────────────────────┼───────────────────────────────────┤
│  SAFETY                 │  READABILITY & FLUENCY           │
│  ├── PHI Leakage Check  │  ├── Perplexity (GPT-2/BioGPT)  │
│  ├── Hallucination Rate │  ├── Flesch-Kincaid Grade        │
│  ├── Contradiction Det. │  ├── Repetition Rate             │
│  └── Omission Analysis  │  └── Length Ratio                │
└─────────────────────────┴───────────────────────────────────┘
```

---

## 3. Semantic Similarity Metrics

### 3.1 BERTScore ⭐ (Highly Recommended)

**What it measures:** Semantic similarity using contextual embeddings, rather than exact word matching.

**How it works:**
1. Encode both generated and reference summaries with a pre-trained language model (e.g., Bio_ClinicalBERT).
2. Compute token-level cosine similarity between all pairs of tokens.
3. Greedily match each token in one text to its most similar token in the other.

$$\text{BERTScore}_{\text{Precision}} = \frac{1}{|\hat{y}|} \sum_{\hat{y}_j \in \hat{y}} \max_{y_i \in y} \cos(\mathbf{h}_{\hat{y}_j}, \mathbf{h}_{y_i})$$

$$\text{BERTScore}_{\text{Recall}} = \frac{1}{|y|} \sum_{y_i \in y} \max_{\hat{y}_j \in \hat{y}} \cos(\mathbf{h}_{y_i}, \mathbf{h}_{\hat{y}_j})$$

$$\text{BERTScore}_{F1} = 2 \cdot \frac{P \cdot R}{P + R}$$

**Why it's better for clinical text:**
- "HTN" and "hypertension" have similar embeddings → correctly recognized as equivalent
- Uses clinical encoders (Bio_ClinicalBERT, BiomedBERT) trained on biomedical text
- Captures semantic meaning, not just surface words

**Best encoder for clinical text:**
1. `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (best)
2. `emilyalsentzer/Bio_ClinicalBERT` (good alternative)

```python
# Already implemented in our clinical_eval.py
from bert_score import score as bscore
P, R, F = bscore(predictions, references, model_type="Bio_ClinicalBERT")
```

---

### 3.2 MoverScore

**What it measures:** Earth Mover's Distance (Wasserstein distance) between embedding distributions of generated and reference summaries.

**How it works:**
- Treats each text as a "distribution" of contextual word embeddings
- Computes the minimum "cost" of transforming one distribution into the other
- Combines word-level similarity with document-level alignment

**Advantage over BERTScore:** Considers global structure, not just greedy token matching. Better at detecting when summaries cover the same information in different order.

```python
# pip install moverscore
from moverscore_v2 import word_mover_score
scores = word_mover_score(refs, preds, n_gram=1, batch_size=32)
```

---

### 3.3 BARTScore

**What it measures:** The probability that a pre-trained BART model would generate the candidate text given the reference (or vice versa).

$$\text{BARTScore}(y, \hat{y}) = \sum_{t=1}^{|\hat{y}|} \log P(\hat{y}_t \mid \hat{y}_{<t}, y; \theta_{\text{BART}})$$

**Variants:**
| Variant | Measures |
|---|---|
| $\text{BARTScore}(y \to \hat{y})$ | Faithfulness (how well does the summary follow from reference?) |
| $\text{BARTScore}(\hat{y} \to y)$ | Adequacy (does the summary contain all reference info?) |
| $\text{BARTScore}(src \to \hat{y})$ | **Source faithfulness** (does the summary follow from the source?) |

The **source faithfulness** variant is especially good for clinical summarization — it checks if the generated summary is consistent with the original clinical note.

```python
# pip install BARTScore
from bart_score import BARTScorer
scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
scores = scorer.score(srcs, preds, batch_size=4)  # source → prediction
```

---

### 3.4 MEDCON (Medical Concept Coverage) ⭐ (Highly Recommended)

**What it measures:** Overlap of medical concepts (UMLS CUIs) between generated and reference summaries, using clinical NLP tools.

**How it works:**
1. Extract medical concepts from both texts using a clinical NER tool (QuickUMLS, MetaMap, or scispaCy)
2. Map concepts to UMLS Concept Unique Identifiers (CUIs)
3. Compute precision, recall, and F1 of concept overlap

$$\text{MEDCON}_{\text{F1}} = \frac{2 \cdot |\text{CUI}_{\text{pred}} \cap \text{CUI}_{\text{ref}}|}{|\text{CUI}_{\text{pred}}| + |\text{CUI}_{\text{ref}}|}$$

**Why it's excellent for clinical text:**
- Normalizes medical terminology: "MI", "myocardial infarction", "heart attack" all map to the same CUI
- Captures clinical meaning independent of phrasing
- Directly measures whether the right medical concepts are preserved

```python
# pip install quickumls scispacy
import scispacy
import spacy
from scispacy.linking import EntityLinker

nlp = spacy.load("en_core_sci_lg")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

def extract_cuis(text):
    doc = nlp(text)
    cuis = set()
    for ent in doc.ents:
        for cui in ent._.kb_ents:
            cuis.add(cui[0])  # CUI string
    return cuis

def medcon_f1(pred, ref):
    pred_cuis = extract_cuis(pred)
    ref_cuis = extract_cuis(ref)
    if not pred_cuis and not ref_cuis:
        return 1.0
    tp = len(pred_cuis & ref_cuis)
    prec = tp / max(len(pred_cuis), 1)
    rec = tp / max(len(ref_cuis), 1)
    return 2 * prec * rec / max(prec + rec, 1e-8)
```

---

## 4. Factual Consistency Metrics

### 4.1 SummaC ⭐ (Highly Recommended)

**What it measures:** Whether the generated summary is factually consistent with the source document, using Natural Language Inference (NLI).

**How it works:**
1. Split the summary into sentences.
2. For each sentence, compute NLI scores against sentences in the source document:
   - **Entailment**: Source supports the summary sentence ✓
   - **Contradiction**: Source contradicts the summary ✗
   - **Neutral**: No clear relationship
3. Aggregate scores using max (SummaC-Conv) or average binning.

**Why it matters:** Detects hallucinated medications, wrong diagnoses, and fabricated lab values that ROUGE completely misses.

```python
# pip install summac
from summac.model_summac import SummaCConv
model = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", device="cuda")
scores = model.score(sources, predictions)
# score["scores"] → list of consistency scores (0 to 1, higher = more consistent)
```

---

### 4.2 QuestEval

**What it measures:** Factual consistency via question generation and answering.

**How it works:**
1. Generate questions from the reference summary (e.g., "What medication was started?")
2. Try to answer these questions using the generated summary
3. If the answers match → factually consistent
4. Also generates questions from the source to check for omissions

**Advantage:** Interpretable — you can see exactly which questions the summary fails to answer.

```python
# pip install questeval
from questeval.questeval_metric import QuestEval
questeval = QuestEval(no_cuda=False)
scores = questeval.corpus_questeval(
    hypothesis=predictions,
    sources=sources,
    list_references=[references]
)
```

---

### 4.3 AlignScore

**What it measures:** Factual alignment between the source document and generated summary, combining information alignment with a unified model.

**How it works:**
- Uses a RoBERTa-based model fine-tuned on 4.7M alignment examples
- Evaluates whether each claim in the summary can be supported by the source
- More robust than NLI-based methods for long documents

```python
# pip install alignscore
from alignscore import AlignScore
scorer = AlignScore(model='roberta-large', batch_size=8, device='cuda',
                    ckpt_path='AlignScore-large.ckpt', evaluation_mode='nli_sp')
scores = scorer.score(contexts=sources, claims=predictions)
```

---

### 4.4 SumFactScore (Numeric Precision)

Custom metric specifically for clinical text that checks **numeric fact preservation**:

- Lab values: "WBC 12.3" in source → should appear correctly in summary
- Dosages: "metformin 500mg" → must be exact
- Vital signs: "BP 140/90" → cannot be altered

$$\text{NumericPrecision} = \frac{|\text{Numbers}_{\text{pred}} \cap \text{Numbers}_{\text{src}}|}{|\text{Numbers}_{\text{pred}}|}$$

This is already implemented in our `clinical_eval.py` as `numeric_conflict_rate`.

---

## 5. Clinical-Specific Metrics

### 5.1 Medical Entity F1 (UMLS-Enriched) ⭐ (Already Implemented)

Measures how well the summary preserves important medical entities from the reference:

| Entity Type | Examples |
|---|---|
| **Diagnoses** | diabetes, HTN, CAD, COPD, CHF |
| **Medications** | metformin, lisinopril, heparin |
| **Lab values** | WBC 12.3, Cr 1.8, Troponin 0.45 |
| **Dosages** | 500mg, 10 units, 2.5mg |
| **Procedures** | cardiac cath, intubation, PCI |

$$\text{Entity F1} = \frac{2 \cdot P \cdot R}{P + R}$$

where:

$$P = \frac{|\text{Entities}_{\text{pred}} \cap \text{Entities}_{\text{ref}}|}{|\text{Entities}_{\text{pred}}|}, \quad R = \frac{|\text{Entities}_{\text{pred}} \cap \text{Entities}_{\text{ref}}|}{|\text{Entities}_{\text{ref}}|}$$

**Already implemented** in our `clinical_eval.py` with regex-based extraction.

**Enhancement:** Use scispaCy or MetaMap for more robust entity extraction.

---

### 5.2 Negation Consistency ⭐ (Already Implemented)

Checks for dangerous **negation flips** between source and summary:

```
Source: "Patient denies chest pain"   ← negated
Summary: "Patient has chest pain"      ← affirmed
→ NEGATION FLIP DETECTED ✗
```

**Already implemented** in our `clinical_eval.py` as `check_negation_consistency()`.

---

### 5.3 ICD Code Coverage

**What it measures:** Whether the generated summary captures all diagnoses from the source, mapped to ICD-10 codes.

**How it works:**
1. Use a clinical NLP tool to extract ICD-10 codes from both the source and generated summary.
2. Compute recall: what percentage of source ICD codes are mentioned in the summary?

```python
# Using medcat for ICD-10 extraction
# pip install medcat
from medcat.cat import CAT
cat = CAT.load_model_pack('path/to/medcat_model')

def extract_icd_codes(text):
    doc = cat.get_entities(text)
    codes = set()
    for ent in doc['entities'].values():
        if ent.get('icd10'):
            for icd in ent['icd10']:
                codes.add(icd['code'])
    return codes
```

---

### 5.4 Temporal Ordering Score

Clinical summaries should follow a logical temporal sequence:
1. Presentation/admission
2. Workup/diagnostics
3. Treatment
4. Outcome/discharge plan

**How to measure:** Use temporal relation extraction to check if events in the summary maintain the same temporal order as in the source.

---

### 5.5 Clinical Coherence Score

**What it measures:** Whether the summary is logically coherent from a clinical perspective.

Uses a clinical language model (BioGPT, PMC-LLaMA, or Meditron) to score the perplexity of the generated summary:

$$\text{Coherence} = \exp\left(-\frac{1}{T}\sum_{t=1}^T \log P(y_t \mid y_{<t}; \theta_{\text{ClinicalLM}})\right)$$

Lower perplexity = more clinically coherent (the summary "reads like" typical clinical text).

---

## 6. Safety Metrics

### 6.1 PHI Leakage Detection ⭐ (Already Implemented)

Scans for Protected Health Information in generated summaries:
- Social Security Numbers
- Phone numbers
- Email addresses
- Medical Record Numbers (MRN)
- Physical addresses
- Patient names

**Already implemented** in our `clinical_eval.py` as `check_phi_leakage()`.

### 6.2 Hallucination Detection ⭐ (Already Implemented)

Detects entities in the summary that don't appear anywhere in the source document.

**Already implemented** as `check_factual_consistency_proxy()`.

### 6.3 Omission Analysis

Detects critical information in the reference that is **missing** from the generated summary:
- Active diagnoses not mentioned
- Current medications omitted
- Pending follow-up items missing

$$\text{Omission Rate} = 1 - \text{Entity Recall}$$

---

## 7. Readability & Fluency Metrics

### 7.1 Clinical Perplexity

Use a clinical language model to evaluate how natural the generated text sounds:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")

def clinical_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()
```

### 7.2 Repetition Rate ⭐ (Already Implemented)

Measures n-gram repetition (sign of degenerate generation):

$$\text{Rep}_n = 1 - \frac{|\text{unique n-grams}|}{|\text{total n-grams}|}$$

**Already implemented** in our `clinical_eval.py` as `check_repetition()`.

### 7.3 Length Ratio ⭐ (Already Implemented)

Measures whether summaries are appropriately concise:

$$\text{Length Ratio} = \frac{\text{avg\_pred\_length}}{\text{avg\_ref\_length}}$$

Ideal: 0.8–1.2 (within 20% of reference length).

---

## 8. Comprehensive Evaluation Framework

### Gated Evaluation Pipeline (Our Implementation)

Our `clinical_eval.py` already implements a **gated evaluation hierarchy**:

```
Gate 1: SAFETY (Hard Pass/Fail)
  ├── Negation consistency → error rate < 2%
  ├── PHI leakage → 0% tolerance
  ├── Factual consistency → contradiction rate < 15%
  └── Entity preservation → recall check
  
Gate 2: QUALITY (Score)
  ├── Repetition rates (bigram/trigram)
  ├── Length statistics
  └── Coherence check

Gate 3: SIMILARITY (Score)
  ├── ROUGE-1/2/L
  └── BERTScore (with clinical encoder)
```

### Proposed Enhanced Framework

```
Gate 1: SAFETY (Hard Pass/Fail) — MUST PASS
  ├── PHI leakage detection           [Already implemented]
  ├── Negation consistency             [Already implemented]
  ├── Hallucination detection          [Already implemented]
  ├── Dosage/numeric accuracy          [Already implemented]
  └── SummaC factual consistency       [NEW — recommended]

Gate 2: CLINICAL ACCURACY (Score ≥ 0.6)
  ├── MEDCON (UMLS concept F1)         [NEW — recommended]
  ├── Medical Entity F1                [Already implemented]
  ├── ICD-10 code coverage             [NEW — optional]
  └── Temporal ordering                [NEW — optional]

Gate 3: SEMANTIC QUALITY (Score ≥ 0.5)
  ├── BERTScore (BiomedBERT)           [Already implemented]
  ├── BARTScore (source faithfulness)  [NEW — recommended]
  ├── MoverScore                       [NEW — optional]
  └── Clinical perplexity (BioGPT)     [NEW — optional]

Gate 4: SURFACE OVERLAP (Informational)
  ├── ROUGE-1/2/L                      [Already implemented]
  ├── Repetition rates                 [Already implemented]
  └── Length ratio                     [Already implemented]
```

---

## 9. Implementation Guide

### Priority 1: Add to Existing Pipeline (Easy, High-Impact)

These can be added to `clinical_eval.py` with minimal effort:

#### SummaC (Factual Consistency)

```python
# pip install summac
def compute_summac(sources, preds):
    from summac.model_summac import SummaCConv
    model = SummaCConv(models=["vitc"], bins='percentile',
                       granularity="sentence", device="cuda")
    result = model.score(sources, preds)
    return {
        "summac_avg": sum(result["scores"]) / len(result["scores"]),
        "summac_scores": result["scores"]
    }
```

#### BARTScore (Source Faithfulness)

```python
# pip install git+https://github.com/neulab/BARTScore.git
def compute_bartscore(sources, preds):
    from bart_score import BARTScorer
    scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
    scores = scorer.score(sources, preds, batch_size=4)
    return {
        "bartscore_avg": sum(scores) / len(scores),
        "bartscore_per_sample": scores
    }
```

### Priority 2: Clinical NLP Integration (Medium Effort, Very High Impact)

#### MEDCON with scispaCy

```python
# pip install scispacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

import scispacy
import spacy

def compute_medcon(preds, refs):
    nlp = spacy.load("en_core_sci_lg")
    
    def extract_concepts(text):
        doc = nlp(text)
        return set(ent.text.lower() for ent in doc.ents)
    
    f1_scores = []
    for pred, ref in zip(preds, refs):
        pred_concepts = extract_concepts(pred)
        ref_concepts = extract_concepts(ref)
        
        if not ref_concepts:
            f1_scores.append(1.0)
            continue
        
        tp = len(pred_concepts & ref_concepts)
        prec = tp / max(len(pred_concepts), 1)
        rec = tp / max(len(ref_concepts), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        f1_scores.append(f1)
    
    return {"medcon_f1": sum(f1_scores) / len(f1_scores)}
```

### Priority 3: Advanced Metrics (Higher Effort)

- QuestEval, AlignScore, ICD-10 coverage — require model downloads and more compute.

---

## 10. Recommended Metric Suite

### Minimum Viable Evaluation (for Every Checkpoint)

| Metric | Type | Status | Why |
|---|---|---|---|
| **ROUGE-1/2/L** | Surface | ✅ Implemented | Baseline comparison with literature |
| **BERTScore** | Semantic | ✅ Implemented | Semantic similarity with clinical encoder |
| **Entity F1** | Clinical | ✅ Implemented | Medical entity preservation |
| **Negation Consistency** | Safety | ✅ Implemented | Catch dangerous negation flips |
| **PHI Leakage** | Safety | ✅ Implemented | Zero-tolerance for privacy violations |
| **Repetition Rate** | Quality | ✅ Implemented | Detect degenerate outputs |

### Recommended Additions (for Final Evaluation)

| Metric | Type | Priority | Install |
|---|---|---|---|
| **SummaC** | Factual Consistency | ⭐ HIGH | `pip install summac` |
| **MEDCON** | Clinical Accuracy | ⭐ HIGH | `pip install scispacy` + model |
| **BARTScore** | Source Faithfulness | ⭐ HIGH | `pip install bart-score` |
| **MoverScore** | Semantic | MEDIUM | `pip install moverscore` |
| **Clinical Perplexity** | Fluency | MEDIUM | `pip install transformers` (BioGPT) |
| **QuestEval** | Factual | LOW (compute heavy) | `pip install questeval` |

### Metric Comparison: What Each Captures

```
                    ROUGE  BERTScore  MEDCON  SummaC  BARTScore  Entity-F1
Synonym handling      ✗       ✓        ✓       ✓        ✓          ✗
Negation detection    ✗       ✗        ✗       ✓        ~          ✗*
Numeric accuracy      ✗       ✗        ✗       ✓        ~          ✓
Hallucination detect  ✗       ✗        ✗       ✓        ✓          ✗
Clinical concepts     ✗       ~        ✓       ✗        ✗          ✓
Source faithfulness    ✗       ✗        ✗       ✓        ✓          ✗
Speed (200 samples)   <1s     ~30s     ~60s    ~120s    ~60s       <1s

✗ = doesn't capture  ~ = partially captures  ✓ = captures well
* = captured by our separate negation checker
```

---

## 11. References

1. **BERTScore:** Zhang et al. "BERTScore: Evaluating Text Generation with BERT" (ICLR 2020)
2. **SummaC:** Laban et al. "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization" (TACL 2022)
3. **BARTScore:** Yuan et al. "BARTScore: Evaluating Generated Text as Text Generation" (NeurIPS 2021)
4. **QuestEval:** Scialom et al. "QuestEval: Summarization Asks for Fact-based Evaluation" (EMNLP 2021)
5. **MoverScore:** Zhao et al. "MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance" (EMNLP 2019)
6. **MEDCON:** Moramarco et al. "Human Evaluation and Correlation with Automatic Metrics in Consultation Note Generation" (ACL 2022)
7. **AlignScore:** Zha et al. "AlignScore: Evaluating Factual Consistency with a Unified Alignment Function" (ACL 2023)
8. **Clinical BERTScore:** Alsentzer et al. "Publicly Available Clinical BERT Embeddings" (NAACL Clinical NLP 2019)
9. **scispaCy:** Neumann et al. "ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing" (BioNLP 2019)
10. **BioGPT:** Luo et al. "BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining" (Bioinformatics 2022)
