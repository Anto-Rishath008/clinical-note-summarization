"""
Clinical Note Summarization — Web Application
==============================================

A Gradio-based web UI for the Mamba-Transformer Hybrid V2 model.

Features:
  - Text input: Paste or type clinical notes
  - Speech input: Speak clinical notes via microphone (uses browser speech-to-text)
  - Model inference: Generates summaries using the trained model
  - Configurable generation: Adjust max length, beam size, temperature
  - Example inputs: Pre-loaded clinical note examples

Usage:
  python app.py --checkpoint checkpoints/v2_run/best_model.pt --config configs/full_train.yaml

Requirements:
  pip install gradio>=4.0.0
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple

warnings.filterwarnings("ignore")

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import MambaTransformerConfig, MambaTransformerModel, build_model
from data_loader import load_tokenizer


# ─────────────────────────── Global State ──────────────────────────────────

MODEL: Optional[MambaTransformerModel] = None
TOKENIZER = None
DEVICE = None
CONFIG = None


# ─────────────────────────── Model Loading ─────────────────────────────────

def load_model(checkpoint_path: str, config_path: str = None):
    """Load model from checkpoint."""
    global MODEL, TOKENIZER, DEVICE, CONFIG

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    cp = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    cfg = cp.get("config", {})
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    CONFIG = cfg

    # Build model
    model_cfg = MambaTransformerConfig.from_dict(cfg.get("model", {}))
    MODEL = build_model(model_cfg)
    MODEL.load_state_dict(cp["model_state_dict"])
    MODEL = MODEL.to(DEVICE).eval()

    step = cp.get("step", 0)
    best_rl = cp.get("best_rouge_l", 0.0)
    print(f"Model loaded — Step: {step} | Best ROUGE-L: {best_rl:.4f}")

    # Load tokenizer
    data_cfg = cfg.get("data", {})
    tokenizer_path = data_cfg.get("tokenizer_path", "data/tokenizer/spm.model")
    TOKENIZER = load_tokenizer(tokenizer_path)
    print(f"Tokenizer loaded — vocab size: {TOKENIZER.GetPieceSize()}")


# ─────────────────────────── Inference ─────────────────────────────────────

def summarize(
    text: str,
    max_length: int = 256,
    decoding_strategy: str = "Greedy",
    beam_size: int = 4,
    temperature: float = 1.0,
    no_repeat_ngram: int = 3,
    repetition_penalty: float = 1.0,
    min_length: int = 20,
) -> Tuple[str, str]:
    """Generate a clinical summary from input text."""
    if MODEL is None:
        return "⚠️ Model not loaded. Please restart with --checkpoint.", ""

    if not text or not text.strip():
        return "⚠️ Please enter some clinical text to summarize.", ""

    text = text.strip()
    start_time = time.time()

    # Tokenize
    max_src_len = CONFIG.get("model", {}).get("max_src_len", 4096)
    src_ids = TOKENIZER.EncodeAsIds(text)[:max_src_len]

    src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    src_mask = (src == MODEL.config.pad_id)

    # Generate
    with torch.no_grad():
        if decoding_strategy == "Beam Search":
            # Import beam search from clinical_eval
            try:
                from clinical_eval import generate_beam
                generated = generate_beam(
                    MODEL, src, src_mask,
                    beam_size=beam_size,
                    max_len=max_length,
                    no_repeat_ngram_size=no_repeat_ngram,
                    min_len=min_length,
                    length_penalty=0.6,
                )
            except ImportError:
                # Fallback to greedy if beam search not available
                generated = MODEL.generate(
                    src, src_mask,
                    max_len=max_length,
                    greedy=True,
                    no_repeat_ngram_size=no_repeat_ngram,
                    repetition_penalty=repetition_penalty,
                )
        elif decoding_strategy == "Sampling":
            generated = MODEL.generate(
                src, src_mask,
                max_len=max_length,
                greedy=False,
                temperature=temperature,
                top_k=50,
                no_repeat_ngram_size=no_repeat_ngram,
                repetition_penalty=repetition_penalty,
            )
        else:  # Greedy
            generated = MODEL.generate(
                src, src_mask,
                max_len=max_length,
                greedy=True,
                no_repeat_ngram_size=no_repeat_ngram,
                repetition_penalty=repetition_penalty,
            )

    # Decode output
    gen_ids = generated[0].cpu().tolist()
    # Remove special tokens
    gen_ids_clean = [t for t in gen_ids if t not in [0, 1, 2, 3]]
    summary = TOKENIZER.DecodeIds(gen_ids_clean)

    elapsed = time.time() - start_time

    # Build metadata
    src_tokens = len(src_ids)
    out_tokens = len(gen_ids_clean)
    meta = (
        f"⏱️ {elapsed:.2f}s | "
        f"📥 {src_tokens} source tokens | "
        f"📤 {out_tokens} output tokens | "
        f"🔧 {decoding_strategy}"
    )
    if decoding_strategy == "Beam Search":
        meta += f" (beam={beam_size})"
    elif decoding_strategy == "Sampling":
        meta += f" (temp={temperature})"

    return summary, meta


# ─────────────────────────── Example Inputs ────────────────────────────────

EXAMPLE_NOTES = [
    # Example 1: Cardiac
    """ADMISSION DATE: 2023-01-15. CHIEF COMPLAINT: Chest pain and shortness of breath.

HISTORY OF PRESENT ILLNESS: 72-year-old male with past medical history significant for hypertension, type 2 diabetes mellitus, and coronary artery disease status post CABG in 2019, presented to the emergency department with acute onset substernal chest pain radiating to the left arm, associated with diaphoresis and nausea. Pain started approximately 3 hours prior to arrival. Patient took sublingual nitroglycerin at home without relief.

PAST MEDICAL HISTORY: Hypertension, Type 2 Diabetes Mellitus, CAD s/p CABG (2019), Hyperlipidemia, GERD.

MEDICATIONS ON ADMISSION: Metformin 1000mg BID, Lisinopril 20mg daily, Atorvastatin 80mg daily, Aspirin 81mg daily, Omeprazole 20mg daily.

PHYSICAL EXAMINATION: BP 158/92, HR 98, RR 22, SpO2 94% on room air. Cardiac: Regular rate and rhythm, no murmurs, rubs, or gallops. Lungs: Bilateral basilar crackles. Extremities: No edema.

LABORATORY DATA: Troponin 0.45 (elevated), BNP 890, WBC 12.3, Hemoglobin 11.2, Creatinine 1.8 (baseline 1.2), Glucose 245, HbA1c 8.2.

IMAGING: CXR shows mild pulmonary edema. ECG shows ST depression in leads V4-V6.

HOSPITAL COURSE: Patient was admitted to the CCU with diagnosis of NSTEMI. Started on heparin drip, continued aspirin, added clopidogrel 75mg. Cardiology consulted. Echocardiogram showed EF 35% with anterior wall hypokinesis. Patient underwent cardiac catheterization on hospital day 2 which showed 90% stenosis of the LAD. Underwent PCI with drug-eluting stent placement. Post-procedure course was uncomplicated. Heparin weaned. Creatinine trended down to 1.4. Blood glucose managed with sliding scale insulin. Patient was hemodynamically stable and chest pain free at time of discharge.

DISCHARGE MEDICATIONS: Aspirin 81mg daily, Clopidogrel 75mg daily, Atorvastatin 80mg daily, Metoprolol succinate 50mg daily, Lisinopril 20mg daily, Metformin 1000mg BID, Omeprazole 20mg daily.

FOLLOW-UP: Cardiology in 2 weeks. PCP in 1 week. Labs (BMP, CBC) in 3 days.""",

    # Example 2: Respiratory
    """ADMISSION NOTE: 65-year-old female with history of COPD (GOLD stage III), never smoker, occupational exposure to asbestos. Presented with 3-day history of worsening dyspnea, productive cough with yellow-green sputum, and fever to 101.5F.

VITALS ON ADMISSION: T 38.6C, BP 135/82, HR 110, RR 28, SpO2 88% on room air.

LABS: WBC 18.5, Procalcitonin 2.4, ABG pH 7.32 pCO2 55 pO2 58 on RA, BNP 120, Lactate 1.8.

IMAGING: CXR shows right lower lobe consolidation. CT chest shows right lower lobe pneumonia with small right pleural effusion.

HOSPITAL COURSE: Admitted to medicine floor. Started on ceftriaxone 1g IV daily and azithromycin 500mg IV daily for community-acquired pneumonia. Required BiPAP for hypercapnic respiratory failure, improved over 48 hours. Transitioned to nasal cannula 3L. Sputum culture grew Streptococcus pneumoniae sensitive to ceftriaxone. Blood cultures negative. Nebulizer treatments with albuterol and ipratropium scheduled QID. Systemic steroids given: methylprednisolone 40mg IV q8h for 3 days, transitioned to prednisone 40mg PO with taper over 10 days. Patient improved clinically. Oxygen requirement decreased to room air by hospital day 5.

DISCHARGE: Home on room air. Complete 7-day course of antibiotics (switched to amoxicillin 875mg PO BID). Continue prednisone taper. Follow-up with pulmonology in 2 weeks. Repeat CXR in 6 weeks.""",

    # Example 3: Surgical
    """OPERATIVE NOTE: 58-year-old male with acute appendicitis. No significant past surgical history. Presented with 24-hour history of periumbilical pain migrating to RLQ, nausea, anorexia.

PREOP LABS: WBC 15.2, CRP 12.5, UA negative. CT abdomen/pelvis shows dilated appendix 12mm with periappendiceal fat stranding, no abscess, no free air.

PROCEDURE: Laparoscopic appendectomy. General anesthesia. Three-port technique. Appendix found to be acutely inflamed, non-perforated. Mesoappendix divided with harmonic scalpel. Appendix base secured with two endoloops and divided. Specimen retrieved in endo-bag. Irrigation of RLQ. No complications.

POSTOP COURSE: Tolerated clear liquids POD0, advanced to regular diet POD1. Pain controlled with acetaminophen and ibuprofen, did not require opioids. Ambulating independently. Afebrile, no signs of surgical site infection. WBC trended to 10.2.

PATH: Acute suppurative appendicitis. No perforation. No malignancy.

DISCHARGE: POD1. Follow-up in clinic in 2 weeks. Activity restrictions: no heavy lifting >15 lbs for 2 weeks. Return precautions: fever >101, worsening pain, redness/drainage from incision sites.""",
]


# ─────────────────────────── Gradio UI ─────────────────────────────────────

def build_ui():
    """Build the Gradio web interface."""
    try:
        import gradio as gr
    except ImportError:
        print("ERROR: Gradio not installed. Run: pip install gradio>=4.0.0")
        sys.exit(1)

    # Custom CSS
    css = """
    .main-title {
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 1em;
    }
    .output-box {
        min-height: 150px;
    }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="Clinical Note Summarizer — Mamba-Transformer V2",
        css=css,
        theme=gr.themes.Soft(),
    ) as demo:

        # ── Header ──
        gr.Markdown(
            """
            # 🏥 Clinical Note Summarizer
            ### Mamba-Transformer Hybrid V2 — MIMIC-IV Trained
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(
                    "**Input a clinical note** via text or speech, and the model will generate a "
                    "Brief Hospital Course (BHC) summary."
                )
            with gr.Column(scale=1):
                device_info = "🟢 GPU" if torch.cuda.is_available() else "🟡 CPU"
                gr.Markdown(f"**Device:** {device_info}")

        gr.Markdown("---")

        # ── Input Section ──
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input Clinical Note")

                # Text input
                input_text = gr.Textbox(
                    label="Clinical Note Text",
                    placeholder="Paste or type a clinical note here...\n\nOr use the microphone below to dictate.",
                    lines=12,
                    max_lines=30,
                )

                # Speech input (microphone)
                audio_input = gr.Audio(
                    label="🎤 Speech Input (Dictate Clinical Note)",
                    sources=["microphone"],
                    type="filepath",
                )

                # Transcribe button
                transcribe_btn = gr.Button(
                    "🎤 Transcribe Audio to Text",
                    variant="secondary",
                    size="sm",
                )

                # Example selector
                gr.Markdown("**📋 Or try an example:**")
                example_dropdown = gr.Dropdown(
                    choices=[
                        "Example 1: Cardiac (NSTEMI, PCI)",
                        "Example 2: Respiratory (Pneumonia, COPD)",
                        "Example 3: Surgical (Appendectomy)",
                    ],
                    label="Load Example",
                    value=None,
                )

        # ── Settings ──
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚙️ Generation Settings")
                with gr.Row():
                    max_length = gr.Slider(
                        minimum=50, maximum=512, value=256, step=10,
                        label="Max Output Length (tokens)",
                    )
                    decoding = gr.Radio(
                        choices=["Greedy", "Beam Search", "Sampling"],
                        value="Greedy",
                        label="Decoding Strategy",
                    )

                with gr.Row():
                    beam_size = gr.Slider(
                        minimum=2, maximum=8, value=4, step=1,
                        label="Beam Size (for Beam Search)",
                        visible=False,
                    )
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                        label="Temperature (for Sampling)",
                        visible=False,
                    )
                    no_repeat = gr.Slider(
                        minimum=0, maximum=6, value=3, step=1,
                        label="No-Repeat N-gram Size",
                    )
                    rep_penalty = gr.Slider(
                        minimum=1.0, maximum=2.0, value=1.0, step=0.1,
                        label="Repetition Penalty",
                    )

        # ── Generate Button ──
        gr.Markdown("---")
        generate_btn = gr.Button(
            "🚀 Generate Summary",
            variant="primary",
            size="lg",
        )

        # ── Output Section ──
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📋 Generated Summary")
                output_text = gr.Textbox(
                    label="Brief Hospital Course Summary",
                    lines=8,
                    max_lines=20,
                    show_copy_button=True,
                    elem_classes=["output-box"],
                )
                output_meta = gr.Textbox(
                    label="Generation Info",
                    lines=1,
                    interactive=False,
                )

        # ── Event Handlers ──

        # Show/hide settings based on decoding strategy
        def update_visibility(strategy):
            return (
                gr.update(visible=(strategy == "Beam Search")),
                gr.update(visible=(strategy == "Sampling")),
            )

        decoding.change(
            fn=update_visibility,
            inputs=decoding,
            outputs=[beam_size, temperature],
        )

        # Load example
        def load_example(choice):
            if choice and "Example 1" in choice:
                return EXAMPLE_NOTES[0]
            elif choice and "Example 2" in choice:
                return EXAMPLE_NOTES[1]
            elif choice and "Example 3" in choice:
                return EXAMPLE_NOTES[2]
            return ""

        example_dropdown.change(
            fn=load_example,
            inputs=example_dropdown,
            outputs=input_text,
        )

        # Transcribe audio
        def transcribe_audio(audio_path):
            if audio_path is None:
                return "⚠️ No audio recorded. Please use the microphone first."

            try:
                import whisper
            except ImportError:
                try:
                    # Try using SpeechRecognition as fallback
                    import speech_recognition as sr
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(audio_path) as source:
                        audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)
                    return text
                except ImportError:
                    return (
                        "⚠️ Speech recognition not available.\n"
                        "Install one of:\n"
                        "  pip install openai-whisper   (offline, recommended)\n"
                        "  pip install SpeechRecognition (online, Google API)"
                    )
                except Exception as e:
                    return f"⚠️ Transcription error: {str(e)}"

            try:
                model = whisper.load_model("base")
                result = model.transcribe(audio_path, language="en")
                return result["text"]
            except Exception as e:
                return f"⚠️ Whisper transcription error: {str(e)}"

        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=audio_input,
            outputs=input_text,
        )

        # Generate summary
        generate_btn.click(
            fn=summarize,
            inputs=[
                input_text, max_length, decoding,
                beam_size, temperature, no_repeat, rep_penalty,
            ],
            outputs=[output_text, output_meta],
        )

        # Also allow Shift+Enter to generate
        input_text.submit(
            fn=summarize,
            inputs=[
                input_text, max_length, decoding,
                beam_size, temperature, no_repeat, rep_penalty,
            ],
            outputs=[output_text, output_meta],
        )

        # ── Footer ──
        gr.Markdown("---")
        gr.Markdown(
            """
            <div style="text-align: center; color: #888; font-size: 0.9em;">
            <b>Mamba-Transformer Hybrid V2</b> — Clinical Note Summarization<br>
            Model trained on MIMIC-IV BHC dataset (~270K samples) | 79.4M parameters<br>
            Architecture: Bidirectional Mamba SSM Encoder + Transformer Decoder with Memory Compression
            </div>
            """
        )

    return demo


# ─────────────────────────── Main ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clinical Note Summarizer Web App")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/v2_run/best_model.pt",
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/full_train.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to serve on (use 0.0.0.0 for network access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to serve on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    args = parser.parse_args()

    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Available checkpoints:")
        ckpt_dir = os.path.dirname(args.checkpoint) or "checkpoints"
        if os.path.exists(ckpt_dir):
            for f in sorted(os.listdir(ckpt_dir)):
                if f.endswith(".pt"):
                    print(f"  {os.path.join(ckpt_dir, f)}")
        sys.exit(1)

    load_model(args.checkpoint, args.config)

    # Build and launch UI
    demo = build_ui()
    print(f"\n{'='*60}")
    print("  CLINICAL NOTE SUMMARIZER — WEB APP")
    print(f"{'='*60}")
    print(f"  URL: http://{args.host}:{args.port}")
    if args.share:
        print("  Public link will be generated...")
    print(f"{'='*60}\n")

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
