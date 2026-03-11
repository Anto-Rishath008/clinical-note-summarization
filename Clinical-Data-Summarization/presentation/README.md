# Overleaf Presentation - Clinical Data Summarization

## AIE315: Natural Language Processing | S6 AIE Batch B | Group 5

---

## How to Upload to Overleaf

### Step 1: Create a New Project
1. Go to [https://www.overleaf.com](https://www.overleaf.com)
2. Click **"New Project"** → **"Upload Project"**
3. Upload this entire `overleaf_presentation` folder as a ZIP file

### Step 2: Alternative (Manual Upload)
1. Create a **"New Project"** → **"Blank Project"**
2. Replace the auto-generated `main.tex` with the `main.tex` from this folder
3. Create a `figures/` folder in Overleaf
4. Upload ALL files from the local `figures/` folder into the Overleaf `figures/` folder

### Step 3: Compile
1. Set the compiler to **pdfLaTeX**
2. Click **"Recompile"**
3. The presentation should compile without errors

---

## Folder Structure

```
overleaf_presentation/
├── main.tex                        # Main LaTeX Beamer presentation
├── generate_plots.py               # Python script to regenerate plots
├── README.md                       # This file
└── figures/
    ├── rouge_comparison.png        # ROUGE scores grouped bar chart
    ├── rouge_comparison.pdf        # (PDF version)
    ├── rougeL_progression.png      # ROUGE-L improvement across models
    ├── rougeL_progression.pdf
    ├── training_loss_comparison.png # Final training loss bars
    ├── training_loss_comparison.pdf
    ├── mamba_training_curve.png    # Mamba V2 training loss curve
    ├── mamba_training_curve.pdf
    ├── model_parameters.png        # Model size comparison
    ├── model_parameters.pdf
    ├── max_input_length.png        # Max input token comparison
    ├── max_input_length.pdf
    ├── complexity_comparison.png   # O(n²) vs O(n·w) vs O(n) complexity
    ├── complexity_comparison.pdf
    ├── radar_comparison.png        # Multi-metric radar chart
    ├── radar_comparison.pdf
    ├── issue_resolution.png        # Issue identification/resolution timeline
    ├── issue_resolution.pdf
    ├── sample_distribution.png     # Per-sample ROUGE-L distribution
    ├── sample_distribution.pdf
    └── placeholder_logo.png        # Replace with university logo
```

---

## Presentation Outline (28 slides)

1. **Title Slide** - Project title, team members, course info
2. **Outline** - Table of contents
3. **Problem Statement** - Clinical data summarization challenges
4. **Dataset** - MIMIC-IV BHC overview

5. **Model 1: Pointer-Generator**
   - Architecture (empty slide for diagram)
   - Mathematical Formulation
   - Configuration & Training
   - Issues & Limitations

6. **Model 2: LongT5**
   - Architecture (empty slide for diagram)
   - Improvements over Pointer-Generator
   - Mathematical Formulation
   - Configuration & Issues

7. **Model 3: Mamba V1**
   - Architecture (empty slide for diagram)
   - Why Mamba? SSM introduction
   - Architecture Details
   - Critical Bug: Cross-Attention Suppression

8. **Model 4: Mamba V2 (Final)**
   - Architecture (empty slide for diagram)
   - Comprehensive Improvements table
   - Key Mathematical Components
   - Training Pipeline Robustness
   - Complete Architecture Flow (TikZ diagram)

9. **Results and Discussion**
   - ROUGE Score Comparison (bar chart)
   - ROUGE-L Progression (line chart)
   - Training Loss Comparison
   - Model Complexity Analysis
   - Issue Resolution Timeline
   - Per-Sample Performance
   - Comprehensive Metrics Table

10. **Clinical Evaluation Framework**
    - Limitations of ROUGE
    - Clinical Metrics: BERTScore, MEDCON, SummaC

11. **Conclusion** - Summary, key findings, future work
12. **References**
13. **Thank You**

---

## Notes

- **Architecture slides** are left empty (with placeholder text) for you to add your own architecture diagrams.
- **Replace `placeholder_logo.png`** with your university logo if desired.
- All plots are available in both PNG and PDF formats.
- The presentation uses IEEE-inspired formatting with the Madrid Beamer theme.
- All mathematical equations are properly formatted using LaTeX math mode.
