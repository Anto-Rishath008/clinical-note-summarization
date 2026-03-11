"""
Generate all comparison plots for the Clinical Data Summarization presentation.
Run this script to produce PDF figures in the figures/ directory.
Updated with FINAL training results (150K steps completed).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IEEE-style formatting
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ============================================================
# DATA: FINAL Metrics for each model (UPDATED)
# ============================================================
models = ['Pointer\nGenerator', 'LongT5', 'Mamba V1\n(Baseline)', 'Mamba V2\n(Final)']
short_models = ['PG', 'LongT5', 'Mamba V1', 'Mamba V2']

# ROUGE scores — UPDATED with actual final results
# PG: from experiments/results.csv (500 steps, severely undertrained)
# LongT5: from artifacts/logs (step ~7500 best)
# Mamba V1: step 40000 (first eval, before V2 fixes)
# Mamba V2: best_model.pt at step 130000 (ROUGE-L=0.2296)
rouge1 = [0.1814, 0.3517, 0.3484, 0.3840]
rouge2 = [0.0030, 0.1232, 0.1027, 0.1386]
rougeL = [0.0855, 0.2185, 0.1952, 0.2296]

# Baselines (from experiments/baseline_results.csv)
lead100_r1, lead100_r2, lead100_rL = 0.1848, 0.0705, 0.1186
lead150_r1, lead150_r2, lead150_rL = 0.2530, 0.0858, 0.1510

# Model parameters (millions)
params = [22.5, 12.5, 52.1, 79.4]

# Max input length
max_input = [768, 4096, 4096, 4096]

# Final validation loss
final_loss = [4.80, 4.20, 3.25, 2.56]

# Colors
colors = ['#C0392B', '#E67E22', '#2980B9', '#27AE60']
bar_colors = ['#E74C3C', '#F39C12', '#3498DB', '#2ECC71']

# ============================================================
# PLOT 1: ROUGE Score Comparison (Grouped Bar Chart)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(models))
width = 0.22

bars1 = ax.bar(x - width, rouge1, width, label='ROUGE-1', color='#2C3E50', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, rouge2, width, label='ROUGE-2', color='#E74C3C', edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width, rougeL, width, label='ROUGE-L', color='#3498DB', edgecolor='white', linewidth=0.5)

# Add Lead-150 baseline line
ax.axhline(y=lead150_rL, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label='Lead-150 ROUGE-L (0.1510)')

ax.set_xlabel('Model')
ax.set_ylabel('ROUGE Score')
ax.set_title('ROUGE Score Comparison Across Models')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.legend(loc='upper left', framealpha=0.9)
ax.set_ylim(0, 0.50)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=6.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rouge_comparison.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'rouge_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# PLOT 2: ROUGE-L Progression (Line Chart)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
model_order = [1, 2, 3, 4]
rougeL_vals = [0.0855, 0.2185, 0.1952, 0.2296]

ax.plot(model_order, rougeL_vals, 'o-', color='#2C3E50', linewidth=2.5, markersize=10, zorder=5)
for i, (x_val, y_val) in enumerate(zip(model_order, rougeL_vals)):
    ax.annotate(f'{y_val:.4f}', (x_val, y_val), textcoords="offset points",
                xytext=(0, 12), ha='center', fontsize=9, fontweight='bold', color=colors[i])

ax.axhline(y=lead150_rL, color='#E74C3C', linestyle='--', linewidth=1.2, alpha=0.7, label='Lead-150 Baseline (0.1510)')
ax.fill_between(model_order, rougeL_vals, alpha=0.1, color='#2C3E50')

ax.set_xlabel('Model Iteration')
ax.set_ylabel('ROUGE-L Score')
ax.set_title('ROUGE-L Score Progression Across Models')
ax.set_xticks(model_order)
ax.set_xticklabels(short_models, fontsize=10)
ax.set_ylim(0.0, 0.32)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rougeL_progression.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'rougeL_progression.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# PLOT 3: Training Loss Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(short_models, final_loss, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.6)

for bar, val in zip(bars, final_loss):
    ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Model')
ax.set_ylabel('Final Validation Loss')
ax.set_title('Final Validation Loss Comparison')
ax.set_ylim(0, 6.0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss_comparison.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# PLOT 4: Model Parameters Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.barh(short_models, params, color=bar_colors, edgecolor='white', linewidth=1.5, height=0.5)

for bar, val in zip(bars, params):
    ax.annotate(f'{val:.1f}M', xy=(val, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Number of Parameters (Millions)')
ax.set_title('Model Size Comparison')
ax.set_xlim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_parameters.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'model_parameters.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# PLOT 5: Max Input Length Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(short_models, max_input, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.6)

for bar, val in zip(bars, max_input):
    ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Model')
ax.set_ylabel('Maximum Input Length (Tokens)')
ax.set_title('Maximum Input Sequence Length')
ax.set_ylim(0, 5000)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'max_input_length.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'max_input_length.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# PLOT 6: Radar Chart - Multi-Metric Comparison
# ============================================================
categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Max Input\n(norm)', 'Parameters\n(norm inv)', 'Loss\n(norm inv)']
N = len(categories)

# Normalize all metrics to 0-1 scale
def normalize(vals, higher_better=True):
    min_v, max_v = min(vals), max(vals)
    if max_v == min_v:
        return [0.5] * len(vals)
    if higher_better:
        return [(v - min_v) / (max_v - min_v) for v in vals]
    else:
        return [(max_v - v) / (max_v - min_v) for v in vals]

norm_r1 = normalize(rouge1)
norm_r2 = normalize(rouge2)
norm_rL = normalize(rougeL)
norm_input = normalize(max_input)
norm_params = normalize(params, higher_better=False)
norm_loss = normalize(final_loss, higher_better=False)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for i, model in enumerate(short_models):
    values = [norm_r1[i], norm_r2[i], norm_rL[i], norm_input[i], norm_params[i], norm_loss[i]]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i], markersize=5)
    ax.fill(angles, values, alpha=0.08, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=8)
ax.set_ylim(0, 1.1)
ax.set_title('Multi-Metric Model Comparison\n(Normalized)', fontsize=12, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'radar_comparison.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# PLOT 7: Issue Resolution Timeline
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

issues = [
    'Limited Input (768 tokens)',
    'Coverage OOM',
    'Incoherent Output',
    'Hardcoded CUDA',
    'Low ROUGE-2 (0.003)',
    'Attention O(n²)',
    'Cross-Attn Suppression',
    'Memory Bottleneck',
]

issue_model = {
    'Limited Input (768 tokens)':      [1, 2, 0, 0],
    'Coverage OOM':                     [1, 2, 0, 0],
    'Incoherent Output':               [1, 0, 2, 0],
    'Hardcoded CUDA':                   [1, 2, 0, 0],
    'Low ROUGE-2 (0.003)':             [1, 0, 0, 2],
    'Attention O(n²)':                  [0, 1, 2, 0],
    'Cross-Attn Suppression':          [0, 0, 1, 2],
    'Memory Bottleneck':               [0, 0, 1, 2],
}

y_pos = np.arange(len(issues))
for i, issue in enumerate(issues):
    statuses = issue_model[issue]
    for j, s in enumerate(statuses):
        if s == 1:
            ax.scatter(j, i, marker='x', color='#E74C3C', s=120, zorder=5, linewidths=2.5)
        elif s == 2:
            ax.scatter(j, i, marker='o', color='#27AE60', s=120, zorder=5, linewidths=2)

ax.scatter([], [], marker='x', color='#E74C3C', s=80, label='Issue Identified')
ax.scatter([], [], marker='o', color='#27AE60', s=80, label='Issue Resolved')

ax.set_yticks(y_pos)
ax.set_yticklabels(issues, fontsize=8)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(short_models, fontsize=10)
ax.set_xlabel('Model')
ax.set_title('Issue Identification and Resolution Across Models')
ax.legend(loc='lower right')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'issue_resolution.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'issue_resolution.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# PLOT 8: Mamba V2 COMPLETE Training Curves (ACTUAL DATA)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Actual evaluation data from train.log (all 43 eval points)
eval_steps = np.array([
    40000, 42500, 45000, 47500, 50000, 52500, 55000, 57500, 60000, 62500,
    65000, 67500, 70000, 72500, 75000, 77500, 80000, 82500, 85000, 87500,
    90000, 92500, 95000, 97500, 100000, 102500, 105000, 107500, 110000,
    112500, 115000, 117500, 120000, 122500, 125000, 127500, 130000,
    135000, 140000, 145000, 150000
])
val_losses = np.array([
    3.2467, 3.1800, 3.1106, 3.0439, 2.9916, 2.9514, 2.9187, 2.9955, 2.9675, 2.9243,
    2.8844, 2.8564, 2.8632, 2.8492, 2.8208, 2.7905, 2.7955, 2.7794, 2.7609, 2.7341,
    2.7162, 2.6862, 2.7153, 2.7095, 2.7001, 2.6750, 2.6631, 2.6348, 2.6183,
    2.5936, 2.5802, 2.5648, 2.5457, 2.5336, 2.5238, 2.6030, 2.6000,
    2.6420, 2.6207, 2.5941, 2.5577
])
eval_rouge1 = np.array([
    0.3484, 0.3470, 0.3505, 0.3541, 0.3572, 0.3557, 0.3586, 0.3490, 0.3551, 0.3446,
    0.3404, 0.3602, 0.3506, 0.3648, 0.3604, 0.3498, 0.3608, 0.3740, 0.3637, 0.3697,
    0.3726, 0.3599, 0.3595, 0.3724, 0.3676, 0.3681, 0.3776, 0.3628, 0.3744,
    0.3755, 0.3882, 0.3834, 0.3873, 0.3931, 0.3840, 0.3677, 0.3840,
    0.3833, 0.3739, 0.3871, 0.3780
])
eval_rouge2 = np.array([
    0.1027, 0.1026, 0.0994, 0.1038, 0.1132, 0.1101, 0.1130, 0.1133, 0.1047, 0.1055,
    0.1030, 0.1115, 0.1105, 0.1155, 0.1205, 0.1125, 0.1236, 0.1201, 0.1237, 0.1273,
    0.1197, 0.1189, 0.1147, 0.1278, 0.1314, 0.1283, 0.1256, 0.1241, 0.1271,
    0.1283, 0.1329, 0.1329, 0.1311, 0.1350, 0.1283, 0.1260, 0.1386,
    0.1355, 0.1353, 0.1348, 0.1327
])
eval_rougeL = np.array([
    0.1952, 0.1952, 0.1923, 0.1943, 0.2032, 0.2016, 0.2040, 0.2000, 0.1932, 0.1963,
    0.1947, 0.2059, 0.2005, 0.2069, 0.2077, 0.2015, 0.2120, 0.2091, 0.2125, 0.2160,
    0.2099, 0.2075, 0.2079, 0.2173, 0.2205, 0.2190, 0.2146, 0.2122, 0.2172,
    0.2180, 0.2258, 0.2194, 0.2226, 0.2288, 0.2209, 0.2147, 0.2296,
    0.2210, 0.2230, 0.2244, 0.2197
])

# Left: Validation Loss
ax1.plot(eval_steps / 1000, val_losses, '-', color='#2C3E50', linewidth=1.8, alpha=0.9)
ax1.fill_between(eval_steps / 1000, val_losses, alpha=0.08, color='#3498DB')

# Mark best loss
best_loss_idx = np.argmin(val_losses)
ax1.plot(eval_steps[best_loss_idx] / 1000, val_losses[best_loss_idx], 'v', color='#27AE60', markersize=10, zorder=5)
ax1.annotate(f'Best: {val_losses[best_loss_idx]:.4f}\n(Step {eval_steps[best_loss_idx]//1000}K)',
             (eval_steps[best_loss_idx] / 1000, val_losses[best_loss_idx]),
             textcoords="offset points", xytext=(-40, -25), fontsize=8,
             arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

ax1.set_xlabel('Training Steps (×1000)')
ax1.set_ylabel('Validation Loss')
ax1.set_title('Validation Loss Curve')
ax1.set_xlim(38, 155)

# Right: ROUGE-L
ax2.plot(eval_steps / 1000, eval_rougeL, 'o-', color='#3498DB', linewidth=1.8, markersize=3.5, alpha=0.9)
ax2.fill_between(eval_steps / 1000, eval_rougeL, alpha=0.08, color='#3498DB')

# Mark best ROUGE-L
best_rL_idx = np.argmax(eval_rougeL)
ax2.plot(eval_steps[best_rL_idx] / 1000, eval_rougeL[best_rL_idx], '*', color='#E74C3C', markersize=14, zorder=5)
ax2.annotate(f'Best: {eval_rougeL[best_rL_idx]:.4f}\n(Step {eval_steps[best_rL_idx]//1000}K)',
             (eval_steps[best_rL_idx] / 1000, eval_rougeL[best_rL_idx]),
             textcoords="offset points", xytext=(10, -15), fontsize=8,
             arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

ax2.axhline(y=lead150_rL, color='gray', linestyle='--', linewidth=1, alpha=0.6, label=f'Lead-150: {lead150_rL}')
ax2.set_xlabel('Training Steps (×1000)')
ax2.set_ylabel('ROUGE-L Score')
ax2.set_title('ROUGE-L Score During Training')
ax2.set_xlim(38, 155)
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mamba_training_curve.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'mamba_training_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# PLOT 9: Complexity Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))

seq_lengths = np.array([512, 1024, 2048, 4096, 8192])
standard_attn = seq_lengths ** 2 / 1e6
local_global = seq_lengths * 255 / 1e6
mamba_linear = seq_lengths * 512 / 1e6

ax.plot(seq_lengths, standard_attn, 's--', color='#E74C3C', linewidth=2, label=r'Standard Attention $O(n^2)$', markersize=6)
ax.plot(seq_lengths, local_global, '^--', color='#F39C12', linewidth=2, label=r'Local-Global $O(n \cdot w)$', markersize=6)
ax.plot(seq_lengths, mamba_linear, 'o-', color='#27AE60', linewidth=2, label=r'Mamba SSM $O(n)$', markersize=6)

ax.set_xlabel('Sequence Length')
ax.set_ylabel('Relative Computation (×$10^6$)')
ax.set_title('Computational Complexity Comparison')
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'complexity_comparison.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'complexity_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# PLOT 10: Per-Sample ROUGE-L Distribution (Mamba V2)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))

sample_rougeL = [0.1521, 0.1930, 0.2289, 0.2454, 0.1693, 0.2274, 0.2069, 0.1697, 0.1619, 0.2000]
sample_ids = list(range(1, 11))

clrs = ['#3498DB' if v >= lead150_rL else '#E74C3C' for v in sample_rougeL]
bars = ax.bar(sample_ids, sample_rougeL, color=clrs, edgecolor='white', linewidth=1)
ax.axhline(y=np.mean(sample_rougeL), color='#2C3E50', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(sample_rougeL):.4f}')
ax.axhline(y=lead150_rL, color='gray', linestyle=':', linewidth=1.2, label=f'Lead-150: {lead150_rL}')

ax.set_xlabel('Sample Index')
ax.set_ylabel('ROUGE-L Score')
ax.set_title('Per-Sample ROUGE-L Distribution (Mamba V2, n=10)')
ax.set_xticks(sample_ids)
ax.legend()
ax.set_ylim(0, 0.35)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_distribution.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# PLOT 11: All 3 ROUGE Metrics During V2 Training
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(eval_steps / 1000, eval_rouge1, 's-', color='#2C3E50', linewidth=1.5, markersize=3, label='ROUGE-1', alpha=0.85)
ax.plot(eval_steps / 1000, eval_rouge2, '^-', color='#E74C3C', linewidth=1.5, markersize=3, label='ROUGE-2', alpha=0.85)
ax.plot(eval_steps / 1000, eval_rougeL, 'o-', color='#3498DB', linewidth=1.5, markersize=3, label='ROUGE-L', alpha=0.85)

# Annotate best
best_idx = np.argmax(eval_rougeL)
ax.axvline(x=eval_steps[best_idx] / 1000, color='#27AE60', linestyle=':', alpha=0.5, label=f'Best (Step {eval_steps[best_idx]//1000}K)')

ax.axhline(y=lead150_rL, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(152, lead150_rL + 0.003, 'Lead-150', fontsize=7, color='gray')

ax.set_xlabel('Training Steps (×1000)')
ax.set_ylabel('ROUGE Score')
ax.set_title('Mamba V2: ROUGE Scores During Training (150K Steps)')
ax.set_xlim(38, 155)
ax.set_ylim(0.05, 0.45)
ax.legend(loc='center left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'v2_rouge_training.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'v2_rouge_training.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# PLOT 12: ROUGE-2 Improvement (dramatic: 0.003 → 0.1386)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
r2_vals = [0.0030, 0.1232, 0.1027, 0.1386]
bars = ax.bar(short_models, r2_vals, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.6)
ax.axhline(y=lead150_r2, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label=f'Lead-150 ROUGE-2 ({lead150_r2})')

for bar, val in zip(bars, r2_vals):
    ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Model')
ax.set_ylabel('ROUGE-2 Score')
ax.set_title('ROUGE-2 (Bigram) Score Improvement')
ax.set_ylim(0, 0.20)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rouge2_improvement.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'rouge2_improvement.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# PLOT 13: Improvement Percentages
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4))

improvements = {
    'PG → LongT5': ((0.2185 - 0.0855) / 0.0855) * 100,
    'LongT5 → V1': ((0.1952 - 0.2185) / 0.2185) * 100,
    'V1 → V2': ((0.2296 - 0.1952) / 0.1952) * 100,
    'PG → V2\n(Total)': ((0.2296 - 0.0855) / 0.0855) * 100,
}

labels = list(improvements.keys())
values = list(improvements.values())
clrs2 = ['#27AE60' if v > 0 else '#E74C3C' for v in values]

bars = ax.barh(labels, values, color=clrs2, edgecolor='white', linewidth=1.5, height=0.5)
for bar, val in zip(bars, values):
    xpos = val + 2 if val > 0 else val - 2
    ha = 'left' if val > 0 else 'right'
    ax.annotate(f'{val:+.1f}%', xy=(val, bar.get_y() + bar.get_height() / 2),
                xytext=(5 if val > 0 else -5, 0), textcoords="offset points",
                ha=ha, va='center', fontsize=10, fontweight='bold')

ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('ROUGE-L Improvement (%)')
ax.set_title('ROUGE-L Improvement Between Model Iterations')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'improvement_pct.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'improvement_pct.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"All plots generated successfully in: {OUTPUT_DIR}")
print("Files created:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {f} ({size_kb:.1f} KB)")
