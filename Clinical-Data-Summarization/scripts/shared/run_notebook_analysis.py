"""
Run the model inference and visualization analysis as a standalone script
This executes all the cells from the Jupyter notebook
"""

import torch
import torch.nn.functional as F
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Import custom modules
import sys
sys.path.append('.')
from src.core import PointerGeneratorModel
import sentencepiece as spm
from rouge_score import rouge_scorer

print("="*80)
print("MODEL INFERENCE AND VISUALIZATION")
print("="*80)

# ============================================================================
# 1. LOAD CONFIGURATION AND MODEL
# ============================================================================
print("\n1. Loading configuration and model...")

CONFIG_PATH = 'configs/rtx4070_8gb.yaml'
CHECKPOINT_PATH = 'artifacts/checkpoints/full_training_restart/best_model.pt'
TOKENIZER_PATH = 'artifacts/tokenizer/spm.model'
METRICS_CSV = 'artifacts/logs/full_training_restart/metrics.csv'

# Load configuration
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

print(f"✓ Configuration loaded")
print(f"  - Embedding dim: {config['model']['emb_dim']}")
print(f"  - Hidden dim: {config['model']['hidden_dim']}")

# Load tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(TOKENIZER_PATH)
vocab_size = tokenizer.get_piece_size()
print(f"✓ Tokenizer loaded: {vocab_size} vocab")

# Add data config for model
config['data'] = {
    'vocab_size': vocab_size,
    'pad_id': tokenizer.pad_id(),
    'bos_id': tokenizer.bos_id(),
    'eos_id': tokenizer.eos_id()
}

# Load checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
print(f"✓ Checkpoint loaded (step {checkpoint['step']}, Best ROUGE: {checkpoint.get('best_rouge', 0):.4f})")

# Create and load model
model = PointerGeneratorModel(config)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model loaded: {total_params:,} parameters on {device}")

# ============================================================================
# 2. LOAD TEST DATA
# ============================================================================
print("\n2. Loading test dataset...")
test_data = pd.read_parquet('data/tokenized/test.parquet')
print(f"✓ Test dataset: {len(test_data)} samples")

# ============================================================================
# 3. GENERATE PREDICTIONS
# ============================================================================
print("\n3. Generating predictions...")

# Generate for 10 random samples
NUM_SAMPLES = 10
sample_indices = np.random.choice(len(test_data), NUM_SAMPLES, replace=False)
results = []

for idx in tqdm(sample_indices, desc="Generating summaries"):
    row = test_data.iloc[idx]
    src_ids = list(row['src_ids'])
    tgt_ids = list(row['tgt_ids'])
    
    # Truncate source to model's max length
    max_src_len = config['model']['chunk_len'] * config['model']['num_chunks']
    src_ids = src_ids[:max_src_len]
    
    # Use model's generate method with beam search
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    src_lengths = torch.tensor([(src_tensor != tokenizer.pad_id()).sum().item()], dtype=torch.long).to(device)
    
    gen_ids, scores = model.generate(
        src_ids=src_tensor,
        src_lengths=src_lengths,
        beam_size=4,
        max_length=192,
        min_length=50,
        length_penalty=1.0,
        no_repeat_ngram=3
    )
    gen_ids = gen_ids[0].cpu().tolist()
    
    # Decode - convert to native Python ints
    src_text = tokenizer.decode([int(x) for x in src_ids])
    tgt_text = tokenizer.decode([int(x) for x in tgt_ids])
    gen_text = tokenizer.decode([int(x) for x in gen_ids])
    
    results.append({
        'index': int(idx),
        'source': src_text,
        'reference': tgt_text,
        'generated': gen_text,
        'src_ids': src_ids,
        'gen_ids': gen_ids,
        'src_len': len(src_ids),
        'gen_len': len(gen_ids),
        'ref_len': len(tgt_ids),
        'p_gen': [0.5]  # Placeholder
    })

print(f"✓ Generated {len(results)} summaries")

# ============================================================================
# 4. COMPUTE ROUGE SCORES
# ============================================================================
print("\n4. Computing ROUGE scores...")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge_scores = []
for result in results:
    scores = scorer.score(result['reference'], result['generated'])
    rouge_scores.append({
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    })

rouge_df = pd.DataFrame(rouge_scores)
print(f"✓ ROUGE scores computed")
print(f"\n  ROUGE-1: {rouge_df['rouge1'].mean():.4f} (±{rouge_df['rouge1'].std():.4f})")
print(f"  ROUGE-2: {rouge_df['rouge2'].mean():.4f} (±{rouge_df['rouge2'].std():.4f})")
print(f"  ROUGE-L: {rouge_df['rougeL'].mean():.4f} (±{rouge_df['rougeL'].std():.4f})")

# ============================================================================
# 5. DISPLAY SAMPLE PREDICTIONS
# ============================================================================
print("\n5. Sample predictions:")
print("="*80)

for i in range(min(3, len(results))):
    result = results[i]
    print(f"\nSAMPLE {i+1} (Index: {result['index']})")
    print("-"*80)
    print(f"SOURCE ({result['src_len']} tokens): {result['source'][:300]}...")
    print(f"\nREFERENCE: {result['reference']}")
    print(f"\nGENERATED: {result['generated']}")
    print(f"ROUGE-L: {rouge_scores[i]['rougeL']:.4f}")

# ============================================================================
# 6. EXPORT RESULTS
# ============================================================================
print("\n6. Exporting results...")
output_dir = Path('results/notebook_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# Export detailed results
src_lengths = [r['src_len'] for r in results]
ref_lengths = [r['ref_len'] for r in results]
gen_lengths = [r['gen_len'] for r in results]
compression_ratios = [g/s for g, s in zip(gen_lengths, src_lengths)]

export_df = pd.DataFrame({
    'sample_idx': [r['index'] for r in results],
    'source': [r['source'][:200] + '...' for r in results],
    'reference': [r['reference'] for r in results],
    'generated': [r['generated'] for r in results],
    'rouge1': rouge_df['rouge1'],
    'rouge2': rouge_df['rouge2'],
    'rougeL': rouge_df['rougeL'],
    'src_length': src_lengths,
    'ref_length': ref_lengths,
    'gen_length': gen_lengths,
    'compression_ratio': compression_ratios,
    'avg_p_gen': [np.mean(r['p_gen']) for r in results]
})

csv_path = output_dir / 'detailed_results.csv'
export_df.to_csv(csv_path, index=False)
print(f"✓ Results exported to {csv_path}")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n7. Creating visualizations...")

# Figure 1: ROUGE Score Distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, metric in enumerate(['rouge1', 'rouge2', 'rougeL']):
    ax = axes[i]
    ax.hist(rouge_df[metric], bins=10, edgecolor='black', alpha=0.7, color=f'C{i}')
    ax.axvline(rouge_df[metric].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {rouge_df[metric].mean():.3f}')
    ax.set_xlabel('Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{metric.upper()} Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / 'rouge_distributions.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved ROUGE distributions")

# Figure 2: Summary Length Analysis
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
x = np.arange(len(results))
width = 0.25
ax.bar(x - width, src_lengths, width, label='Source', alpha=0.8)
ax.bar(x, ref_lengths, width, label='Reference', alpha=0.8)
ax.bar(x + width, gen_lengths, width, label='Generated', alpha=0.8)
ax.set_xlabel('Sample Index', fontsize=11)
ax.set_ylabel('Length (tokens)', fontsize=11)
ax.set_title('Summary Length Comparison', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

ax = axes[1]
ax.scatter(src_lengths, compression_ratios, s=100, alpha=0.6, edgecolors='black')
ax.axhline(y=np.mean(compression_ratios), color='red', linestyle='--', 
           label=f'Mean: {np.mean(compression_ratios):.2%}')
ax.set_xlabel('Source Length (tokens)', fontsize=11)
ax.set_ylabel('Compression Ratio', fontsize=11)
ax.set_title('Compression Ratio vs Source Length', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2]
ax.scatter(ref_lengths, gen_lengths, s=100, alpha=0.6, edgecolors='black')
max_len = max(max(ref_lengths), max(gen_lengths))
ax.plot([0, max_len], [0, max_len], 'r--', alpha=0.5, label='Perfect match')
ax.set_xlabel('Reference Length (tokens)', fontsize=11)
ax.set_ylabel('Generated Length (tokens)', fontsize=11)
ax.set_title('Generated vs Reference Length', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'length_analysis.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved length analysis")

plt.close('all')

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nResults saved in: {output_dir}")
print(f"  - detailed_results.csv")
print(f"  - attention_heatmap_sample1.png")
print(f"  - rouge_distributions.png")
print(f"  - length_analysis.png")
