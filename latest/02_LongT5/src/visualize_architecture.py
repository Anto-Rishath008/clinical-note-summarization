"""
Model Architecture Visualization for Simplified LongT5
=======================================================

Generates architecture diagrams using:
1. Graphviz (DOT format) - Static diagram
2. Torchviz - Computational graph from forward pass
3. Matplotlib - Fallback visualization

Usage:
    python visualize_architecture.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

# Try to import visualization libraries
try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("graphviz not installed. Install with: pip install graphviz")

try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False
    print("torchviz not installed. Install with: pip install torchviz")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not installed. Install with: pip install matplotlib")

# Import our model
try:
    from model import SimplifiedLongT5, LongT5Config, create_model
except ImportError:
    from longt5_model.model import SimplifiedLongT5, LongT5Config, create_model


def create_matplotlib_diagram(output_path: str = "longt5_architecture_matplotlib"):
    """
    Create architecture diagram using matplotlib (no external dependencies).
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available.")
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 130)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#C8E6C9',      # Light green
        'embedding': '#BBDEFB',   # Light blue
        'encoder': '#FFE0B2',     # Light orange
        'decoder': '#E1BEE7',     # Light purple
        'attention': '#FFCDD2',   # Light red
        'ffn': '#B2EBF2',         # Light cyan
        'output': '#FFF9C4',      # Light yellow
        'norm': '#EEEEEE',        # Light gray
        'cross': '#F8BBD9',       # Light pink
    }
    
    def draw_box(x, y, w, h, label, color, fontsize=9):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.5",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=fontsize, wrap=True)
    
    def draw_arrow(x1, y1, x2, y2, label='', color='black'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x + 1, mid_y, label, fontsize=7, color='gray')
    
    # Title
    ax.text(50, 128, 'Simplified LongT5 Architecture', ha='center', va='center', 
            fontsize=16, fontweight='bold')
    ax.text(50, 125, '(~11.4M Parameters)', ha='center', va='center', fontsize=10, color='gray')
    
    # Input Section
    draw_box(20, 115, 25, 6, 'Input IDs\n(batch, 4096)', colors['input'])
    draw_box(55, 115, 25, 6, 'Decoder Input\n(batch, 512)', colors['input'])
    
    # Shared Embedding
    draw_box(30, 105, 40, 6, 'Shared Embedding\nvocab=16000 → d_model=256', colors['embedding'])
    
    # Arrows from input to embedding
    draw_arrow(32.5, 115, 40, 111)
    draw_arrow(67.5, 115, 60, 111)
    
    # Encoder Section
    y_enc = 95
    draw_box(15, y_enc-2, 35, 4, 'ENCODER (×4 Layers)', colors['encoder'], fontsize=11)
    
    # Encoder Layer Detail
    y_enc_layer = y_enc - 8
    draw_box(5, y_enc_layer, 20, 5, 'LayerNorm', colors['norm'])
    draw_arrow(15, y_enc-2, 15, y_enc_layer+5)
    
    y_enc_layer -= 7
    draw_box(5, y_enc_layer, 20, 6, 'Local-Global\nSelf-Attention\n(8 heads)', colors['attention'])
    draw_arrow(15, y_enc_layer+13, 15, y_enc_layer+6)
    
    y_enc_layer -= 6
    draw_box(5, y_enc_layer, 20, 4, 'Dropout + Residual', colors['norm'])
    draw_arrow(15, y_enc_layer+10, 15, y_enc_layer+4)
    
    y_enc_layer -= 6
    draw_box(5, y_enc_layer, 20, 4, 'LayerNorm', colors['norm'])
    draw_arrow(15, y_enc_layer+10, 15, y_enc_layer+4)
    
    y_enc_layer -= 6
    draw_box(5, y_enc_layer, 20, 5, 'Feed-Forward\n256→1024→256', colors['ffn'])
    draw_arrow(15, y_enc_layer+10, 15, y_enc_layer+5)
    
    y_enc_layer -= 6
    draw_box(5, y_enc_layer, 20, 4, 'Dropout + Residual', colors['norm'])
    draw_arrow(15, y_enc_layer+11, 15, y_enc_layer+4)
    
    # Encoder output
    draw_box(5, y_enc_layer-7, 20, 5, 'Encoder Output\n(batch, 4096, 256)', colors['encoder'])
    draw_arrow(15, y_enc_layer, 15, y_enc_layer-2)
    
    # Decoder Section
    y_dec = 95
    draw_box(55, y_dec-2, 35, 4, 'DECODER (×4 Layers)', colors['decoder'], fontsize=11)
    
    # Decoder Layer Detail
    y_dec_layer = y_dec - 8
    draw_box(60, y_dec_layer, 25, 5, 'LayerNorm', colors['norm'])
    draw_arrow(72.5, y_dec-2, 72.5, y_dec_layer+5)
    
    y_dec_layer -= 7
    draw_box(60, y_dec_layer, 25, 6, 'Causal\nSelf-Attention\n(Masked)', colors['attention'])
    draw_arrow(72.5, y_dec_layer+13, 72.5, y_dec_layer+6)
    
    y_dec_layer -= 6
    draw_box(60, y_dec_layer, 25, 4, 'Dropout + Residual', colors['norm'])
    draw_arrow(72.5, y_dec_layer+10, 72.5, y_dec_layer+4)
    
    y_dec_layer -= 6
    draw_box(60, y_dec_layer, 25, 4, 'LayerNorm', colors['norm'])
    draw_arrow(72.5, y_dec_layer+10, 72.5, y_dec_layer+4)
    
    y_dec_layer -= 7
    draw_box(60, y_dec_layer, 25, 6, 'Cross-Attention\nQ:Dec, K,V:Enc', colors['cross'])
    draw_arrow(72.5, y_dec_layer+10, 72.5, y_dec_layer+6)
    
    # Cross attention from encoder
    draw_arrow(25, 50, 60, y_dec_layer+3, 'K, V', '#FF9800')
    
    y_dec_layer -= 6
    draw_box(60, y_dec_layer, 25, 4, 'Dropout + Residual', colors['norm'])
    draw_arrow(72.5, y_dec_layer+10, 72.5, y_dec_layer+4)
    
    y_dec_layer -= 6
    draw_box(60, y_dec_layer, 25, 4, 'LayerNorm', colors['norm'])
    draw_arrow(72.5, y_dec_layer+10, 72.5, y_dec_layer+4)
    
    y_dec_layer -= 6
    draw_box(60, y_dec_layer, 25, 5, 'Feed-Forward\n256→1024→256', colors['ffn'])
    draw_arrow(72.5, y_dec_layer+10, 72.5, y_dec_layer+5)
    
    y_dec_layer -= 6
    draw_box(60, y_dec_layer, 25, 4, 'Dropout + Residual', colors['norm'])
    draw_arrow(72.5, y_dec_layer+11, 72.5, y_dec_layer+4)
    
    # LM Head
    draw_box(55, 15, 30, 6, 'LM Head\n256→16000\n(Tied Weights)', colors['output'])
    draw_arrow(72.5, y_dec_layer, 72.5, 21)
    
    # Output
    draw_box(55, 5, 30, 6, 'Output Logits\n(batch, 512, 16000)', colors['output'])
    draw_arrow(70, 15, 70, 11)
    
    # Legend
    legend_y = 5
    legend_items = [
        ('Input/Output', colors['input']),
        ('Embedding', colors['embedding']),
        ('Encoder', colors['encoder']),
        ('Decoder', colors['decoder']),
        ('Attention', colors['attention']),
        ('FFN', colors['ffn']),
        ('Cross-Attn', colors['cross']),
    ]
    for i, (name, color) in enumerate(legend_items):
        draw_box(2 + i*6.5, legend_y, 6, 3, name, color, fontsize=6)
    
    # Configuration text
    config_text = """Configuration:
• vocab_size: 16,000
• d_model: 256
• d_ff: 1,024
• num_heads: 8
• encoder_layers: 4
• decoder_layers: 4
• max_len: 4,096
• local_radius: 127
• global_block: 16"""
    ax.text(95, 50, config_text, fontsize=8, va='center', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save
    output_dir = Path(__file__).parent
    output_file = output_dir / f"{output_path}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Matplotlib diagram saved to: {output_file}")
    return fig


def create_attention_heatmap(output_path: str = "attention_pattern_heatmap"):
    """
    Create a heatmap showing local-global attention pattern.
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    import numpy as np
    
    seq_len = 32
    local_radius = 4
    global_block = 8
    
    # Create attention mask
    mask = np.zeros((seq_len, seq_len))
    
    for i in range(seq_len):
        # Local attention window
        start = max(0, i - local_radius)
        end = min(seq_len, i + local_radius + 1)
        mask[i, start:end] = 1
        
        # Global tokens
        if i % global_block == 0:
            mask[i, :] = 1  # Global token attends to all
            mask[:, i] = 1  # All attend to global token
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    cmap = plt.cm.colors.ListedColormap(['white', '#4CAF50'])
    im = ax.imshow(mask, cmap=cmap, interpolation='nearest')
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, seq_len, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Highlight global tokens
    for i in range(0, seq_len, global_block):
        ax.axhline(y=i, color='red', linewidth=2, alpha=0.5)
        ax.axvline(x=i, color='red', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    ax.set_title(f'Local-Global Attention Pattern\n(local_radius={local_radius}, global_block={global_block})', fontsize=14)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Can Attend'),
        Patch(facecolor='white', edgecolor='black', label='Masked'),
        Patch(facecolor='none', edgecolor='red', linewidth=2, label='Global Token'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save
    output_dir = Path(__file__).parent
    output_file = output_dir / f"{output_path}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Attention heatmap saved to: {output_file}")
    return fig


def create_graphviz_diagram(output_path: str = "longt5_architecture"):
    """
    Create a detailed architecture diagram using Graphviz.
    """
    if not GRAPHVIZ_AVAILABLE:
        print("Graphviz not available. Skipping DOT diagram generation.")
        return None
    
    dot = Digraph(comment='Simplified LongT5 Architecture')
    dot.attr(rankdir='TB', size='12,16', dpi='150')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
    
    # Color scheme
    colors = {
        'input': '#E8F5E9',      # Light green
        'embedding': '#E3F2FD',   # Light blue
        'encoder': '#FFF3E0',     # Light orange
        'decoder': '#F3E5F5',     # Light purple
        'attention': '#FFEBEE',   # Light red
        'ffn': '#E0F7FA',         # Light cyan
        'output': '#FFF9C4',      # Light yellow
        'norm': '#F5F5F5',        # Light gray
    }
    
    # === INPUT SECTION ===
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input', style='rounded', color='#4CAF50', fontcolor='#4CAF50')
        c.node('input_ids', 'Input IDs\n(batch, src_len)\nClinical Note Tokens', fillcolor=colors['input'])
        c.node('decoder_input_ids', 'Decoder Input IDs\n(batch, tgt_len)\nShifted Labels', fillcolor=colors['input'])
    
    # === EMBEDDING SECTION ===
    with dot.subgraph(name='cluster_embedding') as c:
        c.attr(label='Shared Embedding Layer', style='rounded', color='#2196F3', fontcolor='#2196F3')
        c.node('shared_emb', 'nn.Embedding\nvocab=16000, d_model=256\n(Shared Encoder/Decoder)', fillcolor=colors['embedding'])
    
    # === ENCODER SECTION ===
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='Encoder (×4 Layers)', style='rounded', color='#FF9800', fontcolor='#FF9800')
        
        # Encoder Layer 1
        with c.subgraph(name='cluster_enc_layer') as layer:
            layer.attr(label='Encoder Layer', style='dashed')
            layer.node('enc_ln1', 'LayerNorm', fillcolor=colors['norm'])
            layer.node('enc_local_global', 'Local-Global\nSelf-Attention\n8 heads, d_head=32\nlocal_radius=127\nglobal_block=16', fillcolor=colors['attention'])
            layer.node('enc_dropout1', 'Dropout(0.1)', fillcolor=colors['norm'])
            layer.node('enc_residual1', '⊕ Residual', shape='circle', fillcolor='white')
            layer.node('enc_ln2', 'LayerNorm', fillcolor=colors['norm'])
            layer.node('enc_ffn', 'Feed-Forward\n256→1024→256\nGELU activation', fillcolor=colors['ffn'])
            layer.node('enc_dropout2', 'Dropout(0.1)', fillcolor=colors['norm'])
            layer.node('enc_residual2', '⊕ Residual', shape='circle', fillcolor='white')
        
        c.node('enc_final_ln', 'Final LayerNorm', fillcolor=colors['norm'])
        c.node('encoder_output', 'Encoder Output\n(batch, src_len, 256)', fillcolor=colors['encoder'])
    
    # === DECODER SECTION ===
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(label='Decoder (×4 Layers)', style='rounded', color='#9C27B0', fontcolor='#9C27B0')
        
        # Decoder Layer
        with c.subgraph(name='cluster_dec_layer') as layer:
            layer.attr(label='Decoder Layer', style='dashed')
            layer.node('dec_ln1', 'LayerNorm', fillcolor=colors['norm'])
            layer.node('dec_self_attn', 'Causal\nSelf-Attention\n8 heads\n(Masked)', fillcolor=colors['attention'])
            layer.node('dec_dropout1', 'Dropout(0.1)', fillcolor=colors['norm'])
            layer.node('dec_residual1', '⊕ Residual', shape='circle', fillcolor='white')
            layer.node('dec_ln2', 'LayerNorm', fillcolor=colors['norm'])
            layer.node('dec_cross_attn', 'Cross-Attention\nQ: Decoder\nK,V: Encoder', fillcolor=colors['attention'])
            layer.node('dec_dropout2', 'Dropout(0.1)', fillcolor=colors['norm'])
            layer.node('dec_residual2', '⊕ Residual', shape='circle', fillcolor='white')
            layer.node('dec_ln3', 'LayerNorm', fillcolor=colors['norm'])
            layer.node('dec_ffn', 'Feed-Forward\n256→1024→256\nGELU activation', fillcolor=colors['ffn'])
            layer.node('dec_dropout3', 'Dropout(0.1)', fillcolor=colors['norm'])
            layer.node('dec_residual3', '⊕ Residual', shape='circle', fillcolor='white')
        
        c.node('dec_final_ln', 'Final LayerNorm', fillcolor=colors['norm'])
    
    # === OUTPUT SECTION ===
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output', style='rounded', color='#FFC107', fontcolor='#FFC107')
        c.node('lm_head', 'LM Head\nLinear(256→16000)\n(Tied with Embedding)', fillcolor=colors['output'])
        c.node('logits', 'Logits\n(batch, tgt_len, 16000)', fillcolor=colors['output'])
        c.node('loss', 'CrossEntropyLoss\nlabel_smoothing=0.1', fillcolor=colors['output'])
    
    # === CONNECTIONS ===
    # Input to Embedding
    dot.edge('input_ids', 'shared_emb', label='encode')
    dot.edge('decoder_input_ids', 'shared_emb', label='decode')
    
    # Embedding to Encoder
    dot.edge('shared_emb', 'enc_ln1', label='encoder input')
    
    # Encoder Layer Flow
    dot.edge('enc_ln1', 'enc_local_global')
    dot.edge('enc_local_global', 'enc_dropout1')
    dot.edge('enc_dropout1', 'enc_residual1')
    dot.edge('enc_residual1', 'enc_ln2')
    dot.edge('enc_ln2', 'enc_ffn')
    dot.edge('enc_ffn', 'enc_dropout2')
    dot.edge('enc_dropout2', 'enc_residual2')
    dot.edge('enc_residual2', 'enc_final_ln', label='×4')
    dot.edge('enc_final_ln', 'encoder_output')
    
    # Skip connections
    dot.edge('enc_ln1', 'enc_residual1', style='dashed', color='gray')
    dot.edge('enc_ln2', 'enc_residual2', style='dashed', color='gray')
    
    # Embedding to Decoder
    dot.edge('shared_emb', 'dec_ln1', label='decoder input')
    
    # Decoder Layer Flow
    dot.edge('dec_ln1', 'dec_self_attn')
    dot.edge('dec_self_attn', 'dec_dropout1')
    dot.edge('dec_dropout1', 'dec_residual1')
    dot.edge('dec_residual1', 'dec_ln2')
    dot.edge('dec_ln2', 'dec_cross_attn')
    dot.edge('encoder_output', 'dec_cross_attn', label='K, V', style='bold', color='#FF9800')
    dot.edge('dec_cross_attn', 'dec_dropout2')
    dot.edge('dec_dropout2', 'dec_residual2')
    dot.edge('dec_residual2', 'dec_ln3')
    dot.edge('dec_ln3', 'dec_ffn')
    dot.edge('dec_ffn', 'dec_dropout3')
    dot.edge('dec_dropout3', 'dec_residual3')
    dot.edge('dec_residual3', 'dec_final_ln', label='×4')
    
    # Skip connections
    dot.edge('dec_ln1', 'dec_residual1', style='dashed', color='gray')
    dot.edge('dec_ln2', 'dec_residual2', style='dashed', color='gray')
    dot.edge('dec_ln3', 'dec_residual3', style='dashed', color='gray')
    
    # Decoder to Output
    dot.edge('dec_final_ln', 'lm_head')
    dot.edge('lm_head', 'logits')
    dot.edge('logits', 'loss')
    
    # Save DOT source file
    output_dir = Path(__file__).parent
    output_file = output_dir / output_path
    
    # Save DOT source
    dot_source = dot.source
    with open(f"{output_file}.dot", 'w', encoding='utf-8') as f:
        f.write(dot_source)
    
    # Try to render PNG/PDF if Graphviz executables are installed
    try:
        dot.render(str(output_file), format='png', cleanup=True)
        dot.render(str(output_file), format='pdf', cleanup=True)
        print(f"✅ Graphviz diagram saved to:")
        print(f"   - {output_file}.png")
        print(f"   - {output_file}.pdf")
    except Exception as e:
        print(f"⚠️  Could not render PNG/PDF (Graphviz executables not found)")
        print(f"   DOT source saved to: {output_file}.dot")
        print(f"   To render: Install Graphviz from https://graphviz.org/download/")
        print(f"   Or use online tool: https://dreampuf.github.io/GraphvizOnline/")
    
    return dot


def create_attention_pattern_diagram(output_path: str = "attention_patterns"):
    """
    Create a diagram showing local-global attention patterns.
    """
    if not GRAPHVIZ_AVAILABLE:
        return None
    
    dot = Digraph(comment='Local-Global Attention Pattern')
    dot.attr(rankdir='LR', size='14,6', dpi='150')
    dot.attr('node', shape='box', style='filled', fontname='Arial', fontsize='10')
    
    # Create sequence of tokens
    seq_len = 16
    global_block = 4  # Every 4th token is global for visualization
    
    # Token nodes
    with dot.subgraph(name='cluster_tokens') as c:
        c.attr(label='Sequence Tokens (showing 16 of 4096)', style='rounded')
        for i in range(seq_len):
            is_global = (i % global_block == 0)
            color = '#FFCDD2' if is_global else '#E3F2FD'  # Red for global, blue for local
            label = f'G{i}' if is_global else f'T{i}'
            c.node(f't{i}', label, fillcolor=color)
    
    # Create attention pattern visualization
    with dot.subgraph(name='cluster_attention') as c:
        c.attr(label='Attention Pattern (Local Radius=2)', style='rounded')
        c.node('pattern', '''
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ G │ L │ L │ L │ G │ L │ L │ L │
├───┼───┼───┼───┼───┼───┼───┼───┤
│ ■ │ ■ │ ■ │ ○ │ ■ │ ○ │ ○ │ ○ │ T0 (Global)
│ ■ │ ■ │ ■ │ ■ │ ■ │ ○ │ ○ │ ○ │ T1
│ ■ │ ■ │ ■ │ ■ │ ■ │ ○ │ ○ │ ○ │ T2
│ ○ │ ■ │ ■ │ ■ │ ■ │ ○ │ ○ │ ○ │ T3
│ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ○ │ T4 (Global)
│ ○ │ ○ │ ○ │ ■ │ ■ │ ■ │ ■ │ ■ │ T5
│ ○ │ ○ │ ○ │ ○ │ ■ │ ■ │ ■ │ ■ │ T6
│ ○ │ ○ │ ○ │ ○ │ ■ │ ■ │ ■ │ ■ │ T7
└───┴───┴───┴───┴───┴───┴───┴───┘
■ = Attends   ○ = Masked
G = Global    L = Local
''', shape='plaintext', fontname='Courier')
    
    # Legend
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Legend', style='rounded')
        c.node('global_tok', 'Global Token\n(attends to ALL)', fillcolor='#FFCDD2')
        c.node('local_tok', 'Local Token\n(attends to window)', fillcolor='#E3F2FD')
    
    output_dir = Path(__file__).parent
    output_file = output_dir / output_path
    
    # Save DOT source
    with open(f"{output_file}.dot", 'w', encoding='utf-8') as f:
        f.write(dot.source)
    
    try:
        dot.render(str(output_file), format='png', cleanup=True)
        print(f"✅ Attention pattern diagram saved to: {output_file}.png")
    except Exception:
        print(f"⚠️  DOT source saved to: {output_file}.dot")
    
    return dot


def create_torchviz_graph(model: nn.Module, output_path: str = "longt5_compute_graph"):
    """
    Create computational graph using torchviz.
    """
    if not TORCHVIZ_AVAILABLE:
        print("Torchviz not available. Skipping computational graph generation.")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    src_len = 64  # Small for visualization
    tgt_len = 16
    
    input_ids = torch.randint(0, 1000, (batch_size, src_len), device=device)
    decoder_input_ids = torch.randint(0, 1000, (batch_size, tgt_len), device=device)
    labels = torch.randint(0, 1000, (batch_size, tgt_len), device=device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
    
    # For torchviz, we need gradients
    model.train()
    outputs = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        labels=labels
    )
    
    # Create graph
    dot = make_dot(
        outputs['loss'],
        params=dict(model.named_parameters()),
        show_attrs=False,
        show_saved=False
    )
    
    output_dir = Path(__file__).parent
    output_file = output_dir / output_path
    
    # Save DOT source
    with open(f"{output_file}.dot", 'w', encoding='utf-8') as f:
        f.write(dot.source)
    
    try:
        dot.render(str(output_file), format='png', cleanup=True)
        print(f"✅ Torchviz graph saved to: {output_file}.png")
    except Exception:
        print(f"⚠️  DOT source saved to: {output_file}.dot")
    
    return dot


def create_layer_detail_diagram(output_path: str = "layer_details"):
    """
    Create detailed diagrams of individual layers.
    """
    if not GRAPHVIZ_AVAILABLE:
        return None
    
    dot = Digraph(comment='Layer Details')
    dot.attr(rankdir='TB', size='16,12', dpi='150')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
    
    colors = {
        'input': '#E8F5E9',
        'projection': '#BBDEFB',
        'attention': '#FFCDD2',
        'output': '#FFF9C4',
        'math': '#F5F5F5',
    }
    
    # === MULTI-HEAD ATTENTION DETAIL ===
    with dot.subgraph(name='cluster_mha') as c:
        c.attr(label='Multi-Head Attention (8 heads)', style='rounded', color='#F44336')
        
        # Input
        c.node('mha_input', 'Input\n(batch, seq, 256)', fillcolor=colors['input'])
        
        # Projections
        c.node('q_proj', 'Q Projection\nLinear(256→256)', fillcolor=colors['projection'])
        c.node('k_proj', 'K Projection\nLinear(256→256)', fillcolor=colors['projection'])
        c.node('v_proj', 'V Projection\nLinear(256→256)', fillcolor=colors['projection'])
        
        # Split heads
        c.node('split', 'Split Heads\n(batch, 8, seq, 32)', fillcolor=colors['math'])
        
        # Attention computation
        c.node('qk', 'Q × Kᵀ\nScaled by √32', fillcolor=colors['math'])
        c.node('pos_bias', 'Add Position Bias\n(Relative)', fillcolor=colors['attention'])
        c.node('mask', 'Apply Mask\n(Causal/Padding)', fillcolor=colors['attention'])
        c.node('softmax', 'Softmax', fillcolor=colors['math'])
        c.node('dropout_attn', 'Dropout(0.1)', fillcolor=colors['math'])
        c.node('attn_v', 'Attention × V', fillcolor=colors['math'])
        
        # Merge and output
        c.node('merge', 'Merge Heads\n(batch, seq, 256)', fillcolor=colors['math'])
        c.node('o_proj', 'Output Projection\nLinear(256→256)', fillcolor=colors['projection'])
        c.node('mha_output', 'Output\n(batch, seq, 256)', fillcolor=colors['output'])
        
        # Edges
        c.edge('mha_input', 'q_proj')
        c.edge('mha_input', 'k_proj')
        c.edge('mha_input', 'v_proj')
        c.edge('q_proj', 'split')
        c.edge('k_proj', 'split')
        c.edge('v_proj', 'split')
        c.edge('split', 'qk')
        c.edge('qk', 'pos_bias')
        c.edge('pos_bias', 'mask')
        c.edge('mask', 'softmax')
        c.edge('softmax', 'dropout_attn')
        c.edge('dropout_attn', 'attn_v')
        c.edge('split', 'attn_v', style='dashed', label='V')
        c.edge('attn_v', 'merge')
        c.edge('merge', 'o_proj')
        c.edge('o_proj', 'mha_output')
    
    # === FEED-FORWARD DETAIL ===
    with dot.subgraph(name='cluster_ffn') as c:
        c.attr(label='Feed-Forward Network', style='rounded', color='#2196F3')
        
        c.node('ffn_input', 'Input\n(batch, seq, 256)', fillcolor=colors['input'])
        c.node('wi', 'Expand\nLinear(256→1024)', fillcolor=colors['projection'])
        c.node('gelu', 'GELU\nActivation', fillcolor=colors['math'])
        c.node('dropout_ffn', 'Dropout(0.1)', fillcolor=colors['math'])
        c.node('wo', 'Contract\nLinear(1024→256)', fillcolor=colors['projection'])
        c.node('ffn_output', 'Output\n(batch, seq, 256)', fillcolor=colors['output'])
        
        c.edge('ffn_input', 'wi')
        c.edge('wi', 'gelu')
        c.edge('gelu', 'dropout_ffn')
        c.edge('dropout_ffn', 'wo')
        c.edge('wo', 'ffn_output')
    
    # === RELATIVE POSITION BIAS ===
    with dot.subgraph(name='cluster_rpb') as c:
        c.attr(label='Relative Position Bias', style='rounded', color='#4CAF50')
        
        c.node('positions', 'Query/Key\nPositions', fillcolor=colors['input'])
        c.node('rel_pos', 'Compute Relative\nPositions', fillcolor=colors['math'])
        c.node('bucket', 'Bucket\n(32 buckets)', fillcolor=colors['math'])
        c.node('bias_emb', 'Bias Embedding\nnn.Embedding(32, 8)', fillcolor=colors['projection'])
        c.node('bias_out', 'Position Bias\n(1, 8, seq, seq)', fillcolor=colors['output'])
        
        c.edge('positions', 'rel_pos')
        c.edge('rel_pos', 'bucket')
        c.edge('bucket', 'bias_emb')
        c.edge('bias_emb', 'bias_out')
    
    output_dir = Path(__file__).parent
    output_file = output_dir / output_path
    
    # Save DOT source
    with open(f"{output_file}.dot", 'w', encoding='utf-8') as f:
        f.write(dot.source)
    
    try:
        dot.render(str(output_file), format='png', cleanup=True)
        print(f"✅ Layer detail diagram saved to: {output_file}.png")
    except Exception:
        print(f"⚠️  DOT source saved to: {output_file}.dot")
    
    return dot


def print_model_summary(model: nn.Module):
    """
    Print a summary of model parameters.
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY: Simplified LongT5")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Parameter Count:")
    print(f"   Total Parameters:     {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Model Size:           ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    print(f"\n📐 Architecture:")
    print(f"   Vocabulary Size:      {model.config.vocab_size:,}")
    print(f"   Hidden Dimension:     {model.config.d_model}")
    print(f"   FFN Dimension:        {model.config.d_ff}")
    print(f"   Encoder Layers:       {model.config.num_encoder_layers}")
    print(f"   Decoder Layers:       {model.config.num_decoder_layers}")
    print(f"   Attention Heads:      {model.config.num_heads}")
    print(f"   Head Dimension:       {model.config.d_model // model.config.num_heads}")
    print(f"   Max Sequence Length:  {model.config.max_position_embeddings}")
    print(f"   Local Attention Radius: {model.config.local_radius}")
    print(f"   Global Block Size:    {model.config.global_block_size}")
    
    print(f"\n📦 Layer Breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"   {name}: {params:,} params")
    
    print("\n" + "="*60)


def main():
    """Generate all architecture visualizations."""
    print("🎨 Generating LongT5 Architecture Visualizations...")
    print("="*60)
    
    # Create model with default config
    config = LongT5Config(
        vocab_size=16000,
        d_model=256,
        d_ff=1024,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=8,
        dropout=0.1,
        max_position_embeddings=4096,
        local_radius=127,
        global_block_size=16,
    )
    model = SimplifiedLongT5(config)
    
    # Print model summary
    print_model_summary(model)
    
    # Generate visualizations
    print("\n🖼️  Generating diagrams...")
    
    # 1. Matplotlib architecture diagram (always works)
    create_matplotlib_diagram("longt5_architecture_matplotlib")
    
    # 2. Attention pattern heatmap (matplotlib)
    create_attention_heatmap("attention_pattern_heatmap")
    
    # 3. Main architecture diagram (Graphviz DOT)
    create_graphviz_diagram("longt5_architecture_graphviz")
    
    # 4. Attention pattern diagram (Graphviz DOT)
    create_attention_pattern_diagram("attention_patterns_graphviz")
    
    # 5. Layer detail diagram (Graphviz DOT)
    create_layer_detail_diagram("layer_details_graphviz")
    
    # 6. Torchviz computational graph
    create_torchviz_graph(model, "longt5_compute_graph")
    
    print("\n✅ All visualizations complete!")
    print(f"   Output directory: {Path(__file__).parent}")
    print("\n📝 Note: To render .dot files to PNG, install Graphviz:")
    print("   https://graphviz.org/download/")
    print("   Or paste DOT content into: https://dreampuf.github.io/GraphvizOnline/")


if __name__ == "__main__":
    main()
