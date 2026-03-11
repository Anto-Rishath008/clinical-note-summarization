"""Quick verification that checkpoint loads properly with new additive gating."""
import sys
sys.path.insert(0, 'src')
import torch
import yaml
from model import MambaTransformerConfig, build_model

# Load config
with open('configs/full_train.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_config = MambaTransformerConfig.from_dict(config.get('model', {}))
model = build_model(model_config)
model = model.to('cuda')

# Load checkpoint
ckpt = torch.load('checkpoints/v2_run/checkpoint_step_29800.pt', map_location='cuda', weights_only=False)
model_state = ckpt['model_state_dict']
current_state = model.state_dict()

# Check compatibility
missing = set(current_state.keys()) - set(model_state.keys())
unexpected = set(model_state.keys()) - set(current_state.keys())
print(f"Missing keys (new params): {len(missing)}")
print(f"Unexpected keys (removed): {len(unexpected)}")
if missing:
    for k in sorted(missing):
        print(f"  NEW: {k}")
if unexpected:
    for k in sorted(unexpected):
        print(f"  OLD: {k}")

# Load compatible keys
compatible = {k: v for k, v in model_state.items() if k in current_state and v.shape == current_state[k].shape}
incompatible = {k for k in model_state if k in current_state and model_state[k].shape != current_state[k].shape}
print(f"Compatible keys loaded: {len(compatible)}/{len(model_state)}")
if incompatible:
    print(f"Shape mismatch: {incompatible}")

current_state.update(compatible)
model.load_state_dict(current_state)
print("Model loaded successfully!")

# Show gate values before and after boost
print("\nCross-attention gate values (from checkpoint):")
for name, param in model.named_parameters():
    if 'cross_gate' in name:
        val = param.item()
        gate = 1.0 + torch.sigmoid(torch.tensor(val))
        print(f"  {name}: raw={val:.4f}, new additive factor=1+sigmoid({val:.2f})={gate:.4f}")

# Boost gates
with torch.no_grad():
    for name, param in model.named_parameters():
        if 'cross_gate' in name:
            param.fill_(1.0)

print("\nAfter boost:")
for name, param in model.named_parameters():
    if 'cross_gate' in name:
        val = param.item()
        gate = 1.0 + torch.sigmoid(torch.tensor(val))
        print(f"  {name}: raw={val:.4f}, additive factor={gate:.4f}")

# Quick forward pass test
print("\nRunning forward pass test...")
src = torch.randint(0, 100, (1, 512)).cuda()
tgt = torch.randint(0, 100, (1, 32)).cuda()
with torch.no_grad():
    logits = model(src, tgt)
    print(f"Forward pass OK! Output shape: {logits.shape}")

# Quick generate test
print("\nRunning generate test...")
with torch.no_grad():
    gen = model.generate(src, max_len=20, min_len=5, greedy=True, no_repeat_ngram_size=3, repetition_penalty=1.3)
    print(f"Generate OK! Output shape: {gen.shape}")

print("\n✓ All checks passed!")
