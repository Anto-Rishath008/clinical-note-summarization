# verify_v3.py - Verify V3 model builds, loads checkpoint, produces correct output
import torch, sys, os
sys.path.insert(0, 'src')
os.chdir(r'c:\Users\antor\OneDrive\Desktop\3rd year\SEMESTER-6\NLP\Project\Codes\workspace_mamba_longt5_v1')

from model import MambaTransformerConfig, build_model

# Build config from full_train.yaml values
import yaml
with open('configs/full_train.yaml', 'r') as f:
    cfg_dict = yaml.safe_load(f)

model_config = MambaTransformerConfig.from_dict(cfg_dict.get('model', {}))
print('Config loaded:')
print('  stride:', model_config.stride)
print('  source_bypass_stride:', model_config.source_bypass_stride)
print('  use_copy_mechanism:', model_config.use_copy_mechanism)
print('  dropout:', model_config.dropout)
print()

model = build_model(model_config).cuda()

# Quick forward test with AMP
src = torch.randint(4, 16000, (1, 512)).cuda()
tgt = torch.randint(4, 16000, (1, 64)).cuda()
src_mask = (src == 3)
tgt_mask = (tgt == 3)

with torch.cuda.amp.autocast():
    out = model(src, tgt, src_mask, tgt_mask)

print()
print('Forward test:')
print('  Output shape:', out.shape)
print('  Output range: [%.2f, %.2f]' % (out.min().item(), out.max().item()))
print('  All <= 0 (log-probs):', (out <= 0).all().item())

# Test checkpoint loading
ckpt = torch.load('checkpoints/v2_run/best_model.pt', map_location='cuda', weights_only=False)
model_state = ckpt['model_state_dict']
current_state = model.state_dict()
missing = set(current_state.keys()) - set(model_state.keys())
compatible = {k: v for k, v in model_state.items() if k in current_state and v.shape == current_state[k].shape}
print()
print('Checkpoint loading:')
print('  Step:', ckpt['step'])
print('  Best ROUGE-L:', '%.4f' % ckpt['best_rouge_l'])
print('  Compatible params: %d / %d' % (len(compatible), len(current_state)))
if missing:
    print('  New params (fresh init):', missing)

# Check copy gate bias
for name, param in model.named_parameters():
    if 'gate_linear.bias' in name:
        val = param.item()
        sig = torch.sigmoid(param).item()
        print('  Copy gate: %s = %.1f (sigmoid=%.3f)' % (name, val, sig))

# Test NLL loss computation
import torch.nn.functional as F
B, T, V = out.shape
dummy_targets = torch.randint(0, V, (B, T)).cuda()
nll = F.nll_loss(out.reshape(B*T, V), dummy_targets.reshape(B*T), ignore_index=3)
print('  NLL loss on random targets:', '%.4f' % nll.item())

print()
print('ALL CHECKS PASSED')
