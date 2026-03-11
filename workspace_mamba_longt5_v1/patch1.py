"""Patch model.py with V3 features: source bypass + copy mechanism"""
import re

with open('src/model.py', 'r', encoding='utf-8') as f:
    content = f.read()

edits_applied = 0

# EDIT 1: Insert CopyMechanism class before MambaTransformerModel
if 'class CopyMechanism' not in content:
    copy_class = '\n\nclass CopyMechanism(nn.Module):\n'
    copy_class += '    """Pointer-generator network for copying source tokens."""\n'
    copy_class += '    def __init__(self, d_model: int, vocab_size: int):\n'
    copy_class += '        super().__init__()\n'
    copy_class += '        self.gate_linear = nn.Linear(d_model, 1)\n'
    copy_class += '        self.copy_proj = nn.Linear(d_model, d_model, bias=False)\n'
    copy_class += '        self.vocab_size = vocab_size\n'
    copy_class += '        nn.init.constant_(self.gate_linear.bias, 3.0)\n\n'
    copy_class += '    def forward(self, decoded, vocab_logits, source_detail, source_ids, source_mask):\n'
    copy_class += '        B, T, D = decoded.shape\n'
    copy_class += '        S = source_detail.size(1)\n'
    copy_class += '        V = self.vocab_size\n'
    copy_class += '        p_gen = torch.sigmoid(self.gate_linear(decoded))\n'
    copy_class += '        vocab_probs = F.softmax(vocab_logits, dim=-1)\n'
    copy_class += '        query = self.copy_proj(decoded)\n'
    copy_class += '        attn_scores = torch.bmm(query, source_detail.transpose(1, 2)) / math.sqrt(D)\n'
    copy_class += '        S_raw = source_mask.size(1)\n'
    copy_class += '        if S < S_raw:\n'
    copy_class += '            stride = max(1, S_raw // S)\n'
    copy_class += '            bypass_mask = source_mask[:, ::stride][:, :S]\n'
    copy_class += '        else:\n'
    copy_class += '            bypass_mask = source_mask[:, :S]\n'
    copy_class += '        attn_scores = attn_scores.masked_fill(bypass_mask.unsqueeze(1), float("-inf"))\n'
    copy_class += '        copy_attn = F.softmax(attn_scores, dim=-1)\n'
    copy_class += '        if S < S_raw:\n'
    copy_class += '            stride = max(1, S_raw // S)\n'
    copy_class += '            bypass_ids = source_ids[:, ::stride][:, :S]\n'
    copy_class += '        else:\n'
    copy_class += '            bypass_ids = source_ids[:, :S]\n'
    copy_class += '        copy_probs = torch.zeros(B, T, V, device=decoded.device, dtype=decoded.dtype)\n'
    copy_class += '        bypass_ids_exp = bypass_ids.unsqueeze(1).expand(B, T, S)\n'
    copy_class += '        copy_probs.scatter_add_(2, bypass_ids_exp, copy_attn)\n'
    copy_class += '        combined = p_gen * vocab_probs + (1.0 - p_gen) * copy_probs\n'
    copy_class += '        log_probs = torch.log(combined + 1e-12)\n'
    copy_class += '        return log_probs\n\n\n'
    marker = 'class MambaTransformerModel(nn.Module):'
    idx = content.find(marker)
    if idx >= 0:
        content = content[:idx] + copy_class + content[idx:]
        edits_applied += 1
        print('1. CopyMechanism class inserted')
    else:
        print('1. ERROR: marker not found')
else:
    print('1. CopyMechanism already present')

with open('src/model.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Edits applied:', edits_applied)
