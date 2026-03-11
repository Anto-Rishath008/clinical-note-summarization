# patch3.py - Fix remaining V3 issues in model.py
with open('src/model.py', 'r', encoding='utf-8') as f:
    content = f.read()

edits = 0

# FIX 1: CopyMechanism gate bias 0.0 -> 3.0
old1 = "nn.init.constant_(self.gate_linear.bias, 0.0)  # sigmoid(0) = 0.5 balanced"
new1 = "nn.init.constant_(self.gate_linear.bias, 3.0)  # sigmoid(3) = 0.95, mostly-generate start"
if old1 in content:
    content = content.replace(old1, new1, 1)
    edits += 1
    print('FIX 1: Gate bias 0.0 -> 3.0  DONE')
else:
    print('FIX 1: gate bias text not found')

# FIX 2: Add copy mechanism to greedy generate
# Insert after "next_logits = logits[:, -1, :]" in greedy path
old2 = """            logits = self.decode(generated, memory, causal_mask)
            next_logits = logits[:, -1, :]

            # Block EOS before min_len"""
new2 = """            logits = self.decode(generated, memory, causal_mask)
            next_logits = logits[:, -1, :]

            # V3: Apply copy mechanism if available
            if self.copy_mechanism is not None and self._source_detail is not None:
                last_hidden = self._last_decoded[:, -1:, :]
                step_logits = next_logits.unsqueeze(1)
                step_log_probs = self.copy_mechanism(
                    last_hidden, self._source_detail,
                    self._source_ids, step_logits, self._source_mask,
                )
                next_logits = step_log_probs.squeeze(1)

            # Block EOS before min_len"""
if old2 in content:
    content = content.replace(old2, new2, 1)
    edits += 1
    print('FIX 2: Greedy generate copy mechanism  DONE')
else:
    print('FIX 2: greedy generate text not found')

# FIX 3: Add source detail expansion in beam search
old3 = """            mem_b = memory[b:b+1].expand(beam_size, -1, -1)

            # Initialize beams"""
new3 = """            mem_b = memory[b:b+1].expand(beam_size, -1, -1)

            # V3: Expand source detail for beams
            detail_b = None
            src_ids_b = None
            src_mask_b = None
            if self.copy_mechanism is not None and self._source_detail is not None:
                detail_b = self._source_detail[b:b+1].expand(beam_size, -1, -1)
                src_ids_b = self._source_ids[b:b+1].expand(beam_size, -1)
                src_mask_b = self._source_mask[b:b+1].expand(beam_size, -1)

            # Initialize beams"""
if old3 in content:
    content = content.replace(old3, new3, 1)
    edits += 1
    print('FIX 3: Beam search source detail expansion  DONE')
else:
    print('FIX 3: beam search init text not found')

# FIX 4: Add copy mechanism to beam search decode step
old4 = """                logits = self.decode(beam_seqs, mem_b, causal_mask)
                next_logits = logits[:, -1, :]  # [beam_size, vocab]
                
                # Apply repetition penalty"""
new4 = """                logits = self.decode(beam_seqs, mem_b, causal_mask)
                next_logits = logits[:, -1, :]  # [beam_size, vocab]

                # V3: Apply copy mechanism in beam search
                if self.copy_mechanism is not None and detail_b is not None:
                    last_hidden = self._last_decoded[:, -1:, :]
                    step_logits = next_logits.unsqueeze(1)
                    step_log_probs = self.copy_mechanism(
                        last_hidden, detail_b,
                        src_ids_b, step_logits, src_mask_b,
                    )
                    next_logits = step_log_probs.squeeze(1)
                
                # Apply repetition penalty"""
if old4 in content:
    content = content.replace(old4, new4, 1)
    edits += 1
    print('FIX 4: Beam search copy mechanism  DONE')
else:
    print('FIX 4: beam search decode text not found')

# FIX 5: Update build_model() with V3 features
old5 = "    if config.use_gated_cross_attn: v2_features.append('GatedCrossAttn')"
new5 = """    if config.use_gated_cross_attn: v2_features.append('GatedCrossAttn')
    if getattr(config, 'source_bypass_stride', 0) > 0: v2_features.append(f'SourceBypass(stride={config.source_bypass_stride})')
    if getattr(config, 'use_copy_mechanism', False): v2_features.append('CopyMechanism')"""
if old5 in content:
    content = content.replace(old5, new5, 1)
    edits += 1
    print('FIX 5: build_model() V3 features  DONE')
else:
    print('FIX 5: build_model text not found')

with open('src/model.py', 'w', encoding='utf-8') as f:
    f.write(content)

print()
print('Total fixes applied:', edits, '/ 5')
