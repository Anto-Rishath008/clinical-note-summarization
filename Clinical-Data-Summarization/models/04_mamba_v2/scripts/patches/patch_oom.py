# patch_oom.py - Add OOM protection to training loop
lines = open('src/train.py', 'r', encoding='utf-8').readlines()

# Find line 660 (0-indexed 659): "# Forward pass with AMP"
# Replace lines 659-676 with try/except wrapped version
new_block = [
    '            # Forward pass with AMP + OOM protection\n',
    '            try:\n',
    '                with autocast("cuda", enabled=training_config.use_amp):\n',
    '                    log_probs = model(src, tgt_input, src_mask, tgt_mask)\n',
    '                    loss = F.nll_loss(\n',
    '                        log_probs.reshape(-1, log_probs.size(-1)),\n',
    '                        tgt_output.reshape(-1),\n',
    '                        ignore_index=pad_id,\n',
    '                    )\n',
    '                    loss = loss / training_config.gradient_accumulation_steps\n',
    '\n',
    '                # Check for NaN loss - skip batch if NaN\n',
    '                if torch.isnan(loss) or torch.isinf(loss):\n',
    '                    flush_print(f"WARNING: NaN/Inf loss at batch {batch_idx}, skipping...")\n',
    '                    optimizer.zero_grad()\n',
    '                    continue\n',
    '\n',
    '                # Backward pass\n',
    '                scaler.scale(loss).backward()\n',
    '            except RuntimeError as oom_err:\n',
    '                if "out of memory" in str(oom_err):\n',
    '                    flush_print(f"WARNING: OOM at batch {batch_idx}, skipping...")\n',
    '                    torch.cuda.empty_cache()\n',
    '                    optimizer.zero_grad()\n',
    '                    continue\n',
    '                else:\n',
    '                    raise\n',
]

# Lines 659 through 676 (0-indexed) are the forward+backward block
# Line 659 = "            # Forward pass with AMP"
# Line 676 = "            scaler.scale(loss).backward()"
start_idx = 659
end_idx = 677  # exclusive - line 677 is "            # Accumulate..."

new_lines = lines[:start_idx] + new_block + lines[end_idx:]
open('src/train.py', 'w', encoding='utf-8').writelines(new_lines)
print(f"Patched: replaced lines {start_idx+1}-{end_idx} with OOM-protected version")
