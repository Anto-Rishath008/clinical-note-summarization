lines = open('src/train.py', 'r', encoding='utf-8').readlines()

# Find where the forward pass block starts and ends
# The block starts after "tokens_processed += batch_tokens"
# and ends before "running_loss += loss.item()"
start = None
end = None
for i, line in enumerate(lines):
    if 'tokens_processed += batch_tokens' in line:
        start = i + 1  # next line after this
    if start and 'running_loss += loss.item()' in line:
        end = i
        break

print(f"Replacing lines {start+1} to {end+1}")

# Print what we're replacing
for i in range(start, end):
    print(f"  OLD {i+1}: {lines[i].rstrip()}")

# New correct block
new_block = '''
            # Forward pass with AMP + OOM protection
            try:
                with autocast('cuda', enabled=training_config.use_amp):
                    log_probs = model(src, tgt_input, src_mask, tgt_mask)
                    loss = F.nll_loss(
                        log_probs.reshape(-1, log_probs.size(-1)),
                        tgt_output.reshape(-1),
                        ignore_index=pad_id,
                    )
                    loss = loss / training_config.gradient_accumulation_steps
                
                # Check for NaN loss - skip batch if NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    flush_print(f"WARNING: NaN/Inf loss at batch {batch_idx}, skipping...")
                    optimizer.zero_grad()
                    continue
                
                # Backward pass
                scaler.scale(loss).backward()
            except RuntimeError as oom_err:
                if 'out of memory' in str(oom_err):
                    flush_print(f"WARNING: OOM at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise
'''

new_lines = lines[:start] + [l + '\n' for l in new_block.split('\n')] + lines[end:]
open('src/train.py', 'w', encoding='utf-8').writelines(new_lines)
print("Clean rewrite done")
