# patch_train.py - Fix train.py for log-probs output
with open('src/train.py', 'r', encoding='utf-8') as f:
    content = f.read()

edits = 0

# FIX 1: Eval criterion - replace LabelSmoothingLoss with F.nll_loss
old1 = "    criterion = LabelSmoothingLoss(smoothing=0.0, ignore_index=3)\n    \n    for batch_idx"
new1 = "    # Model returns log-probs; use F.nll_loss directly\n    \n    for batch_idx"
if old1 in content:
    content = content.replace(old1, new1, 1)
    edits += 1
    print('FIX 1: Removed eval LabelSmoothingLoss  DONE')
else:
    print('FIX 1: eval criterion not found')

# FIX 2: Eval loss computation
old2 = "            logits = model(src, tgt_input, src_mask, tgt_mask)\n            loss = criterion(logits, tgt_output)"
new2 = "            log_probs = model(src, tgt_input, src_mask, tgt_mask)\n            B, T, V = log_probs.shape\n            loss = F.nll_loss(log_probs.reshape(B * T, V), tgt_output.reshape(B * T), ignore_index=3)"
if old2 in content:
    content = content.replace(old2, new2, 1)
    edits += 1
    print('FIX 2: Eval loss -> F.nll_loss  DONE')
else:
    print('FIX 2: eval loss not found')

# FIX 3: Training criterion - remove LabelSmoothingLoss
old3 = "    # Loss\n    criterion = LabelSmoothingLoss(\n        smoothing=training_config.label_smoothing,\n        ignore_index=data_config.pad_id,\n    )"
new3 = "    # Loss - model returns log-probs, use F.nll_loss\n    pad_id = data_config.pad_id"
if old3 in content:
    content = content.replace(old3, new3, 1)
    edits += 1
    print('FIX 3: Removed training LabelSmoothingLoss  DONE')
else:
    print('FIX 3: training criterion not found')

# FIX 4: Training forward + loss
old4 = "                logits = model(src, tgt_input, src_mask, tgt_mask)\n                loss = criterion(logits, tgt_output)"
new4 = "                log_probs = model(src, tgt_input, src_mask, tgt_mask)\n                B, T, V = log_probs.shape\n                loss = F.nll_loss(log_probs.reshape(B * T, V), tgt_output.reshape(B * T), ignore_index=pad_id)"
if old4 in content:
    content = content.replace(old4, new4, 1)
    edits += 1
    print('FIX 4: Training loss -> F.nll_loss  DONE')
else:
    print('FIX 4: training loss not found')

# FIX 5: Eval min_len 50 -> 100
old5 = "            min_len=50,              # enforce minimum length"
new5 = "            min_len=100,             # enforce minimum length for clinical summaries"
if old5 in content:
    content = content.replace(old5, new5, 1)
    edits += 1
    print('FIX 5: min_len 50 -> 100  DONE')
else:
    print('FIX 5: min_len not found')

# FIX 6: Eval repetition_penalty 1.3 -> 1.0
old6 = "            repetition_penalty=1.3,"
new6 = "            repetition_penalty=1.0,  # no penalty - let copy mechanism work"
if old6 in content:
    content = content.replace(old6, new6, 1)
    edits += 1
    print('FIX 6: repetition_penalty 1.3 -> 1.0  DONE')
else:
    print('FIX 6: repetition_penalty not found')

with open('src/train.py', 'w', encoding='utf-8') as f:
    f.write(content)

print()
print('Total train.py fixes applied:', edits, '/ 6')
