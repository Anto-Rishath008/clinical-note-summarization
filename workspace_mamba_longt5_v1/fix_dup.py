# fix_dup.py - Remove duplicate OOM block and stale comment
lines = open('src/train.py', 'r', encoding='utf-8').readlines()

# Remove line 660 (0-indexed 659): "            # Forward pass with AMP"  (duplicate)
# Remove lines 688-696 (0-indexed 687-695): duplicate try-except+scaler block
# After fix: line 660 becomes "# Forward pass with AMP + OOM protection"

del lines[659]  # removes the old "# Forward pass with AMP" comment at line 660

# After deletion, the duplicate block shifts up by 1
# Lines 688-696 become lines 687-695
# Let's find and remove the duplicate
# Line 687 should now be "                scaler.scale(loss).backward()"
# Line 688 should be "            except RuntimeError as oom_err:"

# Find the second "except RuntimeError as oom_err:" after the first one
first_except = None
second_except = None
for i, line in enumerate(lines):
    if 'except RuntimeError as oom_err:' in line:
        if first_except is None:
            first_except = i
        else:
            second_except = i
            break

if first_except and second_except:
    # Also remove the orphaned "scaler.scale(loss).backward()" before second except
    # It should be at second_except - 1
    start_del = second_except - 1  # the stale scaler line
    end_del = second_except + 7  # 7 lines: scaler, except, if, print, empty_cache, zero_grad, continue, else, raise
    # Actually let's count: from the orphaned scaler to "raise"
    # Find the "raise" after second_except
    for j in range(second_except, min(second_except + 10, len(lines))):
        if lines[j].strip() == 'raise':
            end_del = j + 1  # include the raise line
            break
    print(f"Removing duplicate lines {start_del+1} to {end_del}")
    del lines[start_del:end_del]
    
open('src/train.py', 'w', encoding='utf-8').writelines(lines)
print("Fixed duplicate OOM block")
