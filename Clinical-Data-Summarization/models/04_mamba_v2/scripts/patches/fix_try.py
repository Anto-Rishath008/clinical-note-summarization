lines = open('src/train.py', 'r', encoding='utf-8').readlines()
# Line 660 (0-indexed 659) is "            # Forward pass with AMP"
# Need to change it to comment + try: on next line
# Currently line 661 starts "                with autocast..." (extra indent for try)
lines[659] = '            # Forward pass with AMP + OOM protection\n'
# Insert the try: line
lines.insert(660, '            try:\n')
open('src/train.py', 'w', encoding='utf-8').writelines(lines)
print("Added try: line")
