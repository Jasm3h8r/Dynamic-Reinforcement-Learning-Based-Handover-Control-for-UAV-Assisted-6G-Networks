import re

with open('src/sim/simulator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and remove the PolicyNetwork class
idx_start = content.find('# ============================================================\n# POLICY GRADIENT (REINFORCE)')
idx_end = content.find('# ============================================================\n# DATA CLASSES', idx_start)

if idx_start != -1 and idx_end != -1:
    before = content[:idx_start]
    after = content[idx_end:]
    new_content = before + after
    with open('src/sim/simulator.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('PolicyNetwork class removed successfully')
else:
    print('Could not find markers')
