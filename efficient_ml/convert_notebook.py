import json
import re

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# English Markdown Generation
en_md_lines = [
    "# Jupyter Notebook: Lab3.ipynb\n",
    "# **MIT 6.5940 EfficientML.ai Lab 3: Neural Architecture Search**\n"
]

for idx, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    source = ''.join(cell['source'])
    source_clean = re.sub(r'!\[(.*?)\]\(data:image/[^;]+;base64,[^\)]+\)', r'![\1](assets/\1)', source)
    
    en_md_lines.append(f"### [Cell {idx}] ({ctype.capitalize()})\n")
    if ctype == 'markdown':
        en_md_lines.append(source_clean)
        en_md_lines.append("\n")
    elif ctype == 'code':
        en_md_lines.append("```python")
        en_md_lines.append(source_clean)
        en_md_lines.append("```\n")

with open('Lab3.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(en_md_lines))

print("Saved Lab3.md successfully!")
