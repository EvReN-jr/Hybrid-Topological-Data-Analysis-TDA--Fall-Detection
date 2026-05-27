#!/usr/bin/env python3
"""
transform_sections.py
Converts standalone *_section.tex files into itutez-compatible ch*.tex files.

Steps for each file:
1. Extract body between \\begin{document} and \\end{document}
2. Remove preamble commands (\\maketitle, \\tableofcontents, \\newpage,
   \\title{...}, \\author{...}, \\date{...})
3. Promote section hierarchy:
     \\subsubsection  ->  \\subsection
     \\subsection     ->  \\section
     \\section        ->  \\chapter
4. Prepend \\phantomsection before the first \\chapter{
5. Write result to Thesis_Final/ch<n>.tex
"""

import re
import os

SECTIONS_DIR = "/home/evo/YL_TEZ/Move/Graduate_Sections"
OUTPUT_DIR   = "/home/evo/YL_TEZ/Thesis_Final"

FILES = [
    ("introduction_section.tex",  "ch1.tex"),
    ("related_work_section.tex",  "ch2.tex"),
    ("methodology_section.tex",   "ch3.tex"),
    ("experiment_section.tex",    "ch4.tex"),
    ("results_section.tex",       "ch5.tex"),
    ("conclusion_section.tex",    "ch6.tex"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_body(text):
    """Return text between \\begin{document} and \\end{document}."""
    m = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', text, re.DOTALL)
    if m is None:
        raise ValueError("Could not find \\begin{document}...\\end{document}")
    return m.group(1)


def remove_preamble_cmds(text):
    """Remove \\maketitle, \\tableofcontents, bare \\newpage,
    \\title{...}, \\author{...}, \\date{...} lines."""
    # Simple single-line commands
    for cmd in (r'\\maketitle', r'\\tableofcontents'):
        text = re.sub(cmd + r'\s*\n?', '', text)

    # \\newpage on its own line (not inside a group)
    text = re.sub(r'\\newpage\s*\n', '\n', text)

    # \\title{...} — may span multiple lines; match balanced braces
    text = re.sub(r'\\title\{[^}]*(?:\{[^}]*\}[^}]*)?\}', '', text, flags=re.DOTALL)
    # Also the multi-line form used: \title{...\\ ... \\ ...}
    # Use a more aggressive pattern for these standalone article preamble lines
    text = re.sub(r'\\title\{.*?\n.*?\}', '', text, flags=re.DOTALL)

    # \\author{} and \\date{}
    text = re.sub(r'\\author\{[^}]*\}\s*\n?', '', text)
    text = re.sub(r'\\date\{[^}]*\}\s*\n?', '', text)

    return text


def promote_sections(text):
    """
    Promote section hierarchy using placeholder markers to avoid
    double-replacement:
        \\subsubsection  ->  §§SUBSUBSEC§§
        \\subsection     ->  §§SUBSEC§§
        \\section        ->  §§SEC§§
    then replace markers with target commands.
    """
    # Step 1: mark every variant
    text = re.sub(r'\\subsubsection', '§§SUBSUBSEC§§', text)
    text = re.sub(r'\\subsection',    '§§SUBSEC§§',    text)
    text = re.sub(r'\\section',       '§§SEC§§',       text)

    # Step 2: replace markers with promoted commands
    text = text.replace('§§SUBSUBSEC§§', r'\subsection')
    text = text.replace('§§SUBSEC§§',    r'\section')
    text = text.replace('§§SEC§§',       r'\chapter')

    return text


def add_phantomsection(text):
    """Insert \\phantomsection\\n before the very first \\chapter{ in text."""
    # Find first occurrence of \chapter{  (not \chapter*)
    idx = text.find(r'\chapter{')
    if idx == -1:
        # Also try starred form
        idx = text.find(r'\chapter*{')
    if idx == -1:
        print("  WARNING: no \\chapter found — phantomsection not added")
        return text
    return text[:idx] + '\\phantomsection\n' + text[idx:]


def process_file(src_name, dst_name):
    src_path = os.path.join(SECTIONS_DIR, src_name)
    dst_path = os.path.join(OUTPUT_DIR,   dst_name)

    print(f"Processing {src_name} -> {dst_name} ...", end=' ', flush=True)

    with open(src_path, 'r', encoding='utf-8') as fh:
        raw = fh.read()

    body = extract_body(raw)
    body = remove_preamble_cmds(body)
    body = promote_sections(body)
    body = add_phantomsection(body)

    # Strip leading blank lines
    body = body.lstrip('\n')

    with open(dst_path, 'w', encoding='utf-8') as fh:
        fh.write(body)

    lines = body.count('\n')
    print(f"OK  ({lines} lines written)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for src, dst in FILES:
        process_file(src, dst)
    print("\nAll sections transformed successfully.")
