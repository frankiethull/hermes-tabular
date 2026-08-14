#!/usr/bin/env python3
import re, pathlib

repo = pathlib.Path('./skills')
for skill_dir in sorted(repo.iterdir()):
    if not skill_dir.is_dir(): continue
    md = skill_dir / 'SKILL.md'
    if not md.is_file(): continue
    text = md.read_text(encoding='utf-8')
    m = re.search(r'^---\s*\nname: ([^\n]+)\ndescription: ([^\n]+)', text, re.MULTILINE)
    if m:
        desc = m.group(2).strip()
        print(f'{m.group(1):25} {len(desc):3d} chars: {desc[:60]}...' if len(desc) > 60 else f'{m.group(1):25} {len(desc):3d} chars: {desc}')
    else:
        print(f'{"no frontmatter":25} SKIP')