#!/usr/bin/env python3
"""Validate the hermes-tabular skill pack.

Checks
------
1. Every skill directory contains a SKILL.md.
2. Every SKILL.md has YAML frontmatter with `name` and `description`.
3. The declared `name` matches the directory name.
4. Descriptions are present and a reasonable length.
5. Every `tabular-*` skill referenced by the orchestrator exists.
6. defaults.yaml parses and contains the expected top-level sections.

Usage
-----
    python scripts/validate_skills.py
    python scripts/validate_skills.py --quiet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
ORCHESTRATOR = "tabular"

REQUIRED_KEYS = ("name", "description")
EXPECTED_DEFAULTS_SECTIONS = (
    "dataset_size",
    "task_detection",
    "split",
    "metrics",
    "models",
    "uncertainty",
    "checkpoint",
)

MIN_DESCRIPTION_CHARS = 40
MAX_DESCRIPTION_CHARS = 500

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
SKILL_REF_RE = re.compile(r"\btabular-[a-z]+\b")


def parse_frontmatter(text: str) -> tuple[dict | None, str | None]:
    """Minimal frontmatter parser. Avoids a hard pyyaml dependency for a flat block."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None, "missing YAML frontmatter delimited by ---"

    block = m.group(1)
    data: dict[str, str] = {}
    key = None
    for raw in block.splitlines():
        if not raw.strip():
            continue
        if raw[:1].isspace() and key:          # folded continuation line
            data[key] = f"{data[key]} {raw.strip()}".strip()
            continue
        if ":" not in raw:
            return None, f"malformed frontmatter line: {raw!r}"
        key, value = raw.split(":", 1)
        key = key.strip()
        data[key] = value.strip().strip('"').strip("'")
    return data, None


def validate_skill(path: Path) -> tuple[list[str], list[str], dict]:
    errors: list[str] = []
    warnings: list[str] = []
    skill_md = path / "SKILL.md"

    if not skill_md.is_file():
        return [f"{path.name}: no SKILL.md"], warnings, {}

    text = skill_md.read_text(encoding="utf-8")
    data, err = parse_frontmatter(text)
    if err:
        return [f"{path.name}: {err}"], warnings, {}

    for key in REQUIRED_KEYS:
        if key not in data or not data[key]:
            errors.append(f"{path.name}: frontmatter missing '{key}'")

    if data.get("name") and data["name"] != path.name:
        errors.append(
            f"{path.name}: frontmatter name '{data['name']}' != directory '{path.name}'"
        )

    desc = data.get("description", "")
    if desc:
        if len(desc) < MIN_DESCRIPTION_CHARS:
            warnings.append(f"{path.name}: description is very short ({len(desc)} chars)")
        if len(desc) > MAX_DESCRIPTION_CHARS:
            warnings.append(f"{path.name}: description is long ({len(desc)} chars)")

    body = text[len(FRONTMATTER_RE.match(text).group(0)):]
    if len(body.strip()) < 200:
        warnings.append(f"{path.name}: body looks thin ({len(body.strip())} chars)")

    return errors, warnings, data


def validate_defaults() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    path = SKILLS_DIR / ORCHESTRATOR / "defaults.yaml"

    if not path.is_file():
        return [f"{ORCHESTRATOR}/defaults.yaml: missing"], warnings

    try:
        import yaml
    except ImportError:
        warnings.append("pyyaml not installed; skipped defaults.yaml parse check")
        return errors, warnings

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"defaults.yaml: invalid YAML — {exc}"], warnings

    if not isinstance(data, dict):
        return ["defaults.yaml: top level is not a mapping"], warnings

    for section in EXPECTED_DEFAULTS_SECTIONS:
        if section not in data:
            errors.append(f"defaults.yaml: missing section '{section}'")

    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the hermes-tabular skill pack.")
    parser.add_argument("--quiet", action="store_true", help="only print problems")
    args = parser.parse_args(argv)

    if not SKILLS_DIR.is_dir():
        print(f"error: {SKILLS_DIR} not found", file=sys.stderr)
        return 2

    dirs = sorted(d for d in SKILLS_DIR.iterdir() if d.is_dir())
    all_errors: list[str] = []
    all_warnings: list[str] = []
    names: set[str] = set()

    for d in dirs:
        errors, warnings, data = validate_skill(d)
        all_errors += errors
        all_warnings += warnings
        if data.get("name"):
            names.add(data["name"])
        if not args.quiet and not errors:
            desc = data.get("description", "")
            print(f"  ok  {d.name:<24} {desc[:58]}{'...' if len(desc) > 58 else ''}")

    # Cross-reference check on the orchestrator.
    orch_md = SKILLS_DIR / ORCHESTRATOR / "SKILL.md"
    if orch_md.is_file():
        referenced = set(SKILL_REF_RE.findall(orch_md.read_text(encoding="utf-8")))
        dangling = sorted(referenced - names)
        if dangling:
            all_errors.append(
                f"{ORCHESTRATOR}: references non-existent skill(s): {', '.join(dangling)}"
            )
        unreferenced = sorted(names - referenced - {ORCHESTRATOR})
        if unreferenced:
            all_warnings.append(
                f"{ORCHESTRATOR}: never delegates to {', '.join(unreferenced)}"
            )
    else:
        all_errors.append(f"{ORCHESTRATOR}/SKILL.md: missing orchestrator skill")

    d_err, d_warn = validate_defaults()
    all_errors += d_err
    all_warnings += d_warn

    print()
    for w in all_warnings:
        print(f"  warn  {w}")
    for e in all_errors:
        print(f"  FAIL  {e}")

    print(
        f"\n{len(dirs)} skill(s) checked — "
        f"{len(all_errors)} error(s), {len(all_warnings)} warning(s)"
    )
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
