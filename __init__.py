"""hermes-tabular — Hermes plugin wrapper around the tabular skill pack.

The skills themselves are plain Markdown under skills/<name>/SKILL.md and are
meant to be adapted, not imported. This plugin simply registers each skill
file with Hermes so the catalog install path works; the project-local
copy flow in the README is unchanged.
"""

from pathlib import Path

# Entry point first, then delegates in invocation order.
SKILLS = [
    "tabular",
    "tabular-inspect",
    "tabular-load",
    "tabular-split",
    "tabular-baseline",
    "tabular-catboost",
    "tabular-tabicl",
    "tabular-ensemble",
    "tabular-evaluate",
    "tabular-uncertainty",
    "tabular-checkpoint",
]


def _description(skill_md: Path) -> str:
    """Pull the description from the SKILL.md frontmatter, best effort."""
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError:
        return ""
    if not text.startswith("---"):
        return ""
    end = text.find("---", 3)
    if end == -1:
        return ""
    for line in text[3:end].splitlines():
        if line.startswith("description:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return ""


def register(ctx):
    here = Path(__file__).parent
    for name in SKILLS:
        skill_md = here / "skills" / name / "SKILL.md"
        if skill_md.is_file():
            ctx.register_skill(name, str(skill_md), _description(skill_md))
