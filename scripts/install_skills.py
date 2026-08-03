#!/usr/bin/env python3
"""Add or remove hermes-tabular skills in a project.

Skills are installed into <target>/.hermes/skills/<skill-name>/.

Examples
--------
    python scripts/install_skills.py add --all --target ../my-project
    python scripts/install_skills.py add tabular tabular-catboost --target ../my-project
    python scripts/install_skills.py add --all --target ../my-project --link
    python scripts/install_skills.py remove tabular-tabicl --target ../my-project
    python scripts/install_skills.py remove --all --target ../my-project
    python scripts/install_skills.py list --target ../my-project
    python scripts/install_skills.py available
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_SKILLS = REPO_ROOT / "skills"
DEFAULT_SKILLS_DIR = ".hermes/skills"

# The orchestrator delegates to every other skill; installing it alone is a footgun.
ORCHESTRATOR = "tabular"

# Minimum useful set if someone wants a lean install.
CORE_SET = [
    "tabular",
    "tabular-inspect",
    "tabular-load",
    "tabular-split",
    "tabular-baseline",
    "tabular-catboost",
    "tabular-evaluate",
]


# --------------------------------------------------------------------------- helpers


def available_skills() -> list[str]:
    if not SOURCE_SKILLS.is_dir():
        return []
    return sorted(
        d.name for d in SOURCE_SKILLS.iterdir() if d.is_dir() and (d / "SKILL.md").is_file()
    )


def skills_root(target: Path, skills_dir: str) -> Path:
    return (target / skills_dir).resolve()


def installed_skills(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        d.name
        for d in root.iterdir()
        if (d.is_dir() or d.is_symlink()) and (d / "SKILL.md").is_file()
    )


def resolve_selection(names: list[str], use_all: bool, use_core: bool) -> list[str]:
    avail = available_skills()
    if use_all:
        return avail
    if use_core:
        return [s for s in CORE_SET if s in avail]

    unknown = [n for n in names if n not in avail]
    if unknown:
        print(f"error: unknown skill(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(avail)}", file=sys.stderr)
        raise SystemExit(2)
    return names


def warn_missing_dependencies(selected: list[str], root: Path) -> None:
    """The orchestrator references every other skill. Warn on dangling references."""
    if ORCHESTRATOR not in selected and ORCHESTRATOR not in installed_skills(root):
        return
    present = set(selected) | set(installed_skills(root))
    missing = [s for s in available_skills() if s != ORCHESTRATOR and s not in present]
    if missing:
        print(
            f"  note: '{ORCHESTRATOR}' delegates to skills that are not installed: "
            f"{', '.join(missing)}"
        )
        print("        It will still load, but those delegations will not resolve.")


# --------------------------------------------------------------------------- commands


def cmd_add(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser().resolve()
    root = skills_root(target, args.skills_dir)
    selected = resolve_selection(args.skills, args.all, args.core)

    if not selected:
        print("Nothing to install.")
        return 0

    verb = "Would install" if args.dry_run else "Installing"
    mode = "symlink" if args.link else "copy"
    print(f"{verb} {len(selected)} skill(s) into {root}  [{mode}]")

    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    for name in selected:
        src = SOURCE_SKILLS / name
        dst = root / name
        exists = dst.is_symlink() or dst.exists()
        action = "update" if exists else "add"

        if args.dry_run:
            print(f"  [{action}] {name}")
            continue

        if exists:
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)

        if args.link:
            dst.symlink_to(src, target_is_directory=True)
        else:
            shutil.copytree(src, dst)
        print(f"  [{action}] {name}")

    if not args.dry_run:
        warn_missing_dependencies(selected, root)
        print(f"\nDone. {len(installed_skills(root))} skill(s) installed.")
        if ORCHESTRATOR in selected:
            print(f"Invoke the '{ORCHESTRATOR}' skill; it delegates to the rest.")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser().resolve()
    root = skills_root(target, args.skills_dir)
    present = installed_skills(root)

    if not present:
        print(f"No hermes-tabular skills found in {root}")
        return 0

    selected = present if args.all else [n for n in args.skills if n in present]
    absent = [] if args.all else [n for n in args.skills if n not in present]
    for name in absent:
        print(f"  [skip] {name} (not installed)")

    if not selected:
        print("Nothing to remove.")
        return 0

    verb = "Would remove" if args.dry_run else "Removing"
    print(f"{verb} {len(selected)} skill(s) from {root}")

    for name in selected:
        dst = root / name
        if args.dry_run:
            print(f"  [remove] {name}")
            continue
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst)
        print(f"  [remove] {name}")

    if not args.dry_run:
        remaining = installed_skills(root)
        # Clean up an empty skills dir we created.
        if not remaining and root.is_dir() and not any(root.iterdir()):
            root.rmdir()
            print(f"Removed empty {root}")
        else:
            warn_missing_dependencies([], root)
        print(f"\nDone. {len(remaining)} skill(s) remain.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser().resolve()
    root = skills_root(target, args.skills_dir)
    present = installed_skills(root)

    if not present:
        print(f"No hermes-tabular skills installed in {root}")
        return 0

    print(f"{len(present)} skill(s) installed in {root}:")
    for name in present:
        kind = "link" if (root / name).is_symlink() else "copy"
        print(f"  {name:<24} [{kind}]")
    return 0


def cmd_available(_: argparse.Namespace) -> int:
    avail = available_skills()
    print(f"{len(avail)} skill(s) available in {SOURCE_SKILLS}:")
    for name in avail:
        marker = "  <- entry point" if name == ORCHESTRATOR else ""
        print(f"  {name}{marker}")
    print(f"\ncore set: {', '.join(CORE_SET)}")
    return 0


# --------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="install_skills.py",
        description="Add or remove hermes-tabular skills in a project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--target", default=".", help="project root (default: .)")
        p.add_argument(
            "--skills-dir",
            default=DEFAULT_SKILLS_DIR,
            help=f"skills path relative to target (default: {DEFAULT_SKILLS_DIR})",
        )

    p_add = sub.add_parser("add", help="install skills into a project")
    p_add.add_argument("skills", nargs="*", help="skill names")
    p_add.add_argument("--all", action="store_true", help="install every skill")
    p_add.add_argument("--core", action="store_true", help="install the core set only")
    p_add.add_argument("--link", action="store_true", help="symlink instead of copy")
    p_add.add_argument("--dry-run", action="store_true", help="preview only")
    add_common(p_add)
    p_add.set_defaults(func=cmd_add)

    p_rm = sub.add_parser("remove", help="remove skills from a project")
    p_rm.add_argument("skills", nargs="*", help="skill names")
    p_rm.add_argument("--all", action="store_true", help="remove every installed skill")
    p_rm.add_argument("--dry-run", action="store_true", help="preview only")
    add_common(p_rm)
    p_rm.set_defaults(func=cmd_remove)

    p_ls = sub.add_parser("list", help="list skills installed in a project")
    add_common(p_ls)
    p_ls.set_defaults(func=cmd_list)

    p_av = sub.add_parser("available", help="list skills available in this repo")
    p_av.set_defaults(func=cmd_available)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in ("add", "remove") and not args.skills:
        if not getattr(args, "all", False) and not getattr(args, "core", False):
            parser.error(f"{args.command}: give skill names, or --all (or --core for add)")

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
