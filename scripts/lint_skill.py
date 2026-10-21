"""Lint a generated skill's SKILL.md against its schema.json (plan 28 B3).

Runs the generator's two static checks on an already-generated skill directory:
write rules that set fields on an id-only reference's elements, and SKILL.md
assignments of a value or field schema.json forbids (null into a non-nullable
field, a string outside an enum, an undeclared field). Prints one violation per
line, prefixed with the skill directory. Reads only; writes nothing.

Exit status: 0 when every directory is clean, 1 when any has a violation, 2 on
a usage error (missing SKILL.md / schema.json).

Usage:
    uv run python scripts/lint_skill.py <skill-dir> [<skill-dir> ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from simulation_harness.skills.generation.lint import lint_skill_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("skill_dirs", type=Path, nargs="+", metavar="skill-dir")
    args = parser.parse_args(argv)

    status = 0
    for skill_dir in args.skill_dirs:
        missing = [
            n for n in ("SKILL.md", "schema.json") if not (skill_dir / n).is_file()
        ]
        if missing:
            print(f"{skill_dir}: missing {', '.join(missing)}", file=sys.stderr)
            return 2
        violations = lint_skill_dir(skill_dir)
        for violation in violations:
            print(f"{skill_dir}: {violation}")
        if violations:
            status = 1
        else:
            print(f"{skill_dir}: ok", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
