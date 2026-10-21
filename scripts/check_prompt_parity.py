"""Check the simulator system prompt against the code the v1 runs used (plan 28 C1).

For each skill directory (``api.json`` + ``schema.json``), renders the system prompt
this checkout builds for each condition and compares it with the prompt the old code
rendered from the same spec:

* ``full`` / ``minimal_skill`` (one branch, ``skill_enabled=True``): byte-identical to
  ``--v1`` (default ``3ce6bd7``, tag ``paper-final-runs``).
* ``no_skill``: ``--v1``'s prompt plus exactly one inserted section, ``## Store Layout``,
  holding the skill's ``schema.json`` (plan 28 S1), and nothing else.
* ``spec_only``: byte-identical to ``--spec-only`` (default ``151e80a``).

The old prompts are rendered from the template file at the old commit
(``git show <ref>:<template>``) through this checkout's ``render_system_prompt``,
with the arguments the old ``DeepAgent.__init__`` passed:

* ``3ce6bd7`` ``deep_agent.py:140-145``: ``(spec)`` / ``(spec, raw_spec=...)``
* ``151e80a`` ``deep_agent.py:153-160``: ``(spec, raw_spec=..., stateless=True)``

That is exact only while ``prompts.py`` is unchanged since each old commit, which is
checked too. Reads only; writes nothing. Rerun at the freeze.

Exit status: 0 when every check passes, 1 on any difference, 2 on a usage error.

Usage:
    uv run python scripts/check_prompt_parity.py <skill-dir> [<skill-dir> ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404 -- runs git on this repo only
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

from simulation_harness.agent import prompts
from simulation_harness.agent.deep_agent import build_system_prompt
from simulation_harness.openapi.parser import OpenAPISpec

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = "src/simulation_harness/agent/templates/simulator_system.jinja2"
PROMPTS = "src/simulation_harness/agent/prompts.py"
V1 = "3ce6bd7"
SPEC_ONLY = "151e80a"


def git_show(ref: str, path: str) -> str:
    """A file's text at ``ref``."""
    return subprocess.run(  # nosec B603 B607 -- fixed argv, git on this repo
        ["git", "-C", str(REPO), "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


class _OldTemplate:
    """Stands in for ``importlib.resources.files(...)``: serves an old template's text."""

    def __init__(self, text: str) -> None:
        self._text = text

    def joinpath(self, _name: str) -> _OldTemplate:
        return self

    def read_text(self, *_args: Any, **_kwargs: Any) -> str:
        return self._text


def render_at(ref: str, spec: OpenAPISpec, **kwargs: Any) -> str:
    """The prompt ``ref``'s template renders, through this checkout's renderer."""
    old = _OldTemplate(git_show(ref, TEMPLATE))
    with patch.object(prompts.importlib.resources, "files", return_value=old):
        return prompts.render_system_prompt(spec, **kwargs)


def _first_diff(a: str, b: str) -> str:
    al, bl = a.splitlines(), b.splitlines()
    for i, (x, y) in enumerate(zip(al, bl)):
        if x != y:
            return f"line {i + 1}: {x[:120]!r} != {y[:120]!r}"
    return f"line {min(len(al), len(bl)) + 1}: lengths {len(al)} vs {len(bl)} lines"


def _without_store_layout(prompt: str, layout: str) -> str | None:
    """``prompt`` minus its S1 section, or None if it has no such section.

    The section is the template's ``\n## Store Layout\n ... ```json\n<layout>\n```\n``:
    from the blank line before its heading through the fence closing ``layout``.
    """
    heading = "\n\n## Store Layout\n"
    block = "```json\n" + layout + "\n```\n"
    start = prompt.find(heading)
    if start < 0:
        return None
    end = prompt.find(block, start)
    if end < 0 or "\n#" in prompt[start + len(heading) : end]:
        return None
    return prompt[: start + 1] + prompt[end + len(block) :]


def check(skill_dir: Path, v1: str = V1, spec_only: str = SPEC_ONLY) -> list[str]:
    """``[(condition, OK|DIFF, detail)]`` lines for one skill; DIFF lines are failures."""
    spec = OpenAPISpec(json.loads((skill_dir / "api.json").read_text(encoding="utf-8")))
    raw_spec = json.dumps(spec.spec_dict, indent=2, ensure_ascii=False)
    layout = json.dumps(
        json.loads((skill_dir / "schema.json").read_text(encoding="utf-8")),
        indent=2,
        ensure_ascii=False,
    )
    lines: list[str] = []

    def head(skill: bool, state: bool) -> str:
        return build_system_prompt(
            spec, skill_dir, skill_enabled=skill, state_enabled=state
        )

    full, old_full = head(True, True), render_at(v1, spec)
    lines.append(
        f"full/minimal_skill vs {v1}: OK byte-identical ({len(full)} chars)"
        if full == old_full
        else f"full/minimal_skill vs {v1}: DIFF {_first_diff(old_full, full)}"
    )

    no_skill, old_no_skill = head(False, True), render_at(v1, spec, raw_spec=raw_spec)
    stripped = _without_store_layout(no_skill, layout)
    if stripped == old_no_skill:
        lines.append(
            f"no_skill vs {v1}: OK only the Store Layout section added "
            f"({len(no_skill) - len(old_no_skill)} chars)"
        )
    else:
        lines.append(
            f"no_skill vs {v1}: DIFF not a single Store Layout insertion: "
            f"{_first_diff(old_no_skill, no_skill)}"
        )

    so, old_so = (
        head(False, False),
        render_at(spec_only, spec, raw_spec=raw_spec, stateless=True),
    )
    lines.append(
        f"spec_only vs {spec_only}: OK byte-identical ({len(so)} chars)"
        if so == old_so
        else f"spec_only vs {spec_only}: DIFF {_first_diff(old_so, so)}"
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("skill_dirs", type=Path, nargs="+", metavar="skill-dir")
    parser.add_argument("--v1", default=V1)
    parser.add_argument("--spec-only", default=SPEC_ONLY)
    args = parser.parse_args(argv)

    status = 0
    current = (REPO / PROMPTS).read_text(encoding="utf-8")
    for ref in (args.v1, args.spec_only):
        if git_show(ref, PROMPTS) != current:
            print(f"prompts.py differs from {ref}: old renders would not be exact")
            status = 1
    for skill_dir in args.skill_dirs:
        missing = [
            n for n in ("api.json", "schema.json") if not (skill_dir / n).is_file()
        ]
        if missing:
            print(f"{skill_dir}: missing {', '.join(missing)}", file=sys.stderr)
            return 2
        for line in check(skill_dir, args.v1, args.spec_only):
            print(f"{skill_dir.name}: {line}")
            if ": DIFF " in line:
                status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
