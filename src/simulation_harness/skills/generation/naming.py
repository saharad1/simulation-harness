"""Frontmatter name forcing for generated SKILL.md (dependency-free)."""

from __future__ import annotations

import re
from datetime import datetime, timezone

# Leading YAML frontmatter block: opening `---`, body, closing `---`.
_FRONTMATTER_RE = re.compile(r"(?s)\A(---[ \t]*\n)(.*?)(\n---[ \t]*(?:\n|\Z))")
# A top-level (unindented) `name:` key within the frontmatter body.
_NAME_LINE_RE = re.compile(r"(?m)^name[ \t]*:.*$")

# Agent Skills specification: a skill name is 1-64 characters of lowercase
# alphanumeric and single hyphens, with no leading, trailing, or consecutive
# hyphens (https://agentskills.io/specification). deepagents validates this and
# warns on violation, so generated skill names are sanitized to comply.
_MAX_SKILL_NAME_LEN = 64
# Runs of one or more hyphens, used to collapse consecutive separators.
_HYPHEN_RUN_RE = re.compile(r"-+")

# Separator between a skill name and its generation stamp in the skills-store
# (`tau2-retail_260906-095804`). It lives HERE, beside the sanitizer, because
# `sanitize_skill_name` is precisely what makes it safe: every character outside
# [a-z0-9] becomes `-`, so a sanitized name can never contain `_`. Change the
# sanitizer's alphabet and this guarantee moves with it.
GENERATION_SEP = "_"
#: Generation stamp format. UTC, and rendered from the manifest's `skill.generatedAt` rather
#: than the wall clock, so the directory name is a view of the manifest and the two cannot
#: disagree. Fixed width, so a lexical sort of generation dirs is a chronological one.
GENERATION_STAMP_FMT = "%y%m%d-%H%M%S"

#: A generation suffix: the separator, then `<yymmdd>-<HHMMSS>`, then end of string.
#: Matching the STAMP SHAPE rather than just the separator, because the store also holds
#: bookkeeping directories that legitimately contain `_` (`_archive/`). A bare
#: "contains the separator" test would classify those as generations.
_GENERATION_SUFFIX_RE = re.compile(rf"{re.escape(GENERATION_SEP)}\d{{6}}-\d{{6}}$")


def is_generation_dirname(dirname: str) -> bool:
    """True for a `<name>_<yymmdd>-<HHMMSS>` generation directory, never for a skill name."""
    return _GENERATION_SUFFIX_RE.search(dirname) is not None


def split_generation_dirname(dirname: str) -> tuple[str, str | None]:
    """`tau2-retail_260906-095804` -> `("tau2-retail", "260906-095804")`.

    An unstamped name returns `(dirname, None)`. Splitting on the LAST match rather than the
    first separator, so a name that itself contains `_` (unreachable via the sanitizer, but
    possible for a hand-placed directory) is not truncated.
    """
    m = _GENERATION_SUFFIX_RE.search(dirname)
    if m is None:
        return dirname, None
    return dirname[: m.start()], dirname[m.start() + len(GENERATION_SEP) :]


def generation_stamp(generated_at: datetime) -> str:
    """Render a manifest `generatedAt` as a directory stamp (UTC)."""
    if generated_at.tzinfo is not None:
        generated_at = generated_at.astimezone(timezone.utc)
    return generated_at.strftime(GENERATION_STAMP_FMT)


def sanitize_skill_name(raw: str) -> str:
    """Sanitize an arbitrary name into a valid Agent Skills skill name.

    Lowercases the input, replaces every character outside lowercase
    alphanumeric with a hyphen, collapses runs of hyphens, trims leading and
    trailing hyphens, and caps the length at 64. Falls back to ``"simulation"``
    if nothing valid remains (empty input, or input made entirely of invalid
    characters such as ``.`` or ``/``).

    Args:
        raw: The proposed name (e.g. a simulation name or OpenAPI title).

    Returns:
        A non-empty name satisfying the Agent Skills specification.
    """
    candidate = "".join(
        c if ((c.isalpha() and c.islower()) or c.isdigit()) else "-"
        for c in (raw or "").lower()
    )
    candidate = _HYPHEN_RUN_RE.sub("-", candidate).strip("-")
    candidate = candidate[:_MAX_SKILL_NAME_LEN].strip("-")
    return candidate or "simulation"


def _force_skill_name(skill_md: str, name: str) -> str:
    """Force the SKILL.md frontmatter ``name`` to equal ``name``."""
    name_line = f"name: {name}"
    match = _FRONTMATTER_RE.match(skill_md)
    if match is None:
        return f"---\n{name_line}\n---\n\n{skill_md}"
    head, body, tail = match.group(1), match.group(2), match.group(3)
    if _NAME_LINE_RE.search(body):
        body = _NAME_LINE_RE.sub(name_line, body, count=1)
    else:
        body = f"{name_line}\n{body}" if body else name_line
    return f"{head}{body}{tail}{skill_md[match.end() :]}"
