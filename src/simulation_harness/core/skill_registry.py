"""SkillRegistry - manages skill generation and reuse."""

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema

from simulation_harness.skills.generation.naming import (
    GENERATION_SEP,
    generation_stamp,
    is_generation_dirname,
)
from simulation_harness.skills.generator import SkillGenerator
from simulation_harness.utils.errors import DatabaseValidationError
from simulation_harness.utils.logging import get_logger

logger = get_logger(__name__)

_REQUIRED_FILES = ("SKILL.md", "schema.json", "db.json", "api.json")
# Keep in sync with `_SUBJECTS` in skills/manifest.py: that list is the
# authority on what the manifest digests, this one on what belongs to a bundle.
_BUNDLE_FILES = (
    "SKILL.md",
    "schema.json",
    "db.json",
    "scenarios.json",
    "api.json",
    "manifest.json",
)
# Bundle files that may legitimately be absent. scenarios.json is written only
# when the scenarios stage produced output; manifest.json exists only for skills
# generated after provenance was introduced. Neither is in _REQUIRED_FILES, so
# neither absence makes a skill incomplete.
_OPTIONAL_BUNDLE_FILES = frozenset({"scenarios.json", "manifest.json"})

# Where a parent skill keeps the variants ABLATED FROM IT. An ablation is not an independent
# skill -- it is a mechanical filter of one specific parent generation, and it is only
# comparable to siblings that descend from the SAME parent draw. A flat store cannot express
# that: `tau2-retail` and `tau2-retail-minskill` sat side by side as peers, so a regenerated
# parent left stale ablations beside it and nothing on disk said they no longer belonged
# together.
#
# The directory name stays the FULL skill name (`ablations/tau2-retail-minskill/`, not
# `ablations/minskill/`), so name<->directory identity is preserved and every config `skill:`
# field, artifact path and eval overlay folder keeps working.
_ABLATIONS_DIR = "ablations"

# --- generation directories -----------------------------------------------------------------
#
# EVERY skill directory is a generation, named for the draw that produced it:
#
#   skills-store/
#     tau2-retail@2026-09-06-095804/     the newest -> what the name `tau2-retail` resolves to
#     tau2-retail@2026-08-31-104036/     the previous one, left in place
#
# Before this, a skill name WAS its directory: two generations were indistinguishable without
# opening manifest.json, and regenerating overwrote the previous bundle. Now the directory name
# states which draw it is, `ls -l skills-store` answers "which generation, since when", and
# nothing is destroyed on regeneration.
#
# A NAME resolves to the NEWEST generation (the stamp is fixed-width, so lexical max is
# chronological max). Only the store speaks in stamps: every config `skill:` field, artifact
# path and eval overlay keeps using the flat name, and the API keeps reporting it. Pin an older
# draw by passing `<name>@<stamp>` explicitly, which resolves to exactly that directory.
#
# WHY `@` AND NOT `-`: see `GENERATION_SEP` in skills/generation/naming.py -- it lives beside
# the sanitizer that guarantees it. In short: a sanitized skill name cannot contain `@`, so
# `"@" in dirname` is a total classifier rather than a heuristic. That is what lets a name and a
# stamped directory share one namespace unambiguously, and it is why a plain `<name>/` directory
# still resolves (a restored bundle, a hand-placed one, a test fixture) as one unstamped
# generation. `-` would be ambiguous with the ablation suffixes (`tau2-retail-minskill` vs
# `tau2-retail-2026-...`) AND producible by the sanitizer.
_GEN_SEP = GENERATION_SEP


class SkillRegistry:
    """Manages skill generation and reuse for simulations."""

    def __init__(self, skills_folder: Path, generator: SkillGenerator) -> None:
        """Initialize the skill registry.

        Args:
            skills_folder: Directory where skills are stored
            generator: Skill generator instance
        """
        self.skills_folder = skills_folder
        self.generator = generator

    def skill_dir(self, simulation_name: str) -> Path:
        """Directory holding ``simulation_name``'s bundle.

        Resolution order:

        1. ``<name>@<stamp>`` -- an explicit generation, returned as-is. This is how an older
           draw is pinned for a comparison run.
        2. The NEWEST ``<store>/<name>@*`` generation. This is the normal path: callers name a
           skill, the store decides which draw that currently means.
        3. ``<store>/<name>/`` as a plain directory -- one unstamped generation (a bundle
           restored over the API, a hand-placed one, a test fixture).
        4. An ablation nested in the newest generation of some parent:
           ``<store>/<parent>@<stamp>/ablations/<name>/``.

        Falls back to ``<store>/<name>`` when nothing matches, so a MISSING skill reports a
        canonical location in errors rather than a speculative one.
        """
        top = self.skills_folder / simulation_name
        # An explicit `<name>@<stamp>`: the caller named a generation, so do not reinterpret it.
        if is_generation_dirname(simulation_name):
            return top
        if not self.skills_folder.is_dir():
            return top

        newest = self.newest_generation(simulation_name)
        if newest is not None:
            return newest
        # Pre-generations layout, or a restored bundle: the name IS the directory.
        if top.is_dir():
            return top

        # Skill names are flat identifiers. A name carrying a separator is not searched for as
        # an ablation (`<parent>/ablations/../..` would escape the store); it falls through to
        # the top-level join, which is what this did for every name before nesting existed --
        # so this adds no traversal surface, it just declines to widen it.
        if "/" in simulation_name or "\\" in simulation_name or simulation_name in (".", ".."):
            return top
        # Ablations are reached through their parent's NEWEST generation only. Every generation
        # keeps its own `ablations/`; resolving into a stale one would score an old variant
        # against a fresh parent -- the exact confound nesting exists to prevent.
        for parent in self._newest_generation_per_skill():
            nested = parent / _ABLATIONS_DIR / simulation_name
            if nested.is_dir():
                return nested
        return top

    def newest_generation(self, simulation_name: str) -> Path | None:
        """The most recent ``<name>@<stamp>`` directory, or None if the name has no generations.

        The stamp is fixed-width, so the lexical maximum is the chronological one -- no stat
        calls, and no dependence on mtimes that a copy or a checkout would rewrite.
        """
        gens = self.generations(simulation_name)
        return gens[-1] if gens else None

    def _newest_generation_per_skill(self) -> list[Path]:
        """One directory per skill name: its newest generation, else its plain directory."""
        newest: dict[str, Path] = {}
        for d in sorted(self.skills_folder.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            name = d.name.split(_GEN_SEP, 1)[0]
            # sorted() ascending + fixed-width stamps means the last write wins, and an
            # unstamped dir is only used when the name has no stamped generation at all.
            if is_generation_dirname(d.name) or name not in newest:
                newest[name] = d
        return list(newest.values())

    def generations(self, simulation_name: str) -> list[Path]:
        """Every generation directory for ``simulation_name``, oldest first.

        The stamp is fixed-width, so a lexical sort is chronological.
        """
        if not self.skills_folder.is_dir():
            return []
        return sorted(
            (
                d
                for d in self.skills_folder.glob(f"{simulation_name}{_GEN_SEP}*")
                if d.is_dir()
            ),
            key=lambda d: d.name,
        )

    def generation_dir(self, simulation_name: str, stamp: str) -> Path:
        """``<store>/<name>@<stamp>``. A plain join -- no probing, no I/O."""
        return self.skills_folder / f"{simulation_name}{_GEN_SEP}{stamp}"

    async def ensure_skill(
        self,
        simulation_name: str,
        openapi_spec: dict[str, Any],
        regenerate: bool = False,
        progress_cb=None,
    ) -> Path:
        """Ensure a skill exists for the simulation.

        If the skill already exists and regenerate is False, reuses it.
        Otherwise, generates a new skill.

        A skill is considered complete only when all four files exist:
        - SKILL.md
        - schema.json
        - db.json
        - api.json

        Args:
            simulation_name: Name of the simulation
            openapi_spec: OpenAPI specification
            regenerate: Whether to regenerate even if skill exists

        Returns:
            Path to the SKILL.md file

        Raises:
            RuntimeError: If skill generation fails
        """
        skill_dir = self.skill_dir(simulation_name)
        skill_file = skill_dir / "SKILL.md"

        # Check if skill is complete (all 4 files exist)
        skill_complete = self.is_complete(simulation_name)

        if skill_complete and not regenerate:
            # Log warning with human-readable mtime and regeneration reminder
            mtime = datetime.fromtimestamp(skill_file.stat().st_mtime)
            logger.warning(
                f"Reusing existing skill '{simulation_name}' "
                f"(SKILL.md modified: {mtime.isoformat()}). "
                f"If the OpenAPI spec has changed, pass regenerate=true to force regeneration."
            )
            return skill_file

        # Log which files are missing if incomplete
        if not regenerate and skill_dir.exists():
            missing_files = self.missing_files(simulation_name)
            if missing_files:
                logger.warning(
                    f"Skill '{simulation_name}' is incomplete. "
                    f"Missing files: {', '.join(missing_files)}. "
                    f"Regenerating..."
                )

        # Generate new skill
        logger.info(
            f"Generating skill: simulation={simulation_name}, regenerate={regenerate}"
        )

        try:
            result_path = await self.generator.generate_skill(
                openapi_spec=openapi_spec,
                simulation_name=simulation_name,
                skills_folder=self.skills_folder,
                progress_cb=progress_cb,
            )
            logger.info(f"Skill generated successfully: path={result_path}")
            # The generation directory itself. Nothing has to be "activated": the new draw has
            # the highest stamp, so `skill_dir(name)` resolves to it from here on. Callers that
            # need the flat skill name take it from the request, never from this basename.
            return result_path

        except Exception as e:
            logger.error(
                f"Skill generation failed: simulation={simulation_name}, error={str(e)}"
            )
            raise

    def _skill_path(self, simulation_name: str, filename: str) -> Path:
        return self.skill_dir(simulation_name) / filename

    def read_schema(self, simulation_name: str) -> dict[str, Any]:
        """Read schema.json for a skill. Raises FileNotFoundError if missing."""
        path = self._skill_path(simulation_name, "schema.json")
        if not path.exists():
            raise FileNotFoundError(
                f"schema.json not found for skill '{simulation_name}'"
            )
        return json.loads(path.read_text())

    def read_db(self, simulation_name: str) -> dict[str, Any]:
        """Read db.json for a skill. Raises FileNotFoundError if missing."""
        path = self._skill_path(simulation_name, "db.json")
        if not path.exists():
            raise FileNotFoundError(f"db.json not found for skill '{simulation_name}'")
        return json.loads(path.read_text())

    def read_api(self, simulation_name: str) -> dict[str, Any]:
        """Read api.json (the raw OpenAPI spec) for a skill. Raises FileNotFoundError if missing."""
        path = self._skill_path(simulation_name, "api.json")
        if not path.exists():
            raise FileNotFoundError(f"api.json not found for skill '{simulation_name}'")
        return json.loads(path.read_text())

    def read_bundle(self, simulation_name: str) -> dict[str, str]:
        """Return verbatim contents of all bundle files for a skill.

        Required files (SKILL.md, schema.json, db.json, api.json) must exist;
        scenarios.json and manifest.json are included only when present. Values
        are the exact on-disk text, so writing them back reproduces the files
        byte-for-byte.

        Raises:
            FileNotFoundError: if a required file is missing.
        """
        files: dict[str, str] = {}
        for fname in _BUNDLE_FILES:
            path = self._skill_path(simulation_name, fname)
            if not path.exists():
                if fname in _OPTIONAL_BUNDLE_FILES:
                    continue
                raise FileNotFoundError(
                    f"{fname} not found for skill '{simulation_name}'"
                )
            files[fname] = path.read_text()
        return files

    def missing_files(self, simulation_name: str) -> list[str]:
        """Return the required artifact files that are absent for this skill."""
        skill_dir = self.skill_dir(simulation_name)
        return [f for f in _REQUIRED_FILES if not (skill_dir / f).exists()]

    def is_complete(self, simulation_name: str) -> bool:
        """True when all four required artifacts exist for this skill."""
        return not self.missing_files(simulation_name)

    def list_complete_skills(self) -> list[str]:
        """FLAT names of complete skills, plus the ablations of each newest generation.

        Names, not directories: `tau2-retail@2026-09-06-095804/` reports as `tau2-retail`, and
        several generations of one skill collapse to a single name. Anything else and one skill
        would enumerate N times -- which loses autostart's single-skill path -- and a STAMPED
        name could reach the API's `name` field, where it would misfile every eval run keyed on
        it (`eval-skills/<name>/`, `artifacts/taubench/<name>/`).

        Only the newest generation's ablations are listed, for the same reason `skill_dir`
        resolves them there: a stale variant is not a condition of the current parent.
        """
        if not self.skills_folder.exists():
            return []
        names: list[str] = []
        for d in self._newest_generation_per_skill():
            name = d.name.split(_GEN_SEP, 1)[0]
            if self.is_complete(name):
                names.append(name)
            # Ablations nest one level only, never recursively, so the store cannot grow a tree
            # deep enough to make a name ambiguous.
            ablations = d / _ABLATIONS_DIR
            if ablations.is_dir():
                names.extend(
                    a.name
                    for a in sorted(ablations.iterdir())
                    if a.is_dir() and not a.name.startswith(".") and self.is_complete(a.name)
                )
        return sorted(names)

    def most_recent_complete_skill(self) -> str | None:
        """The complete skill whose active generation has the newest SKILL.md, else None.

        Resolves through `skill_dir`, so this compares the generations actually in use.
        """
        complete = self.list_complete_skills()
        if not complete:
            return None
        return max(
            complete,
            key=lambda name: self._skill_path(name, "SKILL.md").stat().st_mtime,
        )

    def write_db(self, simulation_name: str, db: dict[str, Any]) -> None:
        """Validate db against the skill's schema.json, then atomically replace db.json.

        Raises:
            FileNotFoundError: if schema.json is missing for this skill.
            DatabaseValidationError: if db does not validate against the schema.
        """
        schema = self.read_schema(simulation_name)  # raises FileNotFoundError

        try:
            jsonschema.validate(instance=db, schema=schema)
        except jsonschema.ValidationError as e:
            json_path = ".".join(str(p) for p in e.absolute_path) or "<root>"
            raise DatabaseValidationError(message=e.message, json_path=json_path) from e
        except jsonschema.SchemaError as e:
            raise DatabaseValidationError(
                message=f"schema is not a valid JSON Schema: {e.message}"
            ) from e

        target = self._skill_path(simulation_name, "db.json")
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(db, indent=2))
        os.replace(tmp, target)


# Made with Bob
