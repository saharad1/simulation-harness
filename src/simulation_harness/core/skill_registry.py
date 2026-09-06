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
# One skill name maps to MANY generations on disk, and exactly one is active:
#
#   skills-store/
#     tau2-retail -> tau2-retail@2026-09-06-095804     symlink, the ACTIVE generation
#     tau2-retail@2026-09-06-095804/                   a generation, uniquely named
#     tau2-retail@2026-08-31-104036/                   the previous one, left in place
#
# Before this, a regenerated skill overwrote its directory: two different generations were
# indistinguishable without opening manifest.json, and rolling back meant `mv`-ing directories
# by hand. Now the directory name states which generation it is, and rollback is repointing a
# symlink.
#
# WHY `@` AND NOT `-`: see `GENERATION_SEP` in skills/generation/naming.py -- it lives beside
# the sanitizer that guarantees it. In short: a sanitized skill name cannot contain `@`, so
# `"@" in dirname` is a total classifier rather than a heuristic, which is what lets a plain
# `<name>/` directory keep resolving unchanged (a restored bundle, a hand-placed one, a test
# fixture) instead of needing a migration. `-` would be ambiguous with the ablation suffixes
# (`tau2-retail-minskill` vs `tau2-retail-2026-...`) AND producible by the sanitizer.
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

        A top-level ``<store>/<name>/`` wins -- whether it is the active-generation symlink or
        a plain directory. Otherwise the name is looked up as an ablation nested under some
        parent: ``<store>/<parent>/ablations/<name>/``. Falls back to the top-level path so a
        MISSING skill still reports the canonical location in errors rather than a speculative
        nested one.

        Returns an UNRESOLVED path, and that is load-bearing: the basename reaches the agent's
        staging directory (`agent/skill_backend.py`) and is validated against SKILL.md's
        front-matter `name:`, so it must stay the flat skill name. Never call `.resolve()` on
        the result.
        """
        top = self.skills_folder / simulation_name
        if top.is_dir():
            return top
        if not self.skills_folder.is_dir():
            return top
        # Skill names are flat identifiers. A name carrying a separator is not searched for as
        # an ablation (`<parent>/ablations/../..` would escape the store); it falls through to
        # the top-level join, which is what this did for every name before nesting existed --
        # so this adds no traversal surface, it just declines to widen it.
        if "/" in simulation_name or "\\" in simulation_name or simulation_name in (".", ".."):
            return top
        for parent in sorted(self.skills_folder.iterdir()):
            if not parent.is_dir() or parent.name.startswith("."):
                continue
            # Only the ACTIVE generation's ablations are reachable by name. Historical
            # generations keep their own `ablations/` -- resolving into one would silently run
            # a stale variant against a fresh parent, which is the exact failure nesting exists
            # to prevent.
            if is_generation_dirname(parent.name):
                continue
            nested = parent / _ABLATIONS_DIR / simulation_name
            if nested.is_dir():
                return nested
        return top

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

    def active_generation(self, simulation_name: str) -> Path | None:
        """The generation the ``<name>`` symlink points at.

        None when ``<name>`` is a plain directory (a restored or hand-placed bundle, which is
        one unstamped generation) or absent. This is the ONLY place a skills path is resolved;
        see `skill_dir`.
        """
        link = self.skills_folder / simulation_name
        if not link.is_symlink():
            return None
        target = os.readlink(link)
        return (self.skills_folder / target) if not os.path.isabs(target) else Path(target)

    def activate(self, simulation_name: str, generation: Path) -> Path:
        """Point ``<name>`` at ``generation``, atomically. Returns the symlink path.

        Adopts a pre-existing plain directory rather than destroying it: a store written before
        generations existed is renamed to its own stamp (read from its manifest) and becomes the
        first generation, so migration is a side effect of the first write. The previous code
        `rmtree`'d whatever sat at `<name>`.

        Raises:
            ValueError: if ``generation`` is not a direct child of the skills folder. The link
                target is stored as a bare basename (so it survives a bind mount), which is
                only meaningful for a sibling -- pointing it outside the store would create a
                dangling link that reads as "skill missing" much later.
        """
        if generation.parent != self.skills_folder:
            raise ValueError(
                f"generation must be a direct child of {self.skills_folder}, got {generation}"
            )
        link = self.skills_folder / simulation_name

        if generation == link:
            # The caller is presenting the flat `<name>/` directory itself as the generation --
            # a writer that predates this layout (or a stub). There is nothing to link to: the
            # bundle already sits at the name. Leave it as one unstamped generation, which
            # `is_generation_dirname` and `list_complete_skills` both handle.
            return link

        if link.exists() and not link.is_symlink():
            # A plain directory from before this layout, and we are about to point the name at
            # a different generation. Preserve it as its own generation rather than destroying
            # it: its manifest knows when it was generated; fall back to its mtime when it
            # predates manifests entirely.
            stamp = self._adopted_stamp(link)
            adopted = self.generation_dir(simulation_name, stamp)
            if adopted.exists():
                logger.warning(
                    f"Cannot adopt {link.name}: {adopted.name} already exists; leaving it in "
                    f"place and replacing the name with a symlink."
                )
                shutil.rmtree(link)
            else:
                os.rename(link, adopted)
                logger.info(
                    f"Adopted pre-generations skill directory: {link.name} -> {adopted.name}"
                )

        # A symlink cannot be swapped in place, so build it beside the target and rename over:
        # `os.replace` onto a symlink replaces the LINK, not what it points at.
        tmp = link.with_name(f".{simulation_name}.link-{uuid.uuid4().hex[:8]}")
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        # Relative target, so the link keeps working under a bind mount (docker-compose maps
        # the host store to /app/skills-store; an absolute target would dangle there).
        os.symlink(generation.name, tmp)
        os.replace(tmp, link)
        return link

    def _adopted_stamp(self, skill_dir: Path) -> str:
        """Generation stamp for a directory that predates generations."""
        manifest = skill_dir / "manifest.json"
        if manifest.exists():
            try:
                raw = json.loads(manifest.read_text())["skill"]["generatedAt"]
                return generation_stamp(datetime.fromisoformat(raw.replace("Z", "+00:00")))
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                logger.warning(
                    f"Could not read generatedAt from {manifest}; stamping from mtime instead"
                )
        return generation_stamp(
            datetime.fromtimestamp(skill_dir.stat().st_mtime, tz=timezone.utc)
        )

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
            # Make the new generation active, and hand back the FLAT-named path. The generator
            # returns `<name>@<stamp>/SKILL.md`; `simulation_creator` derives the agent's
            # skill_dir from this return value via `.parent`, so returning the stamped path
            # would label the run with a directory name instead of the skill name.
            #
            # A generator that wrote OUTSIDE the store cannot be linked to (the link target is
            # a bare basename), so its own path is returned unchanged. That keeps a stubbed or
            # out-of-tree generator working instead of failing a successful generation on a
            # bookkeeping step.
            if result_path.parent.parent == self.skills_folder:
                self.activate(simulation_name, result_path.parent)
                return self.skill_dir(simulation_name) / "SKILL.md"
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
        """Names of complete skills: top-level ones plus each parent's nested ablations.

        Generation directories are SKIPPED, so one skill yields one name however many
        generations sit on disk. Filtering on `@` rather than on `is_symlink()` is deliberate:
        a plain `<name>/` directory is a legitimate single unstamped generation (a bundle
        restored over the API, a hand-placed one, a test fixture), and it must keep enumerating.

        Without this filter one skill enumerates N+1 times (`is_dir()` follows the symlink, and
        every generation is itself a complete directory). That is not cosmetic: it breaks the
        single-skill autostart path, and `most_recent_complete_skill` then ties on mtime between
        a symlink and its target and can return the STAMPED name -- which flows to the API's
        `name` field and corrupts every eval-side path keyed on it.
        """
        if not self.skills_folder.exists():
            return []
        names: list[str] = []
        for d in sorted(self.skills_folder.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if is_generation_dirname(d.name):
                continue
            if self.is_complete(d.name):
                names.append(d.name)
            # Ablations of this parent. Nested one level only, never recursive, so the store
            # cannot grow a tree deep enough to make a name ambiguous.
            ablations = d / _ABLATIONS_DIR
            if ablations.is_dir():
                names.extend(
                    a.name
                    for a in sorted(ablations.iterdir())
                    if a.is_dir() and not a.name.startswith(".") and self.is_complete(a.name)
                )
        return names

    def most_recent_complete_skill(self) -> str | None:
        """The complete skill with the newest SKILL.md mtime, or None if none exist.

        Reads through `skill_dir`, so for a symlinked name this is the ACTIVE generation's
        mtime -- "most recently generated skill", not "most recently touched directory".
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
