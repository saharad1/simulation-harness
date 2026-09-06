"""Assemble a per-simulation skills root for the runtime filesystem backend.

The deepagents SkillsMiddleware discovers skills by scanning a *source*
directory for child directories that each contain a SKILL.md. There is one
active simulation per process, but the skills folder accumulates many, so we
cannot point a source at the whole skills folder (it would expose every past
simulation).

The runtime ``FilesystemBackend`` runs with ``virtual_mode=True``, which
resolves every requested path (following symlinks) and rejects anything that
escapes the backend root. That constrains the layout in two ways:

* We cannot root the backend at a throwaway directory and symlink back to the
  real skill directory — the resolved target would fall outside the root and be
  rejected.
* We must not nest the source *inside* the skill directory and point it back at
  its own parent. The previous design did exactly that with a
  ``<skill_dir>/.skills/<name> -> ..`` symlink, which left a self-referential
  directory cycle inside the published skill directory.

So we stage a *copy* of the skill in a fresh, throwaway directory that holds
nothing but this one skill, and root the backend there::

    <staging>/                       # backend root (created per simulation)
    └── .skills/
        └── <name>/                  # a copy of the skill directory
            ├── SKILL.md
            ├── schema.json
            └── ...

Copying (rather than symlinking) is required by ``virtual_mode``. It does not
risk drift: the staged files are read-only at runtime — mutable session state
flows through the StoreRegistry against the *original* skill directory, never
through this backend — and the staging directory is rebuilt for each
simulation. The caller owns the staging directory and must remove it when the
simulation ends (see ``DeepAgent.shutdown``).

``manifest.json`` is excluded from the copy: it is build provenance with no
value to the simulating agent, and the agent's readable surface is kept to the
artifacts it actually uses.
"""

import shutil
import tempfile
from pathlib import Path

from simulation_harness.skills.generation.naming import GENERATION_SEP


def build_skill_sources(skill_dir: Path) -> tuple[str, list[str]]:
    """Stage ``skill_dir`` in a fresh root and return backend root + sources.

    Creates a new temporary directory and copies ``skill_dir`` into
    ``<staging>/.skills/<skill_dir.name>``. The backend is rooted at
    ``<staging>``, so the SkillsMiddleware discovers exactly this one skill and
    no sibling simulations leak in.

    A stale ``.skills`` directory left inside ``skill_dir`` by the old design is
    skipped during the copy, so it is never propagated into the staging tree.
    ``ablations/`` is skipped too: a parent's ablated variants are other
    conditions of the experiment, and staging them would show the agent thinner
    copies of its own contract.

    Args:
        skill_dir: The simulation's skill directory (contains SKILL.md). Must be
            the flat-named path -- see the ValueError below.

    Returns:
        A tuple ``(root_dir, sources)`` where ``root_dir`` is the staging
        directory for ``FilesystemBackend(root_dir=...)`` and ``sources`` is
        ``["/.skills/"]`` for ``SkillsMiddleware(sources=...)``.

        The caller owns ``root_dir`` and must remove it (e.g. on agent
        shutdown) once the simulation ends.

    Raises:
        ValueError: if a generation directory (``<name>@<stamp>``) was passed
            instead of the flat-named path.
    """
    # `skill_dir.name` becomes the agent-visible skill directory, which deepagents validates
    # against SKILL.md's front-matter `name:` -- and only WARNS on a mismatch. So a generation
    # directory arriving here would stage the skill as `tau2-retail@2026-...`, log one warning
    # nobody reads, and complete the run under a mislabeled simulator. Refuse instead: this is
    # the one place the flat-name invariant is observable, so it is the place to enforce it.
    if GENERATION_SEP in skill_dir.name:
        raise ValueError(
            f"skill_dir must be the flat-named path, got a generation directory: "
            f"{skill_dir.name!r}. Pass the '<name>' symlink (SkillRegistry.skill_dir), not the "
            f"resolved target -- the basename reaches the agent and labels the run."
        )
    staging = Path(tempfile.mkdtemp(prefix="skill-sources-"))
    dest = staging / ".skills" / skill_dir.name
    shutil.copytree(
        skill_dir,
        dest,
        ignore=shutil.ignore_patterns(".skills", "manifest.json", "ablations"),
    )
    return str(staging), ["/.skills/"]
