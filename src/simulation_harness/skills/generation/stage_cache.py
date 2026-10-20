"""Per-stage generation outputs on disk, so a failed generation resumes instead of restarting.

A generation is ~10-25 LLM calls across stages with no checkpoint, so one stage stalling on the
gateway discarded everything: on 2026-10-20 a tau2-retail attempt finished analyze and ~20 calls,
then lost all of it to one stage timeout. Each stage's validated output is written here; a rerun
with identical inputs reads it back instead of calling the model.

Content-addressed: the directory is named by :func:`cache_key`, which covers everything that
determines a stage's output (spec, model, generation config, harness version), so a changed input
never reuses a stale stage. The caller deletes the cache once a generation publishes, so a later
regeneration is a fresh draw rather than a replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from simulation_harness import __version__
from simulation_harness.config.models import GenerationConfig
from simulation_harness.skills.manifest import canonical_spec_digest
from simulation_harness.utils.logging import get_logger

logger = get_logger(__name__)


def cache_key(spec_dict: dict, model: str, gen_config: GenerationConfig) -> str:
    """Hex sha256 over every input that determines a stage's output."""
    material = {
        "spec": canonical_spec_digest(spec_dict),
        "model": model,
        "config": gen_config.model_dump(mode="json"),
        "harness": __version__,
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class StageCache:
    """JSON values keyed by stage name, one file per stage under ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, stage: str) -> Path:
        return self.root / f"{stage}.json"

    def get(self, stage: str) -> Any | None:
        """The stored value, or ``None`` when absent or unreadable.

        Unreadable is a miss, never an error: a half-written file from a killed process
        must cost one stage's rerun, not the whole generation.
        """
        path = self._path(stage)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as e:
            logger.warning("stage cache %s unreadable (%s) — regenerating it", path, e)
            return None

    def put(self, stage: str, value: Any) -> None:
        """Write atomically (temp file, then rename), so a reader never sees a partial file."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(stage)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(json.dumps(value))
        os.replace(tmp, path)

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
