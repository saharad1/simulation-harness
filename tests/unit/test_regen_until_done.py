"""regen_until_done.sh stops at once on a failure that no retry can fix."""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "regen_until_done.sh"


def _run(tmp_path: Path, spec: Path, name: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "HARNESS_SKILLS_FOLDER": str(tmp_path / "store")}
    return subprocess.run(
        ["bash", str(SCRIPT), str(spec), name, "5", "60"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_missing_spec_stops_after_one_attempt(tmp_path: Path) -> None:
    """Observed 2026-10-20: a permanent failure was retried with 5-minute pauses. A missing
    spec file can never succeed, so the script must give up on the first attempt."""
    name = f"t-{uuid.uuid4().hex[:8]}"
    log = REPO / "logs" / f"regen-{name}.log"
    try:
        proc = _run(tmp_path, tmp_path / "nope.json", name)
        assert proc.returncode == 2, proc.stderr
        assert "not retrying" in proc.stderr
        assert len(log.read_text().splitlines()) == 1
    finally:
        log.unlink(missing_ok=True)


def test_invalid_spec_stops_after_one_attempt(tmp_path: Path) -> None:
    name = f"t-{uuid.uuid4().hex[:8]}"
    log = REPO / "logs" / f"regen-{name}.log"
    spec = tmp_path / "bad.json"
    spec.write_text('{"openapi": "3.0.0"}')
    try:
        proc = _run(tmp_path, spec, name)
        assert proc.returncode == 3, proc.stderr
        assert len(log.read_text().splitlines()) == 1
    finally:
        log.unlink(missing_ok=True)
