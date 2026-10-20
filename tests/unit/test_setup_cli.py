"""Tests for the offline setup CLI (fail-fast paths, no LLM)."""

import json
from pathlib import Path

import pytest

from simulation_harness.setup_cli import main


@pytest.fixture
def spec_file(tmp_path: Path) -> Path:
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "acme", "version": "1"},
        "paths": {},
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    return p


def test_main_exits_nonzero_on_missing_file(tmp_path: Path) -> None:
    rc = main([str(tmp_path / "nope.json")])
    assert rc != 0


def test_main_exits_nonzero_on_invalid_spec(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "an openapi spec"}))
    rc = main([str(bad)])
    assert rc != 0


def test_setup_cli_prints_the_cause_chain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The outer "Skill generation failed" says nothing; the stage that stalled must show."""
    from unittest.mock import patch

    from simulation_harness import setup_cli
    from simulation_harness.skills.generation.repair import StageTimeoutError

    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {"openapi": "3.0.0", "info": {"title": "t", "version": "1"}, "paths": {}}
        )
    )

    async def fails(**_: object) -> Path:
        try:
            raise StageTimeoutError("schema", 1500)
        except StageTimeoutError as e:
            raise RuntimeError("Skill generation failed for 't'") from e

    with patch.object(setup_cli, "run_setup", fails):
        assert main([str(spec)]) == 1
    err = capsys.readouterr().err
    assert "Skill generation failed for 't'" in err
    assert "StageTimeoutError" in err and "'schema'" in err
