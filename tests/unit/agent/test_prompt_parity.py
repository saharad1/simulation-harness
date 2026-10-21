"""Plan 28 C1: the v2 system prompts against the code the v1 runs used.

`full`/`minimal_skill` must be byte-identical to v1 (3ce6bd7), `no_skill` must differ only by
the S1 Store Layout section, `spec_only` must be identical to 151e80a. Runs
``scripts/check_prompt_parity.py``, which renders the old templates from git.
"""

import json
import shutil
import subprocess  # nosec B404 -- runs this repo's own script / git in a test
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "check_prompt_parity.py"


def _have_refs(*refs: str) -> bool:
    if shutil.which("git") is None:
        return False
    return all(
        subprocess.run(  # nosec B603 B607
            ["git", "-C", str(_REPO), "cat-file", "-e", f"{ref}^{{commit}}"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
        for ref in refs
    )


pytestmark = pytest.mark.skipif(
    not _have_refs("3ce6bd7", "151e80a"), reason="needs the git history (v1 commits)"
)


def _skill_dir(tmp_path: Path) -> Path:
    d = tmp_path / "petstore"
    d.mkdir()
    (d / "api.json").write_text(
        json.dumps(
            {
                "openapi": "3.0.0",
                "info": {
                    "title": "Pet Store",
                    "version": "1.2.0",
                    "description": "Pets",
                },
                "paths": {
                    "/pets/{pet_id}": {
                        "get": {
                            "operationId": "get_pet",
                            "parameters": [
                                {
                                    "name": "pet_id",
                                    "in": "path",
                                    "required": True,
                                    "schema": {"type": "string"},
                                }
                            ],
                            "responses": {"200": {"description": "A pet"}},
                        }
                    }
                },
            }
        )
    )
    (d / "schema.json").write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {
                    "pets": {"type": "array", "items": {"$ref": "#/$defs/Pet"}}
                },
                "$defs": {
                    "Pet": {
                        "x-primary-key": "pet_id",
                        "properties": {"pet_id": {"type": "string"}},
                    }
                },
            }
        )
    )
    return d


def test_prompts_match_the_v1_and_spec_only_code(tmp_path: Path) -> None:
    result = subprocess.run(  # nosec B603 -- fixed argv, this repo's script
        [sys.executable, str(_SCRIPT), str(_skill_dir(tmp_path))],
        capture_output=True,
        text=True,
        check=False,
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "full/minimal_skill vs 3ce6bd7: OK byte-identical" in out
    assert "no_skill vs 3ce6bd7: OK only the Store Layout section added" in out
    assert "spec_only vs 151e80a: OK byte-identical" in out


def test_parity_check_reports_a_difference(tmp_path: Path) -> None:
    """The check is not vacuous: `spec_only` compared against v1 (whose template has no
    `stateless` branch) is reported as a difference."""
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            str(_SCRIPT),
            "--spec-only",
            "3ce6bd7",
            str(_skill_dir(tmp_path)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "spec_only vs 3ce6bd7: DIFF" in result.stdout
