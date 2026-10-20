"""StageCache: per-stage outputs on disk, so a failed generation resumes."""

from __future__ import annotations

from pathlib import Path

from simulation_harness.config.models import GenerationConfig
from simulation_harness.skills.generation.stage_cache import StageCache, cache_key

SPEC = {"openapi": "3.0.0", "info": {"title": "t", "version": "1"}, "paths": {}}


def test_put_then_get_round_trips(tmp_path: Path) -> None:
    cache = StageCache(tmp_path / "c")
    cache.put("schema", {"a": 1})
    assert cache.get("schema") == {"a": 1}


def test_missing_stage_is_none(tmp_path: Path) -> None:
    assert StageCache(tmp_path / "c").get("schema") is None


def test_root_is_created_lazily(tmp_path: Path) -> None:
    StageCache(tmp_path / "c").get("schema")
    assert not (tmp_path / "c").exists()


def test_corrupt_file_is_a_miss(tmp_path: Path) -> None:
    """A process killed mid-write must not poison the next attempt."""
    root = tmp_path / "c"
    root.mkdir()
    (root / "schema.json").write_text("{not json")
    assert StageCache(root).get("schema") is None


def test_put_leaves_no_tmp_file(tmp_path: Path) -> None:
    cache = StageCache(tmp_path / "c")
    cache.put("schema", {"a": 1})
    assert sorted(p.name for p in (tmp_path / "c").iterdir()) == ["schema.json"]


def test_clear_removes_the_directory(tmp_path: Path) -> None:
    cache = StageCache(tmp_path / "c")
    cache.put("schema", {"a": 1})
    cache.clear()
    assert not (tmp_path / "c").exists()
    cache.clear()  # idempotent


def test_key_changes_with_spec_model_and_config() -> None:
    base = cache_key(SPEC, "m", GenerationConfig())
    assert cache_key({**SPEC, "paths": {"/x": {}}}, "m", GenerationConfig()) != base
    assert cache_key(SPEC, "other", GenerationConfig()) != base
    assert cache_key(SPEC, "m", GenerationConfig(concurrency=1)) != base


def test_key_is_stable_across_dict_order() -> None:
    reordered = {
        "paths": {},
        "info": {"version": "1", "title": "t"},
        "openapi": "3.0.0",
    }
    assert cache_key(reordered, "m", GenerationConfig()) == cache_key(
        SPEC, "m", GenerationConfig()
    )
