from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from simulation_harness.config.models import GenerationConfig
from simulation_harness.skills.generation import pipeline as P
from simulation_harness.skills.generation.ir import (
    Entity,
    Field,
    Operation,
    OperationKind,
    SpecModel,
    StoreMetadata,
)
from simulation_harness.skills.generation.repair import GenerationStageError

SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Aha", "version": "1"},
    "paths": {
        "/features/{id}": {
            "get": {
                "operationId": "getFeature",
                "responses": {"200": {"description": "ok"}},
            }
        }
    },
}

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {"features": {"type": "array", "items": {"$ref": "#/$defs/Feature"}}},
    "$defs": {
        "Feature": {
            "type": "object",
            "additionalProperties": False,
            "x-primary-key": "id",
            "required": ["id"],
            "properties": {"id": {"type": "string"}},
        }
    },
}


def _ir() -> SpecModel:
    return SpecModel(
        api_name="Aha",
        slug="aha",
        entities=[
            Entity(
                name="Feature",
                collection="features",
                primary_key="id",
                fields=[Field(name="id", type="string", required=True)],
            )
        ],
        operations=[
            Operation(
                operation_id="getFeature",
                method="GET",
                path="/features/{id}",
                entity="Feature",
                kind=OperationKind.read,
            )
        ],
        store_metadata=StoreMetadata(
            collections=["features"], pk_map={"features": "id"}
        ),
    )


async def test_run_pipeline_produces_bundle_with_scenarios() -> None:
    phases: list[Any] = []
    with (
        patch.object(P, "analyze", AsyncMock(return_value=_ir())),
        patch.object(P, "generate_schema", AsyncMock(return_value=SCHEMA)),
        patch.object(
            P,
            "generate_scenarios",
            AsyncMock(
                return_value=[
                    __import__(
                        "simulation_harness.skills.generation.ir",
                        fromlist=["Scenario"],
                    ).Scenario(
                        title="Read",
                        intent="Read a feature.",
                        operations=["getFeature"],
                    )
                ]
            ),
        ),
        patch.object(
            P,
            "generate_behavior",
            AsyncMock(return_value="### Derivation Rules\n- total = sum of prices"),
        ),
        patch.object(
            P, "generate_seed", AsyncMock(return_value={"features": [{"id": "f1"}]})
        ),
        patch.object(
            P, "generate_section", AsyncMock(return_value="### /features/{id} GET\nok")
        ),
        patch.object(P, "build_chat"),
    ):
        bundle = await P.run_pipeline(
            SPEC,
            "aha",
            api_key=None,
            base_url=None,
            gen_config=GenerationConfig(),
            model="m",
            progress_cb=phases.append,
        )
    assert bundle.schema == SCHEMA
    assert bundle.db == {"features": [{"id": "f1"}]}
    assert bundle.scenarios == [
        {"title": "Read", "intent": "Read a feature.", "operations": ["getFeature"]}
    ]
    assert "## Example Scenarios" in bundle.skill_md
    assert "designing_schema" in phases
    assert "imagining_scenarios" in phases
    assert "seeding_database" in phases
    assert "assembling" in phases
    assert "describing_behavior" in phases
    assert "### Derivation Rules" in bundle.skill_md


async def test_run_pipeline_disabled_scenarios_skips_stage() -> None:
    phases: list[Any] = []
    with (
        patch.object(P, "analyze", AsyncMock(return_value=_ir())),
        patch.object(P, "generate_schema", AsyncMock(return_value=SCHEMA)),
        patch.object(P, "generate_scenarios", AsyncMock()) as mock_scen,
        patch.object(
            P,
            "generate_behavior",
            AsyncMock(return_value="### Derivation Rules\n- total = sum of prices"),
        ),
        patch.object(
            P, "generate_seed", AsyncMock(return_value={"features": [{"id": "f1"}]})
        ),
        patch.object(
            P, "generate_section", AsyncMock(return_value="### /features/{id} GET\nok")
        ),
        patch.object(P, "build_chat"),
    ):
        bundle = await P.run_pipeline(
            SPEC,
            "aha",
            api_key=None,
            base_url=None,
            gen_config=GenerationConfig(scenarios_enabled=False),
            model="m",
            progress_cb=phases.append,
        )
    mock_scen.assert_not_called()
    assert bundle.scenarios == []
    assert "## Example Scenarios" not in bundle.skill_md
    assert "imagining_scenarios" not in phases


async def test_run_pipeline_disabled_behavior_skips_stage() -> None:
    phases: list[Any] = []
    with (
        patch.object(P, "analyze", AsyncMock(return_value=_ir())),
        patch.object(P, "generate_schema", AsyncMock(return_value=SCHEMA)),
        patch.object(P, "generate_scenarios", AsyncMock(return_value=[])),
        patch.object(P, "generate_behavior", AsyncMock()) as mock_behavior,
        patch.object(
            P, "generate_seed", AsyncMock(return_value={"features": [{"id": "f1"}]})
        ),
        patch.object(
            P, "generate_section", AsyncMock(return_value="### /features/{id} GET\nok")
        ),
        patch.object(P, "build_chat"),
    ):
        bundle = await P.run_pipeline(
            SPEC,
            "aha",
            api_key=None,
            base_url=None,
            gen_config=GenerationConfig(behavior_enabled=False),
            model="m",
            progress_cb=phases.append,
        )
    mock_behavior.assert_not_called()
    assert "describing_behavior" not in phases
    assert "### Derivation Rules" not in bundle.skill_md


async def test_run_pipeline_behavior_stage_failure_skips_gracefully() -> None:
    phases: list[Any] = []
    with (
        patch.object(P, "analyze", AsyncMock(return_value=_ir())),
        patch.object(P, "generate_schema", AsyncMock(return_value=SCHEMA)),
        patch.object(P, "generate_scenarios", AsyncMock(return_value=[])),
        patch.object(
            P,
            "generate_behavior",
            AsyncMock(side_effect=GenerationStageError("behavior", ["boom"])),
        ),
        patch.object(
            P, "generate_seed", AsyncMock(return_value={"features": [{"id": "f1"}]})
        ),
        patch.object(
            P, "generate_section", AsyncMock(return_value="### /features/{id} GET\nok")
        ),
        patch.object(P, "build_chat"),
    ):
        bundle = await P.run_pipeline(
            SPEC,
            "aha",
            api_key=None,
            base_url=None,
            gen_config=GenerationConfig(),
            model="m",
            progress_cb=phases.append,
        )
    assert "### Derivation Rules" not in bundle.skill_md
    assert any(p.startswith("behavior_skipped") for p in phases)
    assert "assembling" in phases


async def test_run_pipeline_behavior_failure_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The degrade to the static preamble must leave a trace in logs/.

    Carrying descriptions into this stage raises its timeout probability, and a
    progress callback alone is invisible after the run.
    """
    import logging

    caplog.set_level(logging.WARNING)
    with (
        patch.object(P, "analyze", AsyncMock(return_value=_ir())),
        patch.object(P, "generate_schema", AsyncMock(return_value=SCHEMA)),
        patch.object(P, "generate_scenarios", AsyncMock(return_value=[])),
        patch.object(
            P,
            "generate_behavior",
            AsyncMock(side_effect=GenerationStageError("behavior", ["boom"])),
        ),
        patch.object(
            P, "generate_seed", AsyncMock(return_value={"features": [{"id": "f1"}]})
        ),
        patch.object(
            P, "generate_section", AsyncMock(return_value="### /features/{id} GET\nok")
        ),
        patch.object(P, "build_chat"),
    ):
        await P.run_pipeline(
            SPEC,
            "aha",
            api_key=None,
            base_url=None,
            gen_config=GenerationConfig(),
            model="m",
        )

    assert any(
        "behavior stage skipped" in message and "GenerationStageError" in message
        for message in caplog.messages
    )


async def test_run_pipeline_scenarios_failure_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The degrade to no scenarios must leave a trace in logs/.

    A progress callback alone is invisible after the run.
    """
    import logging

    caplog.set_level(logging.WARNING)
    with (
        patch.object(P, "analyze", AsyncMock(return_value=_ir())),
        patch.object(P, "generate_schema", AsyncMock(return_value=SCHEMA)),
        patch.object(
            P,
            "generate_scenarios",
            AsyncMock(side_effect=GenerationStageError("scenarios", ["boom"])),
        ),
        patch.object(
            P,
            "generate_behavior",
            AsyncMock(return_value="### Derivation Rules\n- total = sum of prices"),
        ),
        patch.object(
            P, "generate_seed", AsyncMock(return_value={"features": [{"id": "f1"}]})
        ),
        patch.object(
            P, "generate_section", AsyncMock(return_value="### /features/{id} GET\nok")
        ),
        patch.object(P, "build_chat"),
    ):
        await P.run_pipeline(
            SPEC,
            "aha",
            api_key=None,
            base_url=None,
            gen_config=GenerationConfig(),
            model="m",
        )

    assert any(
        "scenarios stage skipped" in message and "GenerationStageError" in message
        for message in caplog.messages
    )


def test_skill_bundle_carries_identity_provenance() -> None:
    bundle = P.SkillBundle(
        skill_md="x",
        schema={},
        db={},
        identity_provenance={"Order": "derived"},
    )
    assert bundle.identity_provenance == {"Order": "derived"}


def test_skill_bundle_identity_provenance_defaults_to_empty() -> None:
    assert P.SkillBundle(skill_md="x", schema={}, db={}).identity_provenance == {}


async def test_run_pipeline_passes_request_timeout_to_every_client() -> None:
    from simulation_harness.skills.generation.ir import Scenario

    calls: list[dict] = []

    def record(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return object()

    with (
        patch.object(P, "analyze", AsyncMock(return_value=_ir())),
        patch.object(P, "generate_schema", AsyncMock(return_value=SCHEMA)),
        patch.object(
            P,
            "generate_scenarios",
            AsyncMock(return_value=[Scenario(title="t", intent="i")]),
        ),
        patch.object(P, "generate_behavior", AsyncMock(return_value="")),
        patch.object(
            P, "generate_seed", AsyncMock(return_value={"features": [{"id": "f1"}]})
        ),
        patch.object(
            P, "generate_section", AsyncMock(return_value="### /features/{id} GET\nok")
        ),
        patch.object(P, "build_chat", side_effect=record),
    ):
        await P.run_pipeline(
            SPEC,
            "aha",
            api_key=None,
            base_url=None,
            gen_config=GenerationConfig(request_timeout_seconds=77),
            model="m",
        )
    assert calls and all(c.get("timeout") == 77 for c in calls), calls


# ── stage cache: a failed generation resumes from its finished stages ────────


def _stage_mocks(seed: Any, behavior: Any) -> dict[str, AsyncMock]:
    from simulation_harness.skills.generation.ir import Scenario

    return {
        "analyze": AsyncMock(return_value=_ir()),
        "generate_schema": AsyncMock(return_value=SCHEMA),
        "generate_scenarios": AsyncMock(
            return_value=[Scenario(title="t", intent="i", operations=["getFeature"])]
        ),
        "generate_behavior": AsyncMock(side_effect=behavior),
        "generate_seed": AsyncMock(side_effect=seed),
        "generate_section": AsyncMock(return_value="### /features/{id} GET\nok"),
    }


async def _run(mocks: dict[str, AsyncMock], cache: Any) -> Any:
    from contextlib import ExitStack

    with ExitStack() as stack:
        for name, mock in mocks.items():
            stack.enter_context(patch.object(P, name, mock))
        stack.enter_context(patch.object(P, "build_chat"))
        return await P.run_pipeline(
            SPEC,
            "aha",
            api_key=None,
            base_url=None,
            gen_config=GenerationConfig(),
            model="m",
            cache=cache,
        )


async def test_rerun_after_seed_failure_reuses_finished_stages(tmp_path: Any) -> None:
    from simulation_harness.skills.generation.stage_cache import StageCache

    cache = StageCache(tmp_path / "c")
    db = {"features": [{"id": "f1"}]}
    mocks = _stage_mocks(
        seed=[GenerationStageError("seed", ["boom"]), db],
        behavior=["### Derivation Rules\n- x", "### Derivation Rules\n- x"],
    )
    with pytest.raises(GenerationStageError):
        await _run(mocks, cache)
    bundle = await _run(mocks, cache)

    assert bundle.db == db
    assert "### Derivation Rules" in bundle.skill_md
    assert bundle.scenarios == [
        {"title": "t", "intent": "i", "operations": ["getFeature"]}
    ]
    for name in (
        "analyze",
        "generate_schema",
        "generate_scenarios",
        "generate_behavior",
        "generate_section",
    ):
        assert mocks[name].await_count == 1, name
    assert mocks["generate_seed"].await_count == 2


async def test_degraded_behavior_is_not_cached(tmp_path: Any) -> None:
    """A stage that fell back to its default must be retried, not frozen as the fallback."""
    from simulation_harness.skills.generation.repair import StageTimeoutError
    from simulation_harness.skills.generation.stage_cache import StageCache

    cache = StageCache(tmp_path / "c")
    mocks = _stage_mocks(
        seed=[GenerationStageError("seed", ["boom"]), {"features": [{"id": "f1"}]}],
        behavior=[StageTimeoutError("behavior", 1), "### Derivation Rules\n- y"],
    )
    with pytest.raises(GenerationStageError):
        await _run(mocks, cache)
    bundle = await _run(mocks, cache)
    assert mocks["generate_behavior"].await_count == 2
    assert "- y" in bundle.skill_md


async def test_no_cache_means_no_files(tmp_path: Any) -> None:
    mocks = _stage_mocks(seed=[{"features": [{"id": "f1"}]}], behavior=[""])
    await _run(mocks, None)
    assert list(tmp_path.iterdir()) == []


async def test_unloadable_cached_value_is_a_miss(tmp_path: Any) -> None:
    """A cache file that parses but no longer validates (e.g. after an IR change) is
    regenerated, not raised — else every retry fails identically."""
    from simulation_harness.skills.generation.stage_cache import StageCache

    cache = StageCache(tmp_path / "c")
    cache.put("analyze", {"not": "a SpecModel"})
    mocks = _stage_mocks(seed=[{"features": [{"id": "f1"}]}], behavior=[""])
    bundle = await _run(mocks, cache)
    assert mocks["analyze"].await_count == 1
    assert bundle.db == {"features": [{"id": "f1"}]}


async def test_seed_is_reseeded_when_its_scenarios_change(tmp_path: Any) -> None:
    """Seed is keyed by its inputs: scenarios that degraded on attempt 1 and succeed on
    attempt 2 must not be paired with a db seeded against no scenarios."""
    from simulation_harness.skills.generation.ir import Scenario
    from simulation_harness.skills.generation.repair import StageTimeoutError
    from simulation_harness.skills.generation.stage_cache import StageCache

    cache = StageCache(tmp_path / "c")
    db = {"features": [{"id": "f1"}]}
    mocks = _stage_mocks(seed=[db, db], behavior=["", ""])
    mocks["generate_scenarios"] = AsyncMock(
        side_effect=[
            StageTimeoutError("scenarios", 1),
            [Scenario(title="t", intent="i", operations=["getFeature"])],
        ]
    )
    with patch.object(P, "validate_bundle", side_effect=[["boom"], []]):
        with pytest.raises(GenerationStageError):
            await _run(mocks, cache)
        await _run(mocks, cache)
    assert mocks["generate_seed"].await_count == 2
