"""Deterministic orchestrator for multi-step skill generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any
from dataclasses import dataclass, field

from pydantic import SecretStr

from simulation_harness.config.models import GenerationConfig
from simulation_harness.openapi.parser import OpenAPISpec
from simulation_harness.skills.generation.ir import Scenario, SpecModel
from simulation_harness.skills.generation.llm import StructuredCallError, build_chat
from simulation_harness.skills.generation.repair import (
    GenerationStageError,
    StageTimeoutError,
    guard_timeout,
)
from simulation_harness.skills.generation.stages.analyze import analyze
from simulation_harness.skills.generation.stages.assemble import (
    assemble_skill,
    render_preamble,
    validate_bundle,
)
from simulation_harness.skills.generation.stages.behavior import generate_behavior
from simulation_harness.skills.generation.stages.operations import (
    generate_section,
    plan_chunks,
)
from simulation_harness.skills.generation.stages.schema import generate_schema
from simulation_harness.skills.generation.stages.scenarios import generate_scenarios
from simulation_harness.skills.generation.stage_cache import StageCache
from simulation_harness.skills.generation.stages.seed import generate_seed
from simulation_harness.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class SkillBundle:
    skill_md: str
    schema: dict
    db: dict
    scenarios: list[dict] = field(default_factory=list)
    identity_provenance: dict[str, str] = field(default_factory=dict)


def _noop(_: str) -> None:
    return None


async def run_pipeline(
    spec_dict: dict,
    slug: str,
    *,
    api_key: SecretStr | None,
    base_url: str | None,
    gen_config: GenerationConfig,
    model: str,
    progress_cb: Callable[[str], None] | None = None,
    cache: StageCache | None = None,
) -> SkillBundle:
    """Run every generation stage and assemble the bundle.

    With ``cache``, each stage's validated output is stored as it finishes and reused on a
    rerun, so a failed generation resumes from its last finished stage. A stage that degraded
    to a fallback (behavior, scenarios) raised before ``cached`` stored anything, so it is
    retried next time rather than frozen as the fallback.
    """
    cb = progress_cb or _noop
    timeout = gen_config.stage_timeout_seconds

    async def cached(
        name: str,
        produce: Callable[[], Awaitable[Any]],
        dump: Callable[[Any], Any] = lambda v: v,
        load: Callable[[Any], Any] = lambda v: v,
    ) -> Any:
        if cache is not None:
            hit = cache.get(name)
            if hit is not None:
                # A value that parses but no longer validates (an IR model changed since it
                # was written) is a miss: raising would fail every retry identically.
                try:
                    value = load(hit)
                except (TypeError, ValueError) as e:
                    logger.warning(
                        "stage %s: cached value unusable (%s) — regenerating", name, e
                    )
                else:
                    # WARNING, not INFO: a resumed stage is not regenerated, and someone
                    # iterating on a failing generation needs to see that.
                    logger.warning("stage %s: reused from cache %s", name, cache.root)
                    return value
        value = await produce()
        if cache is not None:
            cache.put(name, dump(value))
        return value

    def chat(params, json_mode):
        return build_chat(
            api_key=api_key,
            model=model,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            base_url=base_url,
            json_mode=json_mode,
            timeout=params.request_timeout_seconds
            or gen_config.request_timeout_seconds,
        )

    # Stage 1 — analyze (extract → classify fan-out → merge; per-call timeouts inside)
    ir = await cached(
        "analyze",
        lambda: analyze(
            spec_dict,
            slug,
            extract_llm=chat(gen_config.extract, True),
            classify_llm=chat(gen_config.classify, True),
            enrich_llm=chat(gen_config.enrich, True),
            retries=gen_config.repair_retries,
            batch_cap=gen_config.classify_batch_size,
            concurrency=gen_config.concurrency,
            timeout=timeout,
            progress_cb=cb,
        ),
        dump=lambda v: v.model_dump(mode="json"),
        load=SpecModel.model_validate,
    )

    spec = OpenAPISpec(spec_dict)
    chunks = plan_chunks(ir, threshold=gen_config.chunk_threshold)
    total = len(chunks)
    sem = asyncio.Semaphore(gen_config.concurrency)
    done = 0
    lock = asyncio.Lock()

    async def one_section(chunk):
        nonlocal done
        # Keyed by the chunk's operations, so a re-chunked spec never reuses a section.
        ops = "\n".join(op.operation_id for op in chunk)
        name = f"section-{hashlib.sha256(ops.encode('utf-8')).hexdigest()[:12]}"
        async with sem:
            section = await cached(
                name,
                lambda: guard_timeout(
                    generate_section(
                        spec,
                        ir,
                        chunk,
                        chat(gen_config.operation, False),
                        retries=gen_config.repair_retries,
                    ),
                    stage="operations",
                    timeout=timeout,
                ),
            )
        async with lock:
            done += 1
            cb(f"generating_ops {done}/{total}")
        return section

    async def schema_task():
        cb("designing_schema")
        return await cached(
            "schema",
            lambda: guard_timeout(
                generate_schema(
                    ir,
                    chat(gen_config.schema_seed, True),
                    retries=gen_config.repair_retries,
                ),
                stage="schema",
                timeout=timeout,
            ),
        )

    async def scenarios_task():
        if not gen_config.scenarios_enabled:
            return []
        cb("imagining_scenarios")
        try:
            return await cached(
                "scenarios",
                lambda: guard_timeout(
                    generate_scenarios(
                        ir,
                        chat(gen_config.scenarios, True),
                        count=gen_config.scenarios_count,
                        retries=gen_config.repair_retries,
                    ),
                    stage="scenarios",
                    timeout=timeout,
                ),
                dump=lambda v: [s.model_dump(mode="json") for s in v],
                load=lambda v: [Scenario.model_validate(d) for d in v],
            )
        except (GenerationStageError, StructuredCallError, StageTimeoutError) as e:
            logger.warning(
                "scenarios stage skipped (%s: %s) — SKILL.md has no Example "
                "Scenarios section and the seed stage gets no scenarios to "
                "design against",
                type(e).__name__,
                e,
            )
            cb(f"scenarios_skipped {type(e).__name__}")
            return []

    async def behavior_task():
        if not gen_config.behavior_enabled:
            return ""
        cb("describing_behavior")
        try:
            return await cached(
                "behavior",
                lambda: guard_timeout(
                    generate_behavior(
                        ir,
                        chat(gen_config.behavior, False),
                        retries=gen_config.repair_retries,
                    ),
                    stage="behavior",
                    timeout=timeout,
                ),
            )
        except (GenerationStageError, StructuredCallError, StageTimeoutError) as e:
            logger.warning(
                "behavior stage skipped (%s: %s) — SKILL.md falls back to the "
                "static preamble invariants",
                type(e).__name__,
                e,
            )
            cb(f"behavior_skipped {type(e).__name__}")
            return ""

    # Stage: schema ∥ scenarios ∥ operation sections
    schema, scenarios, behavior_section, *sections = await asyncio.gather(
        schema_task(),
        scenarios_task(),
        behavior_task(),
        *[one_section(c) for c in chunks],
    )

    # Stage: seed (depends on schema + scenarios)
    scenario_dicts = [s.model_dump() for s in scenarios]
    cb("seeding_database")
    # Keyed by its inputs: scenarios that degraded on one attempt and succeed on the next
    # must not be paired with a db seeded against none.
    seed_inputs = json.dumps([schema, scenario_dicts], sort_keys=True)
    db = await cached(
        f"seed-{hashlib.sha256(seed_inputs.encode('utf-8')).hexdigest()[:12]}",
        lambda: guard_timeout(
            generate_seed(
                ir,
                schema,
                scenario_dicts,
                chat(gen_config.schema_seed, True),
                retries=gen_config.repair_retries,
            ),
            stage="seed",
            timeout=timeout,
        ),
    )

    # Stage: assemble + final validation
    cb("assembling")
    preamble = render_preamble(ir, scenario_dicts, behavior_section)
    skill_md = assemble_skill(ir, preamble, list(sections))
    errors = validate_bundle(ir, skill_md, schema, db)
    if errors:
        raise GenerationStageError("assemble", errors)
    return SkillBundle(
        skill_md=skill_md,
        schema=schema,
        db=db,
        scenarios=scenario_dicts,
        identity_provenance=dict(ir.identity_provenance),
    )
