"""Plan 28 step 3 failure-path fixes, end to end on the real agent graph.

Each fix acts only when its bug shows up, and the trace record's ``fix_fired`` names it; a
call that does not hit the bug runs and is traced exactly as before (``fix_fired == []``).
Runs the REAL LangGraph agent (SkillsMiddleware + FilesystemMiddleware + state tools) with a
scripted fake chat model.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import SecretStr

from simulation_harness.agent.deep_agent import DeepAgent
from simulation_harness.agent.trace import TRACE_FILE_NAME

_FIXTURE_SKILL_DIR = (
    Path(__file__).resolve().parents[2] / "fixtures" / "restaurant-reservation-api"
)


class _ScriptedChatModel(BaseChatModel):
    """Returns the scripted AI messages in order; records every prompt it was shown."""

    script: list[AIMessage]
    seen: list[list[dict]] = []
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(
            [
                {"type": m.type, "content": m.content}
                for m in messages
                if m.type != "system"
            ]
        )
        message = self.script[self.cursor].model_copy()
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def _operation(operation_id: str = "getRestaurant") -> Any:
    from simulation_harness.openapi.parser import OpenAPIOperation

    op = Mock(spec=OpenAPIOperation)
    op.method = "get"
    op.path = "/restaurants/{id}"
    op.operation_id = operation_id
    op.summary = "Get a restaurant"
    op.description = "Get a restaurant"
    op.parameters = []
    op.request_body = None
    op.get_request_schema = Mock(return_value=None)
    op.get_response_schema = Mock(return_value={"type": "object"})
    return op


def _spec() -> Any:
    from simulation_harness.openapi.parser import OpenAPISpec

    spec = Mock(spec=OpenAPISpec)
    spec.info = {"title": "Test API", "version": "1.0.0", "description": "Test"}
    spec.servers = [{"url": "https://api.example.com"}]
    spec.operations = []
    spec.spec_dict = {"openapi": "3.0.0", "paths": {}}
    return spec


def _agent(
    trace_dir: Path | None,
    script: list[AIMessage],
    *,
    skill_dir: Path = _FIXTURE_SKILL_DIR,
    operations: list[Any] | None = None,
    skill_enabled: bool = True,
) -> tuple[DeepAgent, _ScriptedChatModel]:
    agent = DeepAgent(
        api_key=SecretStr("test-key"),
        model="gpt-4",
        temperature=0.0,
        max_tokens=1000,
        base_url=None,
        spec=_spec(),
        operations=operations or [_operation()],
        skill_dir=skill_dir,
        trace_dir=trace_dir,
        skill_enabled=skill_enabled,
    )
    fake = _ScriptedChatModel(script=script, seen=[])
    agent.llm = fake  # type: ignore[assignment]
    agent.agent = agent._create_agent()
    return agent, fake


def _records(trace_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (trace_dir / TRACE_FILE_NAME).read_text().splitlines()
    ]


def _call(call_id: str, name: str, **args: Any) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"id": call_id, "name": name, "args": args}]
    )


_OK = AIMessage(content='{"id": "rest_001", "name": "The Italian Corner"}')


# --- S3a: a where path that exists in no record --------------------------------------------


@pytest.mark.asyncio
async def test_s3a_bad_path_is_answered_with_real_paths_and_traced(
    tmp_path: Path,
) -> None:
    script = [
        _call("c1", "state_list", store="restaurants", where={"location.zip": "02108"}),
        _call(
            "c2",
            "state_list",
            store="restaurants",
            where={"location.postal_code": "02108"},
        ),
        _OK,
    ]
    agent, fake = _agent(tmp_path / "t", script)
    try:
        out = await agent.generate_response("getRestaurant", {"id": "rest_001"})
    finally:
        await agent.shutdown()
    assert out == {"id": "rest_001", "name": "The Italian Corner"}
    # The model was told its path is wrong and shown the real ones.
    tool_result = json.loads(fake.seen[1][-1]["content"])
    assert tool_result["error"] == "unknown_field"
    assert "location.postal_code" in tool_result["fields"]
    (rec,) = _records(tmp_path / "t")
    assert rec["fix_fired"] == ["S3a_bad_path"]


@pytest.mark.asyncio
async def test_ordinary_call_fires_no_fix(tmp_path: Path) -> None:
    script = [
        _call("c1", "state_list", store="restaurants", where={"cuisine": "Thai"}),
        _OK,
    ]
    agent, fake = _agent(tmp_path / "t", script)
    try:
        out = await agent.generate_response("getRestaurant", {"id": "rest_001"})
    finally:
        await agent.shutdown()
    assert out == {"id": "rest_001", "name": "The Italian Corner"}
    assert fake.seen[1][-1]["content"] == []  # an existing path matching nothing: []
    (rec,) = _records(tmp_path / "t")
    assert rec["fix_fired"] == []


# --- S3b / E1: an unfiltered list that would be evicted ------------------------------------


def _pad_restaurants(agent: DeepAgent, n: int = 400) -> None:
    """Grow `restaurants` past the ~80k-char eviction threshold."""
    assert agent.store_registry is not None
    store = agent.store_registry.for_thread("default")
    base = store.get("restaurants", "rest_001")
    assert base is not None
    for i in range(n):
        store.insert(
            "restaurants", {**base, "id": f"rest_pad_{i:03d}", "description": "x" * 250}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("skill_enabled", [True, False], ids=["full", "no_skill"])
async def test_e1_unfiltered_oversized_list_is_refused_in_every_condition(
    tmp_path: Path, skill_enabled: bool
) -> None:
    script = [
        _call("c1", "state_list", store="restaurants"),
        _call("c2", "state_get", store="restaurants", id="rest_001"),
        _OK,
    ]
    agent, fake = _agent(tmp_path / "t", script, skill_enabled=skill_enabled)
    _pad_restaurants(agent)
    try:
        out = await agent.generate_response("getRestaurant", {"id": "rest_001"})
    finally:
        await agent.shutdown()
    assert out == {"id": "rest_001", "name": "The Italian Corner"}
    refused = json.loads(fake.seen[1][-1]["content"])
    assert refused["error"] == "result_too_large"
    assert "location.postal_code" in refused["fields"]
    (rec,) = _records(tmp_path / "t")
    assert rec["fix_fired"] == ["E1_refused"]
    assert rec["any_evicted"] is False  # nothing reached the eviction path


@pytest.mark.asyncio
async def test_e1_filtered_oversized_list_is_still_evicted_as_before(
    tmp_path: Path,
) -> None:
    """A filtered (or limited) read is not the bug: it is answered, and evicted, as in v1."""
    script = [
        _call("c1", "state_list", store="restaurants", limit=1000),
        _OK,
    ]
    agent, _ = _agent(tmp_path / "t", script)
    _pad_restaurants(agent)
    try:
        await agent.generate_response("getRestaurant", {"id": "rest_001"})
    finally:
        await agent.shutdown()
    (rec,) = _records(tmp_path / "t")
    assert rec["fix_fired"] == []
    assert rec["tool_calls"][0]["evicted"] is True
