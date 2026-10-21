"""HARNESS_TRACE_DIR: one JSONL record per simulated call, and no change to what is simulated.

Runs the REAL LangGraph agent (SkillsMiddleware + FilesystemMiddleware + state tools) with a
scripted fake chat model, so the trace middleware sits in the actual middleware stack and an
oversized ``state_list`` result really goes through deepagents' eviction.
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
from simulation_harness.agent.trace import EVICTED_MARKER, TRACE_FILE_NAME

_FIXTURE_SKILL_DIR = (
    Path(__file__).resolve().parents[2] / "fixtures" / "restaurant-reservation-api"
)
_SKILL_MD = "/.skills/restaurant-reservation-api/SKILL.md"


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


def _script() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "c1",
                    "name": "read_file",
                    "args": {"file_path": _SKILL_MD, "offset": 0, "limit": 20},
                },
                {"id": "c2", "name": "state_list", "args": {"store": "restaurants"}},
                {
                    "id": "c3",
                    "name": "state_get",
                    "args": {"store": "restaurants", "id": "rest_001"},
                },
                {
                    "id": "c4",
                    "name": "state_list",
                    "args": {"store": "restaurants", "where": {"cuisine": "None"}},
                },
            ],
        ),
        AIMessage(content='{"id": "rest_001", "name": "The Italian Corner"}'),
    ]


def _operation() -> Any:
    from simulation_harness.openapi.parser import OpenAPIOperation

    op = Mock(spec=OpenAPIOperation)
    op.method = "get"
    op.path = "/restaurants/{id}"
    op.operation_id = "getRestaurant"
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
    return spec


def _make_agent(trace_dir: Path | None) -> tuple[DeepAgent, _ScriptedChatModel]:
    agent = DeepAgent(
        api_key=SecretStr("test-key"),
        model="gpt-4",
        temperature=0.0,
        max_tokens=1000,
        base_url=None,
        spec=_spec(),
        operations=[_operation()],
        skill_dir=_FIXTURE_SKILL_DIR,
        trace_dir=trace_dir,
    )
    fake = _ScriptedChatModel(script=_script(), seen=[])
    agent.llm = fake
    agent.agent = agent._create_agent()
    # Grow `restaurants` past the ~80k-char eviction threshold so the unfiltered list is evicted.
    store = agent.store_registry.for_thread("default")
    base = store.get("restaurants", "rest_001")
    for i in range(400):
        store.insert(
            "restaurants",
            {**base, "id": f"rest_pad_{i:03d}", "description": "x" * 250},
        )
    return agent, fake


async def _run(trace_dir: Path | None) -> tuple[Any, dict, list, DeepAgent]:
    agent, fake = _make_agent(trace_dir)
    try:
        out = await agent.generate_response("getRestaurant", {"id": "rest_001"})
        return (
            out,
            agent.store_registry.for_thread("default").snapshot(),
            fake.seen,
            agent,
        )
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_tracing_does_not_change_the_simulation(tmp_path: Path) -> None:
    """Same returned body, same store, and the model was shown the same messages."""
    off = await _run(None)
    on = await _run(tmp_path / "trace")

    assert on[0] == off[0] == {"id": "rest_001", "name": "The Italian Corner"}
    assert on[1] == off[1]
    assert on[2] == off[2]
    # The off run wrote nothing anywhere under tmp_path except the on run's dir.
    assert [p.name for p in tmp_path.iterdir()] == ["trace"]


@pytest.mark.asyncio
async def test_trace_record_fields(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    await _run(trace_dir)

    lines = (trace_dir / TRACE_FILE_NAME).read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])

    assert rec["tool"] == "getRestaurant"
    assert rec["arguments"] == {"id": "rest_001"}
    assert rec["thread_id"] == "default"
    assert rec["seq"] == 1
    assert rec["outcome"] == "ok"
    assert rec["seconds"] >= 0
    assert rec["final_response"] == '{"id": "rest_001", "name": "The Italian Corner"}'
    assert rec["returned"] == {"id": "rest_001", "name": "The Italian Corner"}
    assert rec["skill_read"] is True
    assert rec["any_evicted"] is True

    by_id = {c["id"]: c for c in rec["tool_calls"]}
    assert [c["name"] for c in rec["tool_calls"]] == [
        "read_file",
        "state_list",
        "state_get",
        "state_list",
    ]

    read = by_id["c1"]
    assert read["skill_read"] is True
    assert read["args"] == {"file_path": _SKILL_MD, "offset": 0, "limit": 20}
    assert read["evicted"] is False
    assert read["result_chars"] > 0

    listed = by_id["c2"]
    assert listed["args"] == {"store": "restaurants"}
    assert listed["evicted"] is True
    assert listed["result_chars"] > 80_000  # what the tool produced
    assert listed["shown_chars"] < listed["result_chars"]  # what the model saw
    assert listed["result_head"].startswith(EVICTED_MARKER)
    assert listed["seconds"] is not None

    # An empty result is "[]", not an empty string (LangChain keeps a [] tool output as
    # an empty content-block list).
    empty = by_id["c4"]
    assert empty["result_head"] == "[]"
    assert empty["result_chars"] == empty["shown_chars"] == 2

    got = by_id["c3"]
    assert got["evicted"] is False
    assert got["skill_read"] is False
    assert got["result_chars"] == got["shown_chars"]
    assert "rest_001" in got["result_head"]


@pytest.mark.asyncio
async def test_error_outcome_is_traced(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    agent, fake = _make_agent(trace_dir)
    fake.script = [AIMessage(content="not json")]
    try:
        out = await agent.generate_response("getRestaurant", {"id": "rest_001"})
    finally:
        await agent.shutdown()
    assert out["error"] == "Failed to generate valid JSON response"
    rec = json.loads((trace_dir / TRACE_FILE_NAME).read_text().splitlines()[0])
    assert rec["outcome"] == "json_error"
    assert rec["final_response"] == "not json"
    assert rec["returned"] == out
    assert rec["tool_calls"] == []


def test_empty_trace_dir_is_off() -> None:
    agent = DeepAgent(
        api_key=SecretStr("test-key"),
        model="gpt-4",
        temperature=0.0,
        max_tokens=1000,
        base_url=None,
        spec=_spec(),
        operations=[_operation()],
        skill_dir=_FIXTURE_SKILL_DIR,
        trace_dir="",
    )
    assert agent.tracer is None
