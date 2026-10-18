"""Tests that a call ending in an error response leaves the store untouched.

H4: today every state-tool write commits immediately, so a call that ends by
returning ``{"error": ...}`` has already mutated the store. These tests pin
down the fix: ``generate_response`` checkpoints the thread's store before the
LLM call and restores it when the parsed response is an error dict.

The fake LLM is simulated the same way ``tests/unit/agent/test_deep_agent.py``
mocks the model layer: ``agent.agent.ainvoke`` is patched directly rather than
exercising the real LangChain/LangGraph stack. Because the mock replaces the
whole graph invocation (including any state-tool calls the real agent would
have made), the fake's side effect writes to the store itself to stand in for
"the LLM called state_update".
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import SecretStr

from simulation_harness.agent.deep_agent import DeepAgent

_FIXTURE_SKILL_DIR = (
    Path(__file__).resolve().parents[2] / "fixtures" / "restaurant-reservation-api"
)


@pytest.fixture
def mock_spec() -> Any:

    from simulation_harness.openapi.parser import OpenAPISpec

    spec = Mock(spec=OpenAPISpec)
    spec.info = {"title": "Test API", "version": "1.0.0", "description": "Test"}
    spec.servers = [{"url": "https://api.example.com"}]
    return spec


@pytest.fixture
def mock_operation() -> Any:

    from simulation_harness.openapi.parser import OpenAPIOperation

    op = Mock(spec=OpenAPIOperation)
    op.method = "post"
    op.path = "/reservations/{id}/exchange"
    op.operation_id = "exchangeItems"
    op.summary = "Exchange items"
    op.description = "Exchange items on a reservation"
    op.parameters = []
    op.request_body = None
    op.get_request_schema = Mock(return_value=None)
    op.get_response_schema = Mock(return_value={"type": "object"})
    return op


def _make_agent(mock_spec: Any, mock_operation: Any) -> DeepAgent:
    return DeepAgent(
        api_key=SecretStr("test-key"),
        model="gpt-4",
        temperature=0.0,
        max_tokens=1000,
        base_url=None,
        spec=mock_spec,
        operations=[mock_operation],
        skill_dir=_FIXTURE_SKILL_DIR,
    )


def _fake_ainvoke_that_writes_then_errors(agent: DeepAgent, thread_id: str):
    """Build an ainvoke replacement that mutates the store, then returns an error body."""

    async def _ainvoke(_input: Any, config: Any) -> dict:
        store = agent.store_registry.for_thread(thread_id)
        store.update(
            "reservations",
            "reservation_seed_001",
            {"status": "cancelled"},
        )
        message = Mock()
        message.content = '{"error": "insufficient balance"}'
        return {"messages": [message]}

    return _ainvoke


def _fake_ainvoke_that_writes_then_succeeds(agent: DeepAgent, thread_id: str):
    async def _ainvoke(_input: Any, config: Any) -> dict:
        store = agent.store_registry.for_thread(thread_id)
        store.update(
            "reservations",
            "reservation_seed_001",
            {"status": "cancelled"},
        )
        message = Mock()
        message.content = '{"status": "exchanged"}'
        return {"messages": [message]}

    return _ainvoke


@pytest.mark.asyncio
async def test_error_response_rolls_back(mock_spec: Any, mock_operation: Any) -> None:
    """A call whose final body is {"error": ...} leaves the store as it was."""
    agent = _make_agent(mock_spec, mock_operation)
    thread_id = "default"

    before = agent.store_registry.for_thread(thread_id).snapshot()

    agent.agent.ainvoke = AsyncMock(
        side_effect=_fake_ainvoke_that_writes_then_errors(agent, thread_id)
    )

    out = await agent.generate_response(
        tool_name="exchangeItems",
        arguments={"id": "reservation_seed_001"},
        thread_id=thread_id,
    )

    assert out["error"] == "insufficient balance"
    assert agent.store_registry.for_thread(thread_id).snapshot() == before


@pytest.mark.asyncio
async def test_success_response_keeps_writes(
    mock_spec: Any, mock_operation: Any
) -> None:
    """A call whose final body has no "error" key keeps the write it made."""
    agent = _make_agent(mock_spec, mock_operation)
    thread_id = "default"

    agent.agent.ainvoke = AsyncMock(
        side_effect=_fake_ainvoke_that_writes_then_succeeds(agent, thread_id)
    )

    out = await agent.generate_response(
        tool_name="exchangeItems",
        arguments={"id": "reservation_seed_001"},
        thread_id=thread_id,
    )

    assert out == {"status": "exchanged"}
    reservation = agent.store_registry.for_thread(thread_id).get(
        "reservations", "reservation_seed_001"
    )
    assert reservation["status"] == "cancelled"
