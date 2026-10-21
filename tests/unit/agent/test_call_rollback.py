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

import asyncio
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


@pytest.mark.asyncio
async def test_concurrent_error_does_not_erase_other_success(
    mock_spec: Any, mock_operation: Any
) -> None:
    """Two concurrent calls on the SAME thread: one errors, one succeeds.

    The simulation host runs several calls on one thread concurrently (thread "default" is
    shared by every call unless a caller passes an explicit thread_id). Without a per-thread
    lock, the erroring call's checkpoint -- taken before the succeeding call's write landed --
    would restore over that write when it rolls back its own, erasing a committed success.
    This pins the fix: a per-thread lock serializes the two calls, so the erroring call's
    checkpoint/restore cycle completes (and undoes only ITS OWN write) before the succeeding
    call's checkpoint is even taken.
    """
    agent = _make_agent(mock_spec, mock_operation)
    thread_id = "default"

    async def _error_ainvoke(_input: Any, config: Any) -> dict:
        # Checkpoint for this call has already been taken by the time ainvoke runs. Yield here
        # so the OTHER call's checkpoint/write can interleave if nothing is serializing them.
        await asyncio.sleep(0)
        store = agent.store_registry.for_thread(thread_id)
        store.update("reservations", "reservation_seed_001", {"status": "cancelled"})
        message = Mock()
        message.content = '{"error": "insufficient balance"}'
        return {"messages": [message]}

    async def _success_ainvoke(_input: Any, config: Any) -> dict:
        store = agent.store_registry.for_thread(thread_id)
        store.update("reservations", "reservation_seed_002", {"status": "cancelled"})
        message = Mock()
        message.content = '{"status": "exchanged"}'
        return {"messages": [message]}

    calls = iter([_error_ainvoke, _success_ainvoke])

    async def _dispatch(_input: Any, config: Any) -> dict:
        return await next(calls)(_input, config)

    agent.agent.ainvoke = AsyncMock(side_effect=_dispatch)

    error_out, success_out = await asyncio.gather(
        agent.generate_response(
            tool_name="exchangeItems",
            arguments={"id": "reservation_seed_001"},
            thread_id=thread_id,
        ),
        agent.generate_response(
            tool_name="exchangeItems",
            arguments={"id": "reservation_seed_002"},
            thread_id=thread_id,
        ),
    )

    assert error_out["error"] == "insufficient balance"
    assert success_out == {"status": "exchanged"}

    reservation_2 = agent.store_registry.for_thread(thread_id).get(
        "reservations", "reservation_seed_002"
    )
    assert reservation_2["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancelled_call_restores_store(
    mock_spec: Any, mock_operation: Any
) -> None:
    """A call cancelled mid-flight (asyncio.CancelledError) still rolls back its write.

    F3: ``asyncio.CancelledError`` is a ``BaseException``, not an ``Exception`` -- it is
    raised when a client times out or disconnects, which is likely given LLM gateway
    hangs. The old code only restored the checkpoint in ``except Exception``, so a
    cancelled call skipped the restore and left its partial write committed. The fix
    must restore the store on this path too, and must re-raise ``CancelledError``
    rather than swallow it.
    """
    agent = _make_agent(mock_spec, mock_operation)
    thread_id = "default"

    before = agent.store_registry.for_thread(thread_id).snapshot()

    async def _ainvoke(_input: Any, config: Any) -> dict:
        store = agent.store_registry.for_thread(thread_id)
        store.update(
            "reservations",
            "reservation_seed_001",
            {"status": "cancelled"},
        )
        raise asyncio.CancelledError()

    agent.agent.ainvoke = AsyncMock(side_effect=_ainvoke)

    with pytest.raises(asyncio.CancelledError):
        await agent.generate_response(
            tool_name="exchangeItems",
            arguments={"id": "reservation_seed_001"},
            thread_id=thread_id,
        )

    assert agent.store_registry.for_thread(thread_id).snapshot() == before


@pytest.mark.asyncio
async def test_different_threads_not_serialized(
    mock_spec: Any, mock_operation: Any
) -> None:
    """Calls on DIFFERENT threads must still run concurrently, not be serialized by the lock."""
    agent = _make_agent(mock_spec, mock_operation)

    thread_a_entered = asyncio.Event()
    thread_b_entered = asyncio.Event()

    async def _ainvoke_a(_input: Any, config: Any) -> dict:
        thread_a_entered.set()
        # Wait for B to enter too, proving the lock on thread "a" did not block thread "b".
        await asyncio.wait_for(thread_b_entered.wait(), timeout=5)
        message = Mock()
        message.content = '{"status": "ok-a"}'
        return {"messages": [message]}

    async def _ainvoke_b(_input: Any, config: Any) -> dict:
        thread_b_entered.set()
        await asyncio.wait_for(thread_a_entered.wait(), timeout=5)
        message = Mock()
        message.content = '{"status": "ok-b"}'
        return {"messages": [message]}

    calls = iter([_ainvoke_a, _ainvoke_b])

    async def _dispatch(_input: Any, config: Any) -> dict:
        return await next(calls)(_input, config)

    agent.agent.ainvoke = AsyncMock(side_effect=_dispatch)

    out_a, out_b = await asyncio.wait_for(
        asyncio.gather(
            agent.generate_response(
                tool_name="exchangeItems",
                arguments={"id": "reservation_seed_001"},
                thread_id="thread-a",
            ),
            agent.generate_response(
                tool_name="exchangeItems",
                arguments={"id": "reservation_seed_002"},
                thread_id="thread-b",
            ),
        ),
        timeout=5,
    )

    assert out_a == {"status": "ok-a"}
    assert out_b == {"status": "ok-b"}


# --- a failed store write fails the call --------------------------------------
#
# A state tool that rejects a write returns {"error": "validation_error", ...} to the
# LLM as an ordinary tool result. Nothing stopped the LLM from ignoring it, keeping
# the turn's other writes and answering with a success body. These fakes run the
# real state tools, so the ToolMessages are exactly what the agent graph produces.


def _run_tool_calls(config: Any, calls: list[tuple[str, dict]], final: str) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage

    from simulation_harness.state.tools import create_state_tools

    tools = {t.name: t for t in create_state_tools()}
    messages: list[Any] = [HumanMessage(content="API Request: POST /x")]
    for i, (name, args) in enumerate(calls):
        call = {"type": "tool_call", "name": name, "args": args, "id": f"call_{i}"}
        messages.append(AIMessage(content="", tool_calls=[call]))
        messages.append(tools[name].invoke(call, config=config))
    messages.append(AIMessage(content=final))
    return {"messages": messages}


_GOOD_UPDATE = (
    "state_update",
    {
        "store": "reservations",
        "id": "reservation_seed_002",
        "patch": {"status": "cancelled"},
    },
)
_BAD_UPDATE = (
    "state_update",
    {"store": "reservations", "id": "reservation_seed_001", "patch": {"status": 123}},
)
_BAD_INSERT = ("state_insert", {"store": "reservations", "entity": {"id": "r9"}})


async def _call_with(agent: DeepAgent, calls: list[tuple[str, dict]], final: str):
    async def _ainvoke(_input: Any, config: Any) -> dict:
        return _run_tool_calls(config, calls, final)

    agent.agent.ainvoke = AsyncMock(side_effect=_ainvoke)
    return await agent.generate_response(
        tool_name="exchangeItems",
        arguments={"id": "reservation_seed_001"},
        thread_id="default",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [_BAD_INSERT, _BAD_UPDATE])
async def test_an_ignored_failed_write_fails_the_call_and_rolls_back(
    mock_spec: Any, mock_operation: Any, bad: tuple[str, dict]
) -> None:
    agent = _make_agent(mock_spec, mock_operation)
    before = agent.store_registry.for_thread("default").snapshot()

    out = await _call_with(agent, [_GOOD_UPDATE, bad], '{"status": "exchanged"}')

    assert set(out) == {"error", "details"}
    assert "store write failed" in out["error"].lower()
    assert "validation_error" in out["details"]
    # The earlier, successful write of the same turn is undone too.
    assert agent.store_registry.for_thread("default").snapshot() == before


@pytest.mark.asyncio
async def test_a_failed_write_retried_successfully_keeps_the_call(
    mock_spec: Any, mock_operation: Any
) -> None:
    agent = _make_agent(mock_spec, mock_operation)
    retry = (
        "state_update",
        {
            "store": "reservations",
            "id": "reservation_seed_001",
            "patch": {"status": "cancelled"},
        },
    )

    out = await _call_with(agent, [_BAD_UPDATE, retry], '{"status": "exchanged"}')

    assert out == {"status": "exchanged"}
    store = agent.store_registry.for_thread("default")
    assert store.get("reservations", "reservation_seed_001")["status"] == "cancelled"


@pytest.mark.asyncio
async def test_a_success_on_another_row_does_not_clear_a_failed_write(
    mock_spec: Any, mock_operation: Any
) -> None:
    agent = _make_agent(mock_spec, mock_operation)
    before = agent.store_registry.for_thread("default").snapshot()

    out = await _call_with(agent, [_BAD_UPDATE, _GOOD_UPDATE], '{"ok": true}')

    assert "store write failed" in out["error"].lower()
    assert agent.store_registry.for_thread("default").snapshot() == before


@pytest.mark.asyncio
async def test_a_failed_write_answered_with_an_error_is_returned_as_is(
    mock_spec: Any, mock_operation: Any
) -> None:
    agent = _make_agent(mock_spec, mock_operation)

    out = await _call_with(agent, [_BAD_INSERT], '{"error": "cannot book"}')

    assert out == {"error": "cannot book"}


@pytest.mark.asyncio
async def test_a_failed_write_in_an_earlier_call_does_not_fail_this_one(
    mock_spec: Any, mock_operation: Any
) -> None:
    """Only this call's messages count: with a checkpointer the history is the thread's."""
    from langchain_core.messages import AIMessage, HumanMessage

    agent = _make_agent(mock_spec, mock_operation)

    async def _ainvoke(_input: Any, config: Any) -> dict:
        earlier = _run_tool_calls(config, [_BAD_INSERT], '{"error": "x"}')["messages"]
        now = [HumanMessage(content="API Request: GET /y"), AIMessage(content="{}")]
        return {"messages": earlier + now}

    agent.agent.ainvoke = AsyncMock(side_effect=_ainvoke)
    out = await agent.generate_response(
        tool_name="exchangeItems", arguments={}, thread_id="default"
    )

    assert out == {}


# --- I1: a failed write that names no primary key -----------------------------
#
# `state_insert` without the primary key fails with validation_error ("Primary key
# field 'id' is required") under the target (store, None). A later successful insert
# of the complete entity is the recovery, but it is recorded under (store, "<id>"),
# so it never cleared the keyless failure and a recovered call failed closed.

_RESERVATION = {
    "restaurant_id": "rest_001",
    "restaurant_name": "The Italian Corner",
    "date_time": "2025-04-01T19:00:00",
    "party_size": 2,
    "guest_name": "Ann Lee",
    "guest_phone": "+1-555-000-0000",
    "guest_email": "ann@example.com",
    "confirmation_code": "CONF-NEW",
    "created_at": "2025-03-20T10:00:00",
}
_KEYLESS_INSERT = ("state_insert", {"store": "reservations", "entity": _RESERVATION})
_FULL_INSERT = (
    "state_insert",
    {"store": "reservations", "entity": {"id": "r_new", **_RESERVATION}},
)


@pytest.mark.asyncio
async def test_a_keyless_failed_insert_redone_with_its_key_keeps_the_call(
    mock_spec: Any, mock_operation: Any
) -> None:
    agent = _make_agent(mock_spec, mock_operation)

    out = await _call_with(
        agent, [_KEYLESS_INSERT, _FULL_INSERT], '{"id": "r_new", "status": "booked"}'
    )

    assert out == {"id": "r_new", "status": "booked"}
    store = agent.store_registry.for_thread("default")
    assert store.get("reservations", "r_new")["guest_name"] == "Ann Lee"


@pytest.mark.asyncio
async def test_a_keyless_failed_insert_is_not_cleared_by_another_kind_of_write(
    mock_spec: Any, mock_operation: Any
) -> None:
    """Only a later successful insert into the same store redoes a keyless insert: an
    update of some other row is not a redo."""
    agent = _make_agent(mock_spec, mock_operation)
    before = agent.store_registry.for_thread("default").snapshot()

    out = await _call_with(agent, [_KEYLESS_INSERT, _GOOD_UPDATE], '{"ok": true}')

    assert "store write failed" in out["error"].lower()
    assert agent.store_registry.for_thread("default").snapshot() == before


@pytest.mark.asyncio
async def test_a_keyless_failed_insert_is_not_cleared_by_an_insert_elsewhere(
    mock_spec: Any, mock_operation: Any
) -> None:
    agent = _make_agent(mock_spec, mock_operation)
    receipt = (
        "state_insert",
        {
            "store": "cancellations",
            "entity": {
                "reservation_id": "reservation_seed_001",
                "restaurant_name": "The Italian Corner",
                "original_date_time": "2025-03-15T19:00:00",
                "cancelled_at": "2025-03-12T09:00:00",
            },
        },
    )

    out = await _call_with(agent, [_KEYLESS_INSERT, receipt], '{"ok": true}')

    assert "store write failed" in out["error"].lower()


@pytest.mark.asyncio
async def test_a_keyless_failed_insert_after_the_successful_one_still_fails(
    mock_spec: Any, mock_operation: Any
) -> None:
    """Only a LATER success redoes a failed write."""
    agent = _make_agent(mock_spec, mock_operation)

    out = await _call_with(agent, [_FULL_INSERT, _KEYLESS_INSERT], '{"ok": true}')

    assert "store write failed" in out["error"].lower()
