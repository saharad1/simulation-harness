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
async def test_e1_limited_oversized_list_is_refused_too(
    tmp_path: Path,
) -> None:
    """A `limit` does not make a result small: `limit: 1000` on the 500-user retail store
    returned all of it and was evicted (canary smoke, retail rich d3, task 0)."""
    script = [
        _call("c1", "state_list", store="restaurants", limit=1000),
        _call("c2", "state_get", store="restaurants", id="rest_001"),
        _OK,
    ]
    agent, fake = _agent(tmp_path / "t", script)
    _pad_restaurants(agent)
    try:
        await agent.generate_response("getRestaurant", {"id": "rest_001"})
    finally:
        await agent.shutdown()
    refused = json.loads(fake.seen[1][-1]["content"])
    assert refused["error"] == "result_too_large"
    (rec,) = _records(tmp_path / "t")
    assert rec["fix_fired"] == ["E1_refused"]
    assert rec["any_evicted"] is False


# --- S2: malformed JSON final answer --------------------------------------------------------


@pytest.mark.asyncio
async def test_s2_malformed_answer_is_retried_with_the_parse_error(
    tmp_path: Path,
) -> None:
    script = [
        _call("c1", "state_get", store="restaurants", id="rest_001"),
        AIMessage(content='{"id": "rest_001"}{"id": "rest_001"}'),  # "Extra data"
        _OK,
    ]
    agent, fake = _agent(tmp_path / "t", script)
    try:
        out = await agent.generate_response("getRestaurant", {"id": "rest_001"})
    finally:
        await agent.shutdown()
    assert out == {"id": "rest_001", "name": "The Italian Corner"}
    assert fake.cursor == 3
    retry_prompt = fake.seen[2][-1]
    assert retry_prompt["type"] == "human"
    assert "not valid JSON" in retry_prompt["content"]
    assert "Extra data" in retry_prompt["content"]
    (rec,) = _records(tmp_path / "t")
    assert rec["fix_fired"] == ["S2_retry"]
    assert rec["outcome"] == "ok"
    # The record still covers the whole call, from the original request on.
    assert [c["name"] for c in rec["tool_calls"]] == ["state_get"]
    assert rec["n_llm_steps"] == 3


@pytest.mark.asyncio
async def test_s2_gives_up_after_two_retries_with_the_old_error(
    tmp_path: Path,
) -> None:
    script = [AIMessage(content="not json")] * 3
    agent, fake = _agent(tmp_path / "t", script)
    try:
        out = await agent.generate_response("getRestaurant", {"id": "rest_001"})
    finally:
        await agent.shutdown()
    assert out["error"] == "Failed to generate valid JSON response"
    assert fake.cursor == 3  # first answer + 2 retries
    (rec,) = _records(tmp_path / "t")
    assert rec["outcome"] == "json_error"
    assert rec["fix_fired"] == ["S2_retry"]


def _rating_update(call_id: str, rating: Any) -> AIMessage:
    return _call(
        call_id,
        "state_update",
        store="restaurants",
        id="rest_001",
        patch={"rating": rating},
    )


async def _run_rest_001(
    tmp_path: Path, script: list[AIMessage]
) -> tuple[Any, Any, Any]:
    """(returned body, rest_001 before, rest_001 after) of one getRestaurant call."""
    agent, _ = _agent(tmp_path, script)
    assert agent.store_registry is not None
    store = agent.store_registry.for_thread("default")
    before = store.get("restaurants", "rest_001")
    try:
        out = await agent.generate_response("getRestaurant", {"id": "rest_001"})
        after = agent.store_registry.for_thread("default").get(
            "restaurants", "rest_001"
        )
    finally:
        await agent.shutdown()
    return out, before, after


@pytest.mark.asyncio
async def test_s2_retry_is_a_fresh_attempt_and_its_writes_are_checked(
    tmp_path: Path,
) -> None:
    """Each re-ask starts from the pre-call store (I5): only the last attempt's writes
    stand, and only they are checked by the failed-write (B2) check."""
    # The re-ask redoes the write: it is applied once.
    out, _, after = await _run_rest_001(
        tmp_path / "a",
        [
            _rating_update("w1", 4.9),
            AIMessage(content="```json\n{}\n```"),
            _rating_update("w2", 4.9),
            AIMessage(content='{"id": "rest_001", "rating": 4.9}'),
        ],
    )
    assert out == {"id": "rest_001", "rating": 4.9}
    assert after["rating"] == 4.9

    # A write the re-ask fails still fails the call, and the store is as before it.
    out, before, after = await _run_rest_001(
        tmp_path / "b",
        [
            _rating_update("w1", 4.9),
            AIMessage(content="oops"),
            _rating_update("w2", "not a number"),
            AIMessage(content='{"id": "rest_001", "rating": 4.9}'),
        ],
    )
    assert out["error"] == "Store write failed"
    assert after == before

    # A write only the rolled-back attempt failed does not fail the re-ask's answer.
    out, _, after = await _run_rest_001(
        tmp_path / "c",
        [
            _rating_update("w1", "not a number"),
            AIMessage(content="oops"),
            _rating_update("w2", 4.8),
            AIMessage(content='{"id": "rest_001", "rating": 4.8}'),
        ],
    )
    assert out == {"id": "rest_001", "rating": 4.8}
    assert after["rating"] == 4.8


class _DecrementModel(BaseChatModel):
    """A non-idempotent write, decided from what it reads: lower rest_001's rating by 0.5.

    Each attempt reads the rating, writes ``rating - 0.5``, then answers. The first
    attempt's answer is malformed; later ones are valid JSON with the written rating.
    """

    answers: int = 0

    @property
    def _llm_type(self) -> str:
        return "decrement"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_DecrementModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1]
        if last.type == "human":
            message = _call("g", "state_get", store="restaurants", id="rest_001")
        elif (
            last.type == "tool"
            and '"rating"' in str(last.content)
            and (
                messages[-2].tool_calls[0]["name"] == "state_get"  # type: ignore[attr-defined]
            )
        ):
            rating = json.loads(str(last.content))["rating"]
            message = _call(
                "u",
                "state_update",
                store="restaurants",
                id="rest_001",
                patch={"rating": round(rating - 0.5, 2)},
            )
        else:
            self.answers += 1
            rating = json.loads(str(last.content))["rating"]
            message = AIMessage(
                content="oops, not JSON"
                if self.answers == 1
                else json.dumps({"id": "rest_001", "rating": rating})
            )
        # Unique tool-call ids per step, as a real model gives.
        for call in message.tool_calls:
            call["id"] = f"{call['id']}{len(messages)}"
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.mark.asyncio
async def test_s2_retry_starts_from_the_pre_call_store(tmp_path: Path) -> None:
    """I5: the malformed attempt's writes are rolled back before the re-ask, so a
    non-idempotent write the re-ask redoes is applied once, not twice."""
    agent, _ = _agent(tmp_path / "t", [])
    agent.llm = _DecrementModel()  # type: ignore[assignment]
    agent.agent = agent._create_agent()
    assert agent.store_registry is not None
    before = agent.store_registry.for_thread("default").get("restaurants", "rest_001")
    assert before is not None
    try:
        out = await agent.generate_response("getRestaurant", {"id": "rest_001"})
        stored = agent.store_registry.for_thread("default").get(
            "restaurants", "rest_001"
        )
    finally:
        await agent.shutdown()
    once = round(before["rating"] - 0.5, 2)
    assert out == {"id": "rest_001", "rating": once}
    assert stored is not None and stored["rating"] == once
    (rec,) = _records(tmp_path / "t")
    assert rec["fix_fired"] == ["S2_retry"]
    assert rec["outcome"] == "ok"


@pytest.mark.asyncio
async def test_s2_call_usage_sums_only_this_calls_turns(tmp_path: Path) -> None:
    """M7: ``last_call_usage`` of a re-asked call is the sum of ITS turns, even when a
    call on another thread set the agent-wide field between them."""

    def _ai(content: str, tokens: int) -> AIMessage:
        return AIMessage(
            content=content,
            usage_metadata={
                "input_tokens": tokens,
                "output_tokens": 1,
                "total_tokens": tokens + 1,
            },
        )

    foreign = {
        "input_tokens": 1000,
        "output_tokens": 1000,
        "total_tokens": 2000,
        "cached_input_tokens": 0,
    }
    agent, fake = _agent(None, [_ai("not json", 10), _ai(_OK.content, 20)])  # type: ignore[arg-type]
    real_invoke = agent.agent.ainvoke

    async def _ainvoke(payload: Any, config: Any) -> Any:
        if fake.cursor == 1:  # another thread's call finished between the two turns
            agent.last_call_usage = dict(foreign)
        return await real_invoke(payload, config=config)

    agent.agent.ainvoke = _ainvoke  # type: ignore[method-assign]
    try:
        await agent.generate_response("getRestaurant", {"id": "rest_001"})
    finally:
        await agent.shutdown()
    assert agent.last_call_usage == {
        "input_tokens": 30,
        "output_tokens": 2,
        "total_tokens": 32,
        "cached_input_tokens": 0,
    }


@pytest.mark.asyncio
async def test_s2_valid_first_answer_is_not_retried(tmp_path: Path) -> None:
    agent, fake = _agent(tmp_path / "t", [_OK, _OK])
    try:
        await agent.generate_response("getRestaurant", {"id": "rest_001"})
    finally:
        await agent.shutdown()
    assert fake.cursor == 1
    (rec,) = _records(tmp_path / "t")
    assert rec["fix_fired"] == []


# --- S4: "Unknown operation" for an operation the skill defines ---------------------------

# tests/fixtures/restaurant-reservation-api/SKILL.md: "### `/place_reservation` POST" is at
# line 208, its last non-blank line is 259, and the next section starts at line 261.
_PLACE = "place_reservation"


def _place_op() -> Any:
    op = _operation(_PLACE)
    op.method = "post"
    op.path = "/place_reservation"
    return op


def _missing_op() -> Any:
    op = _operation("rate_restaurant")
    op.method = "post"
    op.path = "/rate_restaurant"
    return op


_REFUSAL = AIMessage(content='{"error": "unknown operation: place_reservation"}')
_PLACED = AIMessage(content='{"reservation_id": "res_1", "status": "confirmed"}')


@pytest.mark.asyncio
async def test_s4_refusal_of_a_defined_operation_is_retried_with_its_section(
    tmp_path: Path,
) -> None:
    script = [
        # First attempt writes, then refuses: the refusal rolls the write back.
        _call(
            "w1",
            "state_update",
            store="restaurants",
            id="rest_001",
            patch={"rating": 1.0},
        ),
        _REFUSAL,
        _call(
            "r1",
            "read_file",
            file_path="/.skills/restaurant-reservation-api/SKILL.md",
            offset=207,
            limit=53,
        ),
        _PLACED,
    ]
    agent, fake = _agent(tmp_path / "t", script, operations=[_place_op()])
    try:
        out = await agent.generate_response(_PLACE, {"restaurant_id": "rest_001"})
        assert agent.store_registry is not None
        stored = agent.store_registry.for_thread("default").get(
            "restaurants", "rest_001"
        )
    finally:
        await agent.shutdown()
    assert out == {"reservation_id": "res_1", "status": "confirmed"}
    assert (
        stored is not None and stored["rating"] != 1.0
    )  # the refused attempt's write is undone
    retry = fake.seen[2][-1]
    assert retry["type"] == "human"
    assert "### `/place_reservation` POST" in retry["content"]
    assert "lines 208-259" in retry["content"]
    assert "offset=207, limit=52" in retry["content"]
    assert "/.skills/restaurant-reservation-api/SKILL.md" in retry["content"]
    assert '"restaurant_id": "rest_001"' in retry["content"]  # the request, again
    (rec,) = _records(tmp_path / "t")
    assert rec["fix_fired"] == ["S4_section_retry"]
    assert rec["outcome"] == "ok"
    assert [c["name"] for c in rec["tool_calls"]] == ["state_update", "read_file"]


@pytest.mark.asyncio
async def test_s4_retries_once_only(tmp_path: Path) -> None:
    agent, fake = _agent(
        tmp_path / "t", [_REFUSAL, _REFUSAL, _PLACED], operations=[_place_op()]
    )
    try:
        out = await agent.generate_response(_PLACE, {})
    finally:
        await agent.shutdown()
    assert out == {"error": "unknown operation: place_reservation"}
    assert fake.cursor == 2
    (rec,) = _records(tmp_path / "t")
    assert rec["fix_fired"] == ["S4_section_retry"]
    assert rec["outcome"] == "error_body"


@pytest.mark.asyncio
async def test_s4_operation_the_skill_lacks_is_not_retried(tmp_path: Path) -> None:
    refusal = AIMessage(content='{"error": "Operation not found"}')
    agent, fake = _agent(tmp_path / "t", [refusal, _OK], operations=[_missing_op()])
    try:
        out = await agent.generate_response("rate_restaurant", {})
    finally:
        await agent.shutdown()
    assert out == {"error": "Operation not found"}
    assert fake.cursor == 1
    (rec,) = _records(tmp_path / "t")
    assert rec["fix_fired"] == []


@pytest.mark.asyncio
async def test_s4_other_errors_and_no_skill_are_not_retried(tmp_path: Path) -> None:
    other = AIMessage(content='{"error": "Restaurant not found"}')
    agent, fake = _agent(tmp_path / "a", [other, _PLACED], operations=[_place_op()])
    try:
        assert await agent.generate_response(_PLACE, {}) == {
            "error": "Restaurant not found"
        }
    finally:
        await agent.shutdown()
    assert fake.cursor == 1

    # no_skill has no skill to point at: its refusal stands.
    agent, fake = _agent(
        tmp_path / "b",
        [_REFUSAL, _PLACED],
        operations=[_place_op()],
        skill_enabled=False,
    )
    try:
        out = await agent.generate_response(_PLACE, {})
    finally:
        await agent.shutdown()
    assert out == {"error": "unknown operation: place_reservation"}
    assert fake.cursor == 1


_PHRASINGS = json.loads(
    (_FIXTURE_SKILL_DIR.parent / "s4_refusal_phrasings.json").read_text()
)


@pytest.mark.parametrize("text", _PHRASINGS["refusals"])
def test_s4_refusal_phrasings_are_refusals(text: str) -> None:
    """I3: the pinned phrasings (shared with the eval's S4 detector) all count."""
    from simulation_harness.agent.deep_agent import _is_unknown_operation

    assert _is_unknown_operation({"error": text})
    assert _is_unknown_operation({"error": "not_found", "message": text})


@pytest.mark.parametrize("text", _PHRASINGS["not_refusals"])
def test_s4_pinned_ordinary_errors_are_not_refusals(text: str) -> None:
    from simulation_harness.agent.deep_agent import _is_unknown_operation

    assert not _is_unknown_operation({"error": text})


def test_s4_patterns_are_one_module_constant() -> None:
    """I3: the regex is built from UNKNOWN_OPERATION_PATTERNS alone, the list the eval
    detector copies."""
    from simulation_harness.agent import deep_agent

    assert deep_agent._UNKNOWN_OPERATION.pattern == "(?i)" + "|".join(
        f"(?:{p})" for p in deep_agent.UNKNOWN_OPERATION_PATTERNS
    )


@pytest.mark.parametrize(
    "body",
    [
        {"error": "unknown operation: modify_user_address"},
        {"error": "Operation not found"},
        {"error": "Unknown operation"},
        {"error": "unsupported_operation", "message": "x"},
        {"error": "not_found", "message": "The operation is not defined in the skill"},
    ],
)
def test_s4_refusal_texts(body: dict) -> None:
    from simulation_harness.agent.deep_agent import _is_unknown_operation

    assert _is_unknown_operation(body)


@pytest.mark.parametrize(
    "body",
    [
        {"error": "Order not found"},
        {"error": "User not found"},
        {"error": "Failed to generate valid JSON response"},
        {"error": "Payment method not found for this operation"},
    ],
)
def test_s4_ordinary_errors_are_not_refusals(body: dict) -> None:
    from simulation_harness.agent.deep_agent import _is_unknown_operation

    assert not _is_unknown_operation(body)
