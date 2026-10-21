"""Per-call trace of what the simulator LLM read and did (``HARNESS_TRACE_DIR``).

The harness logs only the outer API call and its token totals, so a wrong simulated answer
("user not found" for a user that exists, "Unknown operation" for an operation the skill
defines) cannot be explained after the fact: the state queries and skill-file reads the LLM
made in between are gone. With ``HARNESS_TRACE_DIR`` set, every simulated API call appends
one JSON line to ``<HARNESS_TRACE_DIR>/trajectory.jsonl`` holding:

- the call (operation, arguments, simulation, run label, thread, sequence number),
- every tool call the LLM made during it (``state_*`` queries and ``read_file``/``grep`` on
  SKILL.md alike) with its arguments, the size of the result the tool produced, the size the
  model was actually shown, whether the FilesystemMiddleware evicted the result to a file
  (results over ~80k chars) or truncated a ``read_file`` page, and its wall time,
- the final response text, the body returned to the caller, and the call's wall time.

Pure observation: with the variable unset or empty nothing here is constructed, and with it
set the only addition to the agent is ``ToolTraceMiddleware``, which returns every tool
result unchanged. A failure to write the trace is logged and never fails the call.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from simulation_harness.utils.logging import get_logger

logger = get_logger(__name__)

TRACE_DIR_ENV = "HARNESS_TRACE_DIR"
TRACE_FILE_NAME = "trajectory.jsonl"

#: Opening of the message deepagents' FilesystemMiddleware puts in place of an evicted tool
#: result (``deepagents.middleware._message_eviction.TOO_LARGE_TOOL_MSG``).
EVICTED_MARKER = "Tool result too large, the result of this tool call"
#: What deepagents appends to a ``read_file`` page it cut to the size limit
#: (``deepagents.middleware.filesystem.READ_FILE_TRUNCATION_MSG``).
READ_TRUNCATED_MARKER = "[Output was truncated due to size limits."

#: How much of each tool result (as the model saw it) the record keeps.
RESULT_HEAD_CHARS = 1500


def _text_of(content: Any) -> str:
    """The text of a message ``content`` (a string, or a list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if not content:
            # A tool that returned an empty list (e.g. a state_list with no match):
            # LangChain keeps it as an empty content-block list. Report it as the JSON
            # the tool returned, so "no rows" is not mistaken for "no output".
            return "[]"
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return "" if content is None else str(content)


class ToolTraceMiddleware(AgentMiddleware):
    """Record each tool result's size and wall time before any other middleware sees it.

    Must be the LAST entry of the agent's middleware list: ``create_agent`` composes
    ``wrap_tool_call`` handlers with the first entry outermost, so the last one is wrapped
    directly around the tool and sees its raw output -- before FilesystemMiddleware swaps an
    oversized result for a preview. The result is returned untouched.
    """

    def __init__(self) -> None:
        super().__init__()
        self._raw: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _note(self, request: Any, result: Any, seconds: float) -> None:
        call_id = (getattr(request, "tool_call", None) or {}).get("id")
        if not call_id:
            return
        content = getattr(result, "content", None)
        with self._lock:
            self._raw[call_id] = {
                "raw_chars": len(_text_of(content)) if content is not None else None,
                "seconds": round(seconds, 4),
            }

    def pop(self, call_id: str | None) -> dict[str, Any]:
        """The recorded size/time of ``call_id`` (removed from the buffer), or ``{}``."""
        if not call_id:
            return {}
        with self._lock:
            return self._raw.pop(call_id, {})

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        started = time.perf_counter()
        result = handler(request)
        self._note(request, result, time.perf_counter() - started)
        return result

    async def awrap_tool_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        started = time.perf_counter()
        result = await handler(request)
        self._note(request, result, time.perf_counter() - started)
        return result


def _this_call(messages: list[Any]) -> list[Any]:
    """The messages this call added: everything after the last ``HumanMessage``."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return list(messages[i + 1 :])
    return list(messages)


def _is_skill_read(name: str, args: dict[str, Any]) -> bool:
    if name not in ("read_file", "grep", "glob", "ls"):
        return False
    target = " ".join(
        str(args.get(k) or "") for k in ("file_path", "path", "glob", "pattern")
    )
    return "SKILL.md" in target


def tool_calls_of(
    messages: list[Any], middleware: ToolTraceMiddleware | None
) -> list[dict[str, Any]]:
    """One entry per tool call in this call's messages, in the order the LLM made them."""
    calls: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    step = 0
    for message in _this_call(messages):
        if isinstance(message, AIMessage):
            step += 1
            for call in message.tool_calls or []:
                entry = {
                    "step": step,
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "args": call.get("args") or {},
                    "skill_read": _is_skill_read(
                        call.get("name") or "", call.get("args") or {}
                    ),
                    "result_chars": None,
                    "shown_chars": None,
                    "evicted": None,
                    "truncated": None,
                    "status": None,
                    "seconds": None,
                    "result_head": None,
                }
                calls[call.get("id")] = entry
                out.append(entry)
            continue
        if not isinstance(message, ToolMessage):
            continue
        entry = calls.get(message.tool_call_id)
        if entry is None:
            continue
        shown = _text_of(message.content)
        raw = middleware.pop(message.tool_call_id) if middleware is not None else {}
        entry["shown_chars"] = len(shown)
        entry["result_chars"] = raw.get("raw_chars", len(shown))
        entry["seconds"] = raw.get("seconds")
        entry["evicted"] = shown.startswith(EVICTED_MARKER)
        entry["truncated"] = READ_TRUNCATED_MARKER in shown
        entry["status"] = getattr(message, "status", None)
        entry["result_head"] = shown[:RESULT_HEAD_CHARS]
    return out


class CallTracer:
    """Appends one JSON line per simulated API call to ``<trace_dir>/trajectory.jsonl``."""

    def __init__(
        self, trace_dir: str | Path, *, simulation: str | None, run_label: str | None
    ):
        self.dir = Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / TRACE_FILE_NAME
        self.simulation = simulation
        self.run_label = run_label
        self.middleware = ToolTraceMiddleware()
        self._lock = threading.Lock()
        self._seq = 0

    def write(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        thread_id: str,
        seconds: float,
        messages: list[Any] | None,
        response_text: str | None,
        returned: Any,
        outcome: str,
    ) -> None:
        """Append the record for one call. Never raises."""
        try:
            with self._lock:
                self._seq += 1
                seq = self._seq
            calls = tool_calls_of(messages or [], self.middleware)
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "seq": seq,
                "simulation": self.simulation,
                "run_label": self.run_label,
                "thread_id": thread_id,
                "tool": tool,
                "arguments": arguments,
                "seconds": round(seconds, 4),
                "outcome": outcome,
                "n_llm_steps": sum(
                    1 for m in _this_call(messages or []) if isinstance(m, AIMessage)
                ),
                "n_tool_calls": len(calls),
                "skill_read": any(c["skill_read"] for c in calls),
                "any_evicted": any(bool(c["evicted"]) for c in calls),
                "tool_calls": calls,
                "final_response": response_text,
                "returned": returned,
            }
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as e:  # noqa: BLE001 -- tracing must never fail a call
            logger.warning(f"Trace write failed: tool={tool}, error={e}")
