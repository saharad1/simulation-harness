"""Usage accounting — what the simulator's LLM actually spent.

Every simulated tool call is one or more LLM calls, and until now their token cost was
invisible from outside the process: ``utils/logging.log_tool_call`` declares a ``token_usage``
field (REQUIREMENTS.md §2.4) that nothing ever filled in. An evaluation harness that cannot
report the cost of a simulated task cannot answer whether simulating is cheaper than running
the real thing — which is the whole comparison the simulator exists to make.

This module keeps a monotonic per-instance counter, exposed over the API as
``GET /api/v1/simulation/usage`` (and zeroed with ``DELETE``). A caller brackets a task with
two reads and subtracts; nothing is attached to the tool RESPONSE, deliberately, because
consumers diff those payloads and extra metadata there would corrupt what they measure.

**Tokens, not dollars.** The harness may talk to a gateway whose model name matches no public
price table, and it has no pricing dependency. It reports counts and the model name; whoever
consumes them prices them.

**The de-duplication problem, and why this class exists rather than a one-line sum.** With a
checkpointer, ``agent.ainvoke`` returns the FULL thread history, not just the turns it added,
so summing the returned messages on every call would re-count every earlier turn — a bill that
grows quadratically and looks like a plausible "long tasks cost more". The meter therefore
remembers how many messages of each thread it has already accounted for and only counts past
that mark. ``forget_thread`` must be called when a thread's history is cleared, or the mark
would sit beyond the end of a fresh, shorter history and the new turns would be counted as
already seen.
"""

from __future__ import annotations

from typing import Any, Iterable

from simulation_harness.utils.logging import get_logger

logger = get_logger(__name__)


def message_usage(message: Any) -> dict[str, int]:
    """Token counts for ONE message, from langchain's ``usage_metadata``.

    All zeroes when the message carries no usage — a tool result, a human turn, or a provider
    that did not report any. Zero is the right answer for "this message cost no tokens"; a
    missing *provider* report is indistinguishable from that here and is left to the caller's
    ``calls``-vs-tokens ratio to expose.
    """
    meta = getattr(message, "usage_metadata", None)
    if not isinstance(meta, dict):
        # A dict-shaped message (tests, serialized history) keeps usage in the same place.
        if isinstance(message, dict):
            meta = message.get("usage_metadata")
        if not isinstance(meta, dict):
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "cached_input_tokens": 0}

    def n(value: Any) -> int:
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    inp, out = n(meta.get("input_tokens")), n(meta.get("output_tokens"))
    total = n(meta.get("total_tokens")) or (inp + out)
    details = meta.get("input_token_details")
    cached = n((details or {}).get("cache_read")) if isinstance(details, dict) else 0
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": total,
        "cached_input_tokens": cached,
    }


class UsageMeter:
    """Cumulative LLM usage for one simulation instance.

    Counters only ever grow, until an explicit :meth:`reset`. Notably NOT reset by a session
    reset: the world going back to its seed does not un-spend the tokens, and a consumer that
    resets the world between tasks would otherwise measure every task as free.
    """

    def __init__(self) -> None:
        self._calls = 0
        self._input = 0
        self._output = 0
        self._total = 0
        self._cached = 0
        self._llm_seconds = 0.0
        self._model: str | None = None
        # thread_id -> number of messages already accounted for (see the module docstring).
        self._seen: dict[str, int] = {}

    def record(
        self,
        messages: Iterable[Any],
        *,
        thread_id: str = "default",
        seconds: float | None = None,
        model: str | None = None,
    ) -> dict[str, int]:
        """Account for one ``ainvoke``. Returns just this call's usage (for the per-call log).

        ``messages`` is whatever the agent returned — the full thread history is fine and
        expected; only the part after the previous mark is counted.
        """
        msgs = list(messages or [])
        seen = self._seen.get(thread_id, 0)
        # A history SHORTER than the mark means it was cleared without forget_thread (or
        # trimmed by middleware). Counting from a stale mark would skip real turns, so fall
        # back to counting the whole of what we can see: over-counting a boundary once is
        # recoverable, silently dropping a task's entire bill is not.
        fresh = msgs[seen:] if len(msgs) >= seen else msgs
        self._seen[thread_id] = len(msgs)

        call = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "cached_input_tokens": 0}
        for m in fresh:
            u = message_usage(m)
            for k in call:
                call[k] += u[k]

        self._calls += 1
        self._input += call["input_tokens"]
        self._output += call["output_tokens"]
        self._total += call["total_tokens"]
        self._cached += call["cached_input_tokens"]
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
            self._llm_seconds += float(seconds)
        if model:
            self._model = model
        return call

    def forget_thread(self, thread_id: str | None = None) -> None:
        """Drop the accounted-messages mark for a thread whose history was just cleared.

        Spend is untouched — only the position in the (now empty) history is.
        """
        if thread_id is None:
            self._seen.clear()
        else:
            self._seen.pop(thread_id, None)

    def reset(self) -> None:
        """Zero every counter. Only ever called from an explicit request."""
        self.__init__()  # noqa: PLC2801 — one definition of "zero", not two
        logger.info("Usage counters reset")

    def snapshot(self) -> dict[str, Any]:
        """The counters, as the API returns them."""
        return {
            "calls": self._calls,
            "input_tokens": self._input,
            "output_tokens": self._output,
            "total_tokens": self._total,
            "cached_input_tokens": self._cached,
            "llm_seconds": round(self._llm_seconds, 3),
            "model": self._model,
        }
