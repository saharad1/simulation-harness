"""Tests for the usage meter.

Weighted toward the mistakes that would produce a believable wrong bill rather than a crash:
double-counting a checkpointed thread's history, and losing a task's spend to a session reset.
"""

from simulation_harness.agent.usage import UsageMeter, message_usage


class _AI:
    """Duck-type of a langchain AIMessage carrying usage_metadata."""

    def __init__(self, inp=0, out=0, cached=0, total=None):
        self.usage_metadata = {
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": total if total is not None else inp + out,
            "input_token_details": {"cache_read": cached},
        }


class _Plain:
    """A message with no usage at all (human turn, tool result)."""


class TestMessageUsage:
    def test_reads_langchain_fields_including_cache_read(self):
        u = message_usage(_AI(inp=1000, out=50, cached=800))
        assert u == {
            "input_tokens": 1000,
            "output_tokens": 50,
            "total_tokens": 1050,
            "cached_input_tokens": 800,
        }

    def test_message_without_usage_is_all_zero(self):
        assert message_usage(_Plain())["total_tokens"] == 0

    def test_accepts_a_dict_shaped_message(self):
        u = message_usage({"usage_metadata": {"input_tokens": 5, "output_tokens": 1}})
        assert u["total_tokens"] == 6

    def test_derives_total_when_the_provider_omits_it(self):
        u = message_usage({"usage_metadata": {"input_tokens": 7, "output_tokens": 3}})
        assert u["total_tokens"] == 10

    def test_non_integer_counts_are_ignored_not_crashed(self):
        u = message_usage({"usage_metadata": {"input_tokens": None, "output_tokens": "x"}})
        assert u["total_tokens"] == 0


class TestUsageMeter:
    def test_accumulates_across_calls(self):
        m = UsageMeter()
        m.record([_AI(inp=100, out=10)])
        m.record([_AI(inp=100, out=10), _AI(inp=200, out=20)])   # history grew by one
        snap = m.snapshot()
        assert snap["calls"] == 2
        assert snap["input_tokens"] == 300
        assert snap["output_tokens"] == 30
        assert snap["total_tokens"] == 330

    def test_does_not_recount_a_checkpointed_history(self):
        # The bug this guards: ainvoke returns the WHOLE thread, so a naive sum re-bills every
        # earlier turn and the total grows quadratically while still looking plausible.
        m = UsageMeter()
        history = []
        for _ in range(4):
            history.append(_AI(inp=100, out=10))
            m.record(list(history))
        assert m.snapshot()["input_tokens"] == 400      # not 1000

    def test_returns_only_this_calls_usage(self):
        m = UsageMeter()
        m.record([_AI(inp=100, out=10)])
        call = m.record([_AI(inp=100, out=10), _AI(inp=7, out=3)])
        assert call["input_tokens"] == 7 and call["output_tokens"] == 3

    def test_threads_are_tracked_separately(self):
        m = UsageMeter()
        m.record([_AI(inp=10, out=1)], thread_id="a")
        m.record([_AI(inp=20, out=2)], thread_id="b")   # index 0 of ITS own history
        assert m.snapshot()["input_tokens"] == 30

    def test_forget_thread_lets_a_cleared_history_be_counted_again(self):
        m = UsageMeter()
        m.record([_AI(inp=100, out=10), _AI(inp=100, out=10)])
        m.forget_thread("default")                     # world reset: history cleared
        m.record([_AI(inp=5, out=1)])                  # fresh first turn
        assert m.snapshot()["input_tokens"] == 205

    def test_a_shortened_history_without_forget_still_bills_the_new_turns(self):
        # Middleware trimming, or a reset that skipped forget_thread: the mark now sits past
        # the end. Counting from it would silently drop the whole task; we count what we see.
        m = UsageMeter()
        m.record([_AI(inp=100, out=10), _AI(inp=100, out=10), _AI(inp=100, out=10)])
        m.record([_AI(inp=9, out=1)])
        assert m.snapshot()["input_tokens"] == 309

    def test_session_reset_must_not_zero_the_spend(self):
        # forget_thread is what a session reset calls. If it also zeroed the counters, a
        # consumer that reseeds the world between tasks would measure every task as free.
        m = UsageMeter()
        m.record([_AI(inp=1000, out=100)])
        m.forget_thread()
        assert m.snapshot()["total_tokens"] == 1100

    def test_explicit_reset_zeroes_everything(self):
        m = UsageMeter()
        m.record([_AI(inp=1000, out=100)], seconds=3.0, model="gpt-4o")
        m.reset()
        snap = m.snapshot()
        assert snap == {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "llm_seconds": 0.0,
            "model": None,
        }
        m.record([_AI(inp=1, out=1)])          # still usable after a reset
        assert m.snapshot()["calls"] == 1

    def test_accumulates_seconds_and_keeps_the_model(self):
        m = UsageMeter()
        m.record([_AI(inp=1, out=1)], seconds=1.25, model="gpt-4o")
        m.record([_AI(inp=1, out=1), _AI(inp=1, out=1)], seconds=2.5, model="gpt-4o")
        snap = m.snapshot()
        assert snap["llm_seconds"] == 3.75
        assert snap["model"] == "gpt-4o"

    def test_a_call_that_returned_nothing_still_counts_as_a_call(self):
        # A failed/empty generation spent latency and possibly prompt tokens we cannot see;
        # counting zero calls would make calls-vs-tokens ratios lie.
        m = UsageMeter()
        m.record(None, seconds=0.5)
        assert m.snapshot()["calls"] == 1
        assert m.snapshot()["total_tokens"] == 0

    def test_bad_seconds_are_ignored(self):
        m = UsageMeter()
        m.record([_AI(inp=1, out=1)], seconds=None)
        m.record([_AI(inp=1, out=1), _AI(inp=1, out=1)], seconds="slow")
        assert m.snapshot()["llm_seconds"] == 0.0
