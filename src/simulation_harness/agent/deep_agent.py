"""Simplified Deep Agent implementation for Simulation Harness.

This module provides the main agent that generates mock responses using
LangChain and LangGraph directly (without the deepagents library).
"""

import asyncio
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware, FsToolName
from deepagents.middleware.skills import SkillsMiddleware

from simulation_harness.openapi.parser import OpenAPIOperation, OpenAPISpec
from simulation_harness.state.registry import StoreRegistry
from simulation_harness.agent.calc_tool import create_calculator_tool
from simulation_harness.state.tools import create_state_tools
from simulation_harness.utils.logging import get_logger
from simulation_harness.agent.prompts import render_system_prompt
from simulation_harness.agent.session_manager import SessionManager
from simulation_harness.agent.skill_backend import build_skill_sources
from simulation_harness.agent.trace import CallTracer
from simulation_harness.agent.usage import UsageMeter
from simulation_harness.skills.generation.llm import no_cache_kwargs

logger = get_logger(__name__)

# The agent only ever *reads* the generated skill from the virtual filesystem;
# all mutation goes through the `state_*` tools instead.  So we allowlist the
# read-only filesystem tools and never hand the model a write-capable one.
#
# Omitted from the deepagents default set on purpose:
#   write_file, edit_file, delete — the agent must not mutate the skill
#   execute                       — needs a SandboxBackendProtocol backend,
#                                   which FilesystemBackend is not, so it could
#                                   only ever return an error
#
# This is a stronger guarantee than the deny-rules it replaces: a tool that is
# never exposed cannot be called at all, so there is no glob pattern to get
# wrong (the previous rules needed five patterns to cover dot-prefixed paths,
# because wcmatch's GLOBSTAR does not imply DOTGLOB).
_READONLY_FS_TOOLS: list[FsToolName] = ["ls", "read_file", "glob", "grep"]


# Store writes whose failure means the write did not happen. A rejected write is
# returned to the LLM as an ordinary tool result, so nothing but this check stops the
# LLM from ignoring it and answering with a success body built on the turn's other
# writes. Only these codes: `not_found`/`duplicate_key` are answers an operation may
# legitimately branch on (look up, then insert or update).
_STORE_WRITE_TOOLS = frozenset({"state_insert", "state_update", "state_delete"})
_FAILED_WRITE_ERRORS = frozenset({"validation_error", "internal_error"})


#: S2: how many times a malformed (non-JSON) final answer is re-asked before the call
#: answers "Failed to generate valid JSON response".
JSON_RETRIES = 2

#: ``additional_kwargs`` key marking a HumanMessage the harness added to re-ask within a
#: call (value: the fix's ``fix_fired`` name). Never set on a call's own request.
RETRY_KWARG = "harness_retry"
S2_RETRY = "S2_retry"


S4_SECTION_RETRY = "S4_section_retry"

#: S4 (plan 28 I3): the one list of patterns for an error body refusing the operation
#: itself ("unknown operation: x", "Operation not found", "The operation is not defined
#: ..."), as opposed to a not-found about the data. Matched case-insensitively, any one
#: suffices. The eval repo's S4 detector copies this list (its test pins the phrasings
#: in ``tests/fixtures/s4_refusal_phrasings.json``); change both together.
_SEP = r"[\s_-]+"
_OP = r"(?:operation|endpoint|tool)"
#: An optional name between the noun and its verb: "operation `modify_user_address` is ...".
_NAME = r"(?:[\s_-]+[`'\"]?[\w/{}.:-]+[`'\"]?)?"
#: Not when a condition follows (review M1): "This operation is not supported for
#: non-pending orders" is a business refusal the skill may require, not "undefined".
_NO_CONDITION = (
    rf"(?!{_SEP}(?:for|because|when|while|unless|since|once|if|after|before|as|due)\b)"
)
UNKNOWN_OPERATION_PATTERNS: tuple[str, ...] = (
    # "invalid" only with "endpoint": "Invalid operation: ..." is a business refusal.
    rf"\b(?:(?:unknown|unsupported|undefined|unrecogni[sz]ed){_SEP}{_OP}"
    rf"|invalid{_SEP}endpoint)\b{_NO_CONDITION}",
    rf"\b{_OP}{_NAME}{_SEP}(?:is{_SEP})?not{_SEP}"
    rf"(?:found|defined|supported|recogni[sz]ed|available|implemented)\b{_NO_CONDITION}",
    rf"\b{_OP}{_NAME}{_SEP}does{_SEP}not{_SEP}exist\b{_NO_CONDITION}",
    rf"\bno{_SEP}such{_SEP}{_OP}\b{_NO_CONDITION}",
    rf"\bnot{_SEP}defined{_SEP}(?:in|by){_SEP}(?:the{_SEP})?"
    r"(?:skill|api|spec|specification|simulation)\b",
)
_UNKNOWN_OPERATION = re.compile(
    "(?i)" + "|".join(f"(?:{p})" for p in UNKNOWN_OPERATION_PATTERNS)
)


def _is_unknown_operation(body: dict[str, Any]) -> bool:
    """Whether an error body is the simulator refusing the operation as undefined."""
    text = " ".join(
        str(body[k])
        for k in ("error", "message", "details", "reason")
        if isinstance(body.get(k), str)
    )
    return bool(_UNKNOWN_OPERATION.search(text))


def skill_section(skill_md: str, path: str, method: str) -> tuple[str, int, int] | None:
    """``(heading, first line, last line)`` of an operation's SKILL.md section, 1-based.

    The heading is the generator's ``### <path> <METHOD>`` (backticks around the path
    tolerated); the section runs to the line before the next heading of level 1-3.
    """
    heading = re.compile(
        rf"^###\s+`?{re.escape(path)}`?\s+{re.escape(method.upper())}\s*$"
    )
    lines = skill_md.splitlines()
    for i, line in enumerate(lines):
        if heading.match(line):
            end = len(lines)
            for k in range(i + 1, len(lines)):
                if re.match(r"^#{1,3}\s", lines[k]):
                    end = k
                    break
            while end > i + 1 and not lines[end - 1].strip():
                end -= 1
            return line.strip(), i + 1, end
    return None


def _fired(trace: dict[str, Any] | None, fix: str) -> None:
    """Record that failure-path fix ``fix`` acted on this call (trace record only)."""
    if trace is not None:
        fired = trace.setdefault("fix_fired", [])
        if fix not in fired:
            fired.append(fix)


#: Opt-in (env ``HARNESS_LLM_LENIENT_JSON``, read per call like HARNESS_LLM_NO_CACHE): a final
#: answer that is not pure JSON is accepted if it contains one, instead of being re-asked
#: (S2) with the call's writes rolled back. For comparing serving models whose replies wrap
#: the JSON in prose or a code fence; off by default, so the paper runs are unchanged.
LENIENT_JSON_ENV = "HARNESS_LLM_LENIENT_JSON"
LENIENT_JSON = "lenient_json"


def _lenient_json_enabled() -> bool:
    return os.environ.get(LENIENT_JSON_ENV, "").strip().lower() in ("1", "true", "yes")


#: Opt-in (env ``HARNESS_SKILL_INLINE``, read when the agent is built): skill conditions get
#: SKILL.md's text appended to the system prompt instead of only a pointer to the file. For
#: serving models that never open the file (gpt-4.1 read it in 0 of 26 airline tasks), so
#: that a skill condition tests the skill's content, not whether the model calls read_file.
SKILL_INLINE_ENV = "HARNESS_SKILL_INLINE"


def _skill_inline_enabled() -> bool:
    return os.environ.get(SKILL_INLINE_ENV, "").strip().lower() in ("1", "true", "yes")


def _extract_json(text: str) -> str | None:
    """The JSON object or array embedded in ``text``, or None.

    Of every ``{``/``[`` that starts a complete JSON value, the one that ends last wins,
    and among those the one that starts first: the answer typically follows the prose,
    and an example fragment inside the prose ends earlier than the answer does.
    """
    decoder = json.JSONDecoder()
    best: tuple[int, int] | None = None  # (end, -start)
    for start, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, (dict, list)):
            continue
        if best is None or (end, -start) > best:
            best = (end, -start)
    if best is None:
        return None
    end, neg_start = best
    return text[-neg_start:end]


def _json_retry_message(
    operation: OpenAPIOperation, error: json.JSONDecodeError, request_prompt: str
) -> HumanMessage:
    """S2: re-ask the request as a fresh attempt (on the restored store), with the parse
    error."""
    return HumanMessage(
        content=(
            f"Your previous response was not valid JSON ({error}). It was rolled back: "
            f"every state change you made for it is undone and the store is as it was "
            f"before this request. Answer the request again from the start, making "
            f"again every state change it needs, and reply with ONLY the JSON response "
            f"body ({operation.method.upper()} {operation.path}): a single JSON value, "
            f"no text before or after it, no markdown.\n\n{request_prompt}"
        ),
        additional_kwargs={RETRY_KWARG: S2_RETRY},
    )


def _turn_start(messages: list[Any]) -> int:
    """Index just after the HumanMessage that opened this call's current attempt.

    The request opens the first attempt; each harness re-ask (S2, S4) opens a fresh
    attempt on the store restored to its pre-call checkpoint, so only the last
    attempt's writes stand and only they are checked.
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i + 1
    return 0


def _unrecovered_write_failures(messages: list[Any], store: Any) -> list[str]:
    """Store writes this call's LLM turn failed and never redid successfully.

    Scans the messages of this call's current attempt (with a checkpointer the list is
    the whole thread's history; a harness re-ask opens a fresh attempt on the restored
    store, see `_turn_start`). A failed write is cleared only by a later successful write to the same
    store and primary key -- kept that narrow on purpose, so a success elsewhere can never
    paper over it. A failed write that named no key has no row to match, so the
    narrowest redo there is: a later successful write by the same tool to the same store
    (an insert missing its primary key, re-sent complete). An update or delete of another
    row, or a write to another store, never clears it.
    """
    start = _turn_start(messages)
    calls: dict[str, dict] = {}
    # (store, key) for a keyed write; (store, None, tool) for a write that named no key.
    failed: dict[tuple[str | None, ...], str] = {}
    for message in messages[start:]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                calls[call.get("id")] = call
            continue
        if not isinstance(message, ToolMessage):
            continue
        call = calls.get(message.tool_call_id)
        if call is None or call.get("name") not in _STORE_WRITE_TOOLS:
            continue
        args = call.get("args") or {}
        store_name = str(args.get("store"))
        if call["name"] == "state_insert":
            entity = args.get("entity")
            key = (
                entity.get(store.primary_key(store_name))
                if isinstance(entity, dict)
                else None
            )
        else:
            key = args.get("id")
        target: tuple[str | None, ...] = (
            (store_name, None, call["name"]) if key is None else (store_name, str(key))
        )
        try:
            body = json.loads(message.content)
        except (TypeError, ValueError):
            body = None
        code = body.get("error") if isinstance(body, dict) else None
        if code in _FAILED_WRITE_ERRORS or message.status == "error":
            detail = body.get("message") if isinstance(body, dict) else message.content
            failed[target] = (
                f"{call['name']} on '{store_name}' id '{target[1]}' returned "
                f"{code or 'a tool error'}: {detail}"
            )
        elif code is None and target[1] is not None:
            failed.pop(target, None)
            # A write that named no key (an insert missing its primary key) is redone by
            # the next successful write of the same tool to the same store: its row
            # cannot be named, and re-sending it complete is the recovery (plan 28 I1).
            failed.pop((store_name, None, call["name"]), None)
    return list(failed.values())


def _store_layout(skill_dir: Path | None) -> str | None:
    """The skill's ``schema.json``, pretty-printed for the prompt, or None if there is none."""
    if skill_dir is None:
        return None
    path = Path(skill_dir) / "schema.json"
    if not path.is_file():
        return None
    return json.dumps(
        json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False
    )


def build_system_prompt(
    spec: OpenAPISpec,
    skill_dir: Path | None,
    *,
    skill_enabled: bool,
    state_enabled: bool,
) -> str:
    """The simulator's system prompt for a condition (``scripts/check_prompt_parity.py``).

    `full`/`minimal_skill` (``skill_enabled``): the slim prompt that points at the skill.
    `no_skill`: the spec itself, unmodified, is the only description of the API -- plus
    the layout of the store it writes into (schema.json), and nothing else of the skill.
    Without the layout it could not know the store's composite keys or id-only references,
    so part of the full-vs-no_skill gap was the missing layout (plan 28 S1). `spec_only`
    (``state_enabled=False``) additionally drops every state-tool instruction and the store
    layout: there is no store and no state tool to call.
    """
    if skill_enabled:
        return render_system_prompt(spec)
    # The layout is for no_skill only: spec_only has no store to lay out.
    layout = _store_layout(skill_dir) if state_enabled else None
    if state_enabled and skill_dir is not None and layout is None:
        # Without it the no_skill prompt would silently lack its S1 section; the store
        # cannot be loaded without the file either, so fail now, not at the first call.
        raise ValueError(
            f"no_skill needs the store layout: no schema.json in {skill_dir}"
        )
    return render_system_prompt(
        spec,
        raw_spec=json.dumps(spec.spec_dict, indent=2, ensure_ascii=False),
        store_schema=layout,
        stateless=not state_enabled,
    )


class DeepAgent:
    """Simplified agent for generating mock API responses using LangChain/LangGraph."""

    def __init__(
        self,
        api_key: SecretStr,
        model: str,
        temperature: float,
        max_tokens: int,
        base_url: str | None,
        spec: OpenAPISpec,
        operations: list[OpenAPIOperation],
        session_timeout_seconds: int = 3600,
        skill_dir: Path | None = None,
        skill_name: str | None = None,
        agent_recursion_limit: int = 50,
        skill_enabled: bool = True,
        state_enabled: bool = True,
        streaming: bool = True,
        trace_dir: str | Path | None = None,
        run_label: str | None = None,
    ):
        """Initialize Deep Agent.

        Args:
            api_key: LLM API key (wrapped in SecretStr)
            model: Model name (e.g., "gpt-4")
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            base_url: Optional custom base URL for OpenAI API
            spec: OpenAPI specification
            operations: List of API operations
            session_timeout_seconds: Session timeout in seconds
            skill_dir: Optional path to skill directory for state store. A GENERATION
                directory (``<name>@<stamp>``), so its basename is not the skill name.
            skill_name: Flat skill name for the agent-visible staging directory. Defaults
                to ``skill_dir``'s basename with any generation stamp stripped.
            agent_recursion_limit: Maximum recursion depth for agent (default: 10)
            skill_enabled: Load the compiled skill. False is the `no_skill` ablation: no
                SkillsMiddleware and no skill files; the raw OpenAPI spec is put in the system
                prompt instead, with the store's layout (``schema.json``). The state store and
                its tools are unaffected.
            state_enabled: Build the state store. False (only with ``skill_enabled=False``)
                is the `spec_only` reference: no StoreRegistry, no state tools, no seed; the
                simulator answers each call from the raw spec and the thread's history alone.
            streaming: Stream chat completions (and stream_usage, so usage_metadata is
                still populated). Defaults on: the shared LiteLLM gateway drops any
                non-streamed `/chat/completions` request whose response takes longer
                than about 10 seconds, while streamed requests of any length work.
            trace_dir: ``HARNESS_TRACE_DIR``. When set (non-empty), every simulated call
                appends one record of the LLM's tool calls -- state queries and SKILL.md
                reads, result sizes, evictions -- to ``<trace_dir>/trajectory.jsonl``
                (see ``agent.trace``). None or empty: no tracing, no change to the agent.
            run_label: Stamped on every trace record (``HARNESS_RUN_LABEL``).
        """
        self.spec = spec
        self.operations = operations
        self.session_timeout_seconds = session_timeout_seconds
        self.skill_dir = skill_dir
        self.skill_name = skill_name
        self.agent_recursion_limit = agent_recursion_limit
        self.skill_enabled = skill_enabled
        self.state_enabled = state_enabled
        # A compiled skill's contracts are written against the store; serving one without it is
        # not a condition anything measures. AblationConfig refuses this at config load too.
        if skill_enabled and not state_enabled:
            raise ValueError(
                "state_enabled=False requires skill_enabled=False (spec_only); "
                "skill=on,state=off is not a supported condition"
            )

        # Cumulative LLM spend for this simulation, read over the API as
        # /api/v1/simulation/usage. Lives on the agent because the agent is what makes the
        # LLM calls; the instance only forwards.
        self.usage = UsageMeter()
        self._model_name = model
        # Usage of the most recent LLM call, for the per-call log line. None until the first
        # call, so a log emitted before any generation says "unknown" rather than "free".
        self.last_call_usage: dict[str, int] | None = None

        # Per-simulation staging dir for the skills backend, created lazily in
        # _create_agent and removed in shutdown().
        self._skill_staging_dir: Path | None = None
        # The staged SKILL.md (text, and its path as the agent's filesystem tools see it),
        # when a skill is loaded; used only by the S4 re-ask.
        self._skill_md: str | None = None
        self._skill_md_path: str | None = None

        # Off unless a trace dir is given; built before the graph, whose middleware it joins.
        self.tracer: CallTracer | None = (
            CallTracer(
                trace_dir,
                simulation=skill_name or (skill_dir.name if skill_dir else None),
                run_label=run_label,
            )
            if trace_dir
            else None
        )

        # Initialize LLM
        # `max_tokens` is ChatOpenAI's declared field name; langchain_openai
        # serializes it to the API's `max_completion_tokens` for us (it is the
        # field's alias). Spell it this way rather than as the alias, to match
        # the rest of the codebase and to avoid depending on the alias surviving
        # a langchain_openai upgrade.
        llm_kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": api_key,
            "streaming": streaming,
            "stream_usage": streaming,
        }
        if base_url is not None:
            llm_kwargs["base_url"] = base_url
        # HARNESS_LLM_NO_CACHE: the same gateway-cache opt-out generation uses. Without it
        # an identical replay (a rerun of a kept cell) is answered from the cache.
        llm_kwargs.update(no_cache_kwargs())

        self.llm = ChatOpenAI(**llm_kwargs)

        self.system_prompt = build_system_prompt(
            spec, skill_dir, skill_enabled=skill_enabled, state_enabled=state_enabled
        )

        logger.debug(
            f"Rendered system prompt: length={len(self.system_prompt)}, "
            f"preview={self.system_prompt[:200]}..."
        )

        # Initialize checkpointer for session management
        self.checkpointer = MemorySaver()

        # Initialize state store registry if skill_dir provided (and the store is enabled)
        self.store_registry: StoreRegistry | None = None
        if skill_dir and state_enabled:
            self.store_registry = StoreRegistry(skill_dir)
            logger.info(f"State store registry initialized for skill: {skill_dir}")

        # One lock per thread, created lazily: without it, several calls on the SAME thread
        # (the simulation host runs up to `max_concurrent_queue_depth` concurrently, and they
        # all share thread "default") could interleave their checkpoint/restore around each
        # other's writes -- a later call's restore would wipe out an earlier call's already-
        # committed write. Calls on DIFFERENT threads must stay concurrent, so the lock is keyed
        # per thread rather than being one lock for the whole agent.
        self._thread_locks: dict[str, asyncio.Lock] = {}

        # Initialize session cleanup manager
        self.session_manager = SessionManager(
            checkpointer=self.checkpointer,
            timeout_seconds=session_timeout_seconds,
            max_sessions=100,  # Default max sessions
            store_registry=self.store_registry,
        )

        # Create LangGraph agent
        self.agent = self._create_agent()

        # Note: Session cleanup task is NOT started automatically
        # Call start_session_cleanup() explicitly when ready

        logger.info(
            f"Deep Agent initialized: model={model}, "
            f"operation_count={len(operations)}, "
            f"session_timeout={session_timeout_seconds}s, "
            f"state_store={'enabled' if self.store_registry else 'disabled'}"
        )

    def start_session_cleanup(self) -> None:
        """Start the session cleanup background task.

        This should be called after the agent is initialized and
        an event loop is running.
        """
        self.session_manager.start()

    def _trace_middleware(self) -> list[Any]:
        """The tracer's observe-only tool middleware, or nothing when tracing is off."""
        return [self.tracer.middleware] if self.tracer is not None else []

    def _create_agent(self) -> CompiledStateGraph:
        """Create the runtime agent.

        With a skill directory, build a lean ``create_agent`` graph that loads
        the per-API skill dynamically (progressive disclosure) and exposes the
        state tools. Without one, fall back to the legacy stateless agent.
        """
        if self.skill_dir and not self.state_enabled:
            # `spec_only`: no store and no state tools; the calculator is kept, as in no_skill.
            # The thread's checkpointed history is the only memory between calls.
            agent = create_agent(
                self.llm,
                system_prompt=self.system_prompt,
                tools=[create_calculator_tool()],
                checkpointer=self.checkpointer,
            )
            logger.info(
                f"Created spec-only agent: raw spec in prompt "
                f"({len(self.system_prompt)} chars), no store, no skill sources"
            )
            return agent

        if self.store_registry and self.skill_dir and not self.skill_enabled:
            # `no_skill`: state tools only. No SkillsMiddleware and no filesystem, so no skill
            # file is reachable; the spec and the store layout (schema.json) are in the system
            # prompt. The store itself is the skill dir's, seeded as usual.
            agent = create_agent(
                self.llm,
                system_prompt=self.system_prompt,
                tools=create_state_tools() + [create_calculator_tool()],
                middleware=self._trace_middleware(),
                checkpointer=self.checkpointer,
            )
            logger.info(
                f"Created no-skill agent: raw spec in prompt "
                f"({len(self.system_prompt)} chars), no skill sources"
            )
            return agent

        if self.store_registry and self.skill_dir:
            tools = create_state_tools() + [create_calculator_tool()]

            root_dir, sources = build_skill_sources(self.skill_dir, self.skill_name)
            self._skill_staging_dir = Path(root_dir)
            staged = sorted(self._skill_staging_dir.glob(".skills/*/SKILL.md"))
            if staged:
                # Read once for the S4 fix: where each operation's section is, and the
                # path the agent's read_file uses for it.
                self._skill_md = staged[0].read_text(encoding="utf-8")
                self._skill_md_path = (
                    "/" + staged[0].relative_to(self._skill_staging_dir).as_posix()
                )
            if self._skill_md is not None and _skill_inline_enabled():
                # The skill's text in the prompt, for a serving model that never opens the
                # file (the read tools stay available). Only for skill conditions: no_skill
                # never reaches this branch with a skill to inline.
                self.system_prompt = (
                    f"{self.system_prompt}\n\n## The simulation skill\n\nThe full text of "
                    f"{self._skill_md_path}, given here so you need not read the file:\n\n"
                    f"{self._skill_md}"
                )
                logger.info(
                    f"{SKILL_INLINE_ENV}: SKILL.md inlined into the system prompt "
                    f"({len(self._skill_md)} chars, prompt now {len(self.system_prompt)})"
                )
            backend = FilesystemBackend(root_dir=root_dir, virtual_mode=True)

            # Read-only filesystem enforcement: expose only the non-mutating
            # filesystem tools (public `tools=` allowlist, deepagents 0.7.0+).
            middleware = [
                SkillsMiddleware(backend=backend, sources=sources),
                FilesystemMiddleware(
                    backend=backend,
                    tools=_READONLY_FS_TOOLS,
                ),
                # Last = innermost: sees each tool result before FilesystemMiddleware may evict it.
                *self._trace_middleware(),
            ]

            agent = create_agent(
                self.llm,
                system_prompt=self.system_prompt,
                tools=tools,
                middleware=middleware,
                checkpointer=self.checkpointer,
            )

            logger.info(
                f"Created lean skill-loading agent: {len(tools)} state tools, "
                f"skill sources={sources}"
            )
            return agent

        # Legacy stateless agent (no skill_dir): unchanged.
        def agent_node(state: MessagesState) -> MessagesState:
            messages = state["messages"]
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.system_prompt)] + list(messages)
            response = self.llm.invoke(messages)
            return {"messages": messages + [response]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", agent_node)
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)
        return workflow.compile(checkpointer=self.checkpointer)

    async def generate_response(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        thread_id: str | None = None,
    ) -> Any:
        """Generate mock response for a tool invocation.

        Args:
            tool_name: Name of the tool being invoked
            arguments: Tool arguments
            thread_id: Optional session/thread identifier for context persistence.
                      If provided, conversation history will be maintained across calls.
                      If None, uses "default" thread.

        Returns:
            Generated mock response

        Raises:
            ValueError: If tool is not found or response generation fails
        """
        # Find the operation
        operation = self._find_operation(tool_name)
        if not operation:
            raise ValueError(f"Operation not found for tool: {tool_name}")

        logger.debug(
            f"Processing request: tool={tool_name}, operation={operation.operation_id}, "
            f"arguments={arguments}"
        )

        # Determine effective thread_id
        effective_thread_id = thread_id or "default"

        # Record session activity for cleanup tracking
        self.session_manager.record_activity(effective_thread_id)

        # Serialize calls on the SAME thread: the simulation host runs several calls on one
        # thread concurrently (up to `max_concurrent_queue_depth`), and without this a later
        # call's checkpoint/restore could interleave with an earlier call's write -- restoring
        # to a checkpoint taken before that write would silently erase it. Keyed per thread so
        # calls on DIFFERENT threads are untouched and keep running concurrently. No lock is
        # needed when there is no store to corrupt.
        lock = (
            self._thread_locks.setdefault(effective_thread_id, asyncio.Lock())
            if self.store_registry is not None
            else None
        )

        if lock is not None:
            async with lock:
                return await self._generate_response_locked(
                    tool_name, operation, arguments, effective_thread_id
                )
        return await self._generate_response_locked(
            tool_name, operation, arguments, effective_thread_id
        )

    async def _generate_response_locked(
        self,
        tool_name: str,
        operation: OpenAPIOperation,
        arguments: dict[str, Any],
        effective_thread_id: str,
    ) -> Any:
        """`_simulate_call`, plus one trace record per call when tracing is on."""
        if self.tracer is None:
            return await self._simulate_call(
                tool_name, operation, arguments, effective_thread_id, None
            )
        trace: dict[str, Any] = {}
        returned: Any = None
        started = time.perf_counter()
        try:
            returned = await self._simulate_call(
                tool_name, operation, arguments, effective_thread_id, trace
            )
            return returned
        finally:
            self.tracer.write(
                tool=tool_name,
                arguments=arguments,
                thread_id=effective_thread_id,
                seconds=time.perf_counter() - started,
                messages=trace.get("messages"),
                response_text=trace.get("response_text"),
                returned=returned,
                outcome=trace.get("outcome", "cancelled"),
                fix_fired=trace.get("fix_fired"),
            )

    async def _simulate_call(
        self,
        tool_name: str,
        operation: OpenAPIOperation,
        arguments: dict[str, Any],
        effective_thread_id: str,
        trace: dict[str, Any] | None,
    ) -> Any:
        """The body of `generate_response`, run while the thread's lock is held.

        ``trace`` (tracing on only) collects this call's messages, raw response text and
        outcome for the trace record; it never changes what is returned.

        Checkpoints the thread's store before the call and restores it on every path that
        returns an error body, so a failed call never leaves behind the writes it made along
        the way -- see `generate_response` for why this must run under the per-thread lock.
        """
        # Populated once the thread's store is resolved below; stays None (and the
        # outer except is then a no-op restore) if that resolution never happens.
        store = None
        checkpoint = None

        try:
            # Create prompt for this specific request
            request_prompt = self._create_request_prompt(operation, arguments)

            logger.debug(
                f"Request prompt: tool={tool_name}, prompt={request_prompt}, "
                f"thread_id={effective_thread_id}"
            )

            # Build config with thread_id for session management
            config: RunnableConfig = {
                "configurable": {"thread_id": effective_thread_id},
                "recursion_limit": self.agent_recursion_limit,
            }

            # Add store_registry to config if available
            if self.store_registry:
                config["configurable"]["store_registry"] = self.store_registry

            # Checkpoint the thread's store before the call so a failed call (one whose
            # final body is an "error" dict) can be rolled back to exactly how the store
            # stood beforehand, undoing any state-tool writes the LLM made along the way.
            store = (
                self.store_registry.for_thread(effective_thread_id)
                if self.store_registry
                else None
            )
            checkpoint = store.checkpoint() if store is not None else None

            # Generate response using agent with session context. One attempt, unless the
            # S4 fix below re-asks once (`request` then becomes the re-ask).
            request = HumanMessage(content=request_prompt)
            section_retried = False
            # This call's usage, summed over its turns (re-asks included).
            call_usage: dict[str, int] = {}
            while True:
                messages, response_text = await self._invoke(
                    request, config, effective_thread_id, trace, call_usage
                )
                logger.debug(
                    f"Raw response: tool={tool_name}, response={response_text[:200]}..., "
                    f"length={len(response_text)}"
                )

                # S2: a malformed final answer gets up to JSON_RETRIES re-asks, each with
                # the parse error, before it becomes the error body below. Each re-ask is
                # a fresh attempt on the restored store (as S4's), so a write the
                # malformed attempt made cannot be applied twice (plan 28 I5). Failure
                # path only: a first answer that parses is never re-asked.
                retries_left = JSON_RETRIES
                while retries_left > 0:
                    try:
                        json.loads(response_text)
                        break
                    except json.JSONDecodeError as parse_error:
                        embedded = (
                            _extract_json(response_text)
                            if _lenient_json_enabled()
                            else None
                        )
                        if embedded is not None:
                            logger.warning(
                                f"Lenient JSON: answer extracted from a non-JSON reply: "
                                f"tool={tool_name}, response={response_text[:200]}"
                            )
                            _fired(trace, LENIENT_JSON)
                            response_text = embedded
                            break
                        retries_left -= 1
                        logger.warning(
                            f"JSON parse error, re-asking: tool={tool_name}, "
                            f"error={parse_error}, response={response_text[:200]}"
                        )
                        _fired(trace, S2_RETRY)
                        if store is not None:
                            store.restore(checkpoint)
                        messages, response_text = await self._invoke(
                            _json_retry_message(operation, parse_error, request_prompt),
                            config,
                            effective_thread_id,
                            trace,
                            call_usage,
                        )

                # Try to parse as JSON
                try:
                    parsed_response = json.loads(response_text)

                    logger.debug(
                        f"Parsed response: tool={tool_name}, type={type(parsed_response).__name__}"
                    )

                    if isinstance(parsed_response, dict) and "error" in parsed_response:
                        if store is not None:
                            store.restore(checkpoint)
                        # S4: a refusal of an operation SKILL.md does define gets one
                        # fresh attempt (on the restored store), pointed at its section.
                        retry = (
                            None
                            if section_retried
                            else self._section_retry_message(
                                operation, request_prompt, parsed_response
                            )
                        )
                        if retry is not None:
                            section_retried = True
                            logger.warning(
                                f"Refusal of a defined operation, re-asking with its "
                                f"SKILL.md section: tool={tool_name}, "
                                f"response={response_text[:200]}"
                            )
                            _fired(trace, S4_SECTION_RETRY)
                            request = retry
                            continue
                        if trace is not None:
                            trace["outcome"] = "error_body"
                        return parsed_response

                    # Fail closed: a success body over a store write that was rejected
                    # and never redone describes a state change that did not happen.
                    failures = (
                        _unrecovered_write_failures(messages, store)
                        if store is not None
                        else []
                    )
                    if failures:
                        logger.warning(
                            f"Store write failed: tool={tool_name}, "
                            f"failures={failures}, response={response_text[:200]}"
                        )
                        store.restore(checkpoint)
                        if trace is not None:
                            trace["outcome"] = "write_failed"
                        return {
                            "error": "Store write failed",
                            "details": "; ".join(failures),
                        }

                    if trace is not None:
                        trace["outcome"] = "ok"
                    return parsed_response

                except json.JSONDecodeError as e:
                    logger.error(
                        f"JSON parse error: tool={tool_name}, response={response_text[:500]}, "
                        f"error={str(e)}"
                    )

                    if store is not None:
                        store.restore(checkpoint)
                    if trace is not None:
                        trace["outcome"] = "json_error"

                    # Return error response
                    return {
                        "error": "Failed to generate valid JSON response",
                        "details": str(e),
                    }

        except Exception as e:
            logger.error(
                f"Agent error: tool={tool_name}, error={str(e)}",
                exc_info=True,
            )

            if store is not None:
                store.restore(checkpoint)
            if trace is not None:
                trace["outcome"] = "exception"

            return {
                "error": "Internal error generating mock response",
                "details": str(e),
            }
        except BaseException:
            # Anything that isn't an Exception -- most notably asyncio.CancelledError,
            # raised when a client times out or disconnects mid-call -- still exits
            # through here without a successful body. Restore the store before
            # propagating so a cancelled call never leaves partial writes behind, then
            # re-raise rather than swallow it: cancellation must keep propagating.
            if store is not None:
                store.restore(checkpoint)
            raise

    async def _invoke(
        self,
        message: HumanMessage,
        config: RunnableConfig,
        thread_id: str,
        trace: dict[str, Any] | None,
        call_usage: dict[str, int],
    ) -> tuple[list[Any], str]:
        """One agent turn on the thread: ``(thread messages, last message's text)``.

        Timed and metered: this await is where every simulated token is spent, and the
        usage is otherwise unrecoverable from outside the process (the tool response
        deliberately carries none). `record` counts only the messages this turn added --
        with a checkpointer, `result["messages"]` is the whole thread history. A call that
        re-asks (S2/S4) makes several turns: ``call_usage`` is that call's own running
        sum (never the agent-wide field, which another thread's call may have set in
        between), and ``last_call_usage`` is set to it.
        """
        _t0 = time.perf_counter()
        result = await self.agent.ainvoke({"messages": [message]}, config=config)
        _llm_seconds = time.perf_counter() - _t0
        usage = self.usage.record(
            result.get("messages") if isinstance(result, dict) else None,
            thread_id=thread_id,
            seconds=_llm_seconds,
            model=self._model_name,
        )
        for key, value in usage.items():
            call_usage[key] = call_usage.get(key, 0) + value
        self.last_call_usage = dict(call_usage)

        # Extract the output from the last message
        messages = result.get("messages", [])
        if trace is not None:
            trace["messages"] = messages
        if not messages:
            raise ValueError("No response from agent")

        last_message = messages[-1]
        response_text = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )
        if trace is not None:
            trace["response_text"] = response_text
        return messages, response_text

    def _section_retry_message(
        self,
        operation: OpenAPIOperation,
        request_prompt: str,
        body: dict[str, Any],
    ) -> HumanMessage | None:
        """S4: the re-ask for a refusal of an operation SKILL.md defines, else None.

        In v1 the simulator read SKILL.md's first 1,000 lines, did not see an operation
        specified further down (retail ``modify_user_address`` at line 1654), and refused
        it as undefined without looking. Only when the body is such a refusal AND the
        loaded skill has the operation's section: an operation the skill lacks, any other
        error, and a condition without a skill keep their answer.
        """
        if self._skill_md is None or not _is_unknown_operation(body):
            return None
        found = skill_section(self._skill_md, operation.path, operation.method)
        if found is None:
            return None
        heading, first, last = found
        return HumanMessage(
            content=(
                f"The operation {operation.operation_id} IS defined by the simulation "
                f"skill: {self._skill_md_path}, section `{heading}`, lines {first}-{last} "
                f"(read_file offset={first - 1}, limit={last - first + 1}). Your previous "
                f"answer was rolled back. Read that section, then answer the request again "
                f"following it.\n\n{request_prompt}"
            ),
            additional_kwargs={RETRY_KWARG: S4_SECTION_RETRY},
        )

    def _find_operation(self, tool_name: str) -> OpenAPIOperation | None:
        """Find operation by tool name.

        Args:
            tool_name: Tool name (operation ID)

        Returns:
            Operation if found, None otherwise
        """
        for operation in self.operations:
            if operation.operation_id == tool_name:
                return operation
        return None

    def _create_request_prompt(
        self,
        operation: OpenAPIOperation,
        arguments: dict[str, Any],
    ) -> str:
        """Create prompt for specific request.

        Args:
            operation: API operation
            arguments: Request arguments

        Returns:
            Formatted request prompt
        """
        prompt_parts = [
            f"API Request: {operation.method.upper()} {operation.path}",
            f"Operation: {operation.operation_id}",
            "",
            "Request Parameters:",
            json.dumps(arguments, indent=2),
            "",
            "Generate a realistic mock response that:",
            "1. Matches the response schema exactly",
            "2. Uses the provided request parameters appropriately",
            "3. Maintains consistency with previous responses in this session",
            "4. Returns ONLY valid JSON (no explanations, no markdown)",
        ]

        return "\n".join(prompt_parts)

    async def reset(self, thread_id: str | None = None) -> None:
        """Reset the agent's state and memory.

        Args:
            thread_id: Optional specific thread to reset. If None, resets all sessions.
        """
        if thread_id:
            # Reset specific session
            cleared = self.session_manager.clear_session(thread_id)
            if cleared:
                logger.info(f"Reset agent session: thread_id={thread_id}")
            else:
                logger.warning(f"Session not found: thread_id={thread_id}")

            # Reset store for this thread if registry exists
            if self.store_registry:
                self.store_registry.reset(thread_id)

            # The history this thread's usage mark pointed into is gone; the mark must go with
            # it or the next turns would be counted as already accounted for. Spend is kept —
            # resetting the world does not un-spend tokens.
            self.usage.forget_thread(thread_id)
        else:
            # Reset all sessions
            count = self.session_manager.clear_all_sessions()
            logger.info(f"Reset all agent sessions: count={count}")

            # Drop all stores if registry exists
            if self.store_registry:
                self.store_registry.drop_all()

            self.usage.forget_thread()  # every history cleared; spend kept

    async def shutdown(self) -> None:
        """Shutdown the agent and cleanup resources."""
        # Stop session cleanup task
        await self.session_manager.stop()

        # Drop all stores if registry exists
        if self.store_registry:
            self.store_registry.drop_all()

        # Remove the per-simulation skills staging dir, if one was created.
        if self._skill_staging_dir is not None:
            shutil.rmtree(self._skill_staging_dir, ignore_errors=True)
            self._skill_staging_dir = None

        logger.info("Agent shutdown complete")


# Made with Bob
