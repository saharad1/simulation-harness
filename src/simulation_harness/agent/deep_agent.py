"""Simplified Deep Agent implementation for Simulation Harness.

This module provides the main agent that generates mock responses using
LangChain and LangGraph directly (without the deepagents library).
"""

import asyncio
import json
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


def _unrecovered_write_failures(messages: list[Any], store: Any) -> list[str]:
    """Store writes this call's LLM turn failed and never redid successfully.

    Scans the messages after the last ``HumanMessage`` (this call's request; with a
    checkpointer the list is the whole thread's history). A failed write is cleared
    only by a later successful write to the same store and primary key -- kept that
    narrow on purpose, so a success elsewhere can never paper over it.
    """
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            start = i + 1
            break
    calls: dict[str, dict] = {}
    failed: dict[tuple[str, str | None], str] = {}
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
        target = (store_name, None if key is None else str(key))
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
    return list(failed.values())


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
                prompt instead. The state store and its tools are unaffected.
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

        self.llm = ChatOpenAI(**llm_kwargs)

        # Render system prompt
        # `no_skill`: the spec itself, unmodified, is the only description of the API.
        if skill_enabled:
            self.system_prompt = render_system_prompt(spec)
        else:
            self.system_prompt = render_system_prompt(
                spec, raw_spec=json.dumps(spec.spec_dict, indent=2, ensure_ascii=False)
            )

        logger.debug(
            f"Rendered system prompt: length={len(self.system_prompt)}, "
            f"preview={self.system_prompt[:200]}..."
        )

        # Initialize checkpointer for session management
        self.checkpointer = MemorySaver()

        # Initialize state store registry if skill_dir provided
        self.store_registry: StoreRegistry | None = None
        if skill_dir:
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
            f"state_store={'enabled' if skill_dir else 'disabled'}"
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
        if self.store_registry and self.skill_dir and not self.skill_enabled:
            # `no_skill`: state tools only. No SkillsMiddleware and no filesystem, so nothing of
            # the compiled skill (SKILL.md, schema.json, ...) is reachable; the spec is in the
            # system prompt. The store itself is the skill dir's, seeded as usual.
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

            # Generate response using agent with session context.
            # Timed and metered: this await is where every simulated token is spent, and the
            # usage is otherwise unrecoverable from outside the process (the tool response
            # deliberately carries none). `record` counts only the messages this call added —
            # with a checkpointer, `result["messages"]` is the whole thread history.
            _t0 = time.perf_counter()
            result = await self.agent.ainvoke(
                {"messages": [HumanMessage(content=request_prompt)]},
                config=config,
            )
            _llm_seconds = time.perf_counter() - _t0
            self.last_call_usage = self.usage.record(
                result.get("messages") if isinstance(result, dict) else None,
                thread_id=effective_thread_id,
                seconds=_llm_seconds,
                model=self._model_name,
            )

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

            logger.debug(
                f"Raw response: tool={tool_name}, response={response_text[:200]}..., "
                f"length={len(response_text)}"
            )
            if trace is not None:
                trace["response_text"] = response_text

            # Try to parse as JSON
            try:
                parsed_response = json.loads(response_text)

                logger.debug(
                    f"Parsed response: tool={tool_name}, type={type(parsed_response).__name__}"
                )

                if isinstance(parsed_response, dict) and "error" in parsed_response:
                    if store is not None:
                        store.restore(checkpoint)
                    if trace is not None:
                        trace["outcome"] = "error_body"
                    return parsed_response

                # Fail closed: a success body over a store write that was rejected and
                # never redone describes a state change that did not happen.
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
