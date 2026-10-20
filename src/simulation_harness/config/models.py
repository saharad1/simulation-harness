"""Pydantic models for configuration."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TransportType(str, Enum):
    """Supported transport types for MCP communication."""

    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"


class LLMConfig(BaseModel):
    """LLM provider configuration (non-secret).

    Secret values (API key, optional API base) are loaded separately via
    simulation_harness.config.Secrets. This model contains only non-secret
    tuning parameters.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(..., description="LLM provider (e.g., 'openai')")
    skill_generation_model: str = Field(..., description="Model for skill generation")
    simulation_model: str = Field(..., description="Model for simulation runtime")
    temperature: float = Field(0, ge=0, le=2, description="Sampling temperature")
    max_tokens: Optional[int] = Field(
        None, gt=0, description="Maximum tokens per request"
    )
    # Bounds each simulation-runtime LLM call. Unset keeps the OpenAI client's own default,
    # a 600 s read timeout: an LLM gateway that accepts a request and never answers then holds
    # one tools/call for ten minutes per attempt (observed 2026-10-18, when healthy calls took
    # a median 2.4 s and p99 14.5 s).
    request_timeout_seconds: Optional[float] = Field(
        None, gt=0, description="Per-request timeout for simulation LLM calls (seconds)"
    )
    max_retries: Optional[int] = Field(
        None,
        ge=0,
        description="Retries per simulation LLM call after a timeout or error",
    )


class SkillsConfig(BaseModel):
    """Skills folder configuration."""

    folder: str = Field(..., description="Path to skills directory")


class AblationConfig(BaseModel):
    """Which of the simulator's two components are enabled.

    Both default True — the full method. Turning either off yields one cell of the
    2x2 ablation grid used to attribute fidelity to the components:

        skill=T state=T  full method (default)
        skill=T state=F  semantic contract, no persistent world (refused, see below)
        skill=F state=T  seed corpus + state tools, no operation contract
        skill=F state=F  spec-only emulation (the ToolEmu-style baseline)

    These are EVALUATION conditions, not features. Nothing in normal operation
    should set them; they exist so the paper's component claims are falsifiable.
    """

    skill: bool = Field(
        True,
        description="Load SKILL.md (per-operation contracts, derivation rules, "
        "scenarios) into the agent. When False the agent sees only the OpenAPI "
        "spec in its system prompt.",
    )
    state: bool = Field(
        True,
        description="Enable the seed database and the session-scoped state store "
        "(state_* tools). When False the agent has no persistence and must "
        "fabricate every response from the request alone.",
    )

    @model_validator(mode="after")
    def _refuse_skill_without_state(self) -> "AblationConfig":
        # A compiled skill's contracts are written against the store (read this entity, write
        # that one). Without one the simulator is told to use tools it does not have, which
        # measures nothing we report. Refused here, at config load, so the process never starts
        # and the status endpoint never reports a condition nobody runs.
        if self.skill and not self.state:
            raise ValueError(
                "HARNESS_ABLATION_STATE=false requires HARNESS_ABLATION_SKILL=false: "
                "skill=on,state=off is not a supported condition (spec_only is skill=off,state=off)"
            )
        return self

    @property
    def label(self) -> str:
        """Short condition name for logs, run notes and report columns."""
        return f"skill={'on' if self.skill else 'off'},state={'on' if self.state else 'off'}"


class SessionsConfig(BaseModel):
    """Session limits configuration."""

    max_messages: int = Field(..., gt=0, description="Maximum tool calls per session")
    idle_timeout_seconds: int = Field(..., gt=0, description="Idle timeout in seconds")
    max_concurrent_queue_depth: int = Field(
        8, gt=0, description="Maximum concurrent queue depth"
    )
    agent_recursion_limit: int = Field(
        50,
        gt=0,
        description="LangGraph super-step budget per tool call. The skill-loading "
        "runtime is a multi-step agent (progressive disclosure + state tools), so "
        "this must exceed the legacy single-call default of 10; ~2 super-steps are "
        "consumed per model/tool round-trip.",
    )


class CreationConfig(BaseModel):
    """Async simulation creation configuration."""

    model_config = ConfigDict(extra="forbid")

    max_duration_seconds: int = Field(
        600,
        gt=0,
        description="Wall-clock budget for async simulation creation; "
        "exceeded creations transition to failed with code 'creation_timeout'.",
    )


class StageParams(BaseModel):
    """Per-stage LLM tuning for the generation pipeline."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(0.0, ge=0, le=2)
    max_tokens: int = Field(8000, gt=0)
    # Overrides GenerationConfig.request_timeout_seconds for this stage's client. The
    # large-output stages need more: a healthy 20 000-token reply took 249 s (2026-10-20).
    request_timeout_seconds: float | None = Field(
        None,
        gt=0,
        description="Per-request timeout for this stage's LLM calls (seconds)",
    )


class GenerationConfig(BaseModel):
    """Multi-step skill-generation pipeline tuning (non-secret)."""

    model_config = ConfigDict(extra="forbid")

    concurrency: int = Field(5, gt=0, description="Max simultaneous op-section calls")
    chunk_threshold: int = Field(
        40, gt=0, description="Ops above this are batched by tag instead of per-op"
    )
    classify_batch_size: int = Field(
        40,
        gt=0,
        description="Bin capacity (max operations) per stage-1b classify call",
    )
    repair_retries: int = Field(
        2,
        ge=0,
        description="Retries per artifact before failing (→ retries+1 attempts)",
    )
    stage_timeout_seconds: int = Field(
        120, gt=0, description="Per individual LLM call timeout"
    )
    # Bounds each generation request, so a gateway that accepts a call and never answers
    # costs this long before llm._ainvoke retries it, instead of the whole stage budget.
    # Unset keeps the client default (no timeout). Measured 2026-10-20: the same 145 KB
    # request took 992 s once and 8 s a minute later; a 20 000-token reply took 249 s.
    request_timeout_seconds: float | None = Field(
        None, gt=0, description="Per-request timeout for generation LLM calls (seconds)"
    )
    extract: StageParams = Field(default_factory=lambda: StageParams(max_tokens=8000))
    enrich: StageParams = Field(default_factory=lambda: StageParams(max_tokens=8000))
    classify: StageParams = Field(default_factory=lambda: StageParams(max_tokens=4000))
    schema_seed: StageParams = Field(
        default_factory=lambda: StageParams(max_tokens=12000)
    )
    operation: StageParams = Field(default_factory=lambda: StageParams(max_tokens=4000))
    scenarios: StageParams = Field(
        default_factory=lambda: StageParams(temperature=0.4, max_tokens=3000)
    )
    scenarios_enabled: bool = Field(
        True, description="Generate example user scenarios and seed data to match"
    )
    scenarios_count: int = Field(
        5, gt=0, description="Number of example scenarios to generate"
    )
    behavior: StageParams = Field(default_factory=lambda: StageParams(max_tokens=3000))
    behavior_enabled: bool = Field(
        True,
        description="Generate a global behavior/realism section (numeric ranges, "
        "derivation rules) injected into the skill preamble",
    )


class StartupConfig(BaseModel):
    """Boot-time auto-start configuration."""

    model_config = ConfigDict(extra="forbid")

    autostart_enabled: bool = Field(
        default=False,
        description="Master switch for boot-time autostart. When False (default), "
        "the harness boots idle regardless of autostart_simulation or discovered "
        "skills.",
    )
    autostart_simulation: Optional[str] = Field(
        default=None,
        description="Skill name to auto-start on boot. If unset, auto-discovers baked skills.",
    )


class MCPConfig(BaseModel):
    """MCP transport configuration."""

    transport: TransportType = Field(..., description="MCP transport type")


class ServerSettings(BaseModel):
    """Server host and port configuration."""

    host: str = Field("localhost", description="Host to bind the server to")
    port: int = Field(8000, ge=1, le=65535, description="Port to bind the server to")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(
        "INFO", description="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)"
    )
    destination_folder: str = Field("./logs", description="Directory for log files")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate that level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v


class HarnessConfig(BaseModel):
    """Complete harness configuration (non-secret)."""

    llm: LLMConfig
    skills: SkillsConfig
    sessions: SessionsConfig
    ablation: AblationConfig = Field(
        default_factory=AblationConfig,
        description="Component toggles for evaluation baselines. Defaults to the "
        "full method; see AblationConfig.",
    )
    creation: CreationConfig = Field(
        default_factory=CreationConfig,
        description="Async simulation-creation tuning.",
    )
    generation: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="Multi-step skill-generation pipeline tuning.",
    )
    mcp: MCPConfig
    server: ServerSettings = Field(
        default_factory=ServerSettings, description="Server configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    startup: StartupConfig = Field(
        default_factory=StartupConfig,
        description="Boot-time auto-start configuration.",
    )


# Made with Bob
