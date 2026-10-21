"""API response models for simulation harness."""

from datetime import datetime
from typing import Any, Optional, cast

from pydantic import BaseModel, Field

from simulation_harness.models.domain import SessionState


class ProgressPayload(BaseModel):
    """Progress payload embedded in SimulationResponse."""

    phase: Optional[str] = Field(
        None,
        description="Current phase: skill_generation | agent_init | None",
    )
    started_at: datetime = Field(..., description="Creation start time")
    updated_at: datetime = Field(..., description="Last status change time")


class ErrorPayload(BaseModel):
    """Error payload embedded in SimulationResponse for failed simulations."""

    code: str = Field(..., description="Stable error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional structured fields"
    )


class SimulationResponse(BaseModel):
    """Response model for simulation status."""

    name: str = Field(..., description="Name of the simulation")
    status: str = Field(
        ...,
        description="One of: pending | generating_skill | initializing | ready | failed",
    )
    session_state: Optional[SessionState] = Field(
        None,
        description="Session state — populated only when status=ready.",
    )
    mcp_url: Optional[str] = Field(
        None,
        description="Full MCP endpoint URL — populated only when status=ready.",
    )
    created_at: datetime = Field(
        ..., description="Timestamp when simulation was declared"
    )
    components: dict[str, bool] = Field(
        default_factory=lambda: {"skill": True, "state": True},
        description="Which simulator components are enabled: `skill` (SKILL.md loaded) and "
        "`state` (seed DB + state_* tools). Both true in normal operation; either false means "
        "the harness was started in an ablation configuration. Exposed so an evaluation can "
        "RECORD the configuration it measured instead of being told out-of-band.",
    )
    sampling: dict[str, Optional[float]] = Field(
        # Annotated so the lambda's inferred type is dict[str, float | None] rather than
        # dict[str, None], which does not match the field.
        default_factory=lambda: cast(
            "dict[str, Optional[float]]", {"temperature": None}
        ),
        description="Decoding parameters this process is serving: `temperature` (and `max_tokens` "
        "when set). Exposed for the same reason as `components` — two runs that differ only in "
        "temperature are otherwise indistinguishable in their artifacts, so a determinism study "
        "could not tell a sampled run from a greedy one after the fact. None means the harness "
        "could not read its config, which is reported as unknown rather than assumed to be 0.",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="What produced this simulation's answers: `skill_generation` (the skills-store "
        "directory loaded, e.g. `tau2-retail_261020-165700`, populated when ready), "
        "`harness_commit` and `harness_dirty` (this process's git checkout; None when not a "
        "checkout), and the runtime LLM settings `simulation_model`, `request_timeout_seconds`, "
        "`max_retries`. Exposed like `components` and `sampling`, so an evaluation records the "
        "exact skill and code it measured instead of a bare skill name.",
    )
    progress: ProgressPayload = Field(..., description="Creation progress info")
    error: Optional[ErrorPayload] = Field(
        None, description="Failure details — populated only when status=failed."
    )


class SimulationBundleResponse(BaseModel):
    """Full generated skill bundle for the active skill."""

    name: str = Field(..., description="Name of the active skill/simulation")
    files: dict[str, str] = Field(
        ...,
        description=(
            "Bundle filename -> verbatim file contents. Always includes SKILL.md, "
            "schema.json, db.json, api.json; scenarios.json and manifest.json are "
            "present only when they exist on disk."
        ),
    )
    sizes: dict[str, int] = Field(
        ...,
        description="Bundle filename -> uncompressed content size in bytes.",
    )


# Made with Bob
