"""C1: HARNESS_LLM_NO_CACHE reaches the runtime simulator's LLM requests, not only generation's.

The shared LiteLLM gateway caches whole responses; generation could opt out with
HARNESS_LLM_NO_CACHE, the simulator could not, so an identical replay returned the cached
answer in about a second. The flag must put the same opt-out on the runtime request body.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

from simulation_harness.agent.deep_agent import DeepAgent

_FIXTURE_SKILL_DIR = (
    Path(__file__).resolve().parents[2] / "fixtures" / "restaurant-reservation-api"
)


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The flag resolves through `.env` too; keep the developer's own out of these tests."""
    monkeypatch.chdir(tmp_path)


def _agent() -> DeepAgent:
    from simulation_harness.openapi.parser import OpenAPISpec

    spec = Mock(spec=OpenAPISpec)
    spec.info = {"title": "Test API", "version": "1.0.0", "description": "Test"}
    return DeepAgent(
        api_key=SecretStr("test-key"),
        model="gpt-4",
        temperature=0.0,
        max_tokens=1000,
        base_url=None,
        spec=spec,
        operations=[],
        skill_dir=_FIXTURE_SKILL_DIR,
    )


def _payload(agent: DeepAgent) -> dict[str, Any]:
    return agent.llm._get_request_payload([HumanMessage(content="hi")])


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_no_cache_flag_reaches_runtime_request(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("HARNESS_LLM_NO_CACHE", value)
    assert _payload(_agent())["extra_body"] == {"cache": {"no-cache": True}}


@pytest.mark.parametrize("value", [None, "", "0", "false"])
def test_runtime_request_unchanged_without_flag(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("HARNESS_LLM_NO_CACHE", raising=False)
    else:
        monkeypatch.setenv("HARNESS_LLM_NO_CACHE", value)
    payload = _payload(_agent())
    assert "extra_body" not in payload
    assert "cache" not in payload
