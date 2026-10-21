"""Integration coverage for PATCH /simulation/state/{store}/{record_id}.

Wires the real FastAPI app to a real ``SimulationInstance`` whose agent is mocked but whose
``StoreRegistry`` is real, so the patch goes through the live thread store, the schema
validator and the central error mapping exactly as in production.
"""

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from simulation_harness.core.simulation_instance import SimulationInstance
from simulation_harness.models.domain import SimulationSpec
from simulation_harness.state import StoreRegistry

BILL_URL = "/api/v1/simulation/state/bills/B1"


@pytest.fixture(scope="module", autouse=True)
def _harness_env() -> Iterator[None]:
    config = {
        "llm": {
            "provider": "openai",
            "skill_generation_model": "gpt-4",
            "simulation_model": "gpt-4",
            "temperature": 0,
        },
        "skills": {"folder": "./skills"},
        "sessions": {
            "max_messages": 100,
            "idle_timeout_seconds": 3600,
            "max_concurrent_queue_depth": 8,
        },
        "mcp": {"transport": "sse"},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        path = f.name

    prior_key = os.environ.get("LLM_API_KEY")
    prior_cfg = os.environ.get("HARNESS_CONFIG_PATH")
    os.environ["LLM_API_KEY"] = "test-key"  # pragma: allowlist secret
    os.environ["HARNESS_CONFIG_PATH"] = path
    try:
        yield
    finally:
        os.unlink(path)
        if prior_key is not None:
            os.environ["LLM_API_KEY"] = prior_key
        else:
            os.environ.pop("LLM_API_KEY", None)
        if prior_cfg is not None:
            os.environ["HARNESS_CONFIG_PATH"] = prior_cfg
        else:
            os.environ.pop("HARNESS_CONFIG_PATH", None)


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    """A minimal skill whose ``bills`` store has an enum-typed ``status``."""
    d = tmp_path / "billing"
    d.mkdir()
    (d / "SKILL.md").write_text("# billing")
    (d / "api.json").write_text("{}")
    (d / "schema.json").write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {
                    "bills": {"type": "array", "items": {"$ref": "#/$defs/Bill"}}
                },
                "$defs": {
                    "Bill": {
                        "type": "object",
                        "required": ["bill_id", "status", "amount"],
                        "properties": {
                            "bill_id": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["Draft", "Issued", "Paid", "Overdue"],
                            },
                            "amount": {"type": "number"},
                        },
                        "x-primary-key": "bill_id",
                    }
                },
            }
        )
    )
    (d / "db.json").write_text(
        json.dumps(
            {
                "bills": [
                    {"bill_id": "B1", "status": "Overdue", "amount": 42.0},
                    {"bill_id": "B2", "status": "Paid", "amount": 7.5},
                ]
            }
        )
    )
    return d


@pytest.fixture
def wired(skill_dir: Path) -> Iterator[tuple[FastAPI, SimulationInstance, MagicMock]]:
    """The real app serving a READY simulation backed by a real StoreRegistry."""
    import importlib

    import simulation_harness.main as m

    importlib.reload(m)

    from simulation_harness.api.dependencies import get_simulation_host
    from simulation_harness.core.simulation_host import SimulationHost
    from simulation_harness.core.simulation_record import (
        SimulationRecord,
        SimulationStatus,
    )

    registry = StoreRegistry(skill_dir)
    agent = MagicMock()
    agent.store_registry = registry
    agent.generate_response = AsyncMock(return_value={"ok": True})
    # A real agent reset with no thread id drops every thread store.
    agent.reset = AsyncMock(side_effect=lambda *a, **k: registry.drop_all())
    agent.shutdown = AsyncMock()

    spec = SimulationSpec(
        name="billing",
        openapi_spec={"openapi": "3.0.0", "info": {"title": "B", "version": "1"}},
    )
    with patch(
        "simulation_harness.core.simulation_instance.DeepAgent", return_value=agent
    ):
        instance = SimulationInstance(
            spec=spec,
            max_messages=100,
            idle_timeout_seconds=3600,
            max_queue_depth=8,
            api_key=SecretStr("k"),  # pragma: allowlist secret
            model="gpt-4",
            temperature=0.0,
            max_tokens=100,
            skill_dir=skill_dir,
        )

    host = SimulationHost()
    record = SimulationRecord.declare(name="billing")
    record.transition(SimulationStatus.INITIALIZING, phase="agent_init")
    record.mark_ready(instance)
    host._record = record
    m.app.dependency_overrides[get_simulation_host] = lambda: host
    try:
        yield m.app, instance, agent
    finally:
        m.app.dependency_overrides.clear()


def _bill(state: dict[str, Any], bill_id: str) -> dict[str, Any]:
    return next(b for b in state["bills"] if b["bill_id"] == bill_id)


class TestRecordPatch:
    async def test_patch_persists_in_state_without_resetting_thread(
        self, wired: tuple[FastAPI, SimulationInstance, MagicMock]
    ) -> None:
        app, instance, agent = wired
        # A prior tool call: the thread has context and a counter worth preserving.
        await instance.execute_tool("getBill", {"bill_id": "B1"})
        assert instance.get_session_state().tool_call_count == 1

        with TestClient(app) as client:
            assert client.get("/api/v1/simulation/state").status_code == 200
            resp = client.patch(
                BILL_URL, params={"thread_id": "default"}, json={"status": "Paid"}
            )
            state = client.get("/api/v1/simulation/state").json()

        assert resp.status_code == 200, resp.text
        assert resp.json() == {"bill_id": "B1", "status": "Paid", "amount": 42.0}
        assert _bill(state, "B1")["status"] == "Paid"
        assert _bill(state, "B2") == {"bill_id": "B2", "status": "Paid", "amount": 7.5}
        # The thread survived: no agent reset, counter untouched.
        agent.reset.assert_not_awaited()
        assert instance.get_session_state().tool_call_count == 1

    def test_bad_enum_returns_422_and_leaves_record(
        self, wired: tuple[FastAPI, SimulationInstance, MagicMock]
    ) -> None:
        app, _instance, _agent = wired
        with TestClient(app) as client:
            client.get("/api/v1/simulation/state")
            resp = client.patch(BILL_URL, json={"status": "Settled"})
            state = client.get("/api/v1/simulation/state").json()

        assert resp.status_code == 422, resp.text
        assert "status" in resp.json()["detail"]
        assert _bill(state, "B1")["status"] == "Overdue"

    @pytest.mark.parametrize(
        "url",
        [
            "/api/v1/simulation/state/bills/NOPE",
            "/api/v1/simulation/state/invoices/B1",
        ],
    )
    def test_unknown_record_or_store_returns_404(
        self, wired: tuple[FastAPI, SimulationInstance, MagicMock], url: str
    ) -> None:
        app, _instance, _agent = wired
        with TestClient(app) as client:
            client.get("/api/v1/simulation/state")
            resp = client.patch(url, json={"status": "Paid"})
        assert resp.status_code == 404, resp.text

    def test_no_live_thread_store_after_reset_returns_409(
        self, wired: tuple[FastAPI, SimulationInstance, MagicMock]
    ) -> None:
        app, _instance, _agent = wired
        with TestClient(app) as client:
            client.get("/api/v1/simulation/state")
            assert client.post("/api/v1/simulation/reset").status_code == 200
            resp = client.patch(BILL_URL, json={"status": "Paid"})
        assert resp.status_code == 409, resp.text

    def test_patch_never_creates_a_thread_store(
        self, wired: tuple[FastAPI, SimulationInstance, MagicMock]
    ) -> None:
        app, _instance, agent = wired
        with TestClient(app) as client:
            resp = client.patch(BILL_URL, json={"status": "Paid"})
        assert resp.status_code == 409, resp.text
        assert agent.store_registry.thread_count() == 0

    def test_no_simulation_returns_404(self) -> None:
        import importlib

        import simulation_harness.main as m

        importlib.reload(m)
        from simulation_harness.api.dependencies import get_simulation_host
        from simulation_harness.core.simulation_host import SimulationHost

        m.app.dependency_overrides[get_simulation_host] = lambda: SimulationHost()
        try:
            with TestClient(m.app) as client:
                resp = client.patch(BILL_URL, json={"status": "Paid"})
        finally:
            m.app.dependency_overrides.clear()
        assert resp.status_code == 404, resp.text
