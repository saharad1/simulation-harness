"""Tests for the prompts module."""

from unittest.mock import Mock

from simulation_harness.agent.prompts import render_system_prompt
from simulation_harness.openapi.parser import OpenAPISpec
from unittest.mock import MagicMock


def _mock_spec() -> MagicMock:
    spec = Mock(spec=OpenAPISpec)
    spec.info = {"title": "Test API", "version": "1.0.0", "description": "A test API"}
    spec.servers = [{"url": "https://api.example.com"}]
    return spec


def test_render_includes_api_info() -> None:
    result = render_system_prompt(_mock_spec())
    assert "Test API" in result
    assert "1.0.0" in result


def test_render_keeps_state_mechanism_and_json_contract() -> None:
    result = render_system_prompt(_mock_spec())
    # State tools must still be described in the always-on prompt.
    assert "state_get" in result
    assert "state_insert" in result
    assert "where" in result.lower()
    # JSON-only contract preserved.
    assert "json" in result.lower()


def test_render_points_to_simulation_skill() -> None:
    result = render_system_prompt(_mock_spec())
    assert "skill" in result.lower()


def test_render_does_not_embed_per_operation_detail() -> None:
    # render_system_prompt no longer accepts or renders operations.
    result = render_system_prompt(_mock_spec())
    assert "/users/{id}" not in result
    assert "getUser" not in result


def test_render_raw_spec_replaces_skill_pointer() -> None:
    """`no_skill`: the raw spec is embedded and every pointer to a skill is gone."""
    result = render_system_prompt(
        _mock_spec(), raw_spec='{"paths": {"/x": "RAWSPECMARKER"}}'
    )
    assert "RAWSPECMARKER" in result
    assert "skill" not in result.lower()
    assert "state_get" in result  # the store mechanism is unaffected


def test_render_without_raw_spec_is_unchanged_by_the_flag() -> None:
    """Passing no raw_spec must render exactly what the method always rendered."""
    assert render_system_prompt(_mock_spec()) == render_system_prompt(
        _mock_spec(), raw_spec=None
    )


def test_render_says_a_failed_store_write_fails_the_operation() -> None:
    for result in (
        render_system_prompt(_mock_spec()),
        render_system_prompt(_mock_spec(), raw_spec="{}"),
    ):
        assert "that write did not happen and the operation failed" in result
