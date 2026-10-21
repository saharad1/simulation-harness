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


def test_render_has_no_failed_write_instruction() -> None:
    """Plan 28 C1: B2 is enforced in code (`_unrecovered_write_failures`), not by a prompt
    sentence, so the skill-branch prompt stays byte-identical to the one v1 ran with."""
    for result in (
        render_system_prompt(_mock_spec()),
        render_system_prompt(_mock_spec(), raw_spec="{}"),
    ):
        assert "that write did not happen" not in result


def test_render_stateless_drops_every_state_tool() -> None:
    """`spec_only`: no store, so the prompt must not tell the simulator to use state tools it
    does not have. The raw spec and the calculator stay."""
    result = render_system_prompt(
        _mock_spec(), raw_spec='{"paths": {"/x": "RAWSPECMARKER"}}', stateless=True
    )
    assert "RAWSPECMARKER" in result
    assert "skill" not in result.lower()
    for tool in (
        "state_get",
        "state_list",
        "state_count",
        "state_insert",
        "state_update",
        "state_delete",
    ):
        assert tool not in result
    assert "state_" not in result
    assert "state tools" not in result.lower()
    assert "calculate" in result


def test_render_stateless_keeps_the_raw_spec_section_byte_identical() -> None:
    """The API Specification section is exactly what `no_skill` gets."""
    raw = '{"paths": {"/x": "RAWSPECMARKER"}}'
    with_state = render_system_prompt(_mock_spec(), raw_spec=raw)
    without = render_system_prompt(_mock_spec(), raw_spec=raw, stateless=True)

    def spec_section(text: str) -> str:
        start = text.index("## API Specification")
        return text[start : text.index("```", text.index("```json") + 7) + 3]

    assert spec_section(with_state) == spec_section(without)


def test_render_stateless_false_is_unchanged() -> None:
    """`stateless=False` renders exactly what no_skill always rendered."""
    raw = '{"paths": {}}'
    assert render_system_prompt(_mock_spec(), raw_spec=raw) == render_system_prompt(
        _mock_spec(), raw_spec=raw, stateless=False
    )
