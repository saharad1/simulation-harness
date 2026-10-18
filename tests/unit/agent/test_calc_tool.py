"""Tests for the exact-arithmetic calculator tool used by the simulator."""

import pytest

from simulation_harness.agent.calc_tool import calc, create_calculator_tool


def test_decimal_exact() -> None:
    assert calc("210.15 - 77.15")["result"] == "133.00"


def test_round_and_sum() -> None:
    assert calc("round(sum([19.99, 5.01, 0.006]), 2)")["result"] == "25.01"
    # Decimal, ROUND_HALF_UP -- not float, not banker's rounding.
    assert calc("round(2.675, 2)")["result"] == "2.68"


def test_rejects_names_and_calls() -> None:
    with pytest.raises(ValueError):
        calc("__import__('os')")
    with pytest.raises(ValueError):
        calc("open('x')")


def test_basic_arithmetic() -> None:
    assert calc("1 + 2 * 3")["result"] == "7"
    assert calc("(1 + 2) * 3")["result"] == "9"
    assert calc("-5 + 3")["result"] == "-2"


def test_abs_min_max() -> None:
    assert calc("abs(-5)")["result"] == "5"
    assert calc("min(3, 1, 2)")["result"] == "1"
    assert calc("max(3, 1, 2)")["result"] == "3"


def test_division_by_zero_raises() -> None:
    with pytest.raises(ValueError):
        calc("1 / 0")


def test_rejects_attribute_and_subscript() -> None:
    with pytest.raises(ValueError):
        calc("(1).bit_length()")
    with pytest.raises(ValueError):
        calc("[1, 2][0]")


def test_rejects_disallowed_function() -> None:
    with pytest.raises(ValueError):
        calc("pow(2, 3)")


def test_rejects_keyword_arguments() -> None:
    with pytest.raises(ValueError):
        calc("round(2.675, ndigits=2)")


def test_create_calculator_tool_returns_structured_tool_named_calculate() -> None:
    tool = create_calculator_tool()
    assert tool.name == "calculate"
    result = tool.func(expression="1 + 1")
    assert result == {"result": "2"}


def test_create_calculator_tool_reports_errors_like_state_tools() -> None:
    tool = create_calculator_tool()
    result = tool.func(expression="1 / 0")
    assert isinstance(result, dict)
    assert "error" in result
