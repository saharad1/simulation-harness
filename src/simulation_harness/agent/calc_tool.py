"""An exact-arithmetic calculator tool for the simulating agent.

The simulator sometimes has to do arithmetic (totals, change, discounts) while
writing or returning a response. Doing that in its head is lossy -- it is
occasionally off by exactly $1.00. This tool evaluates a small, whitelisted
arithmetic grammar using ``decimal.Decimal`` so results are exact, and
reports errors the same way the state tools do (a dict with an ``error``
key), rather than raising into the agent loop.

Only ``+ - * / ( )``, decimal literals, unary minus, and the functions
``round`` (with ``decimal.ROUND_HALF_UP``), ``abs``, ``min``, ``max``, and
``sum`` (over a list literal) are supported. The expression is parsed with
``ast.parse(mode="eval")`` and walked node-by-node against an explicit
whitelist; anything else -- names, attributes, subscripts, keyword
arguments, other calls -- raises ``ValueError``. ``eval``/``exec`` are never
used.
"""

import ast
from decimal import Decimal, DivisionByZero, InvalidOperation, ROUND_HALF_UP

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

_ALLOWED_FUNCS = {"round", "abs", "min", "max", "sum"}


class CalculatorArgs(BaseModel):
    """Arguments for the calculate tool."""

    expression: str = Field(
        ...,
        description=(
            "An arithmetic expression using + - * / ( ), decimal literals, "
            "and round/abs/min/max/sum."
        ),
    )


def _literal_to_decimal(node: ast.AST) -> Decimal:
    """Convert a numeric literal node to a Decimal built from its source text.

    Never routes through a binary float: a float literal's token text is
    passed to ``Decimal(str(...))``-equivalent construction so e.g. 2.675
    stays exact.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise ValueError("booleans are not supported")
        # ast.Constant doesn't retain the original source text, but Python's
        # float/int repr of a literal parsed from source round-trips exactly
        # for literals (the float is constructed from the same digits the
        # tokenizer saw), so str() here reproduces the source digits.
        return Decimal(str(node.value))
    raise ValueError(f"unsupported literal: {ast.dump(node)}")


def _eval_node(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        return _literal_to_decimal(node)

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +_eval_node(node.operand)
        raise ValueError(f"unsupported unary operator: {ast.dump(node.op)}")

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            try:
                return left / right
            except (DivisionByZero, InvalidOperation, ZeroDivisionError) as e:
                raise ValueError("division by zero") from e
        raise ValueError(f"unsupported operator: {ast.dump(node.op)}")

    if isinstance(node, ast.List):
        raise ValueError("list literals are only supported as sum(...) arguments")

    if isinstance(node, ast.Call):
        return _eval_call(node)

    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def _eval_list_literal(node: ast.AST) -> list[Decimal]:
    if not isinstance(node, ast.List):
        raise ValueError("sum() requires a list literal argument")
    return [_eval_node(elt) for elt in node.elts]


def _eval_call(node: ast.Call) -> Decimal:
    if not isinstance(node.func, ast.Name):
        raise ValueError("only direct calls to whitelisted functions are allowed")
    name = node.func.id
    if name not in _ALLOWED_FUNCS:
        raise ValueError(f"function not allowed: {name}")
    if node.keywords:
        raise ValueError("keyword arguments are not allowed")

    if name == "round":
        if len(node.args) not in (1, 2):
            raise ValueError("round() takes 1 or 2 positional arguments")
        value = _eval_node(node.args[0])
        ndigits = int(_eval_node(node.args[1])) if len(node.args) == 2 else 0
        quantum = Decimal(1).scaleb(-ndigits)
        return value.quantize(quantum, rounding=ROUND_HALF_UP)

    if name == "abs":
        if len(node.args) != 1:
            raise ValueError("abs() takes exactly 1 positional argument")
        return abs(_eval_node(node.args[0]))

    if name in ("min", "max"):
        if len(node.args) == 1 and isinstance(node.args[0], ast.List):
            values = _eval_list_literal(node.args[0])
        else:
            if not node.args:
                raise ValueError(f"{name}() requires at least one argument")
            values = [_eval_node(a) for a in node.args]
        return min(values) if name == "min" else max(values)

    if name == "sum":
        if len(node.args) != 1:
            raise ValueError("sum() takes exactly 1 argument (a list literal)")
        values = _eval_list_literal(node.args[0])
        total = Decimal(0)
        for v in values:
            total += v
        return total

    raise ValueError(f"function not allowed: {name}")


def calc(expression: str) -> dict:
    """Evaluate an arithmetic expression exactly and return ``{"result": str}``.

    Raises:
        ValueError: on any disallowed syntax, name, call, or division by zero.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid expression: {e}") from e

    result = _eval_node(tree)
    return {"result": str(result)}


def _calculate(expression: str) -> dict:
    """Tool entry point: evaluate ``expression``, reporting errors like the state tools."""
    try:
        return calc(expression)
    except ValueError as e:
        return {"error": "bad_expression", "message": str(e)}
    except Exception as e:  # defensive: never let a bad expression crash the agent loop
        return {"error": "internal_error", "message": str(e)}


def create_calculator_tool() -> StructuredTool:
    """Create the ``calculate`` tool for the simulating agent.

    Returns:
        A StructuredTool named "calculate" taking a single "expression"
        argument and returning ``{"result": str}`` on success, or an error
        dict (matching the state tools' error shape) on failure.
    """
    return StructuredTool(
        name="calculate",
        description=(
            "Evaluate an exact arithmetic expression (+ - * / ( ), decimals, "
            "round, abs, min, max, sum) and return the result as a string. "
            "Use this for every arithmetic result you write or return."
        ),
        func=_calculate,
        args_schema=CalculatorArgs,
    )


# Made with Bob
