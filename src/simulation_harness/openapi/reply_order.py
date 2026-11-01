"""Put a simulated reply in the field order its response schema declares.

The LLM writes a reply's keys in whatever order it likes, and the no-skill simulator leaves out
optional fields that are null. A real service prints every field of its return type, in the
type's order. An agent reads the reply as text, so the same values in another layout can steer
it differently.

The order comes from the schema alone: an object's declared ``properties``, in the order the
spec lists them. (A spec saved with sorted keys lists them alphabetically; the order is only as
good as the spec's.) A declared optional field the reply omits is added as ``null``; a missing
``required`` field is not invented; a key the schema does not declare stays, after the declared
ones. A whole number in a ``number`` (not ``integer``) field is written as a decimal. An object with none of its declared fields (an
error answer) is left as it is. Values are never changed, and a free-form mapping
(``additionalProperties``) keeps the reply's own key order.
"""

from collections.abc import Callable
from typing import Any

RefResolver = Callable[[str], dict[str, Any] | None]


def _const_matches(variant: dict[str, Any], value: Any) -> bool:
    """Whether ``value`` carries every ``const`` the variant declares (a discriminator)."""
    props = variant.get("properties")
    if not isinstance(props, dict) or not isinstance(value, dict):
        return False
    consts = {
        k: p["const"] for k, p in props.items() if isinstance(p, dict) and "const" in p
    }
    return bool(consts) and all(value.get(k) == c for k, c in consts.items())


def _unwrap(
    schema: Any, resolve: RefResolver, value: Any = None, depth: int = 0
) -> dict[str, Any] | None:
    """The object/array schema under ``$ref``, ``allOf``/``anyOf``/``oneOf`` for ``value``.

    A nullable wrapper (``anyOf: [X, {"type": "null"}]``) unwraps to ``X``. Of several
    variants, the one whose ``const`` fields ``value`` matches is used (e.g. a payment
    method's ``source``); without exactly one such match the reply is left as it is
    (``None``).
    """
    if not isinstance(schema, dict) or depth > 32:
        return None
    ref = schema.get("$ref")
    if isinstance(ref, str):
        return _unwrap(resolve(ref), resolve, value, depth + 1)
    for key in ("allOf", "anyOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list) and variants:
            non_null = [
                v
                for v in variants
                if not (isinstance(v, dict) and v.get("type") == "null")
            ]
            if len(non_null) == 1:
                return _unwrap(non_null[0], resolve, value, depth + 1)
            resolved = [_unwrap(v, resolve, value, depth + 1) for v in non_null]
            matching = [
                v for v in resolved if v is not None and _const_matches(v, value)
            ]
            return matching[0] if len(matching) == 1 else None
    return schema


def order_like_schema(value: Any, schema: Any, resolve: RefResolver) -> Any:
    """``value`` with every object's keys in its schema's order (see the module docstring)."""
    s = _unwrap(schema, resolve, value)
    if s is None:
        return value
    if isinstance(value, dict):
        props = s.get("properties")
        if isinstance(props, dict) and props:
            if not any(k in props for k in value):
                return value  # an off-schema answer, e.g. {"error": ...}: not this type
            required = set(s.get("required", []))
            out: dict[str, Any] = {}
            for key in props:
                if key in value:
                    out[key] = order_like_schema(value[key], props[key], resolve)
                elif key not in required:
                    out[key] = None
            for key, v in value.items():
                if key not in out:
                    out[key] = v
            return out
        extra = s.get("additionalProperties")
        if isinstance(extra, dict):
            return {k: order_like_schema(v, extra, resolve) for k, v in value.items()}
        return value
    if isinstance(value, list):
        items = s.get("items")
        if isinstance(items, dict):
            return [order_like_schema(v, items, resolve) for v in value]
    if (
        s.get("type") == "number"
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        return float(value)  # same number; a `number` (not `integer`) prints as 250.0
    return value
