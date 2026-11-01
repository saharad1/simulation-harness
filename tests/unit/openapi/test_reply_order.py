"""Unit tests for putting a simulated reply in its response schema's field order."""

import json
from typing import Any

from simulation_harness.openapi.reply_order import order_like_schema

# `properties` in the type's own order; `status` and `payment_methods` optional, `status` nullable.
SPEC: dict[str, Any] = {
    "components": {
        "schemas": {
            "Passenger": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "dob": {"type": "string"},
                },
                "required": ["first_name", "dob"],
            }
        }
    }
}
RESERVATION: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reservation_id": {"type": "string"},
        "status": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "cabin": {"type": "string"},
        "passengers": {
            "type": "array",
            "items": {"$ref": "#/components/schemas/Passenger"},
        },
        "payment_methods": {
            "type": "object",
            "additionalProperties": {"$ref": "#/components/schemas/Passenger"},
        },
    },
    "required": ["reservation_id", "cabin", "passengers"],
}
ENVELOPE: dict[str, Any] = {
    "type": "object",
    "properties": {"returns": RESERVATION},
    "required": ["returns"],
}


def resolve(ref: str) -> dict[str, Any] | None:
    result = SPEC["components"]["schemas"].get(ref.rsplit("/", 1)[-1])
    return result if isinstance(result, dict) else None


def test_orders_as_declared_and_fills_missing_optional_with_null() -> None:
    reply = {"returns": {"cabin": "economy", "passengers": [], "reservation_id": "R1"}}
    out = order_like_schema(reply, ENVELOPE, resolve)
    # an optional field declared mid-type stays mid-type, not after the required ones
    assert list(out["returns"]) == [
        "reservation_id",
        "status",
        "cabin",
        "passengers",
        "payment_methods",
    ]
    assert out["returns"]["status"] is None


def test_values_unchanged_and_nested_items_ordered() -> None:
    reply = {
        "returns": {
            "status": "cancelled",
            "passengers": [{"dob": "1990-01-01", "first_name": "Ana"}],
            "cabin": "business",
            "reservation_id": "R2",
            "payment_methods": {
                "card_2": {"dob": "x", "first_name": "y"},
                "card_1": {},
            },
        }
    }
    out = order_like_schema(reply, ENVELOPE, resolve)
    assert out == reply
    assert list(out["returns"]["passengers"][0]) == ["first_name", "dob"]
    # a free-form mapping keeps the reply's key order; its values are still ordered
    assert list(out["returns"]["payment_methods"]) == ["card_2", "card_1"]
    assert list(out["returns"]["payment_methods"]["card_2"]) == ["first_name", "dob"]


def test_missing_required_field_is_not_invented() -> None:
    out = order_like_schema({"returns": {"cabin": "economy"}}, ENVELOPE, resolve)
    assert "reservation_id" not in out["returns"]
    assert "passengers" not in out["returns"]


def test_undeclared_keys_kept_after_declared_ones() -> None:
    reply = {
        "returns": {"note": "x", "reservation_id": "R3", "cabin": "c", "passengers": []}
    }
    out = order_like_schema(reply, ENVELOPE, resolve)
    assert list(out["returns"])[-1] == "note"


def test_error_answers_pass_through() -> None:
    top = {"error": "Failed to generate valid JSON response", "details": "..."}
    assert order_like_schema(top, ENVELOPE, resolve) == top
    nested = {"returns": {"error": "Reservation not found"}}
    out = order_like_schema(nested, ENVELOPE, resolve)
    assert out == nested
    assert list(out["returns"]) == ["error"]


def test_no_schema_or_scalar_unchanged() -> None:
    assert order_like_schema({"b": 1, "a": 2}, None, resolve) == {"b": 1, "a": 2}
    assert order_like_schema("ok", ENVELOPE, resolve) == "ok"
    ambiguous = {"anyOf": [{"type": "object"}, {"type": "array"}]}
    assert list(order_like_schema({"b": 1, "a": 2}, ambiguous, resolve)) == ["b", "a"]


def test_union_variant_picked_by_its_const_field() -> None:
    card = {
        "type": "object",
        "properties": {
            "source": {"const": "credit_card", "type": "string"},
            "id": {"type": "string"},
            "last_four": {"type": "string"},
        },
    }
    gift = {
        "type": "object",
        "properties": {
            "source": {"const": "gift_card", "type": "string"},
            "id": {"type": "string"},
            "amount": {"type": "number"},
        },
    }
    methods = {"type": "object", "additionalProperties": {"anyOf": [card, gift]}}
    reply = {
        "g": {"amount": 5, "id": "g", "source": "gift_card"},
        "c": {"last_four": "1", "id": "c", "source": "credit_card"},
        "x": {"id": "x", "source": "paypal"},
    }
    out = order_like_schema(reply, methods, resolve)
    assert list(out["g"]) == ["source", "id", "amount"]
    assert list(out["c"]) == ["source", "id", "last_four"]
    assert list(out["x"]) == ["id", "source"]  # no variant matches: left as it is


def test_whole_number_in_number_field_written_as_decimal() -> None:
    schema = {
        "type": "object",
        "properties": {
            "amount": {"type": "number"},
            "count": {"type": "integer"},
            "flag": {"type": "number"},
        },
    }
    out = order_like_schema({"amount": 250, "count": 2, "flag": True}, schema, resolve)
    assert json.dumps(out) == '{"amount": 250.0, "count": 2, "flag": true}'
