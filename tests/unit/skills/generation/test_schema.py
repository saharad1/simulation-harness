import json
from unittest.mock import AsyncMock, patch

import pytest
from jsonschema.validators import Draft202012Validator

from simulation_harness.openapi.parser import OpenAPISpec
from simulation_harness.skills.generation.ir import (
    Entity,
    Field,
    SpecModel,
    StoreMetadata,
)
from simulation_harness.skills.generation.repair import GenerationStageError
from simulation_harness.skills.generation.stages import schema as S

GOOD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {"features": {"type": "array", "items": {"$ref": "#/$defs/Feature"}}},
    "$defs": {
        "Feature": {
            "type": "object",
            "additionalProperties": False,
            "x-primary-key": "id",
            "required": ["id"],
            "properties": {"id": {"type": "string"}},
        }
    },
}


def _ir() -> SpecModel:
    return SpecModel(
        api_name="Aha",
        slug="aha",
        entities=[
            Entity(
                name="Feature",
                collection="features",
                primary_key="id",
                fields=[Field(name="id", type="string", required=True)],
            )
        ],
        operations=[],
        store_metadata=StoreMetadata(
            collections=["features"], pk_map={"features": "id"}
        ),
    )


def test_validate_schema_accepts_valid_schema() -> None:
    assert S.validate_schema(GOOD_SCHEMA) == []


def test_validate_schema_flags_invalid_schema() -> None:
    assert S.validate_schema({"type": "not-a-type"})


async def test_generate_schema_returns_validated_schema() -> None:
    with patch.object(
        S, "call_json", AsyncMock(return_value={"schema_json": GOOD_SCHEMA})
    ):
        schema = await S.generate_schema(_ir(), llm=object(), retries=2)
    assert schema == GOOD_SCHEMA


def _shop_ir() -> SpecModel:
    return SpecModel(
        api_name="Shop",
        slug="shop",
        entities=[
            Entity(
                name="Order",
                collection="orders",
                primary_key="order_id",
                fields=[Field(name="order_id", type="string", required=True)],
            )
        ],
        operations=[],
        store_metadata=StoreMetadata(
            collections=["orders"], pk_map={"orders": "order_id"}
        ),
    )


def _order_schema() -> dict:
    return {
        "type": "object",
        "properties": {"orders": {"type": "array", "items": {"$ref": "#/$defs/Order"}}},
        "$defs": {
            "Order": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            }
        },
    }


def test_enforce_contract_sets_x_primary_key_on_the_def() -> None:
    schema = S.enforce_contract(_order_schema(), _shop_ir())
    # loader.py reads the pk off the $def, not off the collection property.
    assert schema["$defs"]["Order"]["x-primary-key"] == "order_id"


def test_enforce_contract_overwrites_a_wrong_x_primary_key() -> None:
    schema = _order_schema()
    schema["$defs"]["Order"]["x-primary-key"] = "id"
    assert (
        S.enforce_contract(schema, _shop_ir())["$defs"]["Order"]["x-primary-key"]
        == "order_id"
    )


def test_enforce_contract_adds_the_primary_key_to_required() -> None:
    schema = _order_schema()
    schema["$defs"]["Order"]["required"] = []
    assert S.enforce_contract(schema, _shop_ir())["$defs"]["Order"]["required"] == [
        "order_id"
    ]


def test_validate_schema_without_ir_only_checks_json_schema_validity() -> None:
    assert S.validate_schema(_order_schema()) == []
    assert S.validate_schema({"type": 5}) != []


def test_validate_schema_rejects_a_missing_collection() -> None:
    schema = _order_schema()
    del schema["properties"]["orders"]
    errors = S.validate_schema(schema, _shop_ir())
    assert any("orders" in e for e in errors)


def test_validate_schema_rejects_an_extra_collection() -> None:
    schema = _order_schema()
    schema["properties"]["ghosts"] = {"type": "array", "items": {"$ref": "#/$defs/G"}}
    schema["$defs"]["G"] = {"type": "object", "properties": {}}
    errors = S.validate_schema(schema, _shop_ir())
    assert any("ghosts" in e for e in errors)


def test_validate_schema_rejects_a_collection_without_a_defs_ref() -> None:
    schema = _order_schema()
    schema["properties"]["orders"] = {"type": "array", "items": {"type": "object"}}
    errors = S.validate_schema(schema, _shop_ir())
    # loader.py silently defaults to pk="id" in this case, which is exactly the
    # drift this validator exists to prevent.
    assert any("$ref" in e for e in errors)


def test_validate_schema_rejects_a_dangling_defs_ref() -> None:
    schema = _order_schema()
    del schema["$defs"]["Order"]
    errors = S.validate_schema(schema, _shop_ir())
    assert any("Order" in e for e in errors)


def test_validate_schema_rejects_a_primary_key_missing_from_the_def() -> None:
    schema = _order_schema()
    del schema["$defs"]["Order"]["properties"]["order_id"]
    errors = S.validate_schema(schema, _shop_ir())
    assert any("order_id" in e for e in errors)


async def test_generate_schema_enforces_the_contract_on_the_returned_schema() -> None:
    payload = {"schema_json": _order_schema()}
    with patch.object(S, "call_json", AsyncMock(return_value=payload)):
        schema = await S.generate_schema(_shop_ir(), object(), retries=0)
    assert schema["$defs"]["Order"]["x-primary-key"] == "order_id"


# --- Fix round 1 --------------------------------------------------------
# enforce_contract now runs before validate_schema on every repair attempt
# (force-before-reject), so it no longer gets the shielding
# Draft202012Validator.check_schema used to provide when validate_schema ran
# first. _resolve_def must therefore isinstance-guard every level itself.


def test_enforce_contract_does_not_crash_on_malformed_properties() -> None:
    schema = {"type": "object", "properties": "orders"}
    result = S.enforce_contract(schema, _shop_ir())
    # Nothing resolvable, but the top-level stamp (Fix 2) still lands.
    assert result["required"] == ["orders"]
    assert result["additionalProperties"] is False


def test_enforce_contract_does_not_crash_on_malformed_defs() -> None:
    schema = _order_schema()
    schema["$defs"] = []
    result = S.enforce_contract(schema, _shop_ir())
    assert result["required"] == ["orders"]


def test_validate_schema_rejects_malformed_properties_without_crashing() -> None:
    errors = S.validate_schema({"type": "object", "properties": "orders"}, _shop_ir())
    assert errors


def test_validate_schema_rejects_malformed_defs_without_crashing() -> None:
    schema = _order_schema()
    schema["$defs"] = []
    errors = S.validate_schema(schema, _shop_ir())
    assert errors


def test_validate_schema_rejects_a_bool_schema_without_crashing() -> None:
    # True/False are valid Draft 2020-12 schemas, so check_schema alone
    # doesn't reject them; validate_schema must guard schema itself too.
    errors = S.validate_schema(True, _shop_ir())  # type: ignore[arg-type]
    assert errors


async def test_generate_schema_raises_a_clean_repair_error_for_malformed_properties() -> (
    None
):
    # Reproduces the reviewer's measured live-path crash: call_json returns a
    # schema with 'properties' as a string. Before this fix, enforce_contract
    # raised AttributeError, which escaped validate() -> with_repair ->
    # guard_timeout uncaught. It must now surface as a normal repair failure.
    payload = {"schema_json": {"type": "object", "properties": "orders"}}
    with patch.object(S, "call_json", AsyncMock(return_value=payload)):
        with pytest.raises(GenerationStageError):
            await S.generate_schema(_shop_ir(), object(), retries=0)


def test_enforce_contract_stamps_top_level_required_and_additional_properties() -> None:
    # Fix 2: db.json's collection set is only checked because these land on
    # the schema that reaches the seed stage's own validate_schema_and_db.
    schema = S.enforce_contract(_order_schema(), _shop_ir())
    assert schema["required"] == ["orders"]
    assert schema["additionalProperties"] is False


def test_enforce_contract_overwrites_a_wrong_top_level_contract() -> None:
    schema = _order_schema()
    schema["required"] = ["ghosts"]
    schema["additionalProperties"] = True
    schema = S.enforce_contract(schema, _shop_ir())
    assert schema["required"] == ["orders"]
    assert schema["additionalProperties"] is False


def _shared_def_ir(*, same_pk: bool) -> SpecModel:
    archived_pk = "order_id" if same_pk else "archive_id"
    return SpecModel(
        api_name="Shop",
        slug="shop",
        entities=[
            Entity(
                name="Order",
                collection="orders",
                primary_key="order_id",
                fields=[Field(name="order_id", type="string", required=True)],
            ),
            Entity(
                name="ArchivedOrder",
                collection="archived_orders",
                primary_key=archived_pk,
                fields=[Field(name=archived_pk, type="string", required=True)],
            ),
        ],
        operations=[],
        store_metadata=StoreMetadata(
            collections=["orders", "archived_orders"],
            pk_map={"orders": "order_id", "archived_orders": archived_pk},
        ),
    )


def _shared_def_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "orders": {"type": "array", "items": {"$ref": "#/$defs/Order"}},
            "archived_orders": {"type": "array", "items": {"$ref": "#/$defs/Order"}},
        },
        "$defs": {
            "Order": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "archive_id": {"type": "string"},
                },
                "required": ["order_id"],
            }
        },
    }


def test_validate_schema_rejects_a_shared_def_with_differing_primary_keys() -> None:
    errors = S.validate_schema(_shared_def_schema(), _shared_def_ir(same_pk=False))
    assert any("shared" in e for e in errors)


def test_validate_schema_accepts_a_shared_def_with_the_same_primary_key() -> None:
    errors = S.validate_schema(_shared_def_schema(), _shared_def_ir(same_pk=True))
    assert errors == []


def test_enforce_contract_stamps_the_composed_key_recipe() -> None:
    ir = _shop_ir()
    ir.entities[0] = ir.entities[0].model_copy(
        update={"key_fields": ["shop_id", "order_no"]}
    )
    order = S.enforce_contract(_order_schema(), ir)["$defs"]["Order"]
    assert order["x-key-fields"] == ["shop_id", "order_no"]
    assert order["x-key-separator"] == "|"


def test_enforce_contract_adds_no_key_recipe_for_a_plain_key() -> None:
    order = S.enforce_contract(_order_schema(), _shop_ir())["$defs"]["Order"]
    assert "x-key-fields" not in order


def _status_spec(returns: dict) -> OpenAPISpec:
    s = {"type": "string"}
    return OpenAPISpec(
        {
            "openapi": "3.0.0",
            "info": {"title": "Hotels", "version": "1"},
            "paths": {
                "/get_room_status": {
                    "post": {
                        "operationId": "get_room_status",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"hotel_id": s, "night": s},
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "ok",
                                "content": {
                                    "application/json": {
                                        "schema": {"properties": {"returns": returns}}
                                    }
                                },
                            }
                        },
                    }
                }
            },
        }
    )


def _room_ir() -> SpecModel:
    return SpecModel(
        api_name="Hotels",
        slug="hotels",
        entities=[
            Entity(
                name="Room",
                collection="rooms",
                primary_key="room_key",
                fields=[
                    Field(name="room_key", type="string", required=True),
                    Field(name="hotel_id", type="string"),
                    Field(name="night", type="string"),
                    Field(name="status", type="string", enum=["open"]),
                ],
            )
        ],
        operations=[],
        store_metadata=StoreMetadata(
            collections=["rooms"], pk_map={"rooms": "room_key"}
        ),
    )


def _room_schema() -> dict:
    return {
        "type": "object",
        "properties": {"rooms": {"type": "array", "items": {"$ref": "#/$defs/Room"}}},
        "$defs": {
            "Room": {
                "type": "object",
                "properties": {
                    "room_key": {"type": "string"},
                    "status": {"type": "string", "enum": ["open"]},
                },
            }
        },
    }


def test_a_lookup_returning_free_text_opens_the_field() -> None:
    spec = _status_spec({"type": "string", "description": "The status of the room."})
    open_fields = S.open_value_fields(spec, _room_ir())
    assert open_fields == {"Room": {"status"}}
    ir = S.relax_open_fields_ir(_room_ir(), open_fields)
    assert next(f for f in ir.entities[0].fields if f.name == "status").enum is None
    schema = S.relax_open_fields_schema(_room_schema(), ir, open_fields)
    assert "enum" not in schema["$defs"]["Room"]["properties"]["status"]


@pytest.mark.parametrize(
    "returns",
    [
        {
            "type": "string",
            "enum": ["open", "closed"],
            "description": "The status of the room.",
        },
        {"type": "string", "description": "A confirmation message."},
        {"type": "object", "properties": {"status": {"type": "string"}}},
    ],
)
def test_constrained_or_unrelated_returns_open_nothing(returns: dict) -> None:
    assert S.open_value_fields(_status_spec(returns), _room_ir()) == {}
    schema = S.relax_open_fields_schema(_room_schema(), _room_ir(), {})
    assert schema["$defs"]["Room"]["properties"]["status"]["enum"] == ["open"]


# --- nullability carried over from the spec --------------------------------


def _tool_spec(returns: dict, *, components: dict | None = None, request=None) -> dict:
    """A tool-style spec: one POST whose 200 response wraps ``returns``."""
    op: dict = {
        "operationId": "get_order",
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"returns": returns},
                        }
                    }
                }
            }
        },
    }
    if request is not None:
        op["requestBody"] = {"content": {"application/json": {"schema": request}}}
    spec: dict = {
        "openapi": "3.1.0",
        "info": {"title": "t", "version": "1"},
        "paths": {"/get_order": {"post": op}},
    }
    if components:
        spec["components"] = {"schemas": components}
    return spec


def _order_schema(**extra_props) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"orders": {"type": "array", "items": {"$ref": "#/$defs/Order"}}},
        "$defs": {
            "Order": {
                "type": "object",
                "x-primary-key": "order_id",
                "properties": {
                    "order_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["cancelled"]},
                    "price": {"type": "integer"},
                    **extra_props,
                },
            }
        },
    }


_NULLABLE_ORDER = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "status": {
            "anyOf": [{"const": "cancelled", "type": "string"}, {"type": "null"}],
            "default": None,
        },
        "price": {"type": ["integer", "null"]},
    },
}


def _accepts(schema: dict, row: dict) -> bool:
    validator = Draft202012Validator(
        {"$defs": schema["$defs"], "$ref": "#/$defs/Order"}
    )
    return validator.is_valid(row)


def test_a_nullable_spec_property_accepts_null_in_the_store() -> None:
    schema = _order_schema()
    assert not _accepts(schema, {"order_id": "o1", "status": None})
    S.widen_nullable_fields(schema, _tool_spec(_NULLABLE_ORDER))
    props = schema["$defs"]["Order"]["properties"]
    assert props["status"] == {"type": ["string", "null"], "enum": ["cancelled", None]}
    assert props["price"] == {"type": ["integer", "null"]}
    assert _accepts(schema, {"order_id": "o1", "status": None, "price": None})
    # Still rejects what the spec rejects.
    assert not _accepts(schema, {"order_id": "o1", "status": "pending"})


def test_widen_nullable_fields_is_idempotent() -> None:
    once = S.widen_nullable_fields(_order_schema(), _tool_spec(_NULLABLE_ORDER))
    twice = S.widen_nullable_fields(
        json.loads(json.dumps(once)), _tool_spec(_NULLABLE_ORDER)
    )
    assert twice == once
    assert S.nullable_widenings(once, _tool_spec(_NULLABLE_ORDER)) == []


def test_nullable_widenings_lists_what_the_pass_changes() -> None:
    assert S.nullable_widenings(_order_schema(), _tool_spec(_NULLABLE_ORDER)) == [
        ("Order", "price"),
        ("Order", "status"),
    ]


def test_nullable_true_and_component_refs_are_followed() -> None:
    order = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "status": {"type": "string", "enum": ["cancelled"], "nullable": True},
            "price": {"$ref": "#/components/schemas/Price"},
        },
    }
    spec = _tool_spec(
        {"$ref": "#/components/schemas/Order"},
        components={"Order": order, "Price": {"type": "integer", "nullable": True}},
    )
    schema = S.widen_nullable_fields(_order_schema(), spec)
    props = schema["$defs"]["Order"]["properties"]
    assert props["status"]["type"] == ["string", "null"]
    assert props["price"]["type"] == ["integer", "null"]


def test_the_primary_key_is_never_widened() -> None:
    returns = {
        "type": "object",
        "properties": {
            "order_id": {"type": ["string", "null"]},
            "price": {"type": "integer"},
        },
    }
    schema = S.widen_nullable_fields(_order_schema(), _tool_spec(returns))
    assert schema["$defs"]["Order"]["properties"]["order_id"] == {"type": "string"}


def test_an_object_without_the_key_or_a_request_body_widens_nothing() -> None:
    # A same-named nullable field on an unrelated object (no `order_id`) or on a
    # request (an optional argument) says nothing about the stored row.
    unrelated = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "status": _NULLABLE_ORDER["properties"]["status"],
        },
    }
    spec = _tool_spec(unrelated, request=_NULLABLE_ORDER)
    schema = S.widen_nullable_fields(_order_schema(), spec)
    assert schema == _order_schema()


def test_nested_objects_and_ref_properties_are_widened() -> None:
    schema = _order_schema(
        address={"$ref": "#/$defs/Address"},
        note={"$ref": "#/$defs/Note"},
    )
    schema["$defs"]["Address"] = {
        "type": "object",
        "properties": {"line2": {"type": "string"}},
    }
    schema["$defs"]["Note"] = {"type": "object", "properties": {}}
    returns = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "address": {
                "type": "object",
                "properties": {
                    "line2": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                },
            },
            "note": {"anyOf": [{"type": "object"}, {"type": "null"}]},
        },
    }
    S.widen_nullable_fields(schema, _tool_spec(returns))
    assert schema["$defs"]["Address"]["properties"]["line2"]["type"] == [
        "string",
        "null",
    ]
    note = schema["$defs"]["Order"]["properties"]["note"]
    assert note == {"anyOf": [{"$ref": "#/$defs/Note"}, {"type": "null"}]}
    assert _accepts(
        schema, {"order_id": "o1", "note": None, "address": {"line2": None}}
    )


def test_a_composed_key_entity_matches_on_its_key_fields_and_keeps_them() -> None:
    schema = {
        "type": "object",
        "properties": {
            "flights": {"type": "array", "items": {"$ref": "#/$defs/Flight"}}
        },
        "$defs": {
            "Flight": {
                "type": "object",
                "x-primary-key": "flight_key",
                "x-key-fields": ["date", "flight_number"],
                "properties": {
                    "flight_key": {"type": "string"},
                    "date": {"type": "string"},
                    "flight_number": {"type": "string"},
                    "gate": {"type": "string"},
                },
            }
        },
    }
    returns = {
        "type": "object",
        "properties": {
            "date": {"type": ["string", "null"]},
            "flight_number": {"type": "string"},
            "gate": {"type": ["string", "null"]},
        },
    }
    S.widen_nullable_fields(schema, _tool_spec(returns))
    props = schema["$defs"]["Flight"]["properties"]
    assert props["date"] == {"type": "string"}
    assert props["gate"]["type"] == ["string", "null"]
