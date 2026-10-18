"""A shared record is referenced by id, never stored as a second copy.

tau2-retail and tau2-airline both promote `User.payment_methods` into a top-level
`payment_methods` collection. Before this, the map's values were *also* stamped
with the full record (`balance`, `brand`, ...), so the store held two copies of
each instrument. The simulator updated one balance and the other went stale.
`Product.variants` duplicated the `items` collection the same way.

These tests run the real identity pass (`derive_identity` -> structural entities)
into `enforce_contract`, so they lock the decision where it is made, not only the
stamp.
"""

from __future__ import annotations

from simulation_harness.skills.generation.ir import SpecModel, StoreMetadata
from simulation_harness.skills.generation.stages.analyze.enrich import (
    structural_entities,
)
from simulation_harness.skills.generation.stages.analyze.identity import (
    derive_identity,
)
from simulation_harness.skills.generation.stages.schema import (
    _element_items,
    enforce_contract,
)
from simulation_harness.skills.generation.stages.seed import validate_schema_and_db


def _ir(schemas: dict[str, dict]) -> SpecModel:
    ident = derive_identity(schemas, synthetic=True)
    return SpecModel(
        api_name="shop",
        slug="shop",
        entities=structural_entities(ident),
        operations=[],
        store_metadata=StoreMetadata(
            collections=ident.collections, pk_map=ident.pk_map
        ),
    )


def _schema(ir: SpecModel) -> dict:
    """A schema as the LLM would plausibly write it: every container left open."""
    defs: dict = {}
    for e in ir.entities:
        props: dict = {}
        for f in e.fields:
            props[f.name] = {"type": f.type} if f.type != "unknown" else {}
            if f.type == "array":
                props[f.name]["items"] = {}
        defs[e.name] = {
            "type": "object",
            "additionalProperties": False,
            "properties": props,
        }
    return {
        "type": "object",
        "properties": {
            e.collection: {"type": "array", "items": {"$ref": f"#/$defs/{e.name}"}}
            for e in ir.entities
        },
        "$defs": defs,
    }


def _field(ir: SpecModel, entity: str, name: str):
    e = next(e for e in ir.entities if e.name == entity)
    return next(f for f in e.fields if f.name == name)


_INSTRUMENT: dict = {
    "properties": {
        "id": {"type": "string"},
        "source": {"type": "string"},
        "balance": {"type": "number"},
        "brand": {"type": "string"},
    }
}

_USER_WITH_INSTRUMENT_MAP: dict[str, dict] = {
    "User": {
        "properties": {
            "user_id": {"type": "string"},
            "payment_methods": {
                "type": "object",
                "additionalProperties": _INSTRUMENT,
            },
        }
    }
}


def test_projection_map_element_is_stamped_as_id() -> None:
    ir = _ir(_USER_WITH_INSTRUMENT_MAP)
    assert "payment_methods" in ir.store_metadata.collections
    out = enforce_contract(_schema(ir), ir)
    prop = out["$defs"]["User"]["properties"]["payment_methods"]
    assert prop["additionalProperties"] == {"type": "string"}
    assert prop["x-element-ref"] == {"collection": "payment_methods", "key": "id"}
    assert "balance" not in str(prop)
    # The record itself still lives, once, in its own collection.
    assert "balance" in out["$defs"]["PaymentMethod"]["properties"]


def test_projection_array_element_is_stamped_as_id() -> None:
    schemas: dict[str, dict] = {
        "Item": {
            "properties": {
                "item_id": {"type": "string"},
                "price": {"type": "number"},
            }
        },
        "Order": {
            "properties": {
                "order_id": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "properties": {
                            "item_id": {"type": "string"},
                            "price": {"type": "number"},
                        }
                    },
                },
            }
        },
    }
    ir = _ir(schemas)
    out = enforce_contract(_schema(ir), ir)
    prop = out["$defs"]["Order"]["properties"]["items"]
    assert prop["items"] == {"type": "string"}
    assert prop["x-element-ref"] == {"collection": "items", "key": "item_id"}


def test_map_of_foreign_keyed_projections_is_stamped_as_id() -> None:
    """tau2-retail's Product.variants: keyed by `item_id`, each value carries only
    fields an `Item` already carries. It reaches the foreign-key path (the element
    is not promoted), but it is still a copy of the `items` row."""
    schemas: dict[str, dict] = {
        "Item": {
            "properties": {
                "item_id": {"type": "string"},
                "price": {"type": "number"},
                "available": {"type": "boolean"},
            }
        },
        "Product": {
            "properties": {
                "product_id": {"type": "string"},
                "variants": {
                    "type": "object",
                    "additionalProperties": {
                        "properties": {
                            "item_id": {"type": "string"},
                            "price": {"type": "number"},
                            "available": {"type": "boolean"},
                        }
                    },
                },
            }
        },
    }
    ir = _ir(schemas)
    out = enforce_contract(_schema(ir), ir)
    prop = out["$defs"]["Product"]["properties"]["variants"]
    assert prop["additionalProperties"] == {"type": "string"}
    assert prop["x-element-ref"] == {
        "collection": "items",
        "key": "item_id",
        "field": "item_id",
    }


def test_map_with_a_value_field_the_target_lacks_keeps_its_fields() -> None:
    """Not a copy: `quantity` exists only on the map value, so dropping the value
    would lose data. Stays embedded with its annotation."""
    schemas: dict[str, dict] = {
        "Item": {
            "properties": {"item_id": {"type": "string"}, "price": {"type": "number"}}
        },
        "Cart": {
            "properties": {
                "cart_id": {"type": "string"},
                "lines": {
                    "type": "object",
                    "additionalProperties": {
                        "properties": {
                            "item_id": {"type": "string"},
                            "quantity": {"type": "integer"},
                        }
                    },
                },
            }
        },
    }
    ir = _ir(schemas)
    shape = _field(ir, "Cart", "lines").element
    assert shape is not None
    assert shape.kind == "embedded"
    assert [f.name for f in shape.fields] == ["item_id", "quantity"]


def test_foreign_key_element_keeps_its_fields() -> None:
    """Order.payment_history: each line is its own fact that points at an
    instrument. Not a copy, so every field stays inline."""
    schemas: dict[str, dict] = {
        **_USER_WITH_INSTRUMENT_MAP,
        "Order": {
            "properties": {
                "order_id": {"type": "string"},
                "payment_history": {
                    "type": "array",
                    "items": {
                        "properties": {
                            "amount": {"type": "number"},
                            "payment_method_id": {"type": "string"},
                            "transaction_type": {"type": "string"},
                        }
                    },
                },
            }
        },
    }
    ir = _ir(schemas)
    out = enforce_contract(_schema(ir), ir)
    prop = out["$defs"]["Order"]["properties"]["payment_history"]
    assert set(prop["items"]["properties"]) == {
        "amount",
        "payment_method_id",
        "transaction_type",
    }
    assert prop["x-element-ref"] == {
        "collection": "payment_methods",
        "key": "id",
        "field": "payment_method_id",
    }


def test_seed_gate_accepts_a_map_of_ids_and_rejects_a_map_of_records() -> None:
    """A map reference is a dict of id strings keyed by id, not a list."""
    ir = _ir(_USER_WITH_INSTRUMENT_MAP)
    shape = _field(ir, "User", "payment_methods").element
    assert shape is not None
    assert (shape.kind, shape.container) == ("reference", "map")
    assert _element_items(shape) == {"type": "string"}

    stamped = enforce_contract(_schema(ir), ir)
    instrument = {"id": "gc_1", "source": "gift_card", "balance": 50.0, "brand": ""}
    ids = {
        "users": [{"user_id": "u1", "payment_methods": {"gc_1": "gc_1"}}],
        "payment_methods": [instrument],
    }
    records = {
        "users": [{"user_id": "u1", "payment_methods": {"gc_1": instrument}}],
        "payment_methods": [instrument],
    }
    assert validate_schema_and_db(stamped, ids) == []
    errors = validate_schema_and_db(stamped, records)
    assert errors and "is not of type 'string'" in errors[0]
