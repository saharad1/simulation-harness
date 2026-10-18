"""Element shapes that carry a composed key are linked to the collection it names.

A composed identity (``Entity.key_fields``) comes from the LLM fallback, which
runs after ``derive_identity``. So the identity pass never sees it, and an
embedded element whose fields include every key field gets no
``target_collection``. Then ``_linked_entities`` never shows the target to the
operation prompt, and the generated contract demands the stored record's full
shape on a request item that carries only the key fields.

``link_composed_references`` runs after compose and adds the annotation. The
element stays ``embedded``: it legitimately carries per-parent fields.
"""

from __future__ import annotations

from simulation_harness.skills.generation.ir import (
    KEY_SEPARATOR,
    ElementField,
    ElementShape,
    Entity,
    Field,
    SpecModel,
    StoreMetadata,
)
from simulation_harness.skills.generation.stages.analyze.composite import (
    link_composed_references,
)
from simulation_harness.skills.generation.stages.operations import _linked_entities
from simulation_harness.skills.generation.stages.schema import enforce_contract


def _embedded(*names: str) -> ElementShape:
    return ElementShape(
        kind="embedded",
        fields=tuple(ElementField(name=n, type="string") for n in names),
    )


def _flight(name: str = "Flight", collection: str = "flights") -> Entity:
    return Entity(
        name=name,
        collection=collection,
        primary_key="flight_key",
        key_fields=["date", "flight_number"],
        fields=[
            Field(name="flight_key", type="string", required=True),
            Field(name="flight_number", type="string"),
            Field(name="date", type="string"),
            Field(name="origin", type="string"),
        ],
    )


def _reservation(shape: ElementShape | None) -> Entity:
    return Entity(
        name="Reservation",
        collection="reservations",
        primary_key="reservation_id",
        fields=[
            Field(name="reservation_id", type="string", required=True),
            Field(name="flights", type="array", element=shape),
        ],
    )


def _shape_of(entities: list[Entity], entity: str, field: str) -> ElementShape | None:
    e = next(e for e in entities if e.name == entity)
    return next(f for f in e.fields if f.name == field).element


def test_element_carrying_composed_key_links_to_target() -> None:
    out = link_composed_references(
        [_flight(), _reservation(_embedded("date", "flight_number", "price"))]
    )
    shape = _shape_of(out, "Reservation", "flights")
    assert shape is not None
    assert shape.kind == "embedded"
    assert shape.target_collection == "flights"
    assert shape.target_key == "flight_key"
    assert shape.target_key_fields == ("date", "flight_number")
    # Per-parent fields are kept: the element is annotated, not replaced.
    assert [f.name for f in shape.fields] == ["date", "flight_number", "price"]


def test_partial_key_does_not_link() -> None:
    original = _embedded("flight_number", "price")
    out = link_composed_references([_flight(), _reservation(original)])
    assert _shape_of(out, "Reservation", "flights") == original


def test_ambiguous_match_does_not_link() -> None:
    original = _embedded("date", "flight_number", "price")
    out = link_composed_references(
        [
            _flight(),
            _flight(name="Departure", collection="departures"),
            _reservation(original),
        ]
    )
    assert _shape_of(out, "Reservation", "flights") == original


def test_no_self_link() -> None:
    """An entity whose array holds elements covering its own key fields is not
    a reference to itself."""
    me = _flight()
    me = me.model_copy(
        update={
            "fields": [
                *me.fields,
                Field(
                    name="siblings",
                    type="array",
                    element=_embedded("date", "flight_number"),
                ),
            ]
        }
    )
    out = link_composed_references([me])
    shape = _shape_of(out, "Flight", "siblings")
    assert shape is not None and shape.target_collection is None


def test_existing_single_id_link_untouched() -> None:
    """Retail-like regression guard: no ``key_fields`` anywhere, and an existing
    reference is left exactly as it was."""
    items_ref = ElementShape(
        kind="reference", target_collection="items", target_key="item_id"
    )
    annotated = ElementShape(
        kind="embedded",
        fields=(
            ElementField(name="payment_method_id", type="string"),
            ElementField(name="amount", type="number"),
        ),
        target_collection="payment_methods",
        target_key="id",
        local_key="payment_method_id",
    )
    entities = [
        Entity(
            name="Order",
            collection="orders",
            primary_key="order_id",
            fields=[
                Field(name="order_id", type="string", required=True),
                Field(name="items", type="array", element=items_ref),
                Field(name="payment_history", type="array", element=annotated),
                Field(name="tags", type="array", element=_embedded("label")),
            ],
        ),
        Entity(
            name="Item",
            collection="items",
            primary_key="item_id",
            fields=[Field(name="item_id", type="string", required=True)],
        ),
    ]
    out = link_composed_references(entities)
    assert out == entities


def test_already_linked_shape_is_not_relinked() -> None:
    original = _embedded("date", "flight_number").model_copy(
        update={"target_collection": "elsewhere", "target_key": "x"}
    )
    out = link_composed_references([_flight(), _reservation(original)])
    assert _shape_of(out, "Reservation", "flights") == original


def test_x_element_ref_carries_key_fields() -> None:
    entities = link_composed_references(
        [_flight(), _reservation(_embedded("date", "flight_number", "price"))]
    )
    ir = SpecModel(
        api_name="air",
        slug="air",
        entities=entities,
        operations=[],
        store_metadata=StoreMetadata(
            collections=["flights", "reservations"],
            pk_map={"flights": "flight_key", "reservations": "reservation_id"},
        ),
    )
    schema = {
        "type": "object",
        "properties": {
            "reservations": {
                "type": "array",
                "items": {"$ref": "#/$defs/Reservation"},
            },
            "flights": {"type": "array", "items": {"$ref": "#/$defs/Flight"}},
        },
        "$defs": {
            "Reservation": {
                "type": "object",
                "properties": {
                    "reservation_id": {"type": "string"},
                    "flights": {"type": "array", "items": {}},
                },
            },
            "Flight": {
                "type": "object",
                "properties": {"flight_key": {"type": "string"}},
            },
        },
    }
    out = enforce_contract(schema, ir)
    prop = out["$defs"]["Reservation"]["properties"]["flights"]
    assert prop["x-element-ref"] == {
        "collection": "flights",
        "key": "flight_key",
        "key_fields": ["date", "flight_number"],
        "separator": KEY_SEPARATOR,
    }
    assert prop["items"]["type"] == "object"


def test_linked_entities_includes_composite_target() -> None:
    entities = link_composed_references(
        [_flight(), _reservation(_embedded("date", "flight_number", "price"))]
    )
    ir = SpecModel(
        api_name="air",
        slug="air",
        entities=entities,
        operations=[],
        store_metadata=StoreMetadata(
            collections=["flights", "reservations"],
            pk_map={"flights": "flight_key", "reservations": "reservation_id"},
        ),
    )
    reservation = next(e for e in ir.entities if e.name == "Reservation")
    linked = _linked_entities(ir, reservation)
    assert [e["collection"] for e in linked] == ["flights"]
    assert linked[0]["key_fields"] == ["date", "flight_number"]
