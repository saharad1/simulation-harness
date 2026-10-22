"""A spec field declared nullable stays nullable in the generated ``schema.json``.

The spec says ``{"anyOf": [{"type": "string"}, {"type": "null"}]}`` (pydantic's
``Optional[str]``) or ``{"type": ["string", "null"]}``. Before this was carried, the
identity pass read neither form — no string ``type`` key — and fell back to plain
``"string"``, so the schema prompt was never told the field could be null and wrote
``{"type": "string"}``. The live store then rejected every write that cleared the
field (tau2-telecom ``resume_line`` setting ``suspension_start_date`` back to null).
Nullability is contract, so it is derived in code and stamped, never prompted.
"""

from __future__ import annotations

from simulation_harness.skills.generation.ir import (
    Entity,
    Field,
    SpecModel,
    StoreMetadata,
)
from simulation_harness.skills.generation.stages.analyze.enrich import (
    structural_entities,
)
from simulation_harness.skills.generation.stages.analyze.identity import (
    IdentityModel,
    derive_identity,
)
from simulation_harness.skills.generation.stages.analyze.merge import (
    compose_data_model,
)
from simulation_harness.skills.generation.stages.schema import enforce_contract
from simulation_harness.skills.generation.stages.seed import validate_schema_and_db

_LINE = {
    "type": "object",
    "properties": {
        "line_id": {"type": "string"},
        "plan_id": {"type": "string"},
        "suspension_start_date": {
            "anyOf": [{"type": "string", "format": "date"}, {"type": "null"}],
            "default": None,
        },
        "data_used_gb": {"type": ["number", "null"]},
        "device_id": {"oneOf": [{"type": "null"}, {"type": "string"}]},
    },
    "required": ["line_id", "plan_id"],
}


def _identity() -> IdentityModel:
    return derive_identity({"Line": _LINE}, synthetic=False)


def test_identity_unwraps_the_null_union_to_the_real_type() -> None:
    (line,) = _identity().entities
    fields = dict(line.fields)
    assert fields["suspension_start_date"] == "string"
    assert fields["data_used_gb"] == "number"
    assert fields["device_id"] == "string"
    assert fields["plan_id"] == "string"


def test_identity_records_which_fields_are_nullable() -> None:
    (line,) = _identity().entities
    assert line.nullable_fields == (
        "data_used_gb",
        "device_id",
        "suspension_start_date",
    )


def test_structural_floor_carries_nullability() -> None:
    (line,) = structural_entities(_identity())
    nullable = {f.name: f.nullable for f in line.fields}
    assert nullable["suspension_start_date"] is True
    assert nullable["plan_id"] is False


def test_compose_pins_nullability_over_the_enrich_llm() -> None:
    """The enrich LLM returns fields without the flag (or with it wrong); the
    derived decision wins, and the prose it wrote survives."""
    enriched = Entity(
        name="Line",
        collection="lines",
        primary_key="line_id",
        fields=[
            Field(name="line_id", type="string", required=True),
            Field(name="plan_id", type="string", nullable=True),
            Field(
                name="suspension_start_date",
                type="string",
                description="Start of the suspension.",
            ),
        ],
    )
    model, _ = compose_data_model("telecom", _identity(), [enriched], None)
    fields = {f.name: f for f in model.entities[0].fields}
    assert fields["suspension_start_date"].nullable is True
    assert fields["suspension_start_date"].description == "Start of the suspension."
    assert fields["plan_id"].nullable is False


def _schema(prop: dict) -> dict:
    return {
        "type": "object",
        "properties": {"lines": {"type": "array", "items": {"$ref": "#/$defs/Line"}}},
        "$defs": {
            "Line": {
                "type": "object",
                "properties": {
                    "line_id": {"type": "string"},
                    "suspension_start_date": prop,
                },
            }
        },
    }


def _ir(nullable: bool = True) -> SpecModel:
    return SpecModel(
        api_name="telecom",
        slug="telecom",
        entities=[
            Entity(
                name="Line",
                collection="lines",
                primary_key="line_id",
                fields=[
                    Field(name="line_id", type="string", required=True),
                    Field(
                        name="suspension_start_date", type="string", nullable=nullable
                    ),
                ],
            )
        ],
        operations=[],
        store_metadata=StoreMetadata(
            collections=["lines"], pk_map={"lines": "line_id"}
        ),
    )


def _prop(schema: dict) -> dict:
    prop = schema["$defs"]["Line"]["properties"]["suspension_start_date"]
    assert isinstance(prop, dict)
    return prop


def test_schema_stamps_null_onto_a_nullable_field() -> None:
    out = enforce_contract(_schema({"type": "string", "description": "d"}), _ir())
    assert _prop(out) == {"type": ["string", "null"], "description": "d"}


def test_schema_stamp_is_idempotent() -> None:
    """``enforce_contract`` runs on every repair attempt over the same payload."""
    schema = _schema({"type": "string"})
    once = enforce_contract(schema, _ir())
    twice = enforce_contract(once, _ir())
    assert _prop(twice) == {"type": ["string", "null"]}


def test_schema_stamp_admits_null_into_an_enum() -> None:
    out = enforce_contract(_schema({"type": "string", "enum": ["a", "b"]}), _ir())
    assert _prop(out) == {"type": ["string", "null"], "enum": ["a", "b", None]}


def test_schema_stamp_leaves_a_field_that_already_admits_null() -> None:
    already = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    out = enforce_contract(_schema(dict(already)), _ir())
    assert _prop(out) == already


def test_schema_stamp_leaves_non_nullable_fields_alone() -> None:
    out = enforce_contract(_schema({"type": "string"}), _ir(nullable=False))
    assert _prop(out) == {"type": "string"}


def test_stamped_schema_accepts_a_null_and_still_rejects_a_wrong_type() -> None:
    out = enforce_contract(_schema({"type": "string"}), _ir())
    ok = {"lines": [{"line_id": "L1", "suspension_start_date": None}]}
    assert validate_schema_and_db(out, ok) == []
    bad = {"lines": [{"line_id": "L1", "suspension_start_date": 7}]}
    assert validate_schema_and_db(out, bad) != []
