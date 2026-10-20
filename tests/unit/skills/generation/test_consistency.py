from __future__ import annotations

from pathlib import Path

from simulation_harness.skills.generation.ir import (
    Entity,
    Field,
    Operation,
    OperationKind,
    SpecModel,
    StoreMetadata,
)
from simulation_harness.skills.generation.stages import assemble as A
from simulation_harness.skills.generation.stages.consistency import (
    check_consistency,
)

FIXTURE_DIR = Path(__file__).parent.parent.parent.parent / "fixtures" / "consistency"


def _ir(*, book_kind: OperationKind = OperationKind.create) -> SpecModel:
    return SpecModel(
        api_name="tau2-airline",
        slug="tau2-airline",
        entities=[
            Entity(
                name="User",
                collection="users",
                primary_key="user_id",
                fields=[Field(name="user_id", type="string", required=True)],
            ),
            Entity(
                name="Reservation",
                collection="reservations",
                primary_key="reservation_id",
                fields=[Field(name="reservation_id", type="string", required=True)],
            ),
        ],
        operations=[
            Operation(
                operation_id="book_reservation",
                method="POST",
                path="/book_reservation",
                entity="Reservation",
                kind=book_kind,
            ),
            Operation(
                operation_id="get_user_details",
                method="POST",
                path="/get_user_details",
                entity="User",
                kind=OperationKind.read,
            ),
        ],
        store_metadata=StoreMetadata(
            collections=["users", "reservations"],
            pk_map={"users": "user_id", "reservations": "reservation_id"},
        ),
    )


def _skill_md(*, store_access_section: str) -> str:
    return (
        "---\nname: tau2-airline\n---\n\n"
        "### Derivation Rules\n"
        "- `User.reservations` equals the set of `Reservation.reservation_id` "
        "values for reservations whose `user_id` matches that `User.user_id`.\n\n"
        "## API Operation Simulation\n\n"
        "### /book_reservation POST\n\n"
        "- **Store access**\n"
        f"{store_access_section}\n\n"
        "### /get_user_details POST\n\n"
        "- **Store access**\n"
        "  - Read `users` by primary key `user_id`.\n"
        "  - Do not write any collection.\n"
    )


def test_derived_field_not_maintained_is_flagged() -> None:
    ir = _ir()
    md = _skill_md(store_access_section="  - **Write `reservations`**\n")
    errors = check_consistency(ir, md)
    assert len(errors) == 1
    assert "book_reservation writes reservations but not users" in errors[0]
    assert "User.reservations" in errors[0]


def test_derived_field_maintained_passes() -> None:
    ir = _ir()
    md = _skill_md(
        store_access_section=("  - **Write `reservations`**\n  - **Write `users`**\n")
    )
    assert check_consistency(ir, md) == []


def test_read_only_op_not_flagged() -> None:
    # get_user_details is a read op; a derivation rule naming it should never
    # trigger pattern 1 even if it only reads, not writes, the collection it
    # depends on.
    ir = _ir(book_kind=OperationKind.read)
    md = _skill_md(store_access_section="  - **Write `reservations`**\n")
    assert check_consistency(ir, md) == []


def test_self_contradiction_flagged() -> None:
    ir = _ir()
    md = _skill_md(
        store_access_section=(
            "  - **Write `reservations`**\n"
            "  - **Write `payments`**\n"
            "  - Do not write any fields in `payments`.\n"
        )
    )
    errors = check_consistency(ir, md)
    assert any(
        "book_reservation both writes and forbids writing payments" in e for e in errors
    )


def test_rule_with_unknown_entity_skipped() -> None:
    ir = _ir()
    md = (
        "---\nname: tau2-airline\n---\n\n"
        "### Derivation Rules\n"
        "- `Widget.count` equals the sum of `Gadget.count` across all gadgets.\n\n"
        "## API Operation Simulation\n\n"
        "### /book_reservation POST\n\n"
        "- **Store access**\n"
        "  - **Write `reservations`**\n\n"
        "### /get_user_details POST\n\n"
        "- **Store access**\n"
        "  - Read `users` by primary key `user_id`.\n"
    )
    assert check_consistency(ir, md) == []


def test_validate_bundle_includes_consistency_errors() -> None:
    ir = _ir()
    md = _skill_md(store_access_section="  - **Write `reservations`**\n")
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {},
        "$defs": {},
    }
    errors = A.validate_bundle(ir, md, schema, {})
    assert any(
        "book_reservation writes reservations but not users" in e for e in errors
    )


def test_real_airline_skill_flags_book_reservation() -> None:
    md = (FIXTURE_DIR / "airline_excerpt.md").read_text()
    ir = SpecModel(
        api_name="tau2-airline",
        slug="tau2-airline",
        entities=[
            Entity(
                name="User",
                collection="users",
                primary_key="user_id",
                fields=[Field(name="user_id", type="string", required=True)],
            ),
            Entity(
                name="Reservation",
                collection="reservations",
                primary_key="reservation_id",
                fields=[Field(name="reservation_id", type="string", required=True)],
            ),
            Entity(
                name="Flight",
                collection="flights",
                primary_key="flight_number",
                fields=[Field(name="flight_number", type="string", required=True)],
            ),
            Entity(
                name="Payment",
                collection="payments",
                primary_key="payment_id",
                fields=[Field(name="payment_id", type="string", required=True)],
            ),
        ],
        operations=[
            Operation(
                operation_id="book_reservation",
                method="POST",
                path="/book_reservation",
                entity="Reservation",
                kind=OperationKind.create,
            ),
            Operation(
                operation_id="get_user_details",
                method="POST",
                path="/get_user_details",
                entity="User",
                kind=OperationKind.read,
            ),
        ],
        store_metadata=StoreMetadata(
            collections=["users", "reservations", "flights", "payments"],
            pk_map={
                "users": "user_id",
                "reservations": "reservation_id",
                "flights": "flight_number",
                "payments": "payment_id",
            },
        ),
    )
    errors = check_consistency(ir, md)
    assert any(
        "book_reservation writes reservations but not users" in e for e in errors
    )


def test_qualified_prohibition_is_not_a_contradiction() -> None:
    """ "Do not write any field in `users` other than `address`" limits a write; it does
    not forbid one. Observed as a rejected tau2-retail rich draw (modify_user_address,
    2026-10-20)."""
    ir = _ir()
    for qualifier in ("other than `reservation_id`", "except `reservation_id`"):
        md = _skill_md(
            store_access_section=(
                "  - **Write `reservations`**\n"
                f"  - Do not write any fields in `reservations` {qualifier}.\n"
            )
        )
        assert not any(
            "both writes and forbids" in e for e in check_consistency(ir, md)
        )
