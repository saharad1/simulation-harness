"""B3: SKILL.md text that contradicts the store schema.

v1 retail rich: every SKILL.md told the simulator to set `price` / `options` on the entries of
`orders[order_id].items`, which the schema makes bare `item_id` strings, so every such write
failed validation. The quoted rule lines below are verbatim from the three v1 retail rich
skills (docrich_261020-111947, -114247, -111422).
"""

import json
import subprocess  # nosec B404 -- runs this repo's own script in a test
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from simulation_harness.openapi.parser import OpenAPISpec
from simulation_harness.skills.generation.ir import (
    ElementShape,
    Entity,
    Field,
    Operation,
    OperationKind,
    SpecModel,
    StoreMetadata,
)
from simulation_harness.skills.generation.lint import (
    id_only_element_writes,
    id_only_refs_from_ir,
    id_only_refs_from_schema,
    lint_skill_dir,
    schema_contradictions,
)
from simulation_harness.skills.generation.repair import GenerationStageError
from simulation_harness.skills.generation.stages import operations as O

_REPO = Path(__file__).resolve().parents[4]

_HEADER = "### /modify_pending_order_items POST"

_V1_111947 = """\
- **Write**
  - Write `orders` by key `order_id`.
    - Append one entry to `payment_history`:
      - `payment_method_id` = request `payment_method_id`.
    - Modify `items` in-place, processing pairs in request order:
      - For each pair (`item_id`, `new_item_id`), find the **first** current entry in `orders[order_id].items` whose `item_id` equals that pair's `item_id`.
      - Set that entry's `item_id` to `new_item_id`.
      - Set that entry's `price` to the `price` of the variant looked up for the **last pair in the request lists**, not the current pair's variant.
      - Set that entry's `options` to the `options` of that same last-looked-up variant.
    - Set `status` to `"pending (item modified)"`.
"""

_V1_114247 = """\
    - For each pair `(item_id, new_item_id)` in request order:
      - Find the **first** entry in `orders[order_id].items` whose current `item_id` equals `item_id`.
      - Set that entry's `item_id` = `new_item_id`.
      - Set that entry's `price` = the `price` of the variant looked up for the **last pair in the lists**, not this pair's variant.
      - Set that entry's `options` = the `options` of that same last-looked-up variant.
"""

_V1_111422 = """\
  - For each pair `(item_id, new_item_id)` in request order:
    - take the first entry in `orders[order_id].items` whose current `item_id` equals `item_id`
    - because modified entries change `item_id`, duplicate `item_ids` match successive entries
    - on that matched entry:
      - set `item_id = new_item_id`
      - set `price` to the `price` of the variant looked up for the **last pair in the request lists**, not this pair's own variant
      - set `options` to the `options` of that same **last-looked-up** variant
  - Set `orders[order_id].status = "pending (item modified)"`.
"""

#: What a skill consistent with the schema says: swap the id, nothing else on the entry.
_CLEAN = """\
- **Write**
  - Write `orders` by key `order_id`.
    - For each pair, find the first entry in `orders[order_id].items` equal to `item_id`.
      - Set that entry's `item_id` to `new_item_id` (the entry is the bare id).
      - Do **not** write `price` or `options` on the order; they come from `items`.
    - Set `status` to `"pending (item modified)"`.
  - Update `orders.payment_history`:
    - When an entry in `orders.payment_history` has `transaction_type = "payment"`, set that entry's `payment_method_id` to the request `payment_method_id`.
"""


def _ir() -> SpecModel:
    items_ref = ElementShape(
        kind="reference", target_collection="items", target_key="item_id"
    )
    return SpecModel(
        api_name="Retail",
        slug="retail",
        entities=[
            Entity(
                name="Order",
                collection="orders",
                primary_key="order_id",
                fields=[
                    Field(name="order_id", type="string", required=True),
                    Field(name="status", type="string"),
                    Field(name="items", type="array", element=items_ref),
                    Field(name="payment_history", type="array"),
                ],
            ),
            Entity(
                name="Item",
                collection="items",
                primary_key="item_id",
                fields=[
                    Field(name="item_id", type="string", required=True),
                    Field(name="price", type="number"),
                    Field(name="options", type="object"),
                ],
            ),
        ],
        operations=[
            Operation(
                operation_id="modify_pending_order_items",
                method="POST",
                path="/modify_pending_order_items",
                entity="Order",
                kind=OperationKind.update,
            )
        ],
        store_metadata=StoreMetadata(
            collections=["orders", "items"],
            pk_map={"orders": "order_id", "items": "item_id"},
        ),
    )


def _schema(*, status_nullable: bool = False) -> dict[str, Any]:
    status: dict[str, Any] = {
        "type": ["string", "null"] if status_nullable else "string",
        "enum": ["pending", "pending (item modified)", "cancelled"],
    }
    return {
        "type": "object",
        "properties": {
            "orders": {"type": "array", "items": {"$ref": "#/$defs/Order"}},
            "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
        },
        "$defs": {
            "Order": {
                "type": "object",
                "additionalProperties": False,
                "x-primary-key": "order_id",
                "properties": {
                    "order_id": {"type": "string"},
                    "status": status,
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "x-element-ref": {"collection": "items", "key": "item_id"},
                    },
                    "payment_history": {"type": "array", "items": {"type": "object"}},
                },
            },
            "Item": {
                "type": "object",
                "additionalProperties": False,
                "x-primary-key": "item_id",
                "properties": {
                    "item_id": {"type": "string"},
                    "price": {"type": "number"},
                    "options": {"type": "object"},
                },
            },
        },
    }


# --- B3: id-only element writes -------------------------------------------------------------


@pytest.mark.parametrize("text", [_V1_111947, _V1_114247, _V1_111422])
def test_v1_rich_rules_setting_fields_on_id_only_entries_are_rejected(
    text: str,
) -> None:
    errors = id_only_element_writes(text, id_only_refs_from_ir(_ir()))
    assert len(errors) == 2
    assert all("`Order.items` holds bare `item_id` ids" in e for e in errors)
    assert all("replace the id in place, do not set other fields" in e for e in errors)
    assert any("sets `price`" in e for e in errors)
    assert any("sets `options`" in e for e in errors)


def test_a_rule_that_only_swaps_the_id_passes() -> None:
    assert id_only_element_writes(_CLEAN, id_only_refs_from_ir(_ir())) == []


def test_explicit_path_into_an_id_only_element_is_rejected() -> None:
    text = "- Set `orders[order_id].items[i].price` to the variant's `price`."
    (error,) = id_only_element_writes(text, id_only_refs_from_ir(_ir()))
    assert "sets `price`" in error


@pytest.mark.parametrize(
    "text",
    [
        "- Write `orders[order_id].items[i].options` from the new variant.",
    ],
)
def test_explicit_path_into_an_id_only_element_as_write_target_is_rejected(
    text: str,
) -> None:
    assert len(id_only_element_writes(text, id_only_refs_from_ir(_ir()))) == 1


@pytest.mark.parametrize(
    "text",
    [
        # v2 retail rich d2 try 1 (plan 28 L1): the element's price is read, not written.
        "- Append to `orders[order_id].payment_history` an entry; set `amount` to the sum "
        "of `orders[order_id].items[i].price` over the returned items.",
        "- Set `amount` = sum of `orders[order_id].items[i].price`.",
        "- Store the refund, computed from `Order.items[i].price`, in `amount`.",
    ],
)
def test_reading_an_element_path_on_a_write_line_passes(text: str) -> None:
    assert id_only_element_writes(text, id_only_refs_from_ir(_ir())) == []


def test_refs_from_schema_match_refs_from_ir() -> None:
    ir_refs = [(r.label, r.key, r.target) for r in id_only_refs_from_ir(_ir())]
    schema_refs = [
        (r.label, r.key, r.target) for r in id_only_refs_from_schema(_schema())
    ]
    assert ir_refs == schema_refs == [("Order.items", "item_id", "items")]


async def test_operations_stage_feeds_the_b3_lint_to_its_repair_round() -> None:
    ir = _ir()
    spec = OpenAPISpec(
        {
            "openapi": "3.0.0",
            "info": {"title": "Retail", "version": "1"},
            "paths": {
                "/modify_pending_order_items": {
                    "post": {
                        "operationId": "modify_pending_order_items",
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            },
        }
    )
    attempts = iter([f"{_HEADER}\n{_V1_111947}", f"{_HEADER}\n{_CLEAN}"])
    call = AsyncMock(side_effect=lambda *a, **k: next(attempts))
    with patch.object(O, "call_text", call):
        section = await O.generate_section(
            spec, ir, ir.operations, llm=object(), retries=1
        )
    assert "Set that entry's `price`" not in section
    feedback_prompt = call.call_args_list[1].args[2]
    assert "`Order.items` holds bare `item_id` ids" in feedback_prompt

    with patch.object(
        O, "call_text", AsyncMock(return_value=f"{_HEADER}\n{_V1_111947}")
    ):
        with pytest.raises(GenerationStageError) as excinfo:
            await O.generate_section(spec, ir, ir.operations, llm=object(), retries=0)
    assert "holds bare `item_id` ids" in str(excinfo.value)


# --- schema consistency ---------------------------------------------------------------------


def test_null_into_a_non_nullable_field_is_rejected_unless_nullable() -> None:
    text = "- **Write**\n  - `status` = left at default `None`\n"
    (error,) = schema_contradictions(text, _schema())
    assert "assigns null to `status`" in error
    assert schema_contradictions(text, _schema(status_nullable=True)) == []


def test_enum_value_outside_the_schema_is_rejected() -> None:
    bad = '- set `status` = `"exchange requested"`\n'
    (error,) = schema_contradictions(bad, _schema())
    assert "outside its schema.json enum" in error
    assert schema_contradictions('- set `status` = `"cancelled"`\n', _schema()) == []


def test_write_of_an_undeclared_field_is_rejected() -> None:
    (error,) = schema_contradictions(
        "- Set `orders[order_id].discount` = `0`\n", _schema()
    )
    assert "`orders[order_id].discount`" in error and "does not" in error


def test_response_construction_is_not_a_store_write() -> None:
    text = (
        "  - For each returned item:\n"
        "    - `status` = `null`\n"
        "    - `discount` = `null`\n"
        "- `returns.status` = `null` because stored `status` is left at default `None`\n"
    )
    assert schema_contradictions(text, _schema()) == []


# --- on a finished skill dir, and the script ----------------------------------------------


def _skill_dir(tmp_path: Path, body: str) -> Path:
    d = tmp_path / "skill"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: retail\n---\n{_HEADER}\n{body}")
    (d / "schema.json").write_text(json.dumps(_schema()))
    return d


def test_lint_skill_dir_runs_both_lints(tmp_path: Path) -> None:
    errors = lint_skill_dir(
        _skill_dir(tmp_path, _V1_111947 + "    - `status` = `null`\n")
    )
    assert sum("holds bare" in e for e in errors) == 2
    assert sum("assigns null" in e for e in errors) == 1


def test_lint_skill_script_exit_codes(tmp_path: Path) -> None:
    script = _REPO / "scripts" / "lint_skill.py"
    bad = subprocess.run(  # nosec B603 -- fixed argv, this repo's script
        [sys.executable, str(script), str(_skill_dir(tmp_path / "a", _V1_111947))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 1
    assert bad.stdout.count("holds bare `item_id` ids") == 2
    good = subprocess.run(  # nosec B603
        [sys.executable, str(script), str(_skill_dir(tmp_path / "b", _CLEAN))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert good.returncode == 0, good.stdout + good.stderr
