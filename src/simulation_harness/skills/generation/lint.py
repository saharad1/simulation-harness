"""Static checks of SKILL.md text against the store layout (plan 28 B3).

Two lints, both pure and LLM-free:

- :func:`id_only_element_writes` -- a write rule that sets a non-key field on an element of an
  *id-only reference* (an array or map whose elements are bare identifiers of another
  collection: ``x-element-ref`` with ``items``/``additionalProperties`` ``{"type": "string"}``,
  i.e. a ``reference`` element shape with no ``link_fields``). v1 retail rich: every
  SKILL.md told the simulator to "set that entry's `price` / `options`" on
  ``orders[order_id].items`` entries, which the schema makes ``item_id`` strings; every such
  write failed validation (28 of 37 retail rich ``full`` failures). Run inside the
  operations stage's repair round (the IR knows the element shapes there), and on a
  finished skill directory.
- :func:`schema_contradictions` -- an assignment in SKILL.md of a value ``schema.json``
  forbids: ``null`` into a non-nullable field, a string outside the field's ``enum``, or a
  write to a field the entity does not declare (``additionalProperties: false``). Needs the
  final ``schema.json``, so it runs on a finished skill (:func:`lint_skill_dir`,
  ``scripts/lint_skill.py``).

Deliberately narrow, as ``stages/consistency.py`` is: a false positive rejects a draw that
would otherwise be good, so each pattern is anchored on an explicit write phrasing and a
backticked field name, and anything this module cannot parse confidently is not reported.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simulation_harness.skills.generation.ir import SpecModel


@dataclass(frozen=True)
class IdOnlyRef:
    """A parent field whose elements are bare identifiers of another collection."""

    entity: str
    collection: str
    field: str
    key: str
    #: The collection the identifiers point into.
    target: str
    #: The parent entity's own top-level field names: a write rule naming one of these
    #: near a mention of the reference is about the parent, not the element.
    parent_fields: frozenset[str]

    @property
    def label(self) -> str:
        return f"{self.entity}.{self.field}"


def id_only_refs_from_ir(ir: SpecModel) -> list[IdOnlyRef]:
    """Id-only reference fields, from the IR's derived element shapes."""
    out = []
    for entity in ir.entities:
        parent = frozenset(f.name for f in entity.fields)
        for field in entity.fields:
            shape = field.element
            if (
                shape is None
                or shape.kind != "reference"
                or shape.link_fields
                or not shape.target_collection
            ):
                continue
            out.append(
                IdOnlyRef(
                    entity=entity.name,
                    collection=entity.collection,
                    field=field.name,
                    key=shape.local_key or shape.target_key or "id",
                    target=shape.target_collection,
                    parent_fields=parent,
                )
            )
    return out


def _entity_defs(schema: dict) -> dict[str, tuple[str, dict]]:
    """``collection -> (entity name, $def)`` for every store in ``schema.json``."""
    out: dict[str, tuple[str, dict]] = {}
    defs = schema.get("$defs") if isinstance(schema, dict) else None
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(defs, dict) or not isinstance(props, dict):
        return out
    for collection, sub in props.items():
        ref = sub.get("items", {}).get("$ref", "") if isinstance(sub, dict) else ""
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.split("/")[-1]
            if isinstance(defs.get(name), dict):
                out[collection] = (name, defs[name])
    return out


def id_only_refs_from_schema(schema: dict) -> list[IdOnlyRef]:
    """Id-only reference fields, from a finished ``schema.json``."""
    out = []
    for collection, (name, entity_def) in _entity_defs(schema).items():
        props = entity_def.get("properties")
        if not isinstance(props, dict):
            continue
        parent = frozenset(props)
        for field, prop in props.items():
            if not isinstance(prop, dict) or not isinstance(
                prop.get("x-element-ref"), dict
            ):
                continue
            element = prop.get("items") if prop.get("type") == "array" else None
            if prop.get("type") == "object":
                element = prop.get("additionalProperties")
            if not (isinstance(element, dict) and element.get("type") == "string"):
                continue
            ref = prop["x-element-ref"]
            out.append(
                IdOnlyRef(
                    entity=name,
                    collection=collection,
                    field=field,
                    key=str(ref.get("field") or ref.get("key") or "id"),
                    target=str(ref.get("collection") or ""),
                    parent_fields=parent,
                )
            )
    return out


#: How many lines after a mention of an id-only field a write rule still refers to its
#: elements ("find the first entry in `orders[order_id].items` ..." then "Set that
#: entry's `price` ...").
_CONTEXT_LINES = 8
_HEADING = re.compile(r"^#{1,4}\s")
_WRITE_VERB = r"(?:set|sets|setting|write|writes|store|stores|assign|assigns|overwrite|overwrites)"
_ELEMENT_NOUN = r"(?:entry|entries|element|elements|line\s+item|order\s+item)"
#: "Set that entry's `price` to ...", "set the matched element's `options` = ...": the rule
#: names the element itself as what it writes.
_SETS_ELEMENT_FIELD = re.compile(
    rf"(?i)\b{_WRITE_VERB}\b\s+"
    r"(?:(?:that|the|this|each|matched|matching|same|modified|current|replaced)\s+)*"
    rf"{_ELEMENT_NOUN}(?:'s|’s)\s+"
    r"`(?P<field>[A-Za-z_]\w*)(?:\s*=[^`]*)?`"
)
#: "on that matched entry:" -- a rule header whose sub-bullets all act on the element.
_ON_ELEMENT = re.compile(
    rf"(?i)\b(?:on|for|in|to)\s+(?:that|the|this|each)\s+(?:matched\s+|matching\s+)?"
    rf"{_ELEMENT_NOUN}\b[^`]*:\s*$"
)
#: "set `price` to ...", "set `item_id = new_item_id`" (under an _ON_ELEMENT header).
_SETS_FIELD = re.compile(
    rf"(?i)\b{_WRITE_VERB}\b\s+`(?P<field>[A-Za-z_]\w*)(?:\s*=[^`]*)?`"
)
#: A prohibition ("Do **not** write `brand`") is not a write.
_NEGATED_WRITE = re.compile(
    rf"(?i)\b(?:do\s+not|don't|never|must\s+not|not)\s+(?:\*\*)?\s*(?:read\s+or\s+)?"
    rf"{_WRITE_VERB}\b"
)
_BACKTICKED = re.compile(r"`([^`]+)`")
#: "Set `x`", "write the `x`": a backticked path that is the object of a write verb.
_WRITES_PATH = re.compile(
    rf"(?i)\b{_WRITE_VERB}\b\s+(?:(?:the|that|this|each|its|new)\s+)?`(?P<path>[^`]+)`"
)


def _mentions(ref: IdOnlyRef) -> re.Pattern[str]:
    """A backticked mention of the reference field: `orders[order_id].items`, `Order.items`."""
    owner = rf"(?:{re.escape(ref.collection)}|{re.escape(ref.entity)})"
    return re.compile(
        rf"^{owner}(?:\[[^\]]*\])?\.{re.escape(ref.field)}(?:\[[^\]]*\])?$"
    )


def _element_path(ref: IdOnlyRef) -> re.Pattern[str]:
    """A backticked path INTO an element: `orders[order_id].items[i].price`."""
    owner = rf"(?:{re.escape(ref.collection)}|{re.escape(ref.entity)})"
    return re.compile(
        rf"^{owner}(?:\[[^\]]*\])?\.{re.escape(ref.field)}(?:\[[^\]]*\]|\.\d+)?"
        r"\.(?P<field>[A-Za-z_]\w*)"
    )


def _violation(ref: IdOnlyRef, field: str, line: str) -> str:
    return (
        f"`{ref.label}` holds bare `{ref.key}` ids (an id-only reference into "
        f"`{ref.target}`); replace the id in place, do not set other fields -- this rule "
        f"sets `{field}` on an element: {line.strip()[:160]!r}"
    )


def id_only_element_writes(text: str, refs: Iterable[IdOnlyRef]) -> list[str]:
    """Write rules in ``text`` that set a non-key field on an id-only reference's element."""
    refs = list(refs)
    if not refs:
        return []
    lines = text.splitlines()
    out: list[str] = []
    seen: set[tuple[str, str, int]] = set()

    def report(ref: IdOnlyRef, field: str, index: int) -> None:
        if (ref.label, field, index) not in seen:
            seen.add((ref.label, field, index))
            out.append(_violation(ref, field, lines[index]))

    for ref in refs:
        mention = _mentions(ref)
        into = _element_path(ref)
        owner = rf"(?:{re.escape(ref.collection)}|{re.escape(ref.entity)})"
        sibling = re.compile(rf"^{owner}(?:\[[^\]]*\])?\.(?P<field>[A-Za-z_]\w*)")
        window_end = -1
        on_element = False
        for i, line in enumerate(lines):
            if _HEADING.match(line):
                window_end, on_element = -1, False
                continue
            tokens = _BACKTICKED.findall(line)
            negated = bool(_NEGATED_WRITE.search(line))
            # A path into an element, as the object of a write verb ("Set `items[i].price`
            # to ..."); a path only read on a write line ("set `amount` to the sum of
            # `items[i].price`") is not a write of it (plan 28 L1).
            if not negated:
                for m in _WRITES_PATH.finditer(line):
                    into_m = into.match(m.group("path").strip())
                    if into_m and into_m.group("field") != ref.key:
                        report(ref, into_m.group("field"), i)
            if any(mention.match(t.strip()) for t in tokens):
                window_end, on_element = i + _CONTEXT_LINES, False
            elif any(
                (m := sibling.match(t.strip())) and m.group("field") != ref.field
                for t in tokens
            ):
                # The text moved on to another field of the same record
                # (`orders.payment_history`): its entries are not this field's.
                window_end, on_element = -1, False
            if i > window_end:
                continue
            if _ON_ELEMENT.search(line):
                on_element = True
            if negated:
                continue
            matches = list(_SETS_ELEMENT_FIELD.finditer(line))
            if on_element:
                matches += list(_SETS_FIELD.finditer(line))
            for m in matches:
                field = m.group("field")
                if field in (ref.key, ref.field) or field in ref.parent_fields:
                    continue
                report(ref, field, i)
    return out


# --- schema consistency -----------------------------------------------------------------

#: "`status` = `"cancelled"`", "`status` = left at default `None`", "set `status` to `null`".
_ASSIGNMENT = re.compile(
    r"`(?P<target>[A-Za-z_][\w.\[\]]*)`\s*=\s*(?P<rest>[^\n]{0,60})"
    rf"|(?i:\b{_WRITE_VERB}\b)\s+`(?P<target2>[A-Za-z_][\w.\[\]]*)`\s+(?i:to)\s+"
    r"(?P<rest2>[^\n]{0,60})"
)
_NULLS = frozenset({"null", "None"})


def _value_of(rest: str) -> str | None:
    """The assigned value: the first backticked token in ``rest``, if it comes first."""
    m = re.match(r"\s*(?:left at (?:the )?default\s+)?`([^`]+)`", rest)
    return m.group(1).strip() if m else None


def _owner_and_field(
    target: str, defs: dict[str, tuple[str, dict]]
) -> tuple[list[dict], str] | None:
    """The candidate entity defs a target names, and its field. None: not a store field."""
    if target.startswith("returns") or target.startswith("request"):
        return None
    parts = [p for p in re.split(r"\[[^\]]*\]", target) if p]
    path = "".join(parts).split(".")
    by_name = {name: d for name, d in defs.values()}
    if len(path) == 1:
        field = path[0]
        owners = [
            d
            for _, d in defs.values()
            if isinstance(d.get("properties"), dict) and field in d["properties"]
        ]
        return owners, field
    if len(path) == 2:
        owner, field = path
        entity = by_name.get(owner) or (defs[owner][1] if owner in defs else None)
        return ([entity], field) if entity is not None else None
    return None


def _nullable(prop: dict) -> bool:
    types = prop.get("type")
    if types == "null" or (isinstance(types, list) and "null" in types):
        return True
    if prop.get("nullable") is True:
        return True
    for key in ("anyOf", "oneOf"):
        for variant in prop.get(key) or []:
            if isinstance(variant, dict) and _nullable(variant):
                return True
    return False


#: A list item that introduces response construction ("For each returned item:",
#: "- **Response shape**"): assignments nested under it build the reply, not the store.
_RESPONSE_CONTEXT = re.compile(
    r"(?i)\b(?:returned|returns?|response|reply|output)\b[^\n]*(?::\s*|\*\*\s*)$"
)


def _store_write_lines(skill_md: str) -> Iterable[str]:
    """The lines of ``skill_md`` that are not nested under a response-building item."""
    response_indent: int | None = None
    for line in skill_md.splitlines():
        if not line.strip():
            yield line
            continue
        indent = len(line) - len(line.lstrip())
        if _HEADING.match(line):
            response_indent = None
        elif response_indent is not None and indent <= response_indent:
            response_indent = None
        if response_indent is None and _RESPONSE_CONTEXT.search(line):
            response_indent = indent
            continue
        if response_indent is None:
            yield line


def schema_contradictions(skill_md: str, schema: dict) -> list[str]:
    """Assignments in SKILL.md of a value or field that ``schema.json`` forbids.

    Only store writes count: assignments nested under a response-building item ("For
    each returned item:", "**Response shape**") describe the reply, whose shape the
    store schema does not govern.
    """
    defs = _entity_defs(schema)
    if not defs:
        return []
    out: list[str] = []
    for line in _store_write_lines(skill_md):
        for m in _ASSIGNMENT.finditer(line):
            target = m.group("target") or m.group("target2")
            rest = m.group("rest") if m.group("target") else m.group("rest2")
            value = _value_of(rest or "")
            found = _owner_and_field(target, defs)
            if value is None or found is None:
                continue
            owners, field = found
            props = [
                d["properties"][field]
                for d in owners
                if isinstance(d.get("properties"), dict)
                and isinstance(d["properties"].get(field), dict)
            ]
            if len(owners) == 1 and not props:
                entity = owners[0]
                if entity.get("additionalProperties") is False and re.search(
                    rf"(?i)\b{_WRITE_VERB}\b", line
                ):
                    out.append(
                        f"SKILL.md writes `{target}`, a field schema.json does not "
                        f"declare (additionalProperties is false): {line.strip()[:160]!r}"
                    )
                continue
            if not props:
                continue
            if value in _NULLS:
                if not any(_nullable(p) for p in props):
                    out.append(
                        f"SKILL.md assigns null to `{target}`, which schema.json does "
                        f"not allow to be null: {line.strip()[:160]!r}"
                    )
                continue
            literal = re.fullmatch(r'"([^"]*)"', value)
            if literal is None:
                continue
            enums = [p.get("enum") for p in props]
            if all(isinstance(e, list) for e in enums) and not any(
                literal.group(1) in e for e in enums if isinstance(e, list)
            ):
                allowed = sorted(
                    {str(v) for e in enums if isinstance(e, list) for v in e}
                )
                out.append(
                    f"SKILL.md assigns {value} to `{target}`, outside its schema.json "
                    f"enum {allowed}: {line.strip()[:160]!r}"
                )
    return out


def lint_skill(skill_md: str, schema: dict) -> list[str]:
    """Both lints on a whole SKILL.md against its final ``schema.json``."""
    return id_only_element_writes(
        skill_md, id_only_refs_from_schema(schema)
    ) + schema_contradictions(skill_md, schema)


def lint_skill_dir(skill_dir: str | Path) -> list[str]:
    """Both lints on a generated skill directory (``SKILL.md`` + ``schema.json``)."""
    root = Path(skill_dir)
    skill_md = (root / "SKILL.md").read_text(encoding="utf-8")
    schema: Any = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        return ["schema.json is not a JSON object"]
    return lint_skill(skill_md, schema)
