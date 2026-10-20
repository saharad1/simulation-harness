"""Stage: mechanical check that general rules agree with per-operation sections.

A generated SKILL.md has two independent sources of truth about the same
behaviour: the preamble's ``### Derivation Rules`` (one concurrent stage) and
each operation's ``### /<op> <METHOD>`` section (a different, per-chunk
concurrent stage). Nothing reconciles them — see the airline
``book_reservation`` defect this module exists to catch: the derivation rule
says ``User.reservations`` is derived from ``reservations``, but
``book_reservation``'s Store access only writes ``reservations``, so the
simulator never updates the user's reservation list.

Pure and LLM-free: ``check_consistency`` only parses text already present in
``skill_md`` against the IR. It is deliberately narrow — exactly the two
mechanical patterns documented in the task scoping, no more — because a false
positive here fails a draw that would otherwise have been good. Parsing is
defensive throughout: a derivation bullet or operation section this module
cannot confidently parse produces no error rather than a guess.
"""

from __future__ import annotations

import re

from simulation_harness.skills.generation.ir import OperationKind, SpecModel

# A derivation bullet's left-hand side: "`Entity.field`" at the start of the
# rule (after the leading "- " is stripped), e.g. "`User.reservations`".
_DERIVED_LEFT = re.compile(r"^`(?P<entity>[A-Za-z_][\w]*)\.(?P<field>[\w]+)`")

# Every "`Entity.field`"-shaped backtick reference elsewhere in the rule. The
# right-hand side is "the first one that names a different entity than the
# left", which tolerates the prose in between ("equals the matching ...",
# "equals the set of ... values for ...").
_ENTITY_FIELD = re.compile(r"`(?P<entity>[A-Za-z_][\w]*)\.(?P<field>[\w]+)`")

# "Write `<collection>`" / "**Write `<collection>`**", the one Store-access
# write shape the real bundles use (verified against skills-store/*/SKILL.md).
_WRITE = re.compile(r"Write `(?P<collection>[a-z_][a-z0-9_]*)`")

# A section-local prohibition naming a collection: "Do not write `payments`",
# "Do not write any fields in `payments`", "never write to `payments`". The
# backtick name(s) must follow immediately (modulo this narrow filler) — a
# qualifier like "unrelated"/"other" changes the meaning to "don't touch
# *some* entries of a collection you otherwise write", which is not the
# blanket prohibition this pattern targets, so it is deliberately excluded
# from the filler and the match simply fails to bridge past it.
_FORBID_WRITE = re.compile(
    r"(?i)\b(?:do not|never)\s+write\s+"
    r"(?:to\s+)?(?:any\s+(?:fields?\s+)?(?:in\s+)?)?"
    r"(?P<names>`[a-z_][a-z0-9_]*`(?:\s*,\s*|\s+or\s+)*(?:`[a-z_][a-z0-9_]*`(?:\s*,\s*|\s+or\s+)*)*)"
)
_BACKTICK_NAME = re.compile(r"`([a-z_][a-z0-9_]*)`")
# "... in `users` other than `address`": a qualifier AFTER the name limits a write the
# section makes rather than forbidding the collection, so the match is not a prohibition.
_QUALIFIED = re.compile(r"(?i)\s*(?:other than|except|besides|apart from)\b")

_SECTION_HEADER = re.compile(r"^### (?P<header>\S+ \S+)", re.M)

# Operation kinds whose effect on a collection is creating or deleting a
# record there, per the scoping: pattern 1 only applies to those.
_CREATE_OR_DELETE = {OperationKind.create, OperationKind.delete}


def _entity_collection_map(ir: SpecModel) -> dict[str, str]:
    return {e.name: e.collection for e in ir.entities}


def _derivation_rules_block(skill_md: str) -> str | None:
    match = re.search(
        r"^### Derivation Rules\n(.*?)(?=^### |\Z)", skill_md, re.M | re.S
    )
    return match.group(1) if match else None


def _derivation_pairs(block: str) -> list[tuple[str, str, str, str]]:
    """Yield (left_entity, left_field, right_entity, right_field) per bullet.

    Only bullets that parse cleanly into a left-hand "`A.f`" and at least one
    differently-named entity reference elsewhere in the line are yielded;
    anything else is silently skipped (precision over recall).
    """
    pairs: list[tuple[str, str, str, str]] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        line = line[1:].strip()
        left = _DERIVED_LEFT.match(line)
        if left is None:
            continue
        left_entity = left.group("entity")
        left_field = left.group("field")
        rest = line[left.end() :]
        right = None
        for m in _ENTITY_FIELD.finditer(rest):
            if m.group("entity") != left_entity:
                right = m
                break
        if right is None:
            continue
        pairs.append(
            (left_entity, left_field, right.group("entity"), right.group("field"))
        )
    return pairs


def _sections(skill_md: str) -> list[tuple[str, str]]:
    """[(header_text, section_body)] for each ``### /<op> <METHOD>`` section."""
    headers = list(_SECTION_HEADER.finditer(skill_md))
    out = []
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(skill_md)
        out.append((m.group("header"), skill_md[start:end]))
    return out


def _op_id_for_header(ir: SpecModel, header: str) -> str | None:
    for op in ir.operations:
        if f"{op.path} {op.method}" == header:
            return op.operation_id
    return None


def _written_collections(section_body: str) -> set[str]:
    return set(_WRITE.findall(section_body))


def _forbidden_collections(section_body: str) -> set[str]:
    names: set[str] = set()
    for m in _FORBID_WRITE.finditer(section_body):
        if _QUALIFIED.match(section_body, m.end()):
            continue
        names.update(_BACKTICK_NAME.findall(m.group("names")))
    return names


def check_consistency(ir: SpecModel, skill_md: str) -> list[str]:
    """Return one generic message per contradiction between general rules and
    per-operation sections. Empty list = consistent. Pure, no LLM calls."""
    errors: list[str] = []
    collection_by_entity = _entity_collection_map(ir)
    sections = _sections(skill_md)
    op_by_id = {op.operation_id: op for op in ir.operations}

    # Precompute each section's writes/forbids and op id once.
    parsed_sections: list[tuple[str, set[str], set[str]]] = []
    for header, body in sections:
        op_id = _op_id_for_header(ir, header)
        if op_id is None:
            continue
        parsed_sections.append(
            (op_id, _written_collections(body), _forbidden_collections(body))
        )

    # Pattern 2: self-contradiction within one section.
    for op_id, writes, forbids in parsed_sections:
        for collection in sorted(writes & forbids):
            errors.append(f"{op_id} both writes and forbids writing {collection}")

    # Pattern 1: derived field not maintained by the operations that create or
    # delete the records it depends on.
    block = _derivation_rules_block(skill_md)
    if block is not None:
        for left_entity, left_field, right_entity, _right_field in _derivation_pairs(
            block
        ):
            if left_entity == right_entity:
                continue
            left_collection = collection_by_entity.get(left_entity)
            right_collection = collection_by_entity.get(right_entity)
            if left_collection is None or right_collection is None:
                continue
            for op_id, writes, _forbids in parsed_sections:
                op = op_by_id.get(op_id)
                if op is None or op.kind not in _CREATE_OR_DELETE:
                    continue
                if right_collection in writes and left_collection not in writes:
                    errors.append(
                        f"{op_id} writes {right_collection} but not {left_collection}, "
                        f"which the derivation rule '{left_entity}.{left_field}' depends on"
                    )
    return errors
