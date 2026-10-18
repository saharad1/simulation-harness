"""Link embedded elements to the collection whose composed key they carry.

A composed identity (``Entity.key_fields``) is decided by the LLM fallback in
``extract.py``, which runs after ``derive_identity``. The identity pass that
annotates foreign keys on element shapes therefore never sees it, and an
embedded element whose fields include every key field of another entity is left
with no ``target_collection``. Downstream, ``_linked_entities`` gates on that
annotation, so the operation stage is never shown the target and cannot tell
that a request item carrying only the key fields names an existing record.

This pass runs on the composed entity list, where every entity's ``key_fields``
is final. It only annotates; it never changes an element's ``kind``, because an
element that carries a composed key may also carry fields of its own.
"""

from __future__ import annotations

from simulation_harness.skills.generation.ir import ElementShape, Entity, Field
from simulation_harness.utils.logging import get_logger

logger = get_logger(__name__)


def _composed_targets(
    shape: ElementShape, owner: Entity, candidates: list[Entity]
) -> list[Entity]:
    names = {f.name for f in shape.fields}
    return [t for t in candidates if t is not owner and set(t.key_fields) <= names]


def _link_field(field: Field, owner: Entity, candidates: list[Entity]) -> Field:
    shape = field.element
    if (
        shape is None
        or shape.kind != "embedded"
        or not shape.fields
        or shape.target_collection is not None
    ):
        return field
    matches = _composed_targets(shape, owner, candidates)
    if len(matches) != 1:
        if len(matches) > 1:
            logger.warning(
                "element of '%s.%s' carries the composed key of more than one "
                "collection (%s); leaving it unlinked rather than guessing",
                owner.name,
                field.name,
                ", ".join(sorted(t.collection for t in matches)),
            )
        return field
    target = matches[0]
    linked = shape.model_copy(
        update={
            "target_collection": target.collection,
            "target_key": target.primary_key,
            "target_key_fields": tuple(target.key_fields),
        }
    )
    return field.model_copy(update={"element": linked})


def link_composed_references(entities: list[Entity]) -> list[Entity]:
    """Annotate each unlinked embedded element that carries exactly one other
    entity's full composed key.

    Returns a new list in the same order; entities with nothing to link are
    returned as the same objects. No entity with ``key_fields`` means no change.
    """
    candidates = [e for e in entities if e.key_fields]
    if not candidates:
        return list(entities)
    out: list[Entity] = []
    for entity in entities:
        fields = [_link_field(f, entity, candidates) for f in entity.fields]
        if any(new is not old for new, old in zip(fields, entity.fields, strict=True)):
            out.append(entity.model_copy(update={"fields": fields}))
        else:
            out.append(entity)
    return out
