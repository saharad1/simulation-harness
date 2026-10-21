"""Stage: design schema.json from the IR."""

from __future__ import annotations

import json
import re

import jsonschema
from jsonschema.validators import Draft202012Validator

try:
    from importlib.resources import files
except ImportError:  # pragma: no cover
    from importlib_resources import files  # type: ignore[import-not-found]

from simulation_harness.skills.generation.ir import (
    KEY_SEPARATOR,
    ElementField,
    ElementShape,
    Entity,
    SpecModel,
)
from simulation_harness.openapi.parser import OpenAPISpec
from simulation_harness.skills.generation.llm import call_json
from simulation_harness.skills.generation.repair import with_repair
from simulation_harness.skills.generation.stages.analyze.sources import (
    _request_signatures,
    _unwrap_returns,
)


def _load_prompt() -> str:
    assets = files("simulation_harness.skills.assets")
    return (assets / "generation" / "schema.md").read_text()


def entity_summary(ir: SpecModel) -> str:
    return json.dumps([e.model_dump() for e in ir.entities], indent=2)


def open_value_fields(spec: OpenAPISpec, ir: SpecModel) -> dict[str, set[str]]:
    """Entity fields the API reports as free text, keyed by entity name.

    A lookup — an operation whose top-level request fields are all fields of one
    entity — that returns a plain string (no ``enum``/``const``) describing a field
    of that entity by name is evidence the field holds more than one value. A
    listing's single-value constraint on the same field then records the listing's
    filter, not the field: tau2-airline's searches return only ``available``
    flights, while ``get_flight_status(flight_number, date)`` returns "The status
    of the flight." as an open string.
    """
    out: dict[str, set[str]] = {}
    for op in spec.operations:
        resp = op.get_success_response_schema()
        if not isinstance(resp, dict):
            continue
        resp = _unwrap_returns(resp)
        if resp.get("type") != "string" or "enum" in resp or "const" in resp:
            continue
        desc = (resp.get("description") or "").lower()
        top = next(
            (
                sig
                for path, sig in _request_signatures(op.get_request_schema())
                if path == ""
            ),
            None,
        )
        if not desc or not top:
            continue
        for entity in ir.entities:
            names = {f.name for f in entity.fields}
            if not top <= names:
                continue
            for f in entity.fields:
                if f.name not in top and re.search(
                    rf"\b{re.escape(f.name.lower())}\b", desc
                ):
                    out.setdefault(entity.name, set()).add(f.name)
    return out


def _single_valued(node: dict) -> bool:
    enum = node.get("enum")
    return "const" in node or (isinstance(enum, list) and len(enum) == 1)


def relax_open_fields_ir(ir: SpecModel, open_fields: dict[str, set[str]]) -> SpecModel:
    """Drop a single-value ``enum`` on open fields before later stages read the IR."""
    if not open_fields:
        return ir
    entities = []
    for e in ir.entities:
        wanted = open_fields.get(e.name, set())
        fields = [
            f.model_copy(update={"enum": None})
            if f.name in wanted and f.enum is not None and len(f.enum) <= 1
            else f
            for f in e.fields
        ]
        entities.append(e.model_copy(update={"fields": fields}))
    return ir.model_copy(update={"entities": entities})


def relax_open_fields_schema(
    schema: dict, ir: SpecModel, open_fields: dict[str, set[str]]
) -> dict:
    """Drop a single-value ``enum``/``const`` the schema LLM put on an open field.

    Deterministic, after generation: a prompt rule asking for this was measured
    unreliable (2 of 10 airline draws still pinned ``status`` to ``available``, so
    the real world's landed and cancelled flights could not be seeded).
    """
    for e in ir.entities:
        wanted = open_fields.get(e.name)
        if not wanted:
            continue
        _, entity_def = _resolve_def(schema, e.collection)
        props = (entity_def or {}).get("properties")
        if not isinstance(props, dict):
            continue
        for name in wanted:
            node = props.get(name)
            if isinstance(node, dict) and _single_valued(node):
                node.pop("enum", None)
                node.pop("const", None)
    return schema


# --- nullability carried over from the spec ---------------------------------
#
# The schema LLM is shown the IR, and an IR ``Field`` has no way to say "or null",
# so ``Optional[X]`` in the spec arrives in schema.json as plain ``X``: the store
# then rejects ``status: null`` that the real system writes. Restored here in
# code, from the spec, after generation — the same deterministic post-pass shape
# as ``relax_open_fields_schema``.


def _spec_resolve(spec: dict, node: object) -> object:
    """Follow local ``$ref``s (``#/components/schemas/X`` and the like)."""
    seen: set[str] = set()
    while isinstance(node, dict) and isinstance(node.get("$ref"), str):
        ref = node["$ref"]
        if not ref.startswith("#/") or ref in seen:
            return node
        seen.add(ref)
        target: object = spec
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            target = target.get(part) if isinstance(target, dict) else None
        if not isinstance(target, dict):
            return node
        node = target
    return node


def _spec_allows_null(spec: dict, node: object, depth: int = 0) -> bool:
    if not isinstance(node, dict) or depth > 16:
        return False
    if node.get("nullable") is True:
        return True
    node = _spec_resolve(spec, node)
    if not isinstance(node, dict):
        return False
    declared = node.get("type")
    if declared == "null" or (isinstance(declared, list) and "null" in declared):
        return True
    if node.get("nullable") is True:
        return True
    variants = node.get("anyOf") or node.get("oneOf")
    return isinstance(variants, list) and any(
        _spec_allows_null(spec, v, depth + 1) for v in variants
    )


def _spec_object(spec: dict, node: object) -> dict | None:
    """The object schema a spec property holds, reading through null and arrays."""
    for _ in range(8):
        node = _spec_resolve(spec, node)
        if not isinstance(node, dict):
            return None
        if isinstance(node.get("properties"), dict):
            return node
        variants = node.get("anyOf") or node.get("oneOf")
        if isinstance(variants, list):
            rest = [
                v
                for v in variants
                if not (isinstance(v, dict) and v.get("type") == "null")
            ]
            if len(rest) != 1:
                return None
            node = rest[0]
        elif isinstance(node.get("items"), dict):
            node = node["items"]
        else:
            return None
    return None


def _spec_response_objects(spec: dict) -> list[dict]:
    """Every object schema reachable from a success response or a component.

    Request bodies are left out on purpose: a nullable request field is an
    optional argument, which says nothing about whether the stored value can be
    null.
    """
    roots: list[object] = []
    paths = spec.get("paths")
    for item in (paths or {}).values() if isinstance(paths, dict) else []:
        if not isinstance(item, dict):
            continue
        for op in item.values():
            responses = op.get("responses") if isinstance(op, dict) else None
            if not isinstance(responses, dict):
                continue
            for resp in responses.values():
                if not isinstance(resp, dict):
                    continue
                roots.append(resp.get("schema"))  # Swagger 2
                content = resp.get("content")
                if isinstance(content, dict):
                    roots.extend(
                        media.get("schema")
                        for media in content.values()
                        if isinstance(media, dict)
                    )
    for holder in (
        (spec.get("components") or {}).get("schemas"),
        spec.get("definitions"),
    ):
        if isinstance(holder, dict):
            roots.extend(holder.values())

    out: list[dict] = []
    seen: set[int] = set()
    stack = list(reversed(roots))
    while stack:
        node = _spec_resolve(spec, stack.pop())
        if not isinstance(node, dict) or id(node) in seen:
            continue
        seen.add(id(node))
        props = node.get("properties")
        if isinstance(props, dict):
            out.append(node)
            stack.extend(reversed(list(props.values())))
        for key in ("items", "additionalProperties"):
            if isinstance(node.get(key), dict):
                stack.append(node[key])
        for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
            if isinstance(node.get(key), list):
                stack.extend(reversed(node[key]))
    return out


def _accepts_null(node: dict) -> bool:
    """Whether a generated schema property already admits ``null``."""
    if "$ref" in node:
        return False
    variants = node.get("anyOf") or node.get("oneOf")
    if isinstance(variants, list):
        return any(isinstance(v, dict) and _accepts_null(v) for v in variants)
    declared = node.get("type")
    type_ok = (
        declared is None
        or declared == "null"
        or (isinstance(declared, list) and "null" in declared)
    )
    enum = node.get("enum")
    enum_ok = not isinstance(enum, list) or None in enum
    const_ok = "const" not in node or node["const"] is None
    return type_ok and enum_ok and const_ok


def _make_nullable(node: dict) -> None:
    if "$ref" in node:
        node["anyOf"] = [{"$ref": node.pop("$ref")}, {"type": "null"}]
        return
    for key in ("anyOf", "oneOf"):
        if isinstance(node.get(key), list):
            node[key].append({"type": "null"})
            return
    declared = node.get("type")
    if isinstance(declared, str) and declared != "null":
        node["type"] = [declared, "null"]
    elif isinstance(declared, list) and "null" not in declared:
        declared.append("null")
    if "const" in node and node["const"] is not None:
        node["enum"] = [node.pop("const"), None]
    elif isinstance(node.get("enum"), list) and None not in node["enum"]:
        node["enum"].append(None)


def _def_object(schema: dict, node: object) -> tuple[str | None, dict | None]:
    """``(def name, object schema)`` a generated property holds, through ``$ref``,
    a null union and array ``items``."""
    name = None
    for _ in range(8):
        if not isinstance(node, dict):
            return None, None
        ref = node.get("$ref")
        if isinstance(ref, str):
            if not ref.startswith("#/$defs/"):
                return None, None
            name = ref.split("/")[-1]
            defs = schema.get("$defs")
            node = defs.get(name) if isinstance(defs, dict) else None
            continue
        if isinstance(node.get("properties"), dict):
            return name, node
        variants = node.get("anyOf") or node.get("oneOf")
        if isinstance(variants, list):
            rest = [
                v
                for v in variants
                if not (isinstance(v, dict) and v.get("type") == "null")
            ]
            if len(rest) != 1:
                return None, None
            node = rest[0]
        elif isinstance(node.get("items"), dict):
            node = node["items"]
        else:
            return None, None
    return None, None


def _widen_object(
    schema: dict,
    spec: dict,
    label: str,
    entity_def: dict,
    spec_obj: dict,
    keep: set[str],
    widened: set[tuple[str, str]],
    seen: set[tuple[int, int]],
) -> None:
    if (id(entity_def), id(spec_obj)) in seen:
        return
    seen.add((id(entity_def), id(spec_obj)))
    props = entity_def.get("properties")
    spec_props = spec_obj.get("properties")
    if not isinstance(props, dict) or not isinstance(spec_props, dict):
        return
    for name in sorted(set(props) & set(spec_props)):
        node, spec_node = props[name], spec_props[name]
        if not isinstance(node, dict):
            continue
        # Recurse first: widening a `$ref` property rewrites it into an anyOf.
        nested_name, nested_def = _def_object(schema, node)
        nested_spec = _spec_object(spec, spec_node)
        if nested_def is not None and nested_spec is not None:
            _widen_object(
                schema,
                spec,
                nested_name or f"{label}.{name}",
                nested_def,
                nested_spec,
                set(),
                widened,
                seen,
            )
        if name in keep or not _spec_allows_null(spec, spec_node):
            continue
        if not _accepts_null(node):
            _make_nullable(node)
            widened.add((label, name))


def _widen(schema: dict, spec: dict) -> list[tuple[str, str]]:
    if not isinstance(schema, dict) or not isinstance(spec, dict):
        return []
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return []
    objects = _spec_response_objects(spec)
    widened: set[tuple[str, str]] = set()
    seen: set[tuple[int, int]] = set()
    for collection in sorted(properties):
        def_name, entity_def = _resolve_def(schema, collection)
        if entity_def is None or not isinstance(entity_def.get("properties"), dict):
            continue
        key_fields = entity_def.get("x-key-fields")
        pk = entity_def.get("x-primary-key")
        keys = (
            [k for k in key_fields if isinstance(k, str)]
            if isinstance(key_fields, list) and key_fields
            else [pk]
            if isinstance(pk, str)
            else []
        )
        if not keys:
            continue  # nothing anchors a spec object to this entity
        def_props = set(entity_def["properties"])
        for obj in objects:
            spec_props = set(obj["properties"])
            # The spec object carries this entity's key, and is mostly made of
            # its fields — a payload that merely mentions the key is not it.
            if not set(keys) <= spec_props or 2 * len(spec_props & def_props) < len(
                spec_props
            ):
                continue
            _widen_object(
                schema,
                spec,
                def_name or collection,
                entity_def,
                obj,
                {*keys, pk} - {None},
                widened,
                seen,
            )
    return sorted(widened)


def widen_nullable_fields(schema: dict, spec: dict) -> dict:
    """Let every entity property the spec declares nullable accept ``null``.

    A property is nullable in the spec when it (after ``$ref``s) has an
    ``anyOf``/``oneOf`` variant ``{"type": "null"}``, a ``type`` list containing
    ``"null"``, or ``nullable: true``. The spec object is matched to a collection
    ``$def`` by carrying its key (``x-primary-key``, or every ``x-key-fields``
    entry) with most of its properties being that ``$def``'s; nested objects are
    followed through ``$ref``/``items`` on both sides. Only success responses and
    components count, and key fields are never widened.

    Widening: ``type: X`` becomes ``[X, "null"]``, an ``enum`` gains ``null``, a
    ``const`` becomes ``enum: [const, null]``, a ``$ref`` becomes
    ``anyOf: [{$ref}, {type: null}]``. Mutates and returns ``schema``; idempotent,
    so it is safe on an already-generated skill's schema.json.
    """
    _widen(schema, spec)
    return schema


def nullable_widenings(schema: dict, spec: dict) -> list[tuple[str, str]]:
    """``(def, property)`` pairs :func:`widen_nullable_fields` would change.

    Does not modify ``schema``.
    """
    return _widen(json.loads(json.dumps(schema)), spec)


def _resolve_def(schema: dict, collection: str) -> tuple[str | None, dict | None]:
    """Follow ``properties[collection].items.$ref`` into ``$defs``.

    Mirrors the resolution ``state/loader.py`` performs at runtime, so a schema
    this module accepts is one the loader can read. Every level is
    isinstance-guarded rather than assumed to be a dict: this runs on schemas
    an LLM produced, before (or without) ``validate_schema`` having accepted
    their shape, so ``properties`` or ``$defs`` being a string or a list (not
    just missing) has to fail closed instead of raising.
    """
    if not isinstance(schema, dict):
        return None, None
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None, None
    prop = properties.get(collection)
    if not isinstance(prop, dict):
        return None, None
    items = prop.get("items")
    ref = items.get("$ref") if isinstance(items, dict) else None
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        return None, None
    def_name = ref.split("/")[-1]
    defs = schema.get("$defs")
    entity_def = defs.get(def_name) if isinstance(defs, dict) else None
    return def_name, entity_def if isinstance(entity_def, dict) else None


def validate_schema(schema: dict, ir: SpecModel | None = None) -> list[str]:
    """Check JSON Schema validity and, when ``ir`` is given, the runtime contract.

    Without ``ir`` this is the original validity-only check, so existing callers
    are unaffected.
    """
    try:
        Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as e:
        return [f"schema.json is not a valid JSON Schema: {e.message}"]
    if ir is None:
        return []
    # `True`/`False` are valid Draft 2020-12 schemas (check_schema above passed
    # them), but there is nothing further to compare against the IR.
    if not isinstance(schema, dict):
        return ["schema.json must be a JSON object to check against store_metadata"]
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return [
            "schema.json's top-level 'properties' must be an object mapping "
            "collection names to their array schemas"
        ]
    errors: list[str] = []
    expected = set(ir.store_metadata.collections)
    present = set(properties)
    for missing in sorted(expected - present):
        errors.append(f"schema.json is missing a property for collection '{missing}'")
    for extra in sorted(present - expected):
        errors.append(
            f"schema.json declares collection '{extra}', which is not in "
            f"store_metadata.collections"
        )
    # Collections sharing a $def with different derived primary keys: only one
    # of the pk_map entries can win when enforce_contract stamps the $def, so
    # this has to be rejected rather than forced. Same pk is harmless — don't
    # reject that, or with_repair asks the LLM to fix something it can't.
    def_collections: dict[str, list[str]] = {}
    for collection in sorted(expected & present):
        def_name, entity_def = _resolve_def(schema, collection)
        if def_name is None:
            errors.append(
                f"collection '{collection}' must be an array whose items are a "
                f"'$ref' into '#/$defs/' — the runtime reads the primary key "
                f"from the referenced $def"
            )
            continue
        if entity_def is None:
            errors.append(
                f"collection '{collection}' references '#/$defs/{def_name}', "
                f"which is not defined"
            )
            continue
        pk = ir.store_metadata.pk_map.get(collection)
        if pk and pk not in (entity_def.get("properties") or {}):
            errors.append(
                f"$defs/{def_name} must declare the primary-key property "
                f"'{pk}' for collection '{collection}'"
            )
        pk_prop = (entity_def.get("properties") or {}).get(pk) if pk else None
        if isinstance(pk_prop, dict) and (
            pk_prop.get("type") in ("array", "object")
            or isinstance(pk_prop.get("items"), dict)
            or isinstance(pk_prop.get("properties"), dict)
        ):
            errors.append(
                f"$defs/{def_name} declares primary key '{pk}' for collection "
                f"'{collection}', but that property is not scalar. The runtime keys "
                f"rows by str(pk_value), so a list- or object-valued key silently "
                f"becomes a stringified value."
            )
        def_collections.setdefault(def_name, []).append(collection)
    for def_name, collections in def_collections.items():
        pks = {ir.store_metadata.pk_map.get(c) for c in collections}
        if len(pks) > 1:
            errors.append(
                f"$defs/{def_name} is shared by collections {sorted(collections)} "
                f"whose derived primary keys differ ({sorted(p for p in pks if p)}); "
                f"each needs its own $def, or the primary keys must match"
            )
    return errors


def _object_element(fields: tuple[ElementField, ...]) -> dict:
    """An object subschema for the element, with `required` where declared.

    `required` is what turns a type check into a shape check. Without it a seeded
    element may omit fields the response promises and still validate — retail's
    `products[].variants` was seeded as a flattened `{color, size, price,
    available}`, matching neither the spec's variant nor the `items` row it
    duplicates, and passed. `additionalProperties` stays open: an element may carry
    storage bookkeeping no response declares, and unlike a top-level entity there
    is no union across sources to make closing it safe.
    """
    subschema: dict = {
        "type": "object",
        "properties": {f.name: {"type": f.type} for f in fields},
    }
    required = [f.name for f in fields if f.required]
    if required:
        subschema["required"] = required
    return subschema


def _element_items(shape: ElementShape) -> dict | None:
    """The ``items`` subschema for one derived element shape, or ``None``.

    ``additionalProperties`` is deliberately left open on object elements. The
    fix this supports only needs the element's *type* pinned — a bare id string
    where an object is required fails on ``"type"`` alone — and storage may
    legitimately carry bookkeeping fields no response declares, which for a
    nested element there is no union-across-sources to make safe.
    """
    if shape.kind == "scalar":
        return {"type": shape.type} if shape.type else None
    if shape.kind == "embedded":
        return _object_element(shape.fields)
    if shape.kind == "reference":
        if not shape.link_fields:
            # A pure projection of the target: the parent stores identifiers, and
            # a primary key is a string by the schema stage's own contract.
            return {"type": "string"}
        return _object_element(shape.link_fields)
    return None


# Where a container's element subschema belongs, and the declared `type` that has
# to be there for the stamp to apply.
_CONTAINER_KEYWORD = {
    "array": ("items", "array"),
    "map": ("additionalProperties", "object"),
}


def _stamp_element_shapes(entity_def: dict, entity: Entity) -> None:
    """Write each derived element shape onto its container property.

    ``items`` for an array, ``additionalProperties`` for a map. Skips a property
    whose declared type does not match the container the shape was derived from:
    getting that wrong is a pre-existing shape problem for ``validate_schema`` to
    report, and stamping over it would bury the real problem.
    """
    props = entity_def.get("properties")
    if not isinstance(props, dict):
        return
    for field in entity.fields:
        if field.element is None:
            continue
        keyword_type = _CONTAINER_KEYWORD.get(field.element.container)
        if keyword_type is None:
            continue
        keyword, declared = keyword_type
        prop = props.get(field.name)
        if not isinstance(prop, dict) or prop.get("type") != declared:
            continue
        items = _element_items(field.element)
        if items is None:
            continue
        prop[keyword] = items
        element = field.element
        if element.target_collection is not None:
            # Recorded so the operation stage can see which collection an
            # identifier points at; validators ignore `x-` keywords. `field` is
            # present when the element's own key name differs from the target's.
            ref: dict = {
                "collection": element.target_collection,
                "key": element.target_key,
            }
            if element.local_key is not None:
                ref["field"] = element.local_key
            if element.target_key_fields:
                # The element carries the target's composed key as separate
                # fields; joined in this order with this separator they are the
                # target's primary key value.
                ref["key_fields"] = list(element.target_key_fields)
                ref["separator"] = KEY_SEPARATOR
            prop["x-element-ref"] = ref


def enforce_contract(schema: dict, ir: SpecModel) -> dict:
    """Stamp the derived contract onto the schema.

    Forced rather than validated because the correct values are known: the
    collection set and pk map were derived from the spec in code, not
    invented by the LLM that wrote ``schema``. Safe to call on a schema that
    hasn't passed :func:`validate_schema` yet — every lookup is
    isinstance-guarded via :func:`_resolve_def`, so a collection or ``$def``
    this can't resolve is left alone rather than raising, and the malformed
    shape falls through to ``validate_schema`` as repair feedback instead of
    an exception escaping the repair loop.

    Stamps three things:
    - Top level: ``required`` is set to exactly ``store_metadata.collections``
      and ``additionalProperties`` to ``False``, so ``db.json``'s collection
      set is checked against the IR too (via ``validate_schema_and_db``, in
      the seed stage's own repair loop) — closing the drift `state/loader.py`
      would otherwise take from ``db.json``'s keys rather than the schema.
    - Per collection: the primary key is written onto the ``$def`` the
      collection's items ``$ref`` resolves to (where the runtime reads it
      from), and added to that ``$def``'s ``required`` list.
    - Per array field with a derived element shape: that field's ``items``. The
      prompt cannot transcribe an element shape it was never given, and until it
      is stamped ``items`` stays ``{}``, which accepts anything — so the seed
      stage is free to populate an array in a shape no other stage agreed to.
      Stamping it makes ``validate_schema_and_db`` reject the mismatch and feed
      it back as repair feedback, with no seed-prompt change at all.
    - Per entity with a composed key: ``x-key-fields`` and ``x-key-separator``,
      the recipe for its ``x-primary-key`` value.
    """
    if isinstance(schema, dict):
        schema["required"] = list(ir.store_metadata.collections)
        schema["additionalProperties"] = False
    entities = {e.collection: e for e in ir.entities}
    for collection, pk in ir.store_metadata.pk_map.items():
        _, entity_def = _resolve_def(schema, collection)
        if entity_def is None:
            continue
        entity_def["x-primary-key"] = pk
        required = entity_def.setdefault("required", [])
        if isinstance(required, list) and pk not in required:
            required.append(pk)
        entity = entities.get(collection)
        if entity is not None:
            _stamp_element_shapes(entity_def, entity)
            if entity.key_fields:
                # The recipe for the composed key travels with the key, so anything
                # that seeds or reads this store builds the same value without
                # reverse-engineering it from SKILL.md prose.
                entity_def["x-key-fields"] = list(entity.key_fields)
                entity_def["x-key-separator"] = KEY_SEPARATOR
    return schema


async def generate_schema(ir: SpecModel, llm, *, retries: int) -> dict:
    prompt = _load_prompt()
    base_user = (
        f"# Entities\n```json\n{entity_summary(ir)}\n```\n\n"
        f"# store_metadata\n```json\n{ir.store_metadata.model_dump_json(indent=2)}\n```\n"
    )

    async def produce(feedback):
        user = base_user + (
            "\n# feedback\n" + "\n".join(feedback) + "\n" if feedback else ""
        )
        payload = await call_json(llm, prompt, user)
        if not isinstance(payload, dict):
            return {"_shape_error": "not an object"}
        return payload

    def validate(payload: dict) -> list[str]:
        if "_shape_error" in payload:
            return [payload["_shape_error"]]
        if "schema_json" not in payload:
            return ["payload must contain 'schema_json'"]
        if not isinstance(payload["schema_json"], dict):
            return ["'schema_json' must be an object"]
        # Force what's known (the primary key) before rejecting what isn't (a
        # wrong collection set or a missing $ref) — no point re-prompting the
        # LLM about something we can simply fix.
        enforce_contract(payload["schema_json"], ir)
        return validate_schema(payload["schema_json"], ir)

    payload = await with_repair(produce, validate, stage="schema", retries=retries)
    return payload["schema_json"]
