"""Selects and shapes the schema maps the analyze stage reasons over.

Two maps, on purpose:

``identity``
    What the deterministic rule may look at. For a spec with declared
    ``components.schemas`` that is the components map. For an RPC/tool-style
    spec it is **success response schemas only**, keyed by operation id — a
    request body is a *projection* of an entity (a create body legitimately
    omits the server-assigned id) and feeding it to the identity rule produces
    undecidable noise, not entities.

``enrich``
    What the LLM may look at: requests *and* responses, value-deduplicated.
    Request bodies carry field-level prose responses omit, and tau2-style specs
    repeat one entity shape across a dozen operations.

A third, smaller output — ``deferred`` — names schemas that are in ``identity``
so the orchestrator can *find* them, but that the deterministic rule must not be
run on. See :func:`_identity_responses`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from simulation_harness.openapi.parser import OpenAPISpec


@dataclass(frozen=True)
class SchemaSources:
    synthetic: bool
    identity: dict[str, dict]
    enrich: dict[str, dict]
    # Keys present in ``identity`` that the deterministic rule must skip and
    # report undecidable instead. Always a subset of ``identity``: the
    # orchestrator resolves the fallback's inputs as
    # ``{name: sources.identity[name] for name in identity.undecidable}``, so a
    # name it cannot look up is a KeyError, not a quiet omission.
    deferred: frozenset[str] = frozenset()


def _unwrap_returns(resp: dict) -> dict:
    """Unwrap the tool-style ``{"returns": {...}}`` response envelope one level.

    For tool-driven specs, success responses carry a {"properties": {"returns":
    {...}}} wrapper; this extracts the inner schema.
    """
    props = resp.get("properties") if isinstance(resp, dict) else None
    if isinstance(props, dict) and isinstance(props.get("returns"), dict):
        return props["returns"]
    return resp


def dedup_by_value(schemas: dict[str, dict]) -> dict[str, dict]:
    """Drop keys whose schema is structurally identical to an earlier one.

    Iterates sorted names so the surviving key is a function of the content,
    not of how the input map happened to be built.
    """
    seen: set[str] = set()
    out: dict[str, dict] = {}
    for name in sorted(schemas):
        fingerprint = json.dumps(schemas[name], sort_keys=True, default=str)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out[name] = schemas[name]
    return out


def inline_schema_evidence(spec: OpenAPISpec) -> dict:
    """Synthesize a `name -> JSON schema` map from operation request/response
    bodies, for RPC/tool-style specs that declare no `components.schemas`.

    The output mirrors the shape of `components.schemas` so it can be fed to
    the extract stage unchanged. Keys are suffixed `__request`/`__response`.
    The tool-style ``{"returns": {...}}`` response envelope is unwrapped one
    level so the LLM sees the entity shape directly.
    """
    evidence: dict = {}
    for op in spec.operations:
        req = op.get_request_schema()
        if req:
            evidence[f"{op.operation_id}__request"] = req
        resp = op.get_success_response_schema()
        if resp:
            resp = _unwrap_returns(resp)
            evidence[f"{op.operation_id}__response"] = resp
    return evidence


def _entity_shaped(resp: dict) -> bool:
    """Whether ``resp`` could carry an entity: an object with properties, or an
    array of them.

    Deliberately looser than the strict gate below — it admits an object that
    never declared ``"type": "object"`` and an ``array``/``items`` wrapper; those
    are two of the shapes that used to be dropped on the floor. It is still not
    the complete gate: an array whose items are a ``prefixItems`` tuple rather
    than a single ``items`` schema, and a response modelled via
    ``additionalProperties`` as a map, are both still excluded here even though
    ``identity.py``'s ``_nested_objects`` recognizes both. No entity is lost to
    this on the current corpus; widening this gate to match is deliberately out
    of scope for this branch. A scalar or an empty object is still excluded
    outright either way: it has no fields, so there is nothing for the fallback
    to model and listing it would be noise.
    """
    inner = resp.get("items") if resp.get("type") == "array" else resp
    return isinstance(inner, dict) and bool(inner.get("properties"))


def _identity_responses(spec: OpenAPISpec) -> tuple[dict[str, dict], frozenset[str]]:
    """Success-response schemas keyed by operation id, plus the deferred subset.

    The strict gate — an explicit ``"type": "object"`` with non-empty
    ``properties`` — is what the deterministic rule is allowed to *decide*. A
    response that is entity-shaped but misses that gate used to be excluded from
    the map entirely, which meant it was neither derived nor undecidable: the
    fallback never saw it and its entity was silently lost. (Measured: two on
    tau2-airline, ``list_all_airports`` and ``search_direct_flight``, whose
    Airport and Flight entities are missing from the generated store today.)

    So they are returned in the map *and* named in the second element, for
    :func:`derive_identity` to route straight to ``undecidable``. Routing rather
    than widening the gate is the whole point: see that function for why the skip
    must be explicit and cannot be left to the rule declining on its own.
    """
    out: dict[str, dict] = {}
    deferred: set[str] = set()
    for op in spec.operations:
        resp = op.get_success_response_schema()
        if not isinstance(resp, dict):
            continue
        resp = _unwrap_returns(resp)
        props = resp.get("properties")
        if resp.get("type") == "object" and isinstance(props, dict) and props:
            out[op.operation_id] = resp
        elif _entity_shaped(resp):
            out[op.operation_id] = resp
            deferred.add(op.operation_id)
    return out, frozenset(deferred)


def _request_signatures(schema: object, path: str = "") -> list[tuple[str, frozenset]]:
    """``(path, scalar property names)`` for every object level of a request body.

    Descends into object-valued properties, array items and union variants, so a
    nested ``items: [{a, b}]`` yields its own signature under ``items[]`` — which
    is how an API usually *names* a referenced instance.
    """
    if not isinstance(schema, dict):
        return []
    if schema.get("type") == "array" or "items" in schema:
        return _request_signatures(schema.get("items"), f"{path}[]")
    variants = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(variants, list) and "properties" not in schema:
        return [s for v in variants for s in _request_signatures(v, path)]
    props = schema.get("properties")
    if not isinstance(props, dict):
        return []
    out: list[tuple[str, frozenset]] = []
    scalars = set()
    for name in sorted(props):
        prop = props[name]
        if not isinstance(prop, dict):
            continue
        nested = _request_signatures(prop, f"{path}.{name}" if path else name)
        if nested or prop.get("type") in ("object", "array") or "properties" in prop:
            out.extend(nested)
        else:
            scalars.add(name)
    if scalars:
        out.insert(0, (path, frozenset(scalars)))
    return out


def _returns_summary(op) -> str | None:
    """One line on what an operation returns: type plus description, or field names.

    A lookup returning a single field of an entity is evidence about that field
    which the entity's listing schema cannot give (a free-text value there means a
    listing's const on the same field is the listing's filter).
    """
    resp = op.get_success_response_schema()
    if not isinstance(resp, dict):
        return None
    resp = _unwrap_returns(resp)
    inner = resp.get("items") if resp.get("type") == "array" else resp
    props = inner.get("properties") if isinstance(inner, dict) else None
    if isinstance(props, dict) and props:
        shape = "{" + ", ".join(sorted(props)) + "}"
        return f"array of {shape}" if inner is not resp else f"object {shape}"
    kind = resp.get("type") or ("enum" if "enum" in resp else "value")
    desc = resp.get("description")
    return f"{kind}: {desc}" if desc else str(kind)


def addressing_evidence(
    spec: OpenAPISpec, leftovers: dict[str, dict]
) -> dict[str, list[dict]]:
    """How the API's operations name instances of each leftover shape.

    For every leftover, the request signatures (see :func:`_request_signatures`)
    made up **only** of that shape's own fields. A response schema says what a
    thing looks like; only the requests say what identifies it. Without this the
    fallback judged identity from responses alone (on tau2-airline it keyed
    ``flights`` by ``flight_number``, though every request naming one flight also
    passes ``date``).

    Evidence, not a decision: a search's criteria also qualify (they are fields
    of the item too), and telling a lookup from a search is left to the
    fallback, which sees each operation's summary and what it returns.
    """
    out: dict[str, list[dict]] = {}
    for name in sorted(leftovers):
        schema = leftovers[name]
        item = (
            schema.get("items")
            if isinstance(schema, dict) and schema.get("type") == "array"
            else schema
        )
        props = item.get("properties") if isinstance(item, dict) else None
        if not isinstance(props, dict) or not props:
            continue
        fields = set(props)
        seen: set[tuple[str, str, tuple]] = set()
        rows: list[dict] = []
        for op in spec.operations:
            for where, sig in _request_signatures(op.get_request_schema()):
                if not sig <= fields:
                    continue
                key = (op.operation_id, where, tuple(sorted(sig)))
                if key in seen:
                    continue
                seen.add(key)
                row = {
                    "operation": op.operation_id,
                    "summary": op.summary,
                    "parameter": where or "(request body)",
                    "fields": sorted(sig),
                }
                returns = _returns_summary(op)
                if returns:
                    row["returns"] = returns
                rows.append(row)
        if rows:
            out[name] = rows
    return out


def collect_sources(spec: OpenAPISpec, spec_dict: dict) -> SchemaSources:
    components = spec_dict.get("components", {}).get("schemas", {}) or {}
    if components:
        return SchemaSources(
            synthetic=False,
            identity=dict(components),
            enrich=dict(components),
        )
    identity, deferred = _identity_responses(spec)
    return SchemaSources(
        synthetic=True,
        identity=identity,
        enrich=dedup_by_value(inline_schema_evidence(spec)),
        deferred=deferred,
    )
