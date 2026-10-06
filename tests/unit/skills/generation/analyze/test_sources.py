"""Tests for schema-source selection feeding the identity rule."""

from __future__ import annotations

from simulation_harness.openapi.parser import OpenAPISpec
from simulation_harness.skills.generation.stages.analyze.sources import (
    _unwrap_returns,
    addressing_evidence,
    collect_sources,
    dedup_by_value,
)


def test_dedup_by_value_drops_identical_shapes() -> None:
    shape = {"type": "object", "properties": {"order_id": {"type": "string"}}}
    out = dedup_by_value({"b__response": dict(shape), "a__response": dict(shape)})
    # Sorted iteration means the alphabetically-first name survives, so the
    # survivor does not depend on how the input map was built.
    assert list(out) == ["a__response"]


def test_dedup_by_value_keeps_distinct_shapes() -> None:
    out = dedup_by_value(
        {
            "a": {"properties": {"x": {"type": "string"}}},
            "b": {"properties": {"y": {"type": "string"}}},
        }
    )
    assert list(out) == ["a", "b"]


def test_collect_sources_named_spec_uses_components() -> None:
    spec_dict = {
        "openapi": "3.0.0",
        "info": {"title": "T", "version": "1"},
        "paths": {},
        "components": {"schemas": {"Order": {"properties": {"id": {}}}}},
    }
    sources = collect_sources(OpenAPISpec(spec_dict), spec_dict)
    assert sources.synthetic is False
    assert list(sources.identity) == ["Order"]
    assert list(sources.enrich) == ["Order"]


def test_collect_sources_synthetic_identity_is_responses_only() -> None:
    paths = {
        "/create": {
            "post": {
                "operationId": "create_order",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"note": {"type": "string"}},
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"order_id": {"type": "string"}},
                                }
                            }
                        }
                    }
                },
            }
        }
    }
    spec_dict = {
        "openapi": "3.0.0",
        "info": {"title": "T", "version": "1"},
        "paths": paths,
    }
    sources = collect_sources(OpenAPISpec(spec_dict), spec_dict)
    assert sources.synthetic is True
    # A request body is a projection of an entity, not the entity — it must not
    # reach the identity rule.
    assert list(sources.identity) == ["create_order"]
    assert sorted(sources.enrich) == ["create_order__request", "create_order__response"]


def test_collect_sources_synthetic_unwraps_returns_envelope() -> None:
    inner = {"type": "object", "properties": {"user_id": {"type": "string"}}}
    paths = {
        "/get": {
            "post": {
                "operationId": "get_user",
                "responses": {
                    "200": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"returns": inner},
                                }
                            }
                        }
                    }
                },
            }
        }
    }
    spec_dict = {
        "openapi": "3.0.0",
        "info": {"title": "T", "version": "1"},
        "paths": paths,
    }
    sources = collect_sources(OpenAPISpec(spec_dict), spec_dict)
    assert sources.identity["get_user"] == inner


def test_collect_sources_synthetic_skips_non_object_responses() -> None:
    paths = {
        "/ping": {
            "get": {
                "operationId": "ping",
                "responses": {
                    "200": {
                        "content": {"application/json": {"schema": {"type": "string"}}}
                    }
                },
            }
        }
    }
    spec_dict = {
        "openapi": "3.0.0",
        "info": {"title": "T", "version": "1"},
        "paths": paths,
    }
    sources = collect_sources(OpenAPISpec(spec_dict), spec_dict)
    assert sources.identity == {}


def _room_tool_spec() -> dict:
    """A tool-style spec where rooms are only ever *returned* by a search, but every
    request that names one room passes (hotel_id, night)."""

    def op(op_id: str, summary: str, body: dict, returns: dict | None = None) -> dict:
        resp: dict = {"description": "ok"}
        if returns is not None:
            resp["content"] = {
                "application/json": {"schema": {"properties": {"returns": returns}}}
            }
        return {
            "post": {
                "operationId": op_id,
                "summary": summary,
                "requestBody": {"content": {"application/json": {"schema": body}}},
                "responses": {"200": resp},
            }
        }

    s = {"type": "string"}
    room_ref = {
        "anyOf": [
            {"type": "object", "properties": {"hotel_id": s, "night": s}},
            {"type": "object", "additionalProperties": True},
        ]
    }
    return {
        "openapi": "3.0.0",
        "info": {"title": "Hotels", "version": "1"},
        "paths": {
            "/search": op(
                "search_rooms",
                "Search for open rooms.",
                {"type": "object", "properties": {"city": s, "night": s}},
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "hotel_id": s,
                            "city": s,
                            "night": s,
                            "status": {"const": "open"},
                        },
                    },
                },
            ),
            "/status": op(
                "get_room_status",
                "Get the status of a room.",
                {"type": "object", "properties": {"hotel_id": s, "night": s}},
                {"type": "string", "description": "The status of the room."},
            ),
            "/book": op(
                "book_stay",
                "Book a stay.",
                {
                    "type": "object",
                    "properties": {
                        "guest_id": s,
                        "rooms": {"type": "array", "items": room_ref},
                    },
                },
            ),
        },
    }


def test_addressing_evidence_finds_lookups_and_nested_references() -> None:
    spec_dict = _room_tool_spec()
    leftover = _unwrap_returns(
        spec_dict["paths"]["/search"]["post"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
    )
    rows = addressing_evidence(OpenAPISpec(spec_dict), {"search_rooms": leftover})[
        "search_rooms"
    ]
    got = {(r["operation"], r["parameter"], tuple(r["fields"])) for r in rows}
    # The lookup, and the reference nested in a union-typed list item.
    assert ("get_room_status", "(request body)", ("hotel_id", "night")) in got
    assert ("book_stay", "rooms[]", ("hotel_id", "night")) in got
    # A search's criteria are fields of the item too, so they are evidence as well.
    assert ("search_rooms", "(request body)", ("city", "night")) in got
    # `guest_id` is not a room field, so book's top-level body is not about rooms.
    assert ("book_stay", "(request body)") not in {(o, w) for o, w, _ in got}
    # What each operation returns rides along: a lookup returning one free-text
    # field is the evidence that relaxes a listing's const on that field.
    by_op = {r["operation"]: r.get("returns") for r in rows}
    assert by_op["get_room_status"] == "string: The status of the room."
    assert by_op["search_rooms"] == "array of {city, hotel_id, night, status}"
    assert by_op["book_stay"] is None


def test_addressing_evidence_skips_shapes_without_properties() -> None:
    spec = OpenAPISpec(_room_tool_spec())
    assert addressing_evidence(spec, {"x": {"type": "string"}}) == {}
