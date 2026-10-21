"""S3a: a `where` / sort path that exists in no record of the store is an error, not `[]`.

v1 retail shipped: `find_user_id_by_name_zip` queried `name.first` / `name.last`, the store holds
`name.first_name` / `name.last_name`, and the store silently matched nothing -- so every lookup
answered "user not found" for a user that exists. The fix acts only on that path: a path that
exists (in some record, or declared in the schema) and matches nothing still returns `[]`.
"""

from typing import Any

import pytest

from simulation_harness.state.errors import BadQueryError, UnknownFieldError
from simulation_harness.state.store import SimulationStore
from simulation_harness.state.tools import state_count, state_list

# Mirrors the retail shipped schema: `name` / `address` are bare objects, so their subfields
# are known only from the records.
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "users": {"type": "array", "items": {"$ref": "#/$defs/User"}},
        "audit": {"type": "array", "items": {"$ref": "#/$defs/Audit"}},
    },
    "$defs": {
        "User": {
            "type": "object",
            "x-primary-key": "user_id",
            "properties": {
                "user_id": {"type": "string"},
                "name": {"type": "object"},
                "address": {"type": "object"},
                "email": {"type": "string"},
                "nickname": {"type": "string"},
                "profile": {
                    "type": "object",
                    "properties": {"tier": {"type": "string"}},
                },
            },
        },
        "Audit": {
            "type": "object",
            "x-primary-key": "id",
            "properties": {"id": {"type": "string"}},
        },
    },
}

_SEED: dict[str, list[dict[str, Any]]] = {
    "users": [
        {
            "user_id": "noah_patel_6952",
            "name": {"first_name": "Noah", "last_name": "Patel"},
            "address": {"zip": "10108", "city": "New York"},
            "email": "noah@example.com",
        },
        {
            "user_id": "mei_kovacs_8020",
            "name": {"first_name": "Mei", "last_name": "Kovacs"},
            "address": {"zip": "28236", "city": "Charlotte"},
            "email": "mei@example.com",
            "legacy_flag": True,
        },
    ],
    "audit": [],
}


@pytest.fixture
def store() -> SimulationStore:
    return SimulationStore(_SCHEMA, _SEED, {"users": "user_id", "audit": "id"})


def _config(store: SimulationStore) -> dict[str, Any]:
    class _Registry:
        def for_thread(self, thread_id: str) -> SimulationStore:
            return store

    return {"configurable": {"store_registry": _Registry(), "thread_id": "default"}}


# --- the failure path -------------------------------------------------------------------


def test_where_on_path_in_no_record_is_an_error_listing_real_paths(
    store: SimulationStore,
) -> None:
    where = {
        "name.first": {"$ieq": "Noah"},
        "name.last": {"$ieq": "Patel"},
        "address.zip": "10108",
    }
    with pytest.raises(UnknownFieldError) as exc:
        store.list("users", where=where)
    err = exc.value
    assert isinstance(err, BadQueryError)  # still a bad query to any generic handler
    assert err.paths == ["name.first", "name.last"]
    for real in ("name.first_name", "name.last_name", "address.zip", "email"):
        assert real in err.fields
    assert "name.first_name" in str(err) and "'name.first'" in str(err)


def test_count_on_path_in_no_record_is_an_error(store: SimulationStore) -> None:
    with pytest.raises(UnknownFieldError):
        store.count("users", where={"name.first": "Noah"})


def test_sort_on_path_in_no_record_is_an_error_listing_real_paths(
    store: SimulationStore,
) -> None:
    with pytest.raises(UnknownFieldError) as exc:
        store.list("users", sort=[("name.first", "asc")])
    assert "name.first_name" in exc.value.fields


def test_state_list_tool_returns_unknown_field_error(store: SimulationStore) -> None:
    out = state_list(
        "users", where={"name.first": {"$ieq": "Noah"}}, config=_config(store)
    )
    assert isinstance(out, dict)
    assert out["error"] == "unknown_field"
    assert "name.first_name" in out["fields"]
    assert "name.first_name" in out["message"]
    count = state_count("users", where={"name.first": "x"}, config=_config(store))
    assert isinstance(count, dict) and count["error"] == "unknown_field"


# --- unchanged: everything that did not hit the bug -------------------------------------


def test_existing_path_matching_nothing_is_still_empty(store: SimulationStore) -> None:
    assert store.list("users", where={"name.first_name": {"$ieq": "Nobody"}}) == []
    assert store.count("users", where={"address.zip": "00000"}) == 0


def test_correct_paths_still_find_the_user(store: SimulationStore) -> None:
    rows = store.list(
        "users",
        where={
            "name.first_name": {"$ieq": "noah"},
            "name.last_name": {"$ieq": "patel"},
            "address.zip": "10108",
        },
    )
    assert [r["user_id"] for r in rows] == ["noah_patel_6952"]


def test_path_in_some_records_only_is_unchanged(store: SimulationStore) -> None:
    assert [r["user_id"] for r in store.list("users", where={"legacy_flag": True})] == [
        "mei_kovacs_8020"
    ]
    assert store.list("users", where={"legacy_flag": False}) == []


def test_schema_declared_path_no_record_has_is_unchanged(
    store: SimulationStore,
) -> None:
    """`nickname` / `profile.tier` are declared fields no seeded record carries yet."""
    assert store.list("users", where={"nickname": "Nono"}) == []
    assert store.list("users", where={"profile.tier": "gold"}) == []


def test_exists_predicates_are_unchanged(store: SimulationStore) -> None:
    assert len(store.list("users", where={"name.first": {"$exists": False}})) == 2
    assert store.list("users", where={"name.first": {"$exists": True}}) == []


def test_empty_store_is_unchanged(store: SimulationStore) -> None:
    assert store.list("audit", where={"anything.at.all": 1}) == []
    assert store.count("audit", where={"anything": 1}) == 0


def test_sort_on_existing_path_is_unchanged(store: SimulationStore) -> None:
    rows = store.list("users", sort=[("name.first_name", "asc")])
    assert [r["user_id"] for r in rows] == ["mei_kovacs_8020", "noah_patel_6952"]
    # A path some records lack keeps its old error (not the new one).
    with pytest.raises(BadQueryError) as exc:
        store.list("users", sort=[("legacy_flag", "asc")])
    assert not isinstance(exc.value, UnknownFieldError)


# --- S3b / E1: an unfiltered state_list too large to show is refused ---------------------


def _big_store(n: int) -> SimulationStore:
    seed = {
        "users": [
            {
                "user_id": f"user_{i:04d}",
                "name": {"first_name": f"F{i}", "last_name": f"L{i}"},
                "address": {"zip": f"{i:05d}"},
                "email": f"user{i}@example.com",
                "bio": "x" * 400,
            }
            for i in range(n)
        ],
        "audit": [],
    }
    return SimulationStore(_SCHEMA, seed, {"users": "user_id", "audit": "id"})


def test_e1_threshold_is_deepagents_eviction_limit() -> None:
    import inspect

    from deepagents.middleware.filesystem import (
        NUM_CHARS_PER_TOKEN,
        FilesystemMiddleware,
    )

    from simulation_harness.state.tools import eviction_threshold_chars

    default = (
        inspect.signature(FilesystemMiddleware.__init__)
        .parameters["tool_token_limit_before_evict"]
        .default
    )
    assert eviction_threshold_chars() == NUM_CHARS_PER_TOKEN * default == 80_000


def test_e1_unfiltered_list_over_eviction_limit_is_refused() -> None:
    store = _big_store(300)  # ~170k chars serialized
    out = state_list("users", config=_config(store))
    assert isinstance(out, dict)
    assert out["error"] == "result_too_large"
    assert out["records"] == 300
    assert out["chars"] > 80_000
    assert "name.first_name" in out["fields"] and "email" in out["fields"]
    assert "where" in out["message"]
    # An empty where is no filter either.
    again = state_list("users", where={}, config=_config(store))
    assert isinstance(again, dict) and again["error"] == "result_too_large"


def test_e1_filtered_limited_or_small_lists_are_unchanged() -> None:
    store = _big_store(300)
    cfg = _config(store)
    # Filtered: answered as before, however large.
    rows = state_list("users", where={"bio": "x" * 400}, config=cfg)
    assert isinstance(rows, list) and len(rows) == 300
    # Limited: answered as before.
    rows = state_list("users", limit=250, config=cfg)
    assert isinstance(rows, list) and len(rows) == 250
    # Small store, unfiltered: answered as before.
    small = _big_store(20)
    rows = state_list("users", config=_config(small))
    assert isinstance(rows, list) and len(rows) == 20
