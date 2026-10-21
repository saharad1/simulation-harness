"""Tests for state tools."""

import json
import logging

import pytest

from simulation_harness.state import StoreRegistry, create_state_tools
from simulation_harness.state.tools import EVICT_CHARS, rows_text
from pathlib import Path
from typing import Any

# ``skill_dir`` is provided by tests/unit/state/conftest.py.


@pytest.fixture
def registry(skill_dir: Path) -> StoreRegistry:
    """Create a StoreRegistry instance."""
    return StoreRegistry(skill_dir)


@pytest.fixture
def mock_config(registry: StoreRegistry) -> dict[str, Any]:
    """Create a mock RunnableConfig."""
    return {"configurable": {"store_registry": registry, "thread_id": "test_thread"}}


def _rows(result: Any) -> list[dict]:
    """state_list returns its rows as text, one record per line; parse it back."""
    assert isinstance(result, str), result
    return json.loads(result)


@pytest.fixture
def tools() -> Any:
    """Create the state tools."""
    return create_state_tools()


class TestToolCreation:
    """Test tool creation."""

    def test_create_tools(self, tools: Any) -> None:
        """Test that create_state_tools returns 6 tools."""
        assert len(tools) == 6

        tool_names = [t.name for t in tools]
        assert "state_get" in tool_names
        assert "state_list" in tool_names
        assert "state_count" in tool_names
        assert "state_insert" in tool_names
        assert "state_update" in tool_names
        assert "state_delete" in tool_names


class TestStateGet:
    """Test state_get tool."""

    def test_get_existing_entity(self, tools: Any, mock_config: dict[str, Any]) -> None:
        """Test getting an existing entity."""
        tool = next(t for t in tools if t.name == "state_get")

        result = tool.func(store="restaurants", id="rest_001", config=mock_config)

        assert "error" not in result
        assert result["id"] == "rest_001"
        assert result["name"] == "The Italian Corner"

    def test_get_fields_returns_only_those_fields(
        self, tools: Any, mock_config: dict[str, Any]
    ) -> None:
        tool = next(t for t in tools if t.name == "state_get")

        result = tool.func(
            store="restaurants", id="rest_001", fields=["name"], config=mock_config
        )

        assert result == {"name": "The Italian Corner"}

    def test_get_nonexistent_entity(
        self, tools: Any, mock_config: dict[str, Any]
    ) -> None:
        """Test getting a nonexistent entity returns error."""
        tool = next(t for t in tools if t.name == "state_get")

        result = tool.func(store="restaurants", id="nonexistent", config=mock_config)

        assert "error" in result
        assert result["error"] == "not_found"

    def test_get_unknown_store(self, tools: Any, mock_config: dict[str, Any]) -> None:
        """Test getting from unknown store returns error."""
        tool = next(t for t in tools if t.name == "state_get")

        result = tool.func(store="unknown_store", id="some_id", config=mock_config)

        assert "error" in result
        assert result["error"] == "unknown_store"


class TestStateList:
    """Test state_list tool."""

    def test_list_all(self, tools: Any, mock_config: dict[str, Any]) -> None:
        """Test listing all entities."""
        tool = next(t for t in tools if t.name == "state_list")

        result = _rows(tool.func(store="restaurants", config=mock_config))

        assert len(result) >= 3

    def test_list_with_filter(self, tools: Any, mock_config: dict[str, Any]) -> None:
        """Test listing with a filter."""
        tool = next(t for t in tools if t.name == "state_list")

        result = _rows(
            tool.func(
                store="restaurants", where={"cuisine": "Italian"}, config=mock_config
            )
        )

        assert all(r["cuisine"] == "Italian" for r in result)

    def test_list_with_sort(self, tools: Any, mock_config: dict[str, Any]) -> None:
        """Test listing with sort parameter (verifies list-to-tuple conversion)."""
        tool = next(t for t in tools if t.name == "state_list")

        # Test sorting by name ascending
        result = _rows(
            tool.func(store="restaurants", sort=[["name", "asc"]], config=mock_config)
        )

        assert len(result) >= 3
        # Verify sorting worked (names should be in ascending order)
        names = [r["name"] for r in result]
        assert names == sorted(names)

        # Test sorting by rating descending
        result = _rows(
            tool.func(
                store="restaurants", sort=[["rating", "desc"]], config=mock_config
            )
        )

        ratings = [r["rating"] for r in result]
        assert ratings == sorted(ratings, reverse=True)

    def test_list_with_unknown_where_field_tells_the_agent(
        self, tools: Any, mock_config: dict[str, Any]
    ) -> None:
        """The simulator agent sees an unknown field as a bad_query naming the real
        fields, not as an empty result it would report as "not found"."""
        tool = next(t for t in tools if t.name == "state_list")

        result = tool.func(
            store="restaurants", where={"location.town": "Boston"}, config=mock_config
        )

        assert isinstance(result, dict)
        assert result["error"] == "bad_query"
        assert "location.town" in result["message"]
        assert "city" in result["message"]

    def test_list_with_malformed_sort(
        self, tools: Any, mock_config: dict[str, Any]
    ) -> None:
        """Test that malformed sort parameter returns clear error."""
        tool = next(t for t in tools if t.name == "state_list")

        # Test with single element (missing direction)
        result = tool.func(store="restaurants", sort=[["name"]], config=mock_config)

        assert isinstance(result, dict)
        assert result["error"] == "bad_query"
        assert "must be [field, direction] pairs" in result["message"]

        # Test with too many elements
        result = tool.func(
            store="restaurants", sort=[["name", "asc", "extra"]], config=mock_config
        )

        assert isinstance(result, dict)
        assert result["error"] == "bad_query"
        assert "must be [field, direction] pairs" in result["message"]

    def test_list_with_limit(self, tools: Any, mock_config: dict[str, Any]) -> None:
        """Test listing with a limit."""
        tool = next(t for t in tools if t.name == "state_list")

        result = _rows(tool.func(store="restaurants", limit=2, config=mock_config))

        assert len(result) == 2

    def test_list_fields_returns_only_those_fields(
        self, tools: Any, mock_config: dict[str, Any]
    ) -> None:
        """A projection keeps a listing small: names and ids, not every nested field."""
        tool = next(t for t in tools if t.name == "state_list")

        result = _rows(
            tool.func(store="restaurants", fields=["id", "name"], config=mock_config)
        )

        assert len(result) >= 3
        assert all(set(r) == {"id", "name"} for r in result)

    def test_list_fields_unknown_field_names_the_real_ones(
        self, tools: Any, mock_config: dict[str, Any]
    ) -> None:
        tool = next(t for t in tools if t.name == "state_list")

        result = tool.func(store="restaurants", fields=["nickname"], config=mock_config)

        assert isinstance(result, dict)
        assert result["error"] == "bad_query"
        assert "nickname" in result["message"] and "cuisine" in result["message"]


class TestLargeResults:
    """A result over the agent's tool-result limit is saved to a file and read back with
    read_file, which pages by LINE and caps each read at EVICT_CHARS. One record per line
    keeps every record whole and reachable; a single-line array was cut mid-list (retail:
    41 of 50 products, the rest silently gone)."""

    def test_rows_text_is_one_record_per_line(self) -> None:
        rows = [{"product_id": f"{i:010d}", "blob": "x" * 2000} for i in range(50)]

        text = rows_text(rows)

        assert len(text) > EVICT_CHARS  # large enough to be offloaded
        lines = text.splitlines()
        assert len(lines) == 52  # "[", 50 records, "]"
        assert all(len(line) < EVICT_CHARS for line in lines)
        for line in lines[1:-1]:
            json.loads(line.rstrip(","))  # each record parses on its own line
        assert json.loads(text) == rows

    def test_large_listing_is_logged(
        self, tools: Any, mock_config: dict[str, Any], caplog: Any
    ) -> None:
        tool = next(t for t in tools if t.name == "state_list")
        get = next(t for t in tools if t.name == "state_get")
        big = {**get.func(store="restaurants", id="rest_001", config=mock_config)}
        big.update(id="rest_big", name="y" * EVICT_CHARS)
        inserted = next(t for t in tools if t.name == "state_insert").func(
            store="restaurants", entity=big, config=mock_config
        )
        assert "error" not in inserted, inserted

        with caplog.at_level(logging.WARNING, logger="simulation_harness.state.tools"):
            tool.func(store="restaurants", config=mock_config)

        assert any("offload" in r.getMessage() for r in caplog.records)


class TestStateCount:
    """Test state_count tool."""

    def test_count_all(self, tools: Any, mock_config: dict[str, Any]) -> None:
        """Test counting all entities."""
        tool = next(t for t in tools if t.name == "state_count")

        result = tool.func(store="restaurants", config=mock_config)

        assert isinstance(result, int)
        assert result >= 3

    def test_count_with_filter(self, tools: Any, mock_config: dict[str, Any]) -> None:
        """Test counting with a filter."""
        tool = next(t for t in tools if t.name == "state_count")

        result = tool.func(
            store="restaurants", where={"location.city": "Boston"}, config=mock_config
        )

        assert isinstance(result, int)
        assert result >= 1


class TestStateInsert:
    """Test state_insert tool."""

    def test_insert_valid_entity(self, tools: Any, mock_config: dict[str, Any]) -> None:
        """Test inserting a valid entity."""
        tool = next(t for t in tools if t.name == "state_insert")

        entity = {
            "id": "rest_test",
            "name": "Test Restaurant",
            "cuisine": "Test",
            "price_tier": 1,
            "rating": 3.0,
            "location": {
                "latitude": 42.0,
                "longitude": -71.0,
                "address": "Test",
                "city": "Test",
                "state": "MA",
                "postal_code": "00000",
            },
            "phone": "+1-555-000-0000",
            "accepts_reservations": True,
        }

        result = tool.func(store="restaurants", entity=entity, config=mock_config)

        assert "error" not in result
        assert result["id"] == "rest_test"

    def test_insert_duplicate_key(
        self, tools: Any, mock_config: dict[str, Any]
    ) -> None:
        """Test inserting a duplicate key returns error."""
        tool = next(t for t in tools if t.name == "state_insert")

        entity = {
            "id": "rest_001",  # Already exists
            "name": "Duplicate",
            "cuisine": "Test",
            "price_tier": 1,
            "rating": 3.0,
            "location": {
                "latitude": 42.0,
                "longitude": -71.0,
                "address": "Test",
                "city": "Test",
                "state": "MA",
                "postal_code": "00000",
            },
            "phone": "+1-555-000-0000",
            "accepts_reservations": True,
        }

        result = tool.func(store="restaurants", entity=entity, config=mock_config)

        assert "error" in result
        assert result["error"] == "duplicate_key"

    def test_insert_invalid_entity(
        self, tools: Any, mock_config: dict[str, Any]
    ) -> None:
        """Test inserting an invalid entity returns error."""
        tool = next(t for t in tools if t.name == "state_insert")

        entity = {
            "id": "rest_invalid",
            "name": "Invalid",
            # Missing required fields
        }

        result = tool.func(store="restaurants", entity=entity, config=mock_config)

        assert "error" in result
        assert result["error"] == "validation_error"


class TestStateUpdate:
    """Test state_update tool."""

    def test_update_existing_entity(
        self, tools: Any, mock_config: dict[str, Any]
    ) -> None:
        """Test updating an existing entity."""
        tool = next(t for t in tools if t.name == "state_update")

        result = tool.func(
            store="restaurants",
            id="rest_001",
            patch={"rating": 4.9},
            config=mock_config,
        )

        assert "error" not in result
        assert result["rating"] == 4.9
        assert result["name"] == "The Italian Corner"  # Other fields unchanged

    def test_update_nonexistent_entity(
        self, tools: Any, mock_config: dict[str, Any]
    ) -> None:
        """Test updating a nonexistent entity returns error."""
        tool = next(t for t in tools if t.name == "state_update")

        result = tool.func(
            store="restaurants",
            id="nonexistent",
            patch={"rating": 5.0},
            config=mock_config,
        )

        assert "error" in result
        assert result["error"] == "not_found"


class TestStateDelete:
    """Test state_delete tool."""

    def test_delete_existing_entity(
        self, tools: Any, mock_config: dict[str, Any]
    ) -> None:
        """Test deleting an existing entity."""
        # First insert a test entity
        insert_tool = next(t for t in tools if t.name == "state_insert")
        entity = {
            "id": "rest_delete_test",
            "name": "To Delete",
            "cuisine": "Test",
            "price_tier": 1,
            "rating": 3.0,
            "location": {
                "latitude": 42.0,
                "longitude": -71.0,
                "address": "Test",
                "city": "Test",
                "state": "MA",
                "postal_code": "00000",
            },
            "phone": "+1-555-000-0000",
            "accepts_reservations": True,
        }
        insert_tool.func(store="restaurants", entity=entity, config=mock_config)

        # Now delete it
        delete_tool = next(t for t in tools if t.name == "state_delete")
        result = delete_tool.func(
            store="restaurants", id="rest_delete_test", config=mock_config
        )

        assert "error" not in result
        assert result["id"] == "rest_delete_test"

        # Verify it's gone
        get_tool = next(t for t in tools if t.name == "state_get")
        get_result = get_tool.func(
            store="restaurants", id="rest_delete_test", config=mock_config
        )
        assert "error" in get_result

    def test_delete_nonexistent_entity(
        self, tools: Any, mock_config: dict[str, Any]
    ) -> None:
        """Test deleting a nonexistent entity returns error."""
        tool = next(t for t in tools if t.name == "state_delete")

        result = tool.func(store="restaurants", id="nonexistent", config=mock_config)

        assert "error" in result
        assert result["error"] == "not_found"


class TestThreadIsolation:
    """Test that tools respect thread isolation."""

    def test_different_threads_see_different_data(
        self, tools: Any, registry: StoreRegistry
    ) -> None:
        """Test that different threads have isolated data."""
        config1 = {
            "configurable": {"store_registry": registry, "thread_id": "thread_1"}
        }
        config2 = {
            "configurable": {"store_registry": registry, "thread_id": "thread_2"}
        }

        insert_tool = next(t for t in tools if t.name == "state_insert")
        get_tool = next(t for t in tools if t.name == "state_get")

        # Insert into thread 1
        entity = {
            "id": "rest_thread1",
            "name": "Thread 1 Only",
            "cuisine": "Test",
            "price_tier": 1,
            "rating": 3.0,
            "location": {
                "latitude": 42.0,
                "longitude": -71.0,
                "address": "Test",
                "city": "Test",
                "state": "MA",
                "postal_code": "00000",
            },
            "phone": "+1-555-000-0000",
            "accepts_reservations": True,
        }
        insert_tool.func(store="restaurants", entity=entity, config=config1)

        # Verify it's in thread 1
        result1 = get_tool.func(store="restaurants", id="rest_thread1", config=config1)
        assert "error" not in result1

        # Verify it's NOT in thread 2
        result2 = get_tool.func(store="restaurants", id="rest_thread1", config=config2)
        assert "error" in result2
        assert result2["error"] == "not_found"


# Made with Bob
