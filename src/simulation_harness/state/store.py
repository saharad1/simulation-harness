"""In-memory state store for simulation sessions."""

import builtins
import copy
from typing import Any

from .errors import (
    NotFoundError,
    DuplicateKeyError,
    UnknownStoreError,
    BadQueryError,
    UnknownFieldError,
)
from .validator import validate_entity, validate_primary_key_present

#: How many of a store's field paths an UnknownFieldError lists (most common first).
MAX_LISTED_FIELDS = 60
#: How deep into nested objects field paths are collected.
_MAX_PATH_DEPTH = 4


class SimulationStore:
    """Per-session in-memory state store with schema validation.

    Stores are keyed by name (e.g., "restaurants", "reservations").
    Each store holds entities keyed by their primary key.
    All operations are validated against the schema.
    """

    def __init__(self, schema: dict, seed: dict, primary_keys: dict[str, str]) -> None:
        """Initialize the store with schema and seed data.

        Args:
            schema: The complete schema.json document
            seed: The seed data from db.json (store_name -> list of entities)
            primary_keys: Map of store_name -> primary_key_field_name
        """
        self._schema = schema
        self._primary_keys = primary_keys

        # Resolve $defs for each store - but keep them with full schema context
        self._defs: dict[str, dict] = {}
        for store_name in seed.keys():
            # Get the schema reference for this store's items
            store_schema = schema.get("properties", {}).get(store_name, {})
            items_ref = store_schema.get("items", {}).get("$ref", "")

            if items_ref.startswith("#/$defs/"):
                def_name = items_ref.split("/")[-1]
                # Store the entity schema but we'll pass full schema for validation
                self._defs[store_name] = schema.get("$defs", {}).get(def_name, {})

        # Initialize data storage: store -> pk -> entity
        self._data: dict[str, dict[str, dict]] = {}
        self._seed = seed

        # Load seed data
        self.reset()

    def reset(self) -> None:
        """Reset the store to its seed state."""
        self._data = {}

        for store_name, entities in self._seed.items():
            pk_field = self._primary_keys.get(store_name, "id")
            self._data[store_name] = {}

            for entity in entities:
                # Deep copy to prevent mutation of seed data
                entity_copy = copy.deepcopy(entity)
                pk_value = entity_copy.get(pk_field)
                if pk_value is not None:
                    self._data[store_name][str(pk_value)] = entity_copy

    def _check_store_exists(self, store: str) -> None:
        """Validate that a store exists.

        Args:
            store: The store name to check

        Raises:
            UnknownStoreError: If the store doesn't exist
        """
        if store not in self._data:
            raise UnknownStoreError(store, list(self._data.keys()))

    def primary_key(self, store: str) -> str:
        """The primary-key field name ``insert`` reads for ``store``."""
        return self._primary_keys.get(store, "id")

    def get(self, store: str, id: str) -> dict | None:
        """Get an entity by its primary key.

        Args:
            store: The store name
            id: The primary key value

        Returns:
            Deep copy of the entity, or None if not found

        Raises:
            UnknownStoreError: If the store doesn't exist
        """
        self._check_store_exists(store)
        entity = self._data[store].get(str(id))
        return copy.deepcopy(entity) if entity is not None else None

    def list(
        self,
        store: str,
        where: dict | None = None,
        sort: list[tuple[str, str]] | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """List entities from a store with optional filtering and sorting.

        Args:
            store: The store name
            where: Optional filter predicates (see _matches_where for DSL)
            sort: Optional list of (field_path, direction) tuples where direction is "asc" or "desc"
            limit: Optional maximum number of results

        Returns:
            List of deep-copied entities matching the criteria

        Raises:
            UnknownStoreError: If the store doesn't exist
            BadQueryError: If where clause or sort is malformed
        """
        self._check_store_exists(store)
        self._check_paths(
            store,
            [
                path
                for path, predicate in (where or {}).items()
                if not (isinstance(predicate, dict) and "$exists" in predicate)
            ]
            + [field_path for field_path, _ in (sort or [])],
        )

        # Get all entities
        entities = list(self._data[store].values())

        # Apply filter
        if where:
            entities = [e for e in entities if self._matches_where(e, where)]

        # Apply sort (default: PK ascending)
        if sort:
            try:
                for field_path, direction in reversed(sort):
                    reverse = direction.lower() == "desc"
                    entities.sort(
                        key=lambda e: self._get_nested_value(e, field_path),
                        reverse=reverse,
                    )
            except (KeyError, TypeError) as e:
                raise BadQueryError(f"Invalid sort specification: {e}")
        else:
            # Default: sort by primary key ascending
            pk_field = self._primary_keys.get(store, "id")
            entities.sort(key=lambda e: e.get(pk_field, ""))

        # Apply limit
        if limit is not None and limit > 0:
            entities = entities[:limit]

        # Return deep copies
        return [copy.deepcopy(e) for e in entities]

    def count(self, store: str, where: dict | None = None) -> int:
        """Count entities in a store with optional filtering.

        Args:
            store: The store name
            where: Optional filter predicates

        Returns:
            Count of matching entities

        Raises:
            UnknownStoreError: If the store doesn't exist
            BadQueryError: If where clause is malformed
        """
        self._check_store_exists(store)
        self._check_paths(
            store,
            [
                path
                for path, predicate in (where or {}).items()
                if not (isinstance(predicate, dict) and "$exists" in predicate)
            ],
        )

        if where is None:
            return len(self._data[store])

        return sum(
            1 for e in self._data[store].values() if self._matches_where(e, where)
        )

    def insert(self, store: str, entity: dict) -> dict:
        """Insert a new entity into a store.

        Args:
            store: The store name
            entity: The entity to insert

        Returns:
            Deep copy of the inserted entity

        Raises:
            UnknownStoreError: If the store doesn't exist
            ValidationError: If the entity fails schema validation
            DuplicateKeyError: If an entity with this PK already exists
        """
        self._check_store_exists(store)

        pk_field = self.primary_key(store)

        # Validate primary key is present
        validate_primary_key_present(store, entity, pk_field)

        pk_value = str(entity[pk_field])

        # Check for duplicate
        if pk_value in self._data[store]:
            raise DuplicateKeyError(store, pk_value)

        # Validate against schema
        entity_schema = self._defs.get(store, {})
        if entity_schema:
            validate_entity(store, entity, entity_schema, self._schema)

        # Insert (deep copy to prevent external mutation)
        entity_copy = copy.deepcopy(entity)
        self._data[store][pk_value] = entity_copy

        return copy.deepcopy(entity_copy)

    def update(self, store: str, id: str, patch: dict) -> dict:
        """Update an existing entity by merging a patch.

        Args:
            store: The store name
            id: The primary key value
            patch: Dictionary of fields to update

        Returns:
            Deep copy of the updated entity

        Raises:
            UnknownStoreError: If the store doesn't exist
            NotFoundError: If the entity doesn't exist
            ValidationError: If the merged entity fails schema validation
        """
        self._check_store_exists(store)

        entity = self._data[store].get(str(id))
        if entity is None:
            raise NotFoundError(store, str(id))

        # Merge patch into entity
        merged = {**entity, **patch}

        # Validate the merged entity
        entity_schema = self._defs.get(store, {})
        if entity_schema:
            validate_entity(store, merged, entity_schema, self._schema)

        # Update in place
        self._data[store][str(id)] = merged

        return copy.deepcopy(merged)

    def delete(self, store: str, id: str) -> dict:
        """Delete an entity by its primary key.

        Args:
            store: The store name
            id: The primary key value

        Returns:
            Deep copy of the deleted entity

        Raises:
            UnknownStoreError: If the store doesn't exist
            NotFoundError: If the entity doesn't exist
        """
        self._check_store_exists(store)

        entity = self._data[store].get(str(id))
        if entity is None:
            raise NotFoundError(store, str(id))

        # Remove and return
        deleted = self._data[store].pop(str(id))
        return copy.deepcopy(deleted)

    def snapshot(self) -> dict:
        """Get a complete snapshot of the current state.

        Returns:
            Dictionary mapping store names to lists of entities
        """
        result = {}
        for store_name, entities in self._data.items():
            result[store_name] = [copy.deepcopy(e) for e in entities.values()]
        return result

    def checkpoint(self) -> dict:
        """Capture the current state for a later restore().

        Unlike snapshot() (store -> list of entities, a read-only API view),
        this preserves the internal store -> pk -> entity shape so restore()
        can swap it back in directly.

        Returns:
            A deep copy of the internal data, suitable for restore()
        """
        return copy.deepcopy(self._data)

    def restore(self, cp: dict) -> None:
        """Restore the store to a previously captured checkpoint.

        Args:
            cp: A checkpoint previously returned by checkpoint()
        """
        self._data = copy.deepcopy(cp)

    def _check_paths(self, store: str, paths: builtins.list[str]) -> None:
        """Raise UnknownFieldError for query paths that exist in no record of ``store``.

        Only the failure path: a path that resolves in at least one record, or that the
        store's schema declares (a field no record carries yet), is left to the query as
        before, and so is any path on an empty store. ``$exists`` predicates are exempt
        by the caller -- asking whether a field exists is a valid question either way.
        """
        entities = self._data[store].values()
        if not entities:
            return
        unknown: builtins.list[str] = []
        for path in paths:
            if not isinstance(path, str) or path in unknown:
                continue
            if any(self._resolves(e, path) for e in entities):
                continue
            if self._declared(store, path):
                continue
            unknown.append(path)
        if unknown:
            fields, more = self.field_paths(store)
            raise UnknownFieldError(store, unknown, fields, more)

    def field_paths(
        self, store: str, limit: int = MAX_LISTED_FIELDS
    ) -> tuple[builtins.list[str], int]:
        """The store's dotted field paths, most common across records first.

        Nested objects are walked (up to a few levels); arrays are not, since a ``where``
        path cannot traverse them. Returns ``(paths[:limit], number left out)``. Map-like
        objects keyed by ids contribute paths seen in one record each, so they sort last.
        """
        counts: dict[str, int] = {}

        def walk(obj: dict, prefix: str, depth: int) -> None:
            for key, value in obj.items():
                path = f"{prefix}{key}"
                counts[path] = counts.get(path, 0) + 1
                if isinstance(value, dict) and depth < _MAX_PATH_DEPTH:
                    walk(value, f"{path}.", depth + 1)

        for entity in self._data.get(store, {}).values():
            walk(entity, "", 1)
        ordered = sorted(counts, key=lambda p: (-counts[p], p))
        return ordered[:limit], max(0, len(ordered) - limit)

    def _resolves(self, entity: dict, path: str) -> bool:
        try:
            self._get_nested_value(entity, path)
        except KeyError:
            return False
        return True

    def _declared(self, store: str, path: str) -> bool:
        """Whether every segment of ``path`` is a declared ``properties`` entry.

        An object declared open by ``additionalProperties`` (a map such as retail
        ``Item.options`` or ``payment_methods``) declares any key below it. A bare
        ``{"type": "object"}`` does not: retail ``User.name`` is one, and ``name.first``
        against records holding ``name.first_name`` is exactly the S3a bug.
        """
        node: Any = self._defs.get(store)
        for part in path.split("."):
            node = self._resolve_ref(node)
            if not isinstance(node, dict):
                return False
            props = node.get("properties")
            if isinstance(props, dict) and part in props:
                node = props[part]
                continue
            extra = node.get("additionalProperties")
            if extra is None or extra is False:
                return False
            node = extra if isinstance(extra, dict) else {}
        return True

    def _resolve_ref(self, node: Any) -> Any:
        if isinstance(node, dict) and isinstance(node.get("$ref"), str):
            ref = node["$ref"]
            if ref.startswith("#/$defs/"):
                return self._schema.get("$defs", {}).get(ref.split("/")[-1], {})
        return node

    def _matches_where(self, entity: dict, where: dict) -> bool:
        """Check if an entity matches a where clause.

        Where clause DSL:
            {"field.path": value}                    # equality
            {"field.path": {"$ieq": "value"}}        # case-insensitive equality
            {"field.path": {"$in": [1, 2, 3]}}       # membership
            {"field.path": {"$gt": 5}}               # greater than
            {"field.path": {"$gte": 5}}              # greater than or equal
            {"field.path": {"$lt": 5}}               # less than
            {"field.path": {"$lte": 5}}              # less than or equal
            {"field.path": {"$exists": true}}        # field exists

        Multiple keys are ANDed together.

        Args:
            entity: The entity to check
            where: The where clause

        Returns:
            True if the entity matches all predicates

        Raises:
            BadQueryError: If the where clause is malformed
        """
        for field_path, predicate in where.items():
            try:
                value = self._get_nested_value(entity, field_path)
            except KeyError:
                # Field doesn't exist
                if isinstance(predicate, dict) and predicate.get("$exists") is False:
                    continue
                return False

            # Handle different predicate types
            if isinstance(predicate, dict):
                # Operator-based predicate
                if "$ieq" in predicate:
                    # Case-insensitive equality
                    if not isinstance(value, str) or not isinstance(
                        predicate["$ieq"], str
                    ):
                        return False
                    if value.lower() != predicate["$ieq"].lower():
                        return False

                elif "$in" in predicate:
                    # Membership
                    if value not in predicate["$in"]:
                        return False

                elif "$gt" in predicate:
                    if not (value > predicate["$gt"]):
                        return False

                elif "$gte" in predicate:
                    if not (value >= predicate["$gte"]):
                        return False

                elif "$lt" in predicate:
                    if not (value < predicate["$lt"]):
                        return False

                elif "$lte" in predicate:
                    if not (value <= predicate["$lte"]):
                        return False

                elif "$exists" in predicate:
                    # Field existence check (already handled above)
                    if not predicate["$exists"]:
                        return False

                else:
                    raise BadQueryError(
                        f"Unknown operator in where clause: {list(predicate.keys())}"
                    )

            else:
                # Direct equality
                if value != predicate:
                    return False

        return True

    def _get_nested_value(self, obj: dict, path: str) -> Any:
        """Get a value from a nested object using a dotted path.

        Args:
            obj: The object to traverse
            path: Dotted path (e.g., "location.city")

        Returns:
            The value at the path

        Raises:
            KeyError: If the path doesn't exist
        """
        parts = path.split(".")
        current = obj

        for part in parts:
            if not isinstance(current, dict):
                raise KeyError(f"Cannot traverse non-dict at path '{path}'")
            current = current[part]

        return current


# Made with Bob
