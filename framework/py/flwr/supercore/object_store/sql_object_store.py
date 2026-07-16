# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower SQLAlchemy-based ObjectStore implementation."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from sqlalchemy import MetaData

from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.supercore.inflatable.inflatable_object import (
    get_object_id,
    is_valid_sha256_hash,
    iterate_object_tree,
)
from flwr.supercore.inflatable.inflatable_utils import validate_object_content
from flwr.supercore.sql_mixin import SqlMixin
from flwr.supercore.state.schema.objectstore_tables import create_objectstore_metadata
from flwr.supercore.utils import build_sql_in_params, uint64_to_int64

from .object_store import NoObjectInStoreError, ObjectStore

_objectstore_mutation_lock_held: ContextVar[bool] = ContextVar(
    "objectstore_mutation_lock_held",
    default=False,
)


class SqlObjectStore(ObjectStore, SqlMixin):
    """SQLAlchemy-based implementation of the ObjectStore interface."""

    _MUTATION_LOCK_ID = "mutation"

    def __init__(
        self,
        database_path: str,
        verify: bool = True,
    ) -> None:
        super().__init__(database_path)
        self.verify = verify

    def get_metadata(self) -> MetaData:
        """Return SQLAlchemy MetaData for ObjectStore tables."""
        return create_objectstore_metadata()

    def preregister(self, run_id: int, object_tree: ObjectTree) -> list[str]:
        """Identify and preregister missing objects in the `ObjectStore`."""
        new_objects = []
        tree_nodes = list(iterate_object_tree(object_tree))
        for tree_node in tree_nodes:
            if not is_valid_sha256_hash(tree_node.object_id):
                raise ValueError(f"Invalid object ID format: {tree_node.object_id}")

        with self._mutation_session():
            for tree_node in tree_nodes:
                obj_id = tree_node.object_id
                child_ids = [child.object_id for child in tree_node.children]
                if len(child_ids) != len(set(child_ids)):
                    raise ValueError(f"Object {obj_id} has duplicate children.")

                # Insert new object if it doesn't exist (race-condition safe)
                # RETURNING returns a row only if the insert succeeded
                rows = self.query(
                    "INSERT INTO objects "
                    "(object_id, content, is_available, ref_count) "
                    "VALUES (:object_id, :content, :is_available, :ref_count) "
                    "ON CONFLICT (object_id) DO NOTHING "
                    "RETURNING object_id",
                    {
                        "object_id": obj_id,
                        "content": b"",
                        "is_available": 0,
                        "ref_count": 0,
                    },
                )
                is_new = bool(rows)

                if is_new:
                    new_objects.append(obj_id)
                else:
                    # Object exists: check if unavailable
                    rows = self.query(
                        "SELECT is_available FROM objects WHERE object_id = :object_id",
                        {"object_id": obj_id},
                    )
                    if rows and not rows[0]["is_available"]:
                        new_objects.append(obj_id)
                    rows = self.query(
                        "SELECT child_id FROM object_children "
                        "WHERE parent_id = :parent_id",
                        {"parent_id": obj_id},
                    )
                    existing_child_ids = {row["child_id"] for row in rows}
                    if existing_child_ids != set(child_ids):
                        raise ValueError(
                            f"Object {obj_id} was preregistered with different "
                            "children."
                        )

                # Set up child relationships.
                if is_new:
                    for cid in child_ids:
                        self.query(
                            "INSERT INTO object_children (parent_id, child_id) "
                            "VALUES (:parent_id, :child_id)",
                            {"parent_id": obj_id, "child_id": cid},
                        )
                        self.query(
                            "UPDATE objects SET ref_count = ref_count + 1 "
                            "WHERE object_id = :object_id",
                            {"object_id": cid},
                        )

                # Ensure run mapping
                self.query(
                    "INSERT INTO run_objects (run_id, object_id) "
                    "VALUES (:run_id, :object_id) ON CONFLICT DO NOTHING",
                    {"run_id": uint64_to_int64(run_id), "object_id": obj_id},
                )
        return new_objects

    def get_object_tree(self, object_id: str) -> ObjectTree:
        """Get the object tree for a given object ID."""
        with self.session():
            rows = self.query(
                "SELECT object_id FROM objects WHERE object_id = :object_id",
                {"object_id": object_id},
            )
            if not rows:
                raise NoObjectInStoreError(
                    f"Object {object_id} was not pre-registered."
                )
            children = self.query(
                "SELECT child_id FROM object_children WHERE parent_id = :parent_id",
                {"parent_id": object_id},
            )

            # Build the object trees of all children
            try:
                child_trees = [self.get_object_tree(ch["child_id"]) for ch in children]
            except NoObjectInStoreError as e:
                # Raise an error if any child object is missing
                # This indicates an integrity issue
                raise NoObjectInStoreError(
                    f"Object tree for object ID '{object_id}' contains missing "
                    "children. This may indicate a corrupted object store."
                ) from e

            # Create and return the ObjectTree for the current object
            return ObjectTree(object_id=object_id, children=child_trees)

    def put(self, object_id: str, object_content: bytes) -> None:
        """Put an object into the store."""
        if self.verify:
            # Verify object_id and object_content match
            object_id_from_content = get_object_id(object_content)
            if object_id != object_id_from_content:
                raise ValueError(f"Object ID {object_id} does not match content hash")

            # Validate object content
            validate_object_content(content=object_content)

        with self.session():
            # UPDATE is the authoritative preregistration check: if cleanup
            # deleted the row concurrently, no row is updated and put must fail.
            rows = self.query(
                "UPDATE objects SET content = :content, is_available = 1 "
                "WHERE object_id = :object_id AND is_available = 0 "
                "RETURNING object_id",
                {"content": object_content, "object_id": object_id},
            )
            if rows:
                return

            rows = self.query(
                "SELECT is_available FROM objects WHERE object_id = :object_id",
                {"object_id": object_id},
            )
            if not rows:
                raise NoObjectInStoreError(
                    f"Object with ID '{object_id}' was not pre-registered."
                )

            return

    def get(self, object_id: str) -> bytes | None:
        """Get an object from the store."""
        rows = self.query(
            "SELECT content FROM objects WHERE object_id = :oid", {"oid": object_id}
        )
        return rows[0]["content"] if rows else None

    def delete(self, object_id: str) -> None:
        """Delete an object and its unreferenced descendants from the store."""
        with self._mutation_session():
            rows = self.query(
                "SELECT 1 FROM objects WHERE object_id = :object_id AND ref_count = 0",
                {"object_id": object_id},
            )
            if not rows:
                return

            children = self.query(
                "SELECT child_id FROM object_children WHERE parent_id = :parent_id",
                {"parent_id": object_id},
            )
            child_ids = [child["child_id"] for child in children]

            rows = self.query(
                "DELETE FROM objects "
                "WHERE object_id = :object_id AND ref_count = 0 "
                "RETURNING object_id",
                {"object_id": object_id},
            )
            if not rows:
                return

            if child_ids:
                placeholders, params = build_sql_in_params(child_ids, "cid")
                self.query(
                    "UPDATE objects SET ref_count = ref_count - 1 "
                    f"WHERE object_id IN ({placeholders}) AND ref_count > 0",
                    params,
                )

            for child_id in child_ids:
                self.delete(child_id)

    def delete_objects_in_run(self, run_id: int) -> None:
        """Delete all objects that were registered in a specific run."""
        run_id_sint = uint64_to_int64(run_id)
        with self._mutation_session():
            objs = self.query(
                "SELECT object_id FROM run_objects WHERE run_id = :run_id",
                {"run_id": run_id_sint},
            )
            self.query(
                "DELETE FROM run_objects WHERE run_id = :run_id",
                {"run_id": run_id_sint},
            )
            for obj in objs:
                self.delete(obj["object_id"])

    def clear(self) -> None:
        """Clear the store."""
        with self._mutation_session():
            self.query("DELETE FROM object_children")
            self.query("DELETE FROM run_objects")
            self.query("DELETE FROM objects")

    @contextmanager
    def _mutation_session(self) -> Iterator[None]:
        """Start a mutation transaction and acquire its SQL lock once."""
        with self.session():
            if _objectstore_mutation_lock_held.get():
                yield
                return

            token = _objectstore_mutation_lock_held.set(True)
            try:
                self._lock_objectstore_mutation()
                yield
            finally:
                _objectstore_mutation_lock_held.reset(token)

    def _lock_objectstore_mutation(self) -> None:
        """Serialize structural ObjectStore writes within the active transaction."""
        self.query(
            "INSERT INTO objectstore_locks (lock_id, lock_value) "
            "VALUES (:lock_id, 0) "
            "ON CONFLICT (lock_id) DO UPDATE "
            "SET lock_value = objectstore_locks.lock_value",
            {"lock_id": self._MUTATION_LOCK_ID},
        )

    def __contains__(self, object_id: str) -> bool:
        """Check if an object_id is in the store."""
        rows = self.query(
            "SELECT 1 FROM objects WHERE object_id = :oid", {"oid": object_id}
        )
        return len(rows) > 0

    def __len__(self) -> int:
        """Return the number of objects in the store."""
        rows = self.query("SELECT COUNT(*) AS cnt FROM objects")
        return int(rows[0]["cnt"])
