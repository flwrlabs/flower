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
from flwr.supercore.utils import uint64_to_int64

from .object_store import NoObjectInStoreError, ObjectStore


class SqlObjectStore(ObjectStore, SqlMixin):
    """SQLAlchemy-based implementation of the ObjectStore interface."""

    def __init__(self, database_path: str, verify: bool = True) -> None:
        super().__init__(database_path)
        self.verify = verify

    def get_metadata(self) -> MetaData:
        """Return SQLAlchemy MetaData for ObjectStore tables."""
        return create_objectstore_metadata()

    def preregister(self, run_id: int, object_tree: ObjectTree) -> list[str]:
        """Identify and preregister missing objects in the `ObjectStore`."""
        new_objects = []
        for tree_node in iterate_object_tree(object_tree):
            obj_id = tree_node.object_id
            if not is_valid_sha256_hash(obj_id):
                raise ValueError(f"Invalid object ID format: {obj_id}")

            child_ids = [child.object_id for child in tree_node.children]
            with self.session():
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

                if rows:
                    # New object inserted: set up child relationships
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
                    new_objects.append(obj_id)
                else:
                    # Object exists: check if unavailable
                    rows = self.query(
                        "SELECT is_available FROM objects WHERE object_id = :object_id",
                        {"object_id": obj_id},
                    )
                    if rows and not rows[0]["is_available"]:
                        new_objects.append(obj_id)

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
            # Only allow adding the object if it has been preregistered
            rows = self.query(
                "SELECT is_available FROM objects WHERE object_id = :object_id",
                {"object_id": object_id},
            )
            if not rows:
                raise NoObjectInStoreError(
                    f"Object with ID '{object_id}' was not pre-registered."
                )

            # Return if object is already present in the store
            if rows[0]["is_available"]:
                return

            # Update the object entry in the store
            self.query(
                "UPDATE objects SET content = :content, is_available = 1 "
                "WHERE object_id = :object_id",
                {"content": object_content, "object_id": object_id},
            )

    def get(self, object_id: str) -> bytes | None:
        """Get an object from the store."""
        rows = self.query(
            "SELECT content FROM objects WHERE object_id = :oid", {"oid": object_id}
        )
        return rows[0]["content"] if rows else None

    def delete(self, object_id: str) -> None:
        """Delete an object and its unreferenced descendants from the store."""
        with self.session():
            rows = self.query(
                "SELECT ref_count FROM objects WHERE object_id = :object_id",
                {"object_id": object_id},
            )

            # If the object is not in the store, nothing to delete
            if not rows:
                return

            # Skip deletion if there are still references
            if rows[0]["ref_count"] > 0:
                return

            # Deleting will cascade via FK, but we need to decrement children first
            children = self.query(
                "SELECT child_id FROM object_children WHERE parent_id = :parent_id",
                {"parent_id": object_id},
            )
            child_ids = [child["child_id"] for child in children]

            if child_ids:
                placeholders = ", ".join(f":cid{i}" for i in range(len(child_ids)))
                params = {f"cid{i}": cid for i, cid in enumerate(child_ids)}
                self.query(
                    "UPDATE objects SET ref_count = ref_count - 1 "
                    f"WHERE object_id IN ({placeholders})",
                    params,
                )

            self.query(
                "DELETE FROM objects WHERE object_id = :object_id",
                {"object_id": object_id},
            )

            # Recursively clean children
            for child_id in child_ids:
                self.delete(child_id)

    def delete_objects_in_run(self, run_id: int) -> None:
        """Delete all objects that were registered in a specific run."""
        run_id_sint = uint64_to_int64(run_id)
        with self.session():
            objs = self.query(
                "SELECT object_id FROM run_objects WHERE run_id = :run_id",
                {"run_id": run_id_sint},
            )
            for obj in objs:
                object_id = obj["object_id"]
                rows = self.query(
                    "SELECT ref_count FROM objects WHERE object_id=:object_id",
                    {"object_id": object_id},
                )
                if rows and rows[0]["ref_count"] == 0:
                    self.delete(object_id)
            self.query(
                "DELETE FROM run_objects WHERE run_id = :run_id",
                {"run_id": run_id_sint},
            )

    def clear(self) -> None:
        """Clear the store."""
        with self.session():
            self.query("DELETE FROM object_children")
            self.query("DELETE FROM run_objects")
            self.query("DELETE FROM objects")

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
