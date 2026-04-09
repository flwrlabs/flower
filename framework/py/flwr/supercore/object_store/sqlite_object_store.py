# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower SQLite ObjectStore implementation."""


from typing import cast

from flwr.common.inflatable import (
    get_object_id,
    is_valid_sha256_hash,
    iterate_object_tree,
)
from flwr.common.inflatable_utils import validate_object_content
from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.supercore.sqlite_mixin import SqliteMixin
from flwr.supercore.utils import uint64_to_int64

from .object_store import NoObjectInStoreError, ObjectStore

SQL_CREATE_OBJECTS = """
CREATE TABLE IF NOT EXISTS objects (
    object_id               TEXT PRIMARY KEY,
    content                 BLOB,
    is_available            INTEGER NOT NULL CHECK (is_available IN (0,1)),
    ref_count               INTEGER NOT NULL
);
"""
SQL_CREATE_OBJECT_CHILDREN = """
CREATE TABLE IF NOT EXISTS object_children (
    parent_id               TEXT NOT NULL,
    child_id                TEXT NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES objects(object_id) ON DELETE CASCADE,
    FOREIGN KEY (child_id)  REFERENCES objects(object_id) ON DELETE CASCADE,
    PRIMARY KEY (parent_id, child_id)
);
"""
SQL_CREATE_RUN_OBJECTS = """
CREATE TABLE IF NOT EXISTS run_objects (
    run_id                  INTEGER NOT NULL,
    object_id               TEXT NOT NULL,
    FOREIGN KEY (object_id) REFERENCES objects(object_id) ON DELETE CASCADE,
    PRIMARY KEY (run_id, object_id)
);
"""


class SqliteObjectStore(ObjectStore, SqliteMixin):
    """SQLite-based implementation of the ObjectStore interface."""

    def __init__(self, database_path: str, verify: bool = True) -> None:
        super().__init__(database_path)
        self.verify = verify

    def get_sql_statements(self) -> tuple[str, ...]:
        """Return SQL statements for ObjectStore tables."""
        return (
            SQL_CREATE_OBJECTS,
            SQL_CREATE_OBJECT_CHILDREN,
            SQL_CREATE_RUN_OBJECTS,
        )

    def preregister(self, run_id: int, object_tree: ObjectTree) -> list[str]:
        """Identify and preregister missing objects in the `ObjectStore`."""
        new_objects = []
        for tree_node in iterate_object_tree(object_tree):
            obj_id = tree_node.object_id
            if not is_valid_sha256_hash(obj_id):
                raise ValueError(f"Invalid object ID format: {obj_id}")

            child_ids = [child.object_id for child in tree_node.children]
            with self.conn:
                row = self.conn.execute(
                    "SELECT object_id, is_available FROM objects WHERE object_id=?",
                    (obj_id,),
                ).fetchone()
                if row is None:
                    # Insert new object
                    self.conn.execute(
                        "INSERT INTO objects"
                        "(object_id, content, is_available, ref_count) "
                        "VALUES (?, ?, ?, ?)",
                        (obj_id, b"", 0, 0),
                    )
                    for cid in child_ids:
                        self.conn.execute(
                            "INSERT INTO object_children(parent_id, child_id) "
                            "VALUES (?, ?)",
                            (obj_id, cid),
                        )
                        self.conn.execute(
                            "UPDATE objects SET ref_count = ref_count + 1 "
                            "WHERE object_id = ?",
                            (cid,),
                        )
                    new_objects.append(obj_id)
                else:
                    # Add to the list of new objects if not available
                    if not row["is_available"]:
                        new_objects.append(obj_id)

                # Ensure run mapping
                self.conn.execute(
                    "INSERT OR IGNORE INTO run_objects(run_id, object_id) "
                    "VALUES (?, ?)",
                    (uint64_to_int64(run_id), obj_id),
                )
        return new_objects

    def get_object_tree(self, object_id: str) -> ObjectTree:
        """Get the object tree for a given object ID."""
        with self.conn:
            row = self.conn.execute(
                "SELECT object_id FROM objects WHERE object_id=?", (object_id,)
            ).fetchone()
            if not row:
                raise NoObjectInStoreError(
                    f"Object {object_id} was not pre-registered."
                )
            children = self.query(
                "SELECT child_id FROM object_children WHERE parent_id=?", (object_id,)
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

        with self.conn:
            # Only allow adding the object if it has been preregistered
            row = self.conn.execute(
                "SELECT is_available FROM objects WHERE object_id=?", (object_id,)
            ).fetchone()
            if row is None:
                raise NoObjectInStoreError(
                    f"Object with ID '{object_id}' was not pre-registered."
                )

            # Return if object is already present in the store
            if row["is_available"]:
                return

            # Update the object entry in the store
            self.conn.execute(
                "UPDATE objects SET content=?, is_available=1 WHERE object_id=?",
                (object_content, object_id),
            )

    def get(self, object_id: str) -> bytes | None:
        """Get an object from the store."""
        rows = self.query("SELECT content FROM objects WHERE object_id=?", (object_id,))
        return rows[0]["content"] if rows else None

    def delete(self, object_id: str) -> None:
        """Delete an object and its unreferenced descendants from the store."""
        with self.conn:
            row = self.conn.execute(
                "SELECT ref_count FROM objects WHERE object_id=?", (object_id,)
            ).fetchone()

            # If the object is not in the store, nothing to delete
            if row is None:
                return

            # Skip deletion if there are still references
            if row["ref_count"] > 0:
                return

            # Deleting will cascade via FK, but we need to decrement children first
            children = self.conn.execute(
                "SELECT child_id FROM object_children WHERE parent_id=?", (object_id,)
            ).fetchall()
            child_ids = [child["child_id"] for child in children]

            if child_ids:
                placeholders = ", ".join("?" for _ in child_ids)
                query = f"""
                    UPDATE objects SET ref_count = ref_count - 1
                    WHERE object_id IN ({placeholders})
                """
                self.conn.execute(query, child_ids)

            self.conn.execute("DELETE FROM objects WHERE object_id=?", (object_id,))

            # Recursively clean children
            for child_id in child_ids:
                self.delete(child_id)

    def delete_objects_in_run(self, run_id: int) -> None:
        """Delete all objects that were registered in a specific run."""
        run_id_sint = uint64_to_int64(run_id)
        with self.conn:
            objs = self.conn.execute(
                "SELECT object_id FROM run_objects WHERE run_id=?", (run_id_sint,)
            ).fetchall()
            for obj in objs:
                object_id = obj["object_id"]
                row = self.conn.execute(
                    "SELECT ref_count FROM objects WHERE object_id=?", (object_id,)
                ).fetchone()
                # Keep objects that are still referenced by at least one other run.
                other_runs = self.conn.execute(
                    "SELECT COUNT(*) AS cnt FROM run_objects "
                    "WHERE object_id=? AND run_id != ?",
                    (object_id, run_id_sint),
                ).fetchone()
                if (
                    row
                    and row["ref_count"] == 0
                    and other_runs
                    and other_runs["cnt"] == 0
                ):
                    self.delete(object_id)
            self.conn.execute("DELETE FROM run_objects WHERE run_id=?", (run_id_sint,))

    def clear(self) -> None:
        """Clear the store."""
        with self.conn:
            self.conn.execute("DELETE FROM object_children;")
            self.conn.execute("DELETE FROM run_objects;")
            self.conn.execute("DELETE FROM objects;")

    def __contains__(self, object_id: str) -> bool:
        """Check if an object_id is in the store."""
        row = self.conn.execute(
            "SELECT 1 FROM objects WHERE object_id=?", (object_id,)
        ).fetchone()
        return row is not None

    def __len__(self) -> int:
        """Return the number of objects in the store."""
        row = self.conn.execute("SELECT COUNT(*) AS cnt FROM objects;").fetchone()
        return cast(int, row["cnt"])
