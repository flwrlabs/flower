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
from typing import Any, cast

from sqlalchemy import MetaData, delete, func, insert, select, update
from sqlalchemy.orm import Session

from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.supercore.inflatable.inflatable_object import (
    get_object_id,
    is_valid_sha256_hash,
    iterate_object_tree,
)
from flwr.supercore.inflatable.inflatable_utils import validate_object_content
from flwr.supercore.sql_mixin import SqlMixin
from flwr.supercore.state.schema.objectstore_models import (
    ObjectChild,
    ObjectStoreLock,
    RunObject,
    StoredObject,
)
from flwr.supercore.state.schema.objectstore_tables import create_objectstore_metadata
from flwr.supercore.utils import uint64_to_int64

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

        with self._mutation_session() as session:
            for tree_node in tree_nodes:
                obj_id = tree_node.object_id
                child_ids = [child.object_id for child in tree_node.children]
                if len(child_ids) != len(set(child_ids)):
                    raise ValueError(f"Object {obj_id} has duplicate children.")

                # Insert new object if it doesn't exist (race-condition safe)
                # RETURNING returns a row only if the insert succeeded
                insert_object = cast(Any, self.dialect_insert(StoredObject)).values(
                    object_id=obj_id,
                    content=b"",
                    is_available=0,
                    ref_count=0,
                )
                insert_object = insert_object.on_conflict_do_nothing(
                    index_elements=[StoredObject.object_id]
                ).returning(StoredObject.object_id)
                is_new = session.execute(insert_object).first() is not None

                if is_new:
                    new_objects.append(obj_id)
                else:
                    # Object exists: check if unavailable
                    is_available = session.scalar(
                        select(StoredObject.is_available).where(
                            StoredObject.object_id == obj_id
                        )
                    )
                    if is_available == 0:
                        new_objects.append(obj_id)
                    existing_child_ids = set(
                        session.scalars(
                            select(ObjectChild.child_id).where(
                                ObjectChild.parent_id == obj_id
                            )
                        )
                    )
                    if existing_child_ids != set(child_ids):
                        raise ValueError(
                            f"Object {obj_id} was preregistered with different "
                            "children."
                        )

                # Set up child relationships.
                if is_new:
                    for cid in child_ids:
                        session.execute(
                            insert(ObjectChild).values(parent_id=obj_id, child_id=cid)
                        )
                        session.execute(
                            update(StoredObject)
                            .where(StoredObject.object_id == cid)
                            .values(ref_count=StoredObject.ref_count + 1)
                        )

                # Ensure run mapping
                insert_run_object = cast(Any, self.dialect_insert(RunObject)).values(
                    run_id=uint64_to_int64(run_id), object_id=obj_id
                )
                insert_run_object = insert_run_object.on_conflict_do_nothing()
                session.execute(insert_run_object)
        return new_objects

    def get_object_tree(self, object_id: str) -> ObjectTree:
        """Get the object tree for a given object ID."""
        with self.session() as session:
            object_exists = session.scalar(
                select(StoredObject.object_id)
                .where(StoredObject.object_id == object_id)
                .execution_options(populate_existing=True)
            )
            if object_exists is None:
                raise NoObjectInStoreError(
                    f"Object {object_id} was not pre-registered."
                )

            try:
                child_ids = session.scalars(
                    select(ObjectChild.child_id).where(
                        ObjectChild.parent_id == object_id
                    )
                ).all()
                child_trees = [self.get_object_tree(ch_id) for ch_id in child_ids]
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

        with self.session() as session:
            # UPDATE is the authoritative preregistration check: if cleanup
            # deleted the row concurrently, no row is updated and put must fail.
            updated_object_id = session.scalar(
                update(StoredObject)
                .where(
                    StoredObject.object_id == object_id,
                    StoredObject.is_available == 0,
                )
                .values(content=object_content, is_available=1)
                .returning(StoredObject.object_id)
            )
            if updated_object_id is not None:
                return

            object_exists = session.scalar(
                select(StoredObject.object_id).where(
                    StoredObject.object_id == object_id
                )
            )
            if object_exists is None:
                raise NoObjectInStoreError(
                    f"Object with ID '{object_id}' was not pre-registered."
                )

            return

    def get(self, object_id: str) -> bytes | None:
        """Get an object from the store."""
        with self.session() as session:
            return session.scalar(
                select(StoredObject.content).where(StoredObject.object_id == object_id)
            )

    def delete(self, object_id: str) -> None:
        """Delete an object and its unreferenced descendants from the store."""
        with self._mutation_session() as session:
            ref_count = session.scalar(
                select(StoredObject.ref_count).where(
                    StoredObject.object_id == object_id,
                    StoredObject.ref_count == 0,
                )
            )
            if ref_count is None:
                return

            child_ids = list(
                session.scalars(
                    select(ObjectChild.child_id).where(
                        ObjectChild.parent_id == object_id
                    )
                )
            )

            deleted_object_id = session.scalar(
                delete(StoredObject)
                .where(
                    StoredObject.object_id == object_id,
                    StoredObject.ref_count == 0,
                )
                .returning(StoredObject.object_id)
            )
            if deleted_object_id is None:
                return

            if child_ids:
                session.execute(
                    update(StoredObject)
                    .where(
                        StoredObject.object_id.in_(child_ids),
                        StoredObject.ref_count > 0,
                    )
                    .values(ref_count=StoredObject.ref_count - 1)
                )

            for child_id in child_ids:
                self.delete(child_id)

    def delete_objects_in_run(self, run_id: int) -> None:
        """Delete all objects that were registered in a specific run."""
        run_id_sint = uint64_to_int64(run_id)
        with self._mutation_session() as session:
            object_ids = list(
                session.scalars(
                    select(RunObject.object_id).where(RunObject.run_id == run_id_sint)
                )
            )
            session.execute(delete(RunObject).where(RunObject.run_id == run_id_sint))
            for object_id in object_ids:
                self.delete(object_id)

    def clear(self) -> None:
        """Clear the store."""
        with self._mutation_session() as session:
            session.execute(delete(ObjectChild))
            session.execute(delete(RunObject))
            session.execute(delete(StoredObject))

    @contextmanager
    def _mutation_session(self) -> Iterator[Session]:
        """Start a mutation transaction and acquire its SQL lock once."""
        with self.session() as session:
            if _objectstore_mutation_lock_held.get():
                yield session
                return

            token = _objectstore_mutation_lock_held.set(True)
            try:
                self._lock_objectstore_mutation()
                yield session
            finally:
                _objectstore_mutation_lock_held.reset(token)

    def _lock_objectstore_mutation(self) -> None:
        """Serialize structural ObjectStore writes within the active transaction."""
        stmt = cast(Any, self.dialect_insert(ObjectStoreLock)).values(
            lock_id=self._MUTATION_LOCK_ID,
            lock_value=0,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ObjectStoreLock.lock_id],
            set_={"lock_value": ObjectStoreLock.lock_value},
        )
        with self.session() as session:
            session.execute(stmt)

    def __contains__(self, object_id: str) -> bool:
        """Check if an object_id is in the store."""
        with self.session() as session:
            return (
                session.scalar(
                    select(StoredObject.object_id)
                    .where(StoredObject.object_id == object_id)
                    .execution_options(populate_existing=True)
                )
                is not None
            )

    def __len__(self) -> int:
        """Return the number of objects in the store."""
        with self.session() as session:
            return int(
                session.scalar(select(func.count()).select_from(StoredObject)) or 0
            )
