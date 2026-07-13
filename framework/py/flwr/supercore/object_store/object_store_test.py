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
"""Tests for ObjectStore."""


import tempfile
import threading
import unittest
from abc import abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast
from unittest.mock import Mock, patch

from parameterized import parameterized
from sqlalchemy import Engine, inspect

from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.supercore.inflatable.inflatable_object import (
    get_object_id,
    get_object_tree,
    iterate_object_tree,
)
from flwr.supercore.inflatable.inflatable_object_test import CustomDataClass

from .in_memory_object_store import InMemoryObjectStore
from .object_store import NoObjectInStoreError, ObjectStore
from .sql_object_store import SqlObjectStore


class ObjectStoreTest(unittest.TestCase):
    """Test all ObjectStore implementations."""

    # This is to True in each child class
    __test__ = False

    def setUp(self) -> None:
        """Set up the test case."""
        self.run_id = 110

    @abstractmethod
    def object_store_factory(self) -> ObjectStore:
        """Provide ObjectStore implementation to test."""
        raise NotImplementedError()

    def test_get_non_existent_object_id(self) -> None:
        """Test get method with a non-existent object_id."""
        # Prepare
        object_store = self.object_store_factory()
        object_id = "non_existent_object_id"

        # Execute
        retrieved_value = object_store.get(object_id)

        # Assert
        self.assertIsNone(retrieved_value)

    def test_put_and_get(self) -> None:
        """Test put and get methods."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value")
        object_content = obj.deflate()
        object_id = get_object_id(object_content)
        object_store.preregister(self.run_id, get_object_tree(obj))

        # Execute
        object_store.put(object_id, object_content)
        retrieved_value = object_store.get(object_id)

        # Assert
        self.assertEqual(object_content, retrieved_value)

    def test_put_overwrite(self) -> None:
        """Test put method with an existing object_id."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value")
        object_content = obj.deflate()
        object_id = get_object_id(object_content)
        object_store.preregister(self.run_id, get_object_tree(obj))

        # Execute
        object_store.put(object_id, object_content)
        object_store.put(object_id, object_content)
        retrieved_value = object_store.get(object_id)

        # Assert
        self.assertEqual(object_content, retrieved_value)

    def test_put_object_id_and_content_pair_not_matching(self) -> None:
        """Test put method with an object_id that does not match that of content."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value")
        object_content = obj.deflate()
        object_id = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        object_store.preregister(self.run_id, ObjectTree(object_id=object_id))

        # Execute and assert
        with self.assertRaises(ValueError):
            object_store.put(object_id, object_content)

    def test_delete(self) -> None:
        """Test delete method."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value")
        object_content = obj.deflate()
        object_id = get_object_id(object_content)
        object_store.preregister(self.run_id, get_object_tree(obj))
        object_store.put(object_id, object_content)

        # Execute
        object_store.delete(object_id)
        retrieved_value = object_store.get(object_id)

        # Assert
        self.assertIsNone(retrieved_value)

    def test_delete_non_existent_object_id(self) -> None:
        """Test delete method with a non-existent object_id."""
        # Prepare
        object_store = self.object_store_factory()
        object_id = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

        object_store.delete(object_id)
        # No exception should be raised

    def test_clear(self) -> None:
        """Test clear method."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value1")
        object_content1 = obj.deflate()
        object_id1 = get_object_id(object_content1)
        object_store.preregister(self.run_id, get_object_tree(obj))
        obj = CustomDataClass(data=b"test_value2")
        object_content2 = obj.deflate()
        object_id2 = get_object_id(object_content2)
        object_store.preregister(self.run_id, get_object_tree(obj))

        object_store.put(object_id1, object_content1)
        object_store.put(object_id2, object_content2)

        # Execute
        object_store.clear()

        # Assert
        retrieved_value1 = object_store.get(object_id1)
        retrieved_value2 = object_store.get(object_id2)

        self.assertIsNone(retrieved_value1)
        self.assertIsNone(retrieved_value2)

    def test_clear_empty_store(self) -> None:
        """Test clear method on an empty store."""
        # Prepare
        object_store = self.object_store_factory()

        # Execute
        object_store.clear()
        # No exception should be raised

    def test_contains(self) -> None:
        """Test __contains__ method."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value1")
        object_content = obj.deflate()
        object_id = get_object_id(object_content)
        object_store.preregister(self.run_id, get_object_tree(obj))
        object_store.put(object_id, object_content)
        unavailable = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

        # Execute
        contained = object_id in object_store
        not_contained = unavailable in object_store

        # Assert
        self.assertTrue(contained)
        self.assertFalse(not_contained)

    def test_put_without_preregistering(self) -> None:
        """Test put without preregistering first."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value")
        object_content = obj.deflate()
        object_id = get_object_id(object_content)

        # Execute
        with self.assertRaises(NoObjectInStoreError):
            object_store.put(object_id, object_content)

    def test_preregister(self) -> None:
        """Test preregister functionality."""
        # Prepare
        object_store = self.object_store_factory()
        obj1 = CustomDataClass(data=b"test_value1")
        object_content1 = obj1.deflate()
        object_id1 = get_object_id(object_content1)
        obj2 = CustomDataClass(data=b"test_value2")
        object_content2 = obj2.deflate()
        object_id2 = get_object_id(object_content2)

        # Execute (preregister all)
        not_present = object_store.preregister(self.run_id, get_object_tree(obj1))
        not_present += object_store.preregister(self.run_id, get_object_tree(obj2))

        # Assert (none was present)
        self.assertEqual([object_id1, object_id2], not_present)

        # Execute (pre-register an available object)
        object_store.put(object_id1, object_content1)
        not_present = object_store.preregister(self.run_id, get_object_tree(obj1))

        # Assert none was not present
        self.assertEqual([], not_present)

        # Execute (pre-register an unavailable object)
        not_present = object_store.preregister(self.run_id, get_object_tree(obj2))

        # Assert the unavailable object is returned
        self.assertEqual([object_id2], not_present)

    def test_get_object_tree(self) -> None:
        """Test get_object_tree method."""
        # Prepare
        object_store = self.object_store_factory()
        obj = CustomDataClass(
            data=b"test_value", children=[CustomDataClass(data=b"child")]
        )
        obj_tree = get_object_tree(obj)
        object_store.preregister(self.run_id, get_object_tree(obj))

        # Execute
        retrieved_tree = object_store.get_object_tree(obj_tree.object_id)
        retrieved_tree_traversed = [
            node.object_id for node in iterate_object_tree(retrieved_tree)
        ]
        obj_tree_traversed = [node.object_id for node in iterate_object_tree(obj_tree)]

        # Assert
        self.assertEqual(retrieved_tree_traversed, obj_tree_traversed)

    @parameterized.expand([(""), ("invalid")])  # type: ignore
    def test_preregister_with_invalid_object_id(self, invalid_object_id) -> None:
        """Test preregistering with object_id that is not a valid SHA256."""
        # Prepare
        object_store = self.object_store_factory()

        # Execute
        with self.assertRaises(ValueError):
            object_store.preregister(
                self.run_id, ObjectTree(object_id=invalid_object_id)
            )

    # pylint: disable-next=too-many-locals
    def test_put_get_delete_object_with_children(self) -> None:
        """Test put and get methods with an object that has children."""
        # Prepare: Define object hierarchy
        objects, id_to_content = _create_object_hierarchy()
        ids = list(id_to_content.keys())
        parent1 = objects[3]
        parent2 = objects[4]

        # Execute: Preregister and put all objects
        object_store = self.object_store_factory()
        object_store.preregister(self.run_id, get_object_tree(parent1))
        object_store.preregister(self.run_id, get_object_tree(parent2))
        for obj_id, content in id_to_content.items():
            object_store.put(obj_id, content)

        # Assert: All objects should be in the store
        self.assertEqual(len(object_store), 5)

        # Execute: Retrieve all objects
        for obj_id, content in id_to_content.items():
            retrieved = object_store.get(obj_id)

            # Assert: Retrieved object should match the original content
            self.assertEqual(retrieved, content)

        # Execute: Delete parent1
        object_store.delete(ids[3])

        # Assert: Only parent2 and child2 should remain
        self.assertEqual(len(object_store), 2)
        self.assertTrue(ids[2] in object_store)
        self.assertTrue(ids[4] in object_store)

        # Execute: Delete parent2
        object_store.delete(ids[4])

        # Assert: The store should be empty now
        self.assertEqual(len(object_store), 0)

    def test_delete_objects_in_run(self) -> None:
        """Test deleting objects in a specific run."""
        # Prepare: Define object hierarchy
        objects, id_to_content = _create_object_hierarchy()
        ids = list(id_to_content.keys())
        parent1 = objects[3]
        parent2 = objects[4]

        # Execute: Preregister parent 1 and its descendants for run 1
        object_store = self.object_store_factory()
        object_store.preregister(run_id=1, object_tree=get_object_tree(parent1))

        # Execute: Preregister parent 2 and its descendants for run 2
        object_store.preregister(run_id=2, object_tree=get_object_tree(parent2))

        # Execute: Put all objects
        for obj_id, content in id_to_content.items():
            object_store.put(obj_id, content)

        # Assert: All objects should be in the store
        self.assertEqual(len(object_store), 5)

        # Execute: Delete objects in run 1
        object_store.delete_objects_in_run(run_id=1)

        # Assert: Only parent2 and child2 should remain
        self.assertEqual(len(object_store), 2)
        self.assertTrue(ids[2] in object_store)  # child2
        self.assertTrue(ids[4] in object_store)  # parent2

        # Execute: Delete objects in run 2
        object_store.delete_objects_in_run(run_id=2)

        # Assert: The store should be empty now
        self.assertEqual(len(object_store), 0)


def _create_object_hierarchy() -> tuple[list[CustomDataClass], dict[str, bytes]]:
    """Create a hierarchy of objects for testing.

    - parent1 -> child1, child2
    - parent2 -> child2
    - child1 -> grandchild

    The returned list is in the order:
    [grandchild, child1, child2, parent1, parent2]

    Returns
    -------
    tuple[list[CustomDataClass], dict[str, bytes]]
        A tuple containing a list of CustomDataClass objects and
        a mapping of object IDs to their deflated content.
    """
    grandchild = CustomDataClass(b"grandchild")
    child1 = CustomDataClass(b"child1", children=[grandchild])
    child2 = CustomDataClass(b"child2")
    parent1 = CustomDataClass(b"parent1", children=[child1, child2])
    parent2 = CustomDataClass(b"parent2", children=[child2])

    objects = [grandchild, child1, child2, parent1, parent2]
    id_to_content = {obj.object_id: obj.deflate() for obj in objects}
    return objects, id_to_content


class InMemoryObjectStoreTest(ObjectStoreTest):
    """Test InMemoryObjectStore implementation."""

    __test__ = True

    def object_store_factory(self) -> ObjectStore:
        """Return InMemoryObjectStore."""
        return InMemoryObjectStore()


class SqlObjectStoreTestMixin(unittest.TestCase):
    """Test SQL-specific ObjectStore behavior."""

    __test__ = False

    def object_store_factory(self) -> SqlObjectStore:
        """Provide SQL ObjectStore implementation to test."""
        raise NotImplementedError()

    def test_preregister_rejects_new_children_for_existing_object_id(self) -> None:
        """Ensure existing SQL objects cannot get new children."""
        store = self.object_store_factory()
        parent_id = get_object_id(b"parent")
        child_id = get_object_id(b"child")

        store.preregister(run_id=1, object_tree=ObjectTree(object_id=parent_id))

        with self.assertRaisesRegex(ValueError, "different children"):
            store.preregister(
                run_id=2,
                object_tree=ObjectTree(
                    object_id=parent_id,
                    children=[ObjectTree(object_id=child_id)],
                ),
            )


class SqlInMemoryObjectStoreTest(SqlObjectStoreTestMixin, ObjectStoreTest):
    """Test SqlObjectStore implementation with in-memory database."""

    __test__ = True

    def object_store_factory(self) -> SqlObjectStore:
        """Return SqlObjectStore."""
        store = SqlObjectStore(":memory:")
        store.initialize()
        return store

    def test_in_memory_does_not_create_alembic_version(self) -> None:
        """Ensure in-memory DB uses create_all without Alembic versioning."""
        store = self.object_store_factory()
        table_names = inspect(
            cast(Engine, store._engine)  # pylint: disable=W0212
        ).get_table_names()
        self.assertNotIn("alembic_version", table_names)


class SqlPersistentObjectStoreTestMixin(unittest.TestCase):
    """Test SQL ObjectStore behavior requiring a persistent database."""

    __test__ = False
    run_id: int

    def object_store_factory(self) -> SqlObjectStore:
        """Return a new store connected to the same persistent database."""
        raise NotImplementedError()

    def test_persistent_db_creates_alembic_version(self) -> None:
        """Ensure persistent SQL databases run Alembic migrations."""
        store = self.object_store_factory()
        table_names = inspect(
            cast(Engine, store._engine)  # pylint: disable=W0212
        ).get_table_names()
        self.assertIn("alembic_version", table_names)
        self.assertIn("objectstore_locks", table_names)

    def test_mutation_lock_uses_sql_lock_row(self) -> None:
        """Ensure ObjectStore mutations use a transaction-scoped SQL lock."""
        store = self.object_store_factory()
        store.query = Mock()  # type: ignore[method-assign]

        store._lock_objectstore_mutation()  # pylint: disable=protected-access

        store.query.assert_any_call(
            "INSERT INTO objectstore_locks (lock_id, lock_value) "
            "VALUES (:lock_id, 0) "
            "ON CONFLICT (lock_id) DO UPDATE "
            "SET lock_value = objectstore_locks.lock_value",
            {"lock_id": store._MUTATION_LOCK_ID},  # pylint: disable=W0212
        )
        self.assertEqual(store.query.call_count, 1)

    def test_mutation_session_locks_once(self) -> None:
        """Ensure nested mutation sessions reuse the transaction-scoped lock."""
        store = self.object_store_factory()
        store.query = Mock()  # type: ignore[method-assign]

        with store._mutation_session():  # pylint: disable=protected-access
            with store._mutation_session():  # pylint: disable=protected-access
                pass

        self.assertEqual(store.query.call_count, 1)

    def test_concurrent_preregister_and_run_cleanup(self) -> None:
        """Concurrent run cleanup preserves objects registered by another run."""
        store = self.object_store_factory()
        second_store = self.object_store_factory()
        child = CustomDataClass(b"shared")
        old_parent = CustomDataClass(b"old", children=[child])
        new_parent = CustomDataClass(b"new", children=[child])
        content_by_id = {obj.object_id: obj.deflate() for obj in [child, new_parent]}

        store.preregister(run_id=1, object_tree=get_object_tree(old_parent))
        store.put(child.object_id, child.deflate())
        store.put(old_parent.object_id, old_parent.deflate())
        barrier = threading.Barrier(2)

        def cleanup() -> None:
            barrier.wait()
            store.delete_objects_in_run(run_id=1)

        def preregister() -> None:
            barrier.wait()
            missing = second_store.preregister(
                run_id=2, object_tree=get_object_tree(new_parent)
            )
            for object_id in missing:
                second_store.put(object_id, content_by_id[object_id])

        with ThreadPoolExecutor(max_workers=2) as executor:
            for future in [executor.submit(cleanup), executor.submit(preregister)]:
                future.result()

        self.assertEqual(store.get(child.object_id), child.deflate())
        self.assertEqual(store.get(new_parent.object_id), new_parent.deflate())

    def test_put_raises_if_object_deleted_before_update(self) -> None:
        """A concurrent delete before the write must not report put success."""
        store = self.object_store_factory()
        second_store = self.object_store_factory()
        obj = CustomDataClass(data=b"test_value")
        object_content = obj.deflate()
        object_id = get_object_id(object_content)
        store.preregister(self.run_id, get_object_tree(obj))

        should_delete = True
        original_query = store.query

        def delete_before_update(
            query: str,
            data: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            nonlocal should_delete
            normalized_query = " ".join(query.split())
            if should_delete and normalized_query.startswith(
                "UPDATE objects SET content = :content"
            ):
                should_delete = False

                def delete_object() -> None:
                    with second_store.session():
                        second_store.query(
                            "DELETE FROM objects "
                            "WHERE object_id = :object_id AND ref_count = 0",
                            {"object_id": object_id},
                        )

                # Use a separate thread so the ContextVar-bound session used by put()
                # is not reused; the deletion must run in another transaction.
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(delete_object).result()
            return original_query(query, data)

        with patch.object(store, "query", side_effect=delete_before_update):
            with self.assertRaises(NoObjectInStoreError):
                store.put(object_id, object_content)

        self.assertIsNone(store.get(object_id))


class SqlFileBasedObjectStoreTest(
    SqlPersistentObjectStoreTestMixin, SqlObjectStoreTestMixin, ObjectStoreTest
):
    """Test SqlObjectStore implementation with file-based database."""

    __test__ = True

    def setUp(self) -> None:
        """Set up the test case."""
        super().setUp()
        self.temp_file = tempfile.NamedTemporaryFile()  # pylint: disable=R1732

    def tearDown(self) -> None:
        """Tear down the test case."""
        super().tearDown()
        self.temp_file.close()

    def object_store_factory(self) -> SqlObjectStore:
        """Return SqlObjectStore."""
        store = SqlObjectStore(self.temp_file.name)
        store.initialize()
        return store
