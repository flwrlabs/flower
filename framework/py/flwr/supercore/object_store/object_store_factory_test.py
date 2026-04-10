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
"""Tests for factory class that creates ObjectStore instances."""


import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from .in_memory_object_store import InMemoryObjectStore
from .object_store_factory import ObjectStoreFactory
from .sql_object_store import SqlObjectStore


class TestObjectStoreFactory(unittest.TestCase):
    """Test the ObjectStoreFactory class."""

    def test_store_creates_in_memory_object_store(self) -> None:
        """Test that the factory creates an InMemoryObjectStore instance."""
        factory = ObjectStoreFactory()
        store = factory.store()
        self.assertIsInstance(store, InMemoryObjectStore)

    def test_store_returns_same_instance(self) -> None:
        """Test that the factory returns the same instance on subsequent calls."""
        factory = ObjectStoreFactory()
        store1 = factory.store()
        store2 = factory.store()
        self.assertIs(store1, store2)

    def test_store_initializes_sql_store_once_under_concurrency(self) -> None:
        """Test that concurrent SQL store access initializes only one instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "state.db"
            factory = ObjectStoreFactory(str(db_path))
            barrier = threading.Barrier(9)
            init_calls = 0
            init_calls_lock = threading.Lock()
            returned_stores = []

            def slow_initialize(_self: SqlObjectStore) -> None:
                nonlocal init_calls
                with init_calls_lock:
                    init_calls += 1
                time.sleep(0.01)

            def worker() -> None:
                barrier.wait()
                returned_stores.append(factory.store())

            with patch.object(SqlObjectStore, "initialize", new=slow_initialize):
                threads = [threading.Thread(target=worker) for _ in range(8)]
                for thread in threads:
                    thread.start()
                barrier.wait()
                for thread in threads:
                    thread.join()

            self.assertEqual(init_calls, 1)
            self.assertEqual(len({id(store) for store in returned_stores}), 1)


if __name__ == "__main__":
    unittest.main()
