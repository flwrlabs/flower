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
"""Tests for LinkStateFactory initialization behavior."""


import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.object_store.sql_object_store import SqlObjectStore
from flwr.superlink.federation import NoOpFederationManager

from .linkstate_factory import LinkStateFactory
from .sql_linkstate import SqlLinkState


class TestLinkStateFactory(unittest.TestCase):
    """Test the LinkStateFactory class."""

    def test_state_initializes_sql_state_once_under_concurrency(self) -> None:
        """Test that concurrent SQL state access initializes only one instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "state.db"
            factory = LinkStateFactory(
                str(db_path),
                NoOpFederationManager(),
                ObjectStoreFactory(str(db_path)),
            )
            barrier = threading.Barrier(9)
            init_calls = 0
            init_calls_lock = threading.Lock()
            returned_states = []

            def fast_store_initialize(_self: SqlObjectStore) -> None:
                time.sleep(0.001)

            def slow_state_initialize(_self: SqlLinkState) -> None:
                nonlocal init_calls
                with init_calls_lock:
                    init_calls += 1
                time.sleep(0.01)

            def worker() -> None:
                barrier.wait()
                returned_states.append(factory.state())

            with patch.object(SqlObjectStore, "initialize", new=fast_store_initialize):
                with patch.object(
                    SqlLinkState, "initialize", new=slow_state_initialize
                ):
                    threads = [threading.Thread(target=worker) for _ in range(8)]
                    for thread in threads:
                        thread.start()
                    barrier.wait()
                    for thread in threads:
                        thread.join()

            self.assertEqual(init_calls, 1)
            self.assertEqual(len({id(state) for state in returned_states}), 1)
