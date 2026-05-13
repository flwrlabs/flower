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
"""Factory class that creates State instances."""


from logging import DEBUG
from threading import Lock

from flwr.common.logger import log
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import FederationManager

from .in_memory_linkstate import InMemoryLinkState
from .linkstate import LinkState
from .sql_linkstate import SqlLinkState


class LinkStateFactory:
    """Factory class that creates LinkState instances.

    Parameters
    ----------
    database : str
        A string representing the path to the database file that will be opened.
        Note that passing ':memory:' will open a connection to a database that is
        in RAM, instead of on disk. For more information on special in-memory
        databases, please refer to https://sqlite.org/inmemorydb.html.
    federation_manager : FederationManager
        An instance of FederationManager to manage federations.
    objectstore_factory : ObjectStoreFactory
        An instance of ObjectStoreFactory to create object stores.
    """

    def __init__(
        self,
        database: str,
        federation_manager: FederationManager,
        objectstore_factory: ObjectStoreFactory,
    ) -> None:
        self.database = database
        self.state_instance: LinkState | None = None
        self._state_lock = Lock()
        self.federation_manager = federation_manager
        self.objectstore_factory = objectstore_factory

    def state(self) -> LinkState:
        """Return a State instance and create it, if necessary."""
        # Fast return cached state if it exists
        if self.state_instance is not None:
            return self.state_instance

        with self._state_lock:
            if self.state_instance is not None:
                return self.state_instance

            # Get the ObjectStore instance
            object_store = self.objectstore_factory.store()

            # InMemoryState
            if self.database == FLWR_IN_MEMORY_DB_NAME:
                self.state_instance = InMemoryLinkState(
                    self.federation_manager, object_store
                )
            # SqlLinkState
            else:
                state = SqlLinkState(
                    self.database, self.federation_manager, object_store
                )
                state.initialize()
                self.state_instance = state

            log(DEBUG, "Using %s", type(self.state_instance).__name__)
            return self.state_instance
