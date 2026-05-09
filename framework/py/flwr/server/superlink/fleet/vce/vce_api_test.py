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
"""Test Fleet Simulation Engine API."""


import threading
from json import JSONDecodeError
from pathlib import Path
from time import sleep
from unittest import TestCase
from unittest.mock import patch

from flwr.clientapp.client_app import LoadClientAppError
from flwr.common.typing import Run
from flwr.server.superlink.fleet.vce import vce_api
from flwr.server.superlink.fleet.vce.vce_api import start_vce
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager

TEST_CLIENT_APP_ATTR = "e2e_bare.client_app:app"


def terminate_simulation(f_stop: threading.Event, sleep_duration: int) -> None:
    """Set event to terminate Simulation Engine after `sleep_duration` seconds."""
    sleep(sleep_duration)
    f_stop.set()


def _autoresolve_app_dir(rel_client_app_dir: str = "e2e/e2e-bare") -> str:
    """Correctly resolve working directory for the app."""
    framework_dir = Path(__file__).resolve().parents[6]
    return str(framework_dir / rel_client_app_dir)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def start_and_shutdown(
    backend: str = "ray",
    client_app_attr: str = TEST_CLIENT_APP_ATTR,
    app_dir: str = "",
    num_supernodes: int = 1,
    state_factory: LinkStateFactory | None = None,
    duration: int = 0,
    backend_config: str = "{}",
) -> None:
    """Start Simulation Engine and terminate after specified number of seconds.

    Some tests need to be terminated by triggering externally an threading.Event. This
    is enabled when passing `duration`>0.
    """
    f_stop = threading.Event()

    if duration:

        # Setup thread that will set the f_stop event, triggering the termination of all
        # logic in the Simulation Engine. It will also terminate the Backend.
        termination_th = threading.Thread(
            target=terminate_simulation, args=(f_stop, duration)
        )
        termination_th.start()

    # Resolve working directory if not passed
    if not app_dir:
        app_dir = _autoresolve_app_dir()

    run = Run.create_empty(run_id=1234)
    if state_factory is None:
        state_factory = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), ObjectStoreFactory()
        )

    start_vce(
        num_supernodes=num_supernodes,
        client_app_attr=client_app_attr,
        backend_name=backend,
        backend_config_json_stream=backend_config,
        state_factory=state_factory,
        app_dir=app_dir,
        f_stop=f_stop,
        run=run,
    )

    if duration:
        termination_th.join()


class TestFleetSimulationEngineRayBackend(TestCase):
    """A basic class that enables testing functionalities."""

    def test_erroneous_client_app_attr(self) -> None:
        """Tests attempt to load a ClientApp that can't be found."""
        with patch.object(vce_api.time, "sleep"), self.assertRaises(LoadClientAppError):
            start_and_shutdown(
                client_app_attr="totally_fictitious_app:client",
            )

    def test_erroneous_backend_config(self) -> None:
        """Backend Config should be a JSON stream."""
        with self.assertRaises(JSONDecodeError):
            start_and_shutdown(num_supernodes=50, backend_config="not a proper config")

    def test_with_nonexistent_backend(self) -> None:
        """Test specifying a backend that does not exist."""
        with self.assertRaises(KeyError):
            start_and_shutdown(num_supernodes=50, backend="this-backend-does-not-exist")

    def test_start_and_shutdown(self) -> None:
        """Start Simulation Engine Fleet and terminate it."""
        start_and_shutdown(num_supernodes=50, duration=10)
