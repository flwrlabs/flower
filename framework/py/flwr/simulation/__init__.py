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
"""Flower simulation."""


import importlib
from typing import Any

from flwr.simulation.app import run_simulation_process
from flwr.simulation.run_simulation import run_simulation
from flwr.simulation.simulationio_connection import SimulationIoConnection

RAY_IMPORT_ERROR: str = """Unable to import module `ray`.

To install the necessary dependencies, install `flwr` with the `simulation` extra:

    pip install -U "flwr[simulation]"
"""


def start_simulation(*args: Any, **kwargs: Any) -> Any:
    """Start a Ray-based Flower simulation server."""
    if importlib.util.find_spec("ray") is None:
        raise ImportError(RAY_IMPORT_ERROR)

    # pylint: disable-next=import-outside-toplevel
    from flwr.simulation.legacy_app import start_simulation as start_simulation_legacy

    return start_simulation_legacy(*args, **kwargs)


__all__ = [
    "SimulationIoConnection",
    "run_simulation",
    "run_simulation_process",
    "start_simulation",
]
