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

from flwr.simulation.app import run_simulation_process
from flwr.simulation.simulationio_connection import SimulationIoConnection

is_ray_installed = importlib.util.find_spec("ray") is not None

if is_ray_installed:
    from flwr.simulation.legacy_app import start_simulation
else:
    RAY_IMPORT_ERROR: str = """Unable to import module `ray`.

To install the necessary dependencies, install `flwr` with the `simulation` extra:

    pip install -U "flwr[simulation]"
"""

    def start_simulation(*args, **kwargs):  # type: ignore
        """Log error stating that module `ray` could not be imported."""
        raise ImportError(RAY_IMPORT_ERROR)


__all__ = [
    "SimulationIoConnection",
    "run_simulation_process",
    "start_simulation",
]
