# Copyright 2026 Inria (cyrille kenfack & davide frey). All Rights Reserved.
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
"""Unit tests for simulation CLI arg registration."""

import argparse

from flwr.decentralized.simulation.args import add_simulation_args


def test_add_simulation_args_registers_defaults() -> None:
    """Simulation parser should expose stable defaults."""
    parser = argparse.ArgumentParser()
    add_simulation_args(parser)
    args = parser.parse_args([])

    assert args.nb_nodes == 10
    assert args.sim_timeout == 300
    assert args.enable_sampling is True
    assert args.topology_kind == "ring"


def test_add_simulation_args_boolean_optional_flag_parsing() -> None:
    """BooleanOptionalAction should support explicit disable flag."""
    parser = argparse.ArgumentParser()
    add_simulation_args(parser)
    args = parser.parse_args(["--no-enable-sampling"])

    assert args.enable_sampling is False
