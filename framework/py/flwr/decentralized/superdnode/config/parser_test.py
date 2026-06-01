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
"""Unit tests for SuperDNode run parser."""

from pathlib import Path

from flwr.decentralized.superdnode.config.parser import _parse_args_run


def test_parse_args_run_defaults() -> None:
    """Parser should expose expected deploy/simulation defaults."""
    parser = _parse_args_run()
    args = parser.parse_args(["--context", "ctx"])

    assert args.execution_mode == "simulation"
    assert args.timeout == 500
    assert args.nodeapps_pyproject == Path("pyproject.toml")


def test_parse_args_run_parses_simulation_flags() -> None:
    """Parser should include simulation-specific arguments."""
    parser = _parse_args_run()
    args = parser.parse_args(
        [
            "--context",
            "ctx",
            "--execution-mode",
            "simulation",
            "--nb-nodes",
            "12",
            "--sim-timeout",
            "44",
            "--no-enable-sampling",
        ]
    )

    assert args.nb_nodes == 12
    assert args.sim_timeout == 44
    assert args.enable_sampling is False
