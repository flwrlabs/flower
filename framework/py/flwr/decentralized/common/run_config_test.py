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
"""Unit tests for decentralized run config helpers."""

from flwr.decentralized.common.run_config import DLRunConfig


def test_get_cycles_with_positive_aggregation_steps() -> None:
    """Compute finite cycle count when aggregation steps are enabled."""
    cfg = DLRunConfig(rounds=2, n_aggregation_steps=2)

    assert cfg.get_steps_per_round() == 3
    assert cfg.get_cycles() == 7


def test_get_cycles_disabled_aggregation_returns_infinite_marker() -> None:
    """Return -1 marker when no aggregation steps are configured."""
    cfg = DLRunConfig(rounds=5, n_aggregation_steps=0)

    assert cfg.get_steps_per_round() == 1
    assert cfg.get_cycles() == -1


def test_none_aggregation_steps_defaults_to_one() -> None:
    """Treat `None` aggregation steps as one effective step."""
    cfg = DLRunConfig(rounds=1, n_aggregation_steps=None)  # type: ignore[arg-type]

    assert cfg.get_steps_per_round() == 2
    assert cfg.get_cycles() == 3
