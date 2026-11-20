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
"""Tests for run-level random seeding."""


from __future__ import annotations

import random
import sys
from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest
from pytest import MonkeyPatch

from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.common.constant import PARTITION_ID_KEY
from flwr.common.seeding import (
    RANDOM_SEED_KEY,
    clear_run_seed,
    client_seed,
    derive_seed,
    read_random_seed,
    sample_population,
    seed_client,
    seed_random,
    seed_server,
)
from flwr.supercore.task_identity import TaskIdentity


@pytest.fixture(autouse=True)
def _restore_rng_state() -> Iterator[None]:
    """Restore global RNGs mutated by seeding tests."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    yield
    random.setstate(py_state)
    np.random.set_state(np_state)
    clear_run_seed()


def test_read_random_seed_absent() -> None:
    """Missing config and non-mappings leave seeding disabled."""
    assert read_random_seed({}) is None
    assert read_random_seed(object()) is None


def test_read_random_seed_accepts_integral_values() -> None:
    """Integers and integral floats are valid seeds."""
    assert read_random_seed({RANDOM_SEED_KEY: 7}) == 7
    assert read_random_seed({RANDOM_SEED_KEY: 7.0}) == 7


@pytest.mark.parametrize("value", [True, False, "7", 1.5, -1, 2**32])
def test_read_random_seed_rejects_invalid_values(value: object) -> None:
    """Reject booleans, strings, fractions, and out-of-range numbers."""
    with pytest.raises(ValueError, match=RANDOM_SEED_KEY):
        read_random_seed({RANDOM_SEED_KEY: value})


def test_seed_random_repeats_python_and_numpy_streams() -> None:
    """The same seed reproduces the global Python and NumPy streams."""
    seed_random(9)
    py_draw = random.random()
    np_draw = np.random.random(3).copy()
    seed_random(9)
    assert random.random() == py_draw
    assert np.array_equal(np.random.random(3), np_draw)


def test_seed_random_seeds_already_imported_frameworks(
    monkeypatch: MonkeyPatch,
) -> None:
    """PyTorch and TensorFlow are seeded only when already imported."""
    torch_seeds: list[int] = []
    tf_seeds: list[int] = []

    torch = SimpleNamespace(
        manual_seed=torch_seeds.append,
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    tensorflow = SimpleNamespace(random=SimpleNamespace(set_seed=tf_seeds.append))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow)

    seed_random(3)

    assert torch_seeds == [3]
    assert tf_seeds == [3]


def test_derive_seed_is_stable_and_distinguishes_parts() -> None:
    """Derived seeds depend on every identity part and stay in range."""
    first = derive_seed(4, "partition:0", "train", "", "1")
    assert first == derive_seed(4, "partition:0", "train", "", "1")
    assert 0 <= first <= 2**32 - 1
    assert first != derive_seed(4, "partition:1", "train", "", "1")
    assert first != derive_seed(4, "partition:0", "train", "", "2")
    assert first != derive_seed(5, "partition:0", "train", "", "1")


def test_sample_population_is_stable_when_seeded() -> None:
    """Seeded sampling ignores candidate order and global RNG draws."""
    seed_random(4)
    first = sample_population([10, 2, 7, 4, 8], 3)
    seed_random(4)
    _ = random.uniform(0.0, 1.0)
    second = sample_population([8, 4, 7, 2, 10], 3)
    assert first == second


def test_sample_population_preserves_order_without_a_seed() -> None:
    """Unseeded sampling matches the global random module."""
    clear_run_seed()
    population = [3, 1, 2]
    random.seed(0)
    expected = random.sample(population, 2)
    random.seed(0)
    assert sample_population(population, 2) == expected


def test_seed_server_clears_the_sampler_when_the_key_is_absent() -> None:
    """A later unseeded ServerApp does not keep the previous sampler."""
    seed_server({RANDOM_SEED_KEY: 4})
    assert seed_server({}) is None
    population = [3, 1, 2]
    random.seed(0)
    expected = random.sample(population, 2)
    random.seed(0)
    assert sample_population(population, 2) == expected


def test_seeded_sample_orders_numeric_cids_independently_of_input() -> None:
    """``"1"`` and ``"01"`` stay distinct, and a superscript digit does not crash."""
    population = ["01", "1", "²", "10"]
    seed_random(2)
    first = sample_population(population, len(population))
    seed_random(2)
    second = sample_population(["²", "10", "1", "01"], len(population))
    assert first == second
    assert set(first) == set(population)


def test_seed_server_uses_the_configured_seed() -> None:
    """The server process starts at the configured seed."""
    assert seed_server({}) is None
    assert seed_server({RANDOM_SEED_KEY: 7}) == 7
    draw = random.random()
    random.seed(7)
    assert draw == random.random()


def test_seed_client_depends_on_partition_and_round(
    monkeypatch: MonkeyPatch,
) -> None:
    """Client seeds differ by partition and server round."""
    _set_task_identity(monkeypatch)
    message = _train_message(server_round=1)
    first = _context(partition_id=0)
    second = _context(partition_id=1)
    later = _train_message(server_round=2)

    assert seed_client(first, message) == client_seed(11, first, message)
    assert client_seed(11, first, message) != client_seed(11, second, message)
    assert client_seed(11, first, message) != client_seed(11, first, later)


def _set_task_identity(monkeypatch: MonkeyPatch) -> None:
    """Allow instruction messages to be constructed in unit tests."""
    monkeypatch.setattr(TaskIdentity, "_task_id", 1)
    monkeypatch.setattr(TaskIdentity, "_run_id", 1)
    monkeypatch.setattr(TaskIdentity, "_node_id", 0)


def _context(partition_id: int) -> Context:
    """Return a client context with a partition id and run seed."""
    return Context(
        run_id=1,
        node_id=100 + partition_id,
        node_config={PARTITION_ID_KEY: partition_id},
        state=RecordDict(),
        run_config={RANDOM_SEED_KEY: 11},
    )


def _train_message(server_round: int) -> Message:
    """Return a train message carrying a server round."""
    return Message(
        RecordDict({"config": ConfigRecord({"server-round": server_round})}),
        dst_node_id=1,
        message_type="train",
    )
