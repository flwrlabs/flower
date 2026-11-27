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
"""Tests for the simulation Ray actor."""


from __future__ import annotations

import random
from unittest.mock import Mock

import pytest

from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.common.constant import PARTITION_ID_KEY
from flwr.common.seeding import RANDOM_SEED_KEY, clear_run_seed, client_seed
from flwr.supercore.task_identity import TaskIdentity

from .ray_actor import VirtualClientEngineActor


def test_actor_seeds_before_loading_the_client_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loader's first RNG draw follows the derived client seed."""
    monkeypatch.setattr(TaskIdentity, "_task_id", 1)
    monkeypatch.setattr(TaskIdentity, "_run_id", 1)
    monkeypatch.setattr(TaskIdentity, "_node_id", 0)
    context = Context(
        run_id=1,
        node_id=4,
        node_config={PARTITION_ID_KEY: 2},
        state=RecordDict(),
        run_config={RANDOM_SEED_KEY: 11},
    )
    message = Message(
        RecordDict({"config": ConfigRecord({"server-round": 1})}),
        dst_node_id=4,
        message_type="train",
    )
    py_state = random.getstate()
    try:
        random.seed(client_seed(11, context, message))
        expected = random.random()
        draws: list[float] = []

        def load() -> Mock:
            draws.append(random.random())
            return Mock(return_value=Mock())

        VirtualClientEngineActor().run(load, message, "2", context)
        assert draws == [expected]
    finally:
        random.setstate(py_state)
        clear_run_seed()
