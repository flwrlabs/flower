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
"""Krum tests."""


import numpy as np

from flwr.app import ArrayRecord

from .krum import Krum
from .multikrum import MultiKrum
from .strategy_utils_test import create_mock_reply


def test_aggregate_train_krum() -> None:
    """Tests if Krum is aggregating correctly."""
    # Prepare
    strategy = Krum(num_malicious_nodes=1)
    reply_0 = create_mock_reply(
        ArrayRecord([np.array([0.2, 0.2, 0.2, 0.2])]), num_examples=5
    )
    reply_1 = create_mock_reply(
        ArrayRecord([np.array([0.7, 0.7, 0.7, 0.7])]), num_examples=5
    )
    reply_2 = create_mock_reply(
        ArrayRecord([np.array([1.0, 1.0, 1.0, 1.0])]), num_examples=5
    )
    replies = [reply_0, reply_1, reply_2]

    expected = ArrayRecord([np.array([0.7, 0.7, 0.7, 0.7])])

    # Execute
    actual, _ = strategy.aggregate_train(server_round=1, replies=replies)
    assert actual
    assert actual.object_id == expected.object_id


def test_aggregate_train_multikrum() -> None:
    """Tests if multi-Krum is aggregating correctly."""
    # Prepare
    strategy = MultiKrum(num_malicious_nodes=1, num_nodes_to_select=2)
    reply_0 = create_mock_reply(
        ArrayRecord([np.array([0.2, 0.2, 0.2, 0.2])]), num_examples=5
    )
    reply_1 = create_mock_reply(
        ArrayRecord([np.array([0.5, 0.5, 0.5, 0.5])]), num_examples=5
    )
    reply_2 = create_mock_reply(
        ArrayRecord([np.array([1.0, 1.0, 1.0, 1.0])]), num_examples=5
    )
    replies = [reply_0, reply_1, reply_2]

    expected = ArrayRecord([np.array([0.35, 0.35, 0.35, 0.35])])

    # Execute
    actual, _ = strategy.aggregate_train(server_round=1, replies=replies)
    assert actual
    assert actual.object_id == expected.object_id
