#
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
"""Flad tests."""


from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)

from .strategy import Flad

CLIENT_NAMES = ["client_0", "client_1", "client_2"]
INITIAL_WEIGHTS = np.array([1.0, 1.0])
CLI_WEIGHTS = [np.array([10.0, 10.0]), np.array([20.0, 20.0]), np.array([30.0, 30.0])]


#
# utitily functions
#
def make_strategy() -> Flad:
    """Create a Flad strategy with clients mapped to node IDs."""
    strategy = Flad(client_names=",".join(CLIENT_NAMES))
    for i, client in enumerate(strategy.clients):
        client["id"] = i
    strategy.is_mapped = True
    return strategy


def make_mock_grid(node_ids: list[int]) -> Mock:
    """Create a mock Grid returning the given node IDs."""
    grid = Mock()
    grid.get_node_ids.return_value = node_ids
    return grid


def create_mock_reply(
    node_id: int,
    arrayrecord_key: str = "arrays",
    arrays: ArrayRecord | None = None,
    metrics: dict[str, Any] | None = None,
    has_error: bool = False,
) -> Message:
    """Create a mock Message sent by the client identified by `node_id`."""
    message = Mock(spec=Message)
    message.metadata = Mock(src_node_id=node_id)
    message.has_error.return_value = has_error

    content: dict[str, Any] = {}
    if arrays is not None:
        content[arrayrecord_key] = arrays
    if metrics is not None:
        content["metrics"] = MetricRecord(metrics)
    message.content = RecordDict(content)
    return message


def create_mock_mapping_reply(
    node_id: int, client_name: str, configrecord_key: str = "config"
) -> Message:
    """Create a mock reply to the `_map_clients()` query."""
    message = Mock(spec=Message)
    message.metadata = Mock(src_node_id=node_id)
    message.has_error.return_value = False
    message.content = RecordDict(
        {configrecord_key: ConfigRecord({"name": client_name})}
    )
    return message


#
# configure_train
#
def test_configure_train_maps_clients_and_selects_participants() -> None:
    """Map clients, select participants and build a train message per participant."""
    # Prepare
    strategy = Flad(client_names=",".join(CLIENT_NAMES))
    node_ids = [0, 1, 2]
    grid = make_mock_grid(node_ids)
    grid.send_and_receive.return_value = [
        create_mock_mapping_reply(nid, name)
        for nid, name in zip(node_ids, CLIENT_NAMES, strict=True)
    ]

    arrays = ArrayRecord([INITIAL_WEIGHTS])
    config = ConfigRecord(
        {
            "min_epochs": 1,
            "max_epochs": 5,
            "min_steps": 1,
            "max_steps": 10,
            "patience": 3,
        }
    )

    # Execute
    messages = list(strategy.configure_train(1, arrays, config, grid))

    # Assert
    assert strategy.is_mapped
    assert {client["name"]: client["id"] for client in strategy.clients} == {
        "client_0": 0,
        "client_1": 1,
        "client_2": 2,
    }

    # All clients should be selected since their f1_score (0) is
    # <= last_round_average_f1 (1.0).
    assert len(messages) == len(CLIENT_NAMES)
    for message in messages:
        assert message.metadata.message_type == MessageType.TRAIN


def test_configure_train_selects_only_clients_below_last_round_f1() -> None:
    """Only clients with f1_score <= last_round_average_f1 should be selected."""
    # Prepare
    strategy = make_strategy()
    strategy.last_round_average_f1 = 0.5
    strategy.clients[0]["f1_score"] = 0.9  # excluded
    strategy.clients[1]["f1_score"] = 0.4  # included
    strategy.clients[2]["f1_score"] = 0.3  # included
    grid = make_mock_grid([client["id"] for client in strategy.clients])

    arrays = ArrayRecord([INITIAL_WEIGHTS])
    config = ConfigRecord(
        {
            "min_epochs": 1,
            "max_epochs": 5,
            "min_steps": 1,
            "max_steps": 10,
            "patience": 3,
        }
    )

    # Execute
    messages = list(strategy.configure_train(1, arrays, config, grid))

    # Assert
    selected_names = {client["name"] for client in strategy.round_participants}
    assert selected_names == {"client_1", "client_2"}
    assert len(messages) == 2


#
# aggregate_train
#
def test_aggregate_train_averages_all_clients_unweighted() -> None:
    """Aggregation should be a simple unweighted average across all clients."""
    # Prepare
    strategy = make_strategy()
    strategy.round_participants = strategy.clients[:2]  # client_0, client_1
    # client_2 does not reply, so its arrays are not overwritten by training
    # phase and must be set because aggregate_train averages over all
    # clients, not just selected participants.
    strategy.clients[2][strategy.arrayrecord_key] = ArrayRecord([CLI_WEIGHTS[2]])

    replies = [
        create_mock_reply(
            node_id=strategy.clients[0]["id"],
            arrays=ArrayRecord([CLI_WEIGHTS[0]]),
        ),
        create_mock_reply(
            node_id=strategy.clients[1]["id"],
            arrays=ArrayRecord([CLI_WEIGHTS[1]]),
        ),
    ]

    expected = np.array([(10.0 + 20.0 + 30.0) / 3, (10.0 + 20.0 + 30.0) / 3])

    # Execute
    actual_aggregated, _ = strategy.aggregate_train(1, replies)

    # Assert
    actual = actual_aggregated.to_numpy_ndarrays()[0]
    np.testing.assert_allclose(actual, expected)


#
# configure_evaluate
#
def test_configure_evaluate_selects_all_clients() -> None:
    """configure_evaluate should consider every client, regardless of
    f1_score."""
    # Prepare
    strategy = make_strategy()
    strategy.last_round_average_f1 = 0.0  # would exclude everyone if applied
    strategy.clients[0]["f1_score"] = 0.9
    grid = make_mock_grid([client["id"] for client in strategy.clients])

    arrays = ArrayRecord([INITIAL_WEIGHTS])
    config = ConfigRecord({})

    # Execute
    messages = list(strategy.configure_evaluate(1, arrays, config, grid))

    # Assert
    assert len(messages) == len(strategy.clients)
    for message in messages:
        assert message.metadata.message_type == MessageType.EVALUATE


#
# aggregate_evaluate
#
def test_aggregate_evaluate_averages_f1_scores() -> None:
    """The aggregated 'avg_f1_score' should be the mean of per-client f1
    scores, and 'avg_f1_score_best' should track the best avg f1 score."""
    # Prepare
    strategy = make_strategy()
    strategy.round_participants = strategy.clients
    f1_scores = [0.2, 0.5, 0.8]

    replies = [
        create_mock_reply(
            node_id=client["id"],
            metrics={"f1_score": f1},
        )
        for client, f1 in zip(strategy.clients, f1_scores, strict=True)
    ]

    # Execute
    metric_record = strategy.aggregate_evaluate(1, replies)

    # Assert
    expected_avg = float(np.average(f1_scores))
    assert metric_record["avg_f1_score"] == pytest.approx(expected_avg)
    assert metric_record["avg_f1_score_best"] == pytest.approx(expected_avg)
    assert strategy.best_average_f1 == pytest.approx(expected_avg)
    for client, f1 in zip(strategy.clients, f1_scores, strict=True):
        assert client["f1_score"] == f1


#
# start
#
def test_start() -> None:
    """Test the start() method for a number of rounds until stop condition is met."""
    # Prepare
    strategy = make_strategy()
    node_ids = [client["id"] for client in strategy.clients]
    grid = make_mock_grid(node_ids)
    initial_arrays = ArrayRecord([INITIAL_WEIGHTS])
    train_config = ConfigRecord(
        {
            "min_epochs": 1,
            "max_epochs": 5,
            "min_steps": 1,
            "max_steps": 10,
            "patience": 2,
        }
    )

    # Per-round f1_score reported by every client: a peak at round 2 (0.9),
    # followed by 3 consecutive below-peak rounds. With patience=2, the stop
    # counter needs to exceed 2 (i.e. reach 3) to stop, which happens right
    # at the end of round 5.
    f1_scores = [0.5, 0.9, 0.7, 0.7, 0.7]
    side_effect = []
    for f1_score in f1_scores:
        side_effect.append(
            [
                create_mock_reply(node_id=nid, arrays=ArrayRecord([CLI_WEIGHTS[nid]]))
                for nid in node_ids
            ]
        )
        side_effect.append(
            [
                create_mock_reply(node_id=nid, metrics={"f1_score": f1_score})
                for nid in node_ids
            ]
        )
    grid.send_and_receive.side_effect = side_effect
    expected_avg_f1_scores = 0.7
    expected_avg_f1_best_scores = 0.9
    expected_weights = np.array([(10.0 + 20.0 + 30.0) / 3] * 2)

    # Execute
    actual = strategy.start(grid, initial_arrays, train_config=train_config)

    # Assert
    last_round = len(f1_scores)
    actual_weights = actual.arrays.to_numpy_ndarrays()[0]

    assert actual.evaluate_metrics_clientapp[last_round][
        "avg_f1_score"
    ] == pytest.approx(expected_avg_f1_scores)
    assert actual.evaluate_metrics_clientapp[last_round][
        "avg_f1_score_best"
    ] == pytest.approx(expected_avg_f1_best_scores)
    np.testing.assert_allclose(actual_weights, expected_weights)
