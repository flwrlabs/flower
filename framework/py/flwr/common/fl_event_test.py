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
"""Federated learning lifecycle event tests."""

import pytest

from flwr.supercore.corestate.utils import validate_task_event_data
from flwr.supercore.utils import strict_json_loads

from .fl_event import (
    FL_NODE_EVALUATE_COMPLETED,
    FL_NODE_EVALUATE_FAILED,
    FL_NODE_EVALUATE_STARTED,
    FL_NODE_FIT_COMPLETED,
    FL_NODE_FIT_FAILED,
    FL_NODE_FIT_STARTED,
    FL_ROUND_COMPLETED,
    FL_ROUND_EVALUATE_COMPLETED,
    FL_ROUND_EVALUATE_FAILED,
    FL_ROUND_EVALUATE_STARTED,
    FL_ROUND_FAILED,
    FL_ROUND_FIT_COMPLETED,
    FL_ROUND_FIT_FAILED,
    FL_ROUND_FIT_STARTED,
    FL_ROUND_STARTED,
    FL_RUN_COMPLETED,
    FL_RUN_FAILED,
    FL_RUN_STARTED,
    make_task_event,
)


def test_event_name_constants() -> None:
    """Event name constants follow the dotted naming pattern."""
    assert (FL_RUN_STARTED, FL_RUN_COMPLETED, FL_RUN_FAILED) == (
        "fl.run.started",
        "fl.run.completed",
        "fl.run.failed",
    )
    assert (FL_ROUND_STARTED, FL_ROUND_COMPLETED, FL_ROUND_FAILED) == (
        "fl.round.started",
        "fl.round.completed",
        "fl.round.failed",
    )
    assert (FL_ROUND_FIT_STARTED, FL_ROUND_FIT_COMPLETED, FL_ROUND_FIT_FAILED) == (
        "fl.round.fit.started",
        "fl.round.fit.completed",
        "fl.round.fit.failed",
    )
    assert (
        FL_ROUND_EVALUATE_STARTED,
        FL_ROUND_EVALUATE_COMPLETED,
        FL_ROUND_EVALUATE_FAILED,
    ) == (
        "fl.round.evaluate.started",
        "fl.round.evaluate.completed",
        "fl.round.evaluate.failed",
    )
    assert (FL_NODE_FIT_STARTED, FL_NODE_FIT_COMPLETED, FL_NODE_FIT_FAILED) == (
        "fl.node.fit.started",
        "fl.node.fit.completed",
        "fl.node.fit.failed",
    )
    assert (
        FL_NODE_EVALUATE_STARTED,
        FL_NODE_EVALUATE_COMPLETED,
        FL_NODE_EVALUATE_FAILED,
    ) == (
        "fl.node.evaluate.started",
        "fl.node.evaluate.completed",
        "fl.node.evaluate.failed",
    )


def test_make_task_event_minimal() -> None:
    """Create an event without optional fields."""
    # Execute
    event = make_task_event(FL_RUN_STARTED)

    # Assert
    assert event.event == FL_RUN_STARTED
    assert strict_json_loads(event.data) == {"type": FL_RUN_STARTED}
    validate_task_event_data(event.data)


def test_make_task_event_with_node_id_and_round() -> None:
    """Create an event with node ID and server round."""
    # Execute
    event = make_task_event(FL_NODE_FIT_STARTED, node_id=123, server_round=2)

    # Assert
    assert event.event == FL_NODE_FIT_STARTED
    assert strict_json_loads(event.data) == {
        "type": FL_NODE_FIT_STARTED,
        "node_id": 123,
        "server_round": 2,
    }
    validate_task_event_data(event.data)


def test_make_task_event_with_metadata() -> None:
    """Create an event with additional metadata entries."""
    # Execute
    event = make_task_event(
        FL_ROUND_FIT_COMPLETED,
        server_round=1,
        metadata={"num_results": 3, "num_failures": 1},
    )

    # Assert
    assert strict_json_loads(event.data) == {
        "type": FL_ROUND_FIT_COMPLETED,
        "server_round": 1,
        "num_results": 3,
        "num_failures": 1,
    }


def test_make_task_event_failure_metadata() -> None:
    """Create a failure event carrying `error` and `details`."""
    # Execute
    event = make_task_event(
        FL_RUN_FAILED,
        metadata={"error": "RuntimeError", "details": "boom"},
    )

    # Assert
    assert strict_json_loads(event.data) == {
        "type": FL_RUN_FAILED,
        "error": "RuntimeError",
        "details": "boom",
    }


def test_make_task_event_rejects_non_finite_numbers() -> None:
    """Reject metadata containing non-finite floating-point values."""
    with pytest.raises(ValueError):
        make_task_event(FL_RUN_COMPLETED, metadata={"elapsed_time": float("nan")})
