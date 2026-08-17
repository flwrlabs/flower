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
"""Federated learning lifecycle events."""

from __future__ import annotations

from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.supercore.typing import JSONObject, JSONValue
from flwr.supercore.utils import strict_json_dumps

# Run lifecycle events
FL_RUN_STARTED = "fl.run.started"
FL_RUN_COMPLETED = "fl.run.completed"
FL_RUN_FAILED = "fl.run.failed"

# Round lifecycle events
FL_ROUND_STARTED = "fl.round.started"
FL_ROUND_COMPLETED = "fl.round.completed"
FL_ROUND_FAILED = "fl.round.failed"

# Round fit events
FL_ROUND_FIT_STARTED = "fl.round.fit.started"
FL_ROUND_FIT_COMPLETED = "fl.round.fit.completed"
FL_ROUND_FIT_FAILED = "fl.round.fit.failed"

# Round evaluate events
FL_ROUND_EVALUATE_STARTED = "fl.round.evaluate.started"
FL_ROUND_EVALUATE_COMPLETED = "fl.round.evaluate.completed"
FL_ROUND_EVALUATE_FAILED = "fl.round.evaluate.failed"

# Node fit events
FL_NODE_FIT_STARTED = "fl.node.fit.started"
FL_NODE_FIT_COMPLETED = "fl.node.fit.completed"
FL_NODE_FIT_FAILED = "fl.node.fit.failed"

# Node evaluate events
FL_NODE_EVALUATE_STARTED = "fl.node.evaluate.started"
FL_NODE_EVALUATE_COMPLETED = "fl.node.evaluate.completed"
FL_NODE_EVALUATE_FAILED = "fl.node.evaluate.failed"


def make_task_event(
    event: str,
    *,
    node_id: int | None = None,
    server_round: int | None = None,
    metadata: dict[str, JSONValue] | None = None,
) -> TaskEvent:
    """Create a ``TaskEvent`` for a federated learning lifecycle event.

    The event payload (``TaskEvent.data``) is a JSON object of the form
    ``{"type": <event>, ...}``, including ``node_id`` and ``server_round``
    when provided, followed by any additional ``metadata`` entries (e.g.
    ``error``/``details`` for failure events).

    Parameters
    ----------
    event : str
        The dotted event name (e.g. ``fl.round.fit.completed``).
    node_id : Optional[int] (default: None)
        The ID of the node the event originates from, if applicable.
    server_round : Optional[int] (default: None)
        The federated learning round the event belongs to, if applicable.
    metadata : Optional[dict[str, JSONValue]] (default: None)
        Additional JSON-serializable payload entries.

    Returns
    -------
    TaskEvent
        A ``TaskEvent`` with ``event`` set and ``data`` containing the JSON
        object payload. ``run_id`` and ``task_id`` are left unset; they are
        assigned by the SuperLink from the authenticated identity.
    """
    data: JSONObject = {"type": event}
    if node_id is not None:
        data["node_id"] = node_id
    if server_round is not None:
        data["server_round"] = server_round
    if metadata:
        data.update(metadata)
    return TaskEvent(event=event, data=strict_json_dumps(data, compact=True))
