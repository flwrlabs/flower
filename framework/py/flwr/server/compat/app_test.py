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
"""Tests for lifecycle event delivery in the compatibility server app."""


from unittest.mock import Mock

from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611

from .app import _compose_event_callbacks


def test_composed_event_callbacks_preserve_existing_callback() -> None:
    """Test both the app callback and Grid delivery receive the event."""
    existing_callback = Mock()
    delivery_callback = Mock()
    callback = _compose_event_callbacks(existing_callback, delivery_callback)
    event = TaskEvent(event="fl.run.started", data='{"type":"fl.run.started"}')

    callback(event)

    existing_callback.assert_called_once_with(event)
    delivery_callback.assert_called_once_with(event)


def test_composed_event_callbacks_continue_after_callback_failure() -> None:
    """Test a failing app callback does not prevent Grid delivery."""
    delivery_callback = Mock()
    callback = _compose_event_callbacks(
        Mock(side_effect=RuntimeError("callback failed")), delivery_callback
    )
    event = TaskEvent(event="fl.run.started", data='{"type":"fl.run.started"}')

    callback(event)

    delivery_callback.assert_called_once_with(event)
