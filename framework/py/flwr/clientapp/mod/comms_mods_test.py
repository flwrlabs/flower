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
"""Tests for clientapp communication mods."""


import numpy as np

from flwr.app import Array, ArrayRecord, Context, Message, RecordDict
from flwr.app.message_type import MessageType
from flwr.common import NDArray

from .comms_mods import arrays_size_mod, message_size_mod


def _make_context() -> Context:
    return Context(
        run_id=1,
        node_id=1,
        node_config={},
        state=RecordDict(),
        run_config={},
    )


def _make_message(value: NDArray) -> Message:
    return Message(
        content=RecordDict({"arrays": ArrayRecord({"w": Array(value)})}),
        dst_node_id=1,
        message_type=MessageType.TRAIN,
    )


def test_message_size_mod_calls_next_layer() -> None:
    """message_size_mod should return the next layer's output."""
    msg = _make_message(np.array([1.0, 2.0]))
    out_msg = Message(RecordDict(), reply_to=msg)

    actual = message_size_mod(msg, _make_context(), lambda _msg, _ctxt: out_msg)

    assert actual is out_msg


def test_arrays_size_mod_calls_next_layer() -> None:
    """arrays_size_mod should return the next layer's output."""
    msg = _make_message(np.array(1.0))
    out_msg = Message(RecordDict(), reply_to=msg)

    actual = arrays_size_mod(msg, _make_context(), lambda _msg, _ctxt: out_msg)

    assert actual is out_msg
