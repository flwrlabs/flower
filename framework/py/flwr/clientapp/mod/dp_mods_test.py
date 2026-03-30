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
"""Tests for clientapp DP mods."""


import numpy as np

from flwr.app.message_type import MessageType
from flwr.common import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    NDArray,
    RecordDict,
)
from flwr.common.differential_privacy_constants import KEY_CLIPPING_NORM, KEY_NORM_BIT

from .centraldp_mods import adaptiveclipping_mod, fixedclipping_mod
from .localdp_mod import LocalDpMod


def _make_context() -> Context:
    return Context(
        run_id=1,
        node_id=1,
        node_config={},
        state=RecordDict(),
        run_config={},
    )


def _make_train_message(
    server_value: NDArray, clipping_norm: float | None = None
) -> Message:
    content = RecordDict(
        {
            "arrays": ArrayRecord({"w": Array(server_value)}),
        }
    )
    if clipping_norm is not None:
        content["configs"] = ConfigRecord({KEY_CLIPPING_NORM: clipping_norm})
    return Message(content=content, dst_node_id=1, message_type=MessageType.TRAIN)


def _make_reply_message(
    msg: Message, client_value: NDArray, include_metrics: bool = False
) -> Message:
    content = RecordDict({"arrays": ArrayRecord({"w": Array(client_value)})})
    if include_metrics:
        content["metrics"] = MetricRecord({"loss": 0.5})
    return Message(content=content, reply_to=msg)


def test_fixedclipping_mod_handles_scalar_ndarray() -> None:
    """Fixed clipping should preserve scalar ndarray outputs."""
    msg = _make_train_message(np.array(0.0), clipping_norm=1.0)

    out_msg = fixedclipping_mod(
        msg,
        _make_context(),
        lambda in_msg, _ctxt: _make_reply_message(in_msg, np.array(2.0)),
    )

    out_array = out_msg.content.array_records["arrays"]["w"].numpy()
    assert isinstance(out_array, np.ndarray)
    assert out_array.shape == ()
    np.testing.assert_array_equal(out_array, np.array(2.0))


def test_adaptiveclipping_mod_handles_scalar_ndarray() -> None:
    """Adaptive clipping should preserve scalar ndarray outputs."""
    msg = _make_train_message(np.array(0.0), clipping_norm=1.0)

    out_msg = adaptiveclipping_mod(
        msg,
        _make_context(),
        lambda in_msg, _ctxt: _make_reply_message(
            in_msg, np.array(2.0), include_metrics=True
        ),
    )

    out_array = out_msg.content.array_records["arrays"]["w"].numpy()
    assert isinstance(out_array, np.ndarray)
    assert out_array.shape == ()
    np.testing.assert_array_equal(out_array, np.array(2.0))
    assert out_msg.content.metric_records["metrics"][KEY_NORM_BIT] == 1


def test_localdp_mod_handles_scalar_ndarray() -> None:
    """Local DP should preserve scalar ndarray outputs."""
    msg = _make_train_message(np.array(0.0))
    mod = LocalDpMod(
        clipping_norm=1.0,
        sensitivity=0.0,
        epsilon=1.0,
        delta=0.1,
    )

    out_msg = mod(
        msg,
        _make_context(),
        lambda in_msg, _ctxt: _make_reply_message(in_msg, np.array(2.0)),
    )

    out_array = out_msg.content.array_records["arrays"]["w"].numpy()
    assert isinstance(out_array, np.ndarray)
    assert out_array.shape == ()
    np.testing.assert_array_equal(out_array, np.array(2.0))
