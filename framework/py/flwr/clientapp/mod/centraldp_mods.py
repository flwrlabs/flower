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
"""Clipping modifiers for central DP with client-side clipping."""


from logging import ERROR, INFO
from typing import cast

import numpy as np

from flwr.app import Error
from flwr.clientapp.typing import ClientAppCallable
from flwr.common import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    log,
)
from flwr.common.constant import ErrorCode
from flwr.common.differential_privacy import (
    compute_adaptive_clip_model_update,
    compute_clip_model_update,
)
from flwr.common.differential_privacy_constants import KEY_CLIPPING_NORM, KEY_NORM_BIT


# pylint: disable=too-many-return-statements
def fixedclipping_mod(
    msg: Message, ctxt: Context, call_next: ClientAppCallable
) -> Message:
    """Client-side fixed clipping modifier.

    This mod needs to be used with the `DifferentialPrivacyClientSideFixedClipping`
    server-side strategy wrapper.

    The wrapper sends the clipping_norm value to the client.

    This mod clips the client model updates before sending them to the server.

    It operates on messages of type `MessageType.TRAIN`.

    Notes
    -----
    Consider the order of mods when using multiple.

    Typically, fixedclipping_mod should be the last to operate on params.
    """
    if len(msg.content.array_records) != 1:
        return _handle_multi_record_err("fixedclipping_mod", msg, ArrayRecord)
    if len(msg.content.config_records) != 1:
        return _handle_multi_record_err("fixedclipping_mod", msg, ConfigRecord)

    # Get keys in the single ConfigRecord
    keys_in_config = set(next(iter(msg.content.config_records.values())).keys())
    if KEY_CLIPPING_NORM not in keys_in_config:
        return _handle_no_key_err("fixedclipping_mod", msg)
    # Record array record communicated to client and clipping norm
    original_array_record = next(iter(msg.content.array_records.values()))
    clipping_norm = cast(
        float, next(iter(msg.content.config_records.values()))[KEY_CLIPPING_NORM]
    )

    # Call inner app
    out_msg = call_next(msg, ctxt)

    # Check if the msg has error
    if out_msg.has_error():
        return out_msg

    # Ensure reply has a single ArrayRecord
    if len(out_msg.content.array_records) != 1:
        return _handle_multi_record_err("fixedclipping_mod", out_msg, ArrayRecord)

    new_array_record_key, client_to_server_arrecord = next(
        iter(out_msg.content.array_records.items())
    )
    # Ensure keys in returned ArrayRecord match those in the one sent from server
    if list(original_array_record.keys()) != list(client_to_server_arrecord.keys()):
        return _handle_array_key_mismatch_err("fixedclipping_mod", out_msg)

    client_to_server_ndarrays = client_to_server_arrecord.to_numpy_ndarrays()
    # Clip the client update
    compute_clip_model_update(
        param1=client_to_server_ndarrays,
        param2=original_array_record.to_numpy_ndarrays(),
        clipping_norm=clipping_norm,
    )

    log(
        INFO, "fixedclipping_mod: parameters are clipped by value: %.4f.", clipping_norm
    )
    # Replace outgoing ArrayRecord's Array while preserving their keys
    out_msg.content.array_records[new_array_record_key] = ArrayRecord(
        {
            k: Array(np.asarray(v))
            for k, v in zip(
                client_to_server_arrecord.keys(),
                client_to_server_ndarrays,
                strict=True,
            )
        }
    )
    return out_msg


def adaptiveclipping_mod(
    msg: Message, ctxt: Context, call_next: ClientAppCallable
) -> Message:
    """Client-side adaptive clipping modifier.

    This mod needs to be used with the DifferentialPrivacyClientSideAdaptiveClipping
    server-side strategy wrapper.

    The wrapper sends the clipping_norm value to the client.

    This mod clips the client model updates before sending them to the server.

    It also sends KEY_NORM_BIT to the server for computing the new clipping value.

    It operates on messages of type `MessageType.TRAIN`.

    Notes
    -----
    Consider the order of mods when using multiple.

    Typically, adaptiveclipping_mod should be the last to operate on params.
    """
    if len(msg.content.array_records) != 1:
        return _handle_multi_record_err("adaptiveclipping_mod", msg, ArrayRecord)
    if len(msg.content.config_records) != 1:
        return _handle_multi_record_err("adaptiveclipping_mod", msg, ConfigRecord)

    # Get keys in the single ConfigRecord
    keys_in_config = set(next(iter(msg.content.config_records.values())).keys())
    if KEY_CLIPPING_NORM not in keys_in_config:
        return _handle_no_key_err("adaptiveclipping_mod", msg)

    # Record array record communicated to client and clipping norm
    original_array_record = next(iter(msg.content.array_records.values()))
    clipping_norm = cast(
        float, next(iter(msg.content.config_records.values()))[KEY_CLIPPING_NORM]
    )

    # Call inner app
    out_msg = call_next(msg, ctxt)

    # Check if the msg has error
    if out_msg.has_error():
        return out_msg

    # Ensure reply has a single ArrayRecord
    if len(out_msg.content.array_records) != 1:
        return _handle_multi_record_err("adaptiveclipping_mod", out_msg, ArrayRecord)

    # Ensure reply has a single MetricRecord
    if len(out_msg.content.metric_records) != 1:
        return _handle_multi_record_err("adaptiveclipping_mod", out_msg, MetricRecord)

    new_array_record_key, client_to_server_arrecord = next(
        iter(out_msg.content.array_records.items())
    )

    # Ensure keys in returned ArrayRecord match those in the one sent from server
    if list(original_array_record.keys()) != list(client_to_server_arrecord.keys()):
        return _handle_array_key_mismatch_err("adaptiveclipping_mod", out_msg)

    client_to_server_ndarrays = client_to_server_arrecord.to_numpy_ndarrays()
    # Clip the client update
    norm_bit = compute_adaptive_clip_model_update(
        client_to_server_ndarrays,
        original_array_record.to_numpy_ndarrays(),
        clipping_norm,
    )
    log(
        INFO,
        "adaptiveclipping_mod: ndarrays are clipped by value: %.4f.",
        clipping_norm,
    )
    # Replace outgoing ArrayRecord's Array while preserving their keys
    out_msg.content.array_records[new_array_record_key] = ArrayRecord(
        {
            k: Array(np.asarray(v))
            for k, v in zip(
                client_to_server_arrecord.keys(),
                client_to_server_ndarrays,
                strict=True,
            )
        }
    )
    # Add to the MetricRecords the norm bit (recall reply messages only contain
    # one MetricRecord)
    metric_record_key = list(out_msg.content.metric_records.keys())[0]
    # We cast it to `int` because MetricRecord does not support `bool` values
    out_msg.content.metric_records[metric_record_key][KEY_NORM_BIT] = int(norm_bit)
    return out_msg


def _handle_err(msg: Message, reason: str) -> Message:
    """Log and return error message."""
    log(ERROR, reason)
    return Message(
        Error(code=ErrorCode.MOD_FAILED_PRECONDITION, reason=reason),
        reply_to=msg,
    )


def _handle_multi_record_err(mod_name: str, msg: Message, record_type: type) -> Message:
    """Log and return multi-record error."""
    cnt = sum(isinstance(_, record_type) for _ in msg.content.values())
    return _handle_err(
        msg,
        f"{mod_name} expects exactly one {record_type.__name__}, "
        f"but found {cnt} {record_type.__name__}(s).",
    )


def _handle_no_key_err(mod_name: str, msg: Message) -> Message:
    """Log and return no-key error."""
    return _handle_err(
        msg,
        f"{mod_name} requires the key '{KEY_CLIPPING_NORM}' to be present in the "
        "ConfigRecord, but it was not found. "
        "Please ensure the `DifferentialPrivacyClientSideFixedClipping` wrapper "
        "is used in the ServerApp.",
    )


def _handle_array_key_mismatch_err(mod_name: str, msg: Message) -> Message:
    """Create array-key-mismatch error reasons."""
    return _handle_err(
        msg,
        f"{mod_name} expects the keys in the ArrayRecord of the reply message to match "
        "those from the ArrayRecord that the ClientApp received, but they do not.",
    )
