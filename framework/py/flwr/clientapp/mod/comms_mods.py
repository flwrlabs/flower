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
"""Mods that report statistics about message communication."""


from logging import INFO

import numpy as np

from flwr.app import Context, Message
from flwr.clientapp.typing import ClientAppCallable
from flwr.common.logger import log


def message_size_mod(
    msg: Message, ctxt: Context, call_next: ClientAppCallable
) -> Message:
    """Message size mod.

    This mod logs the size in bytes of the message being transmited.
    """
    # Log the size of the incoming message in bytes
    total_bytes = sum(record.count_bytes() for record in msg.content.values())
    log(INFO, "Incoming message size: %i bytes", total_bytes)

    # Call the next layer
    msg = call_next(msg, ctxt)

    # Log the size of the outgoing message in bytes
    total_bytes = sum(record.count_bytes() for record in msg.content.values())
    log(INFO, "Outgoing message size: %i bytes", total_bytes)
    return msg


def arrays_size_mod(
    msg: Message, ctxt: Context, call_next: ClientAppCallable
) -> Message:
    """Arrays size mod.

    This mod logs the number of array elements transmitted in ``ArrayRecord`` objects
    of the message as well as their sizes in bytes.
    """
    # Log the ArrayRecord size statistics and the total size in the incoming message
    array_record_size_stats = _get_array_record_size_stats(msg)
    total_bytes = sum(stat["bytes"] for stat in array_record_size_stats.values())
    if array_record_size_stats:
        log(INFO, "Incoming `ArrayRecord` size statistics:")
        log(INFO, array_record_size_stats)
    log(INFO, "Total array elements received: %i bytes", total_bytes)

    msg = call_next(msg, ctxt)

    # Log the ArrayRecord size statistics and the total size in the outgoing message
    array_record_size_stats = _get_array_record_size_stats(msg)
    total_bytes = sum(stat["bytes"] for stat in array_record_size_stats.values())
    if array_record_size_stats:
        log(INFO, "Outgoing `ArrayRecord` size statistics:")
        log(INFO, array_record_size_stats)
    log(INFO, "Total array elements sent: %i bytes", total_bytes)
    return msg


def _get_array_record_size_stats(
    msg: Message,
) -> dict[str, dict[str, int]]:
    """Get `ArrayRecord` size statistics from the message."""
    array_record_size_stats = {}
    for record_name, arr_record in msg.content.array_records.items():
        arr_record_bytes = arr_record.count_bytes()
        element_count = 0
        for array in arr_record.values():
            element_count += (
                int(np.prod(array.shape)) if array.shape else array.numpy().size
            )

        array_record_size_stats[record_name] = {
            "elements": element_count,
            "bytes": arr_record_bytes,
        }
    return array_record_size_stats
