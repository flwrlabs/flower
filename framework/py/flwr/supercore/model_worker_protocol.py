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
"""Compatibility imports for the prestarted Model worker protocol."""

from flwr.supercore.task_worker_protocol import (
    MAX_PROTOCOL_MESSAGE_BYTES,
    MessageChannel,
    ProtocolOutputSender,
    encode_message,
    read_message,
    relay_task_output,
    send_message,
)

__all__ = [
    "MAX_PROTOCOL_MESSAGE_BYTES",
    "MessageChannel",
    "ProtocolOutputSender",
    "encode_message",
    "read_message",
    "relay_task_output",
    "send_message",
]
