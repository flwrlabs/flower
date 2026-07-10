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
"""Framing utilities for protobuf-over-HTTP streams."""


from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

from google.protobuf.message import Message

from flwr.supercore.protobuf.constants import FRAME_HEADER_SIZE


def frame_message(message: Message) -> bytes:
    """Prefix a protobuf message with its four-byte payload size."""
    payload = message.SerializeToString()
    return len(payload).to_bytes(FRAME_HEADER_SIZE, "big") + payload


async def async_iter_framed_bytes(
    messages: AsyncIterable[Message],
) -> AsyncIterator[bytes]:
    """Frame every protobuf message from an asynchronous iterator."""
    async for message in messages:
        yield frame_message(message)
