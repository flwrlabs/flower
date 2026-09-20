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
"""Bounded protocol and output relay for the prestarted Model worker."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from queue import Empty, Full, Queue
from typing import Any, BinaryIO, Protocol, TextIO, cast

from flwr.common.constant import FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT
from flwr.supercore.typing import JSONObject

MAX_PROTOCOL_MESSAGE_BYTES = 16_384
_MAX_OUTPUT_FRAME_CHARS = 1_024
_MAX_QUEUED_OUTPUT_FRAMES = 64
# Stop draining best-effort logs promptly after task completion.
_OUTPUT_DRAIN_BUDGET_SECONDS = 0.25
# Leave time for the drain budget and one final bounded socket write.
_OUTPUT_CLOSE_TIMEOUT_SECONDS = 0.5


class MessageChannel(Protocol):
    """Minimal buffered byte-stream interface used by the worker protocol."""

    def readline(self, size: int = -1, /) -> bytes:
        """Read at most one protocol line."""

    def write(self, data: bytes, /) -> int:
        """Write protocol bytes."""

    def flush(self) -> None:
        """Flush pending protocol bytes."""


class ProtocolOutputSender:
    """Send bounded output frames without blocking Model task execution."""

    def __init__(self, channel: MessageChannel) -> None:
        self._channel = channel
        self._queue: Queue[JSONObject] = Queue(maxsize=_MAX_QUEUED_OUTPUT_FRAMES)
        self._done = threading.Event()
        self._failed = threading.Event()
        self._returncode: int | None = None
        self._drain_deadline: float | None = None
        self._thread: threading.Thread | None = None
        try:
            thread = threading.Thread(target=self._send, daemon=True)
            thread.start()
            self._thread = thread
        except RuntimeError:
            self._failed.set()

    def write(self, stream: str, output: str) -> None:
        """Queue one or more bounded frames, dropping output on saturation."""
        if self._done.is_set() or self._failed.is_set():
            return
        for start in range(0, len(output), _MAX_OUTPUT_FRAME_CHARS):
            try:
                self._queue.put_nowait(
                    {
                        "event": "output",
                        "stream": stream,
                        "data": output[start : start + _MAX_OUTPUT_FRAME_CHARS],
                    }
                )
            except Full:
                return

    def close(self, returncode: int | None) -> None:
        """Finish output delivery best-effort within a fixed time bound."""
        self._returncode = returncode
        self._drain_deadline = time.monotonic() + _OUTPUT_DRAIN_BUDGET_SECONDS
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=_OUTPUT_CLOSE_TIMEOUT_SECONDS)

    def _send(self) -> None:
        try:
            self._drain_output()
        except Exception:  # pylint: disable=broad-exception-caught
            self._failed.set()
            self._done.wait()
        if self._returncode is not None:
            try:
                send_message(
                    self._channel,
                    {"event": "finished", "returncode": self._returncode},
                )
            except Exception:  # pylint: disable=broad-exception-caught
                self._failed.set()

    def _drain_output(self) -> None:
        """Send queued output until completion or the drain budget expires."""
        while not self._done.is_set() or not self._queue.empty():
            if (
                self._done.is_set()
                and self._drain_deadline is not None
                and time.monotonic() >= self._drain_deadline
            ):
                break
            try:
                payload = self._queue.get(timeout=0.01)
            except Empty:
                continue
            send_message(self._channel, payload)


class _RelayedBinaryOutput:
    """Relay binary writes while preserving the original stream interface."""

    def __init__(self, relay: _RelayedTextOutput, original: BinaryIO) -> None:
        self._relay = relay
        self._original = original

    def write(self, output: bytes | bytearray | memoryview) -> int:
        """Decode and relay one binary write without exposing task authority."""
        raw = bytes(output)
        self._relay.write(raw.decode(self._relay.encoding, errors="replace"))
        return len(raw)

    def write1(self, output: bytes | bytearray | memoryview) -> int:
        """Relay one buffered binary write."""
        return self.write(output)

    def writelines(self, lines: Iterable[bytes]) -> None:
        """Relay binary lines through the redacting text stream."""
        for line in lines:
            self.write(line)

    @property
    def raw(self) -> _RelayedBinaryOutput:
        """Return a redacting proxy for the underlying raw stream."""
        return _RelayedBinaryOutput(
            self._relay, cast(BinaryIO, cast(Any, self._original).raw)
        )

    def flush(self) -> None:
        """Keep the relay's redaction suffix buffered."""
        self._relay.flush()

    def __getattr__(self, name: str) -> Any:
        """Delegate the remaining binary stream interface."""
        return getattr(self._original, name)


class _RelayedTextOutput:
    """Redact authority markers and enqueue one ordered text stream."""

    def __init__(
        self,
        sender: ProtocolOutputSender,
        stream: str,
        secrets: tuple[str, ...],
        original: TextIO,
    ) -> None:
        self._sender = sender
        self._stream = stream
        self._secrets = tuple(secret for secret in secrets if secret)
        self._original = original
        self._pending = ""
        self._lock = threading.Lock()
        self._retained_chars = max(map(len, self._secrets), default=1) - 1

    @property
    def buffer(self) -> _RelayedBinaryOutput:
        """Return a binary view that preserves redaction and output relay."""
        return _RelayedBinaryOutput(
            self, cast(BinaryIO, cast(Any, self._original).buffer)
        )

    @property
    def encoding(self) -> str:
        """Return the original stream encoding."""
        return self._original.encoding or "utf-8"

    def __getattr__(self, name: str) -> Any:
        """Delegate the remaining text stream interface."""
        return getattr(self._original, name)

    def writable(self) -> bool:
        """Return whether the relay accepts text writes."""
        return True

    def isatty(self) -> bool:
        """Return whether the protocol output channel is a terminal."""
        return False

    def write(self, output: str) -> int:
        """Redact and enqueue output without waiting for socket delivery."""
        with self._lock:
            redacted = self._redact(self._pending + output)
            if len(redacted) > self._retained_chars:
                split_at = len(redacted) - self._retained_chars
                self._sender.write(self._stream, redacted[:split_at])
                self._pending = redacted[split_at:]
            else:
                self._pending = redacted
        return len(output)

    def writelines(self, lines: Iterable[str]) -> None:
        """Relay lines instead of delegating writes to the original stream."""
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        """Keep a bounded suffix so secrets split across writes stay redacted."""

    def finish(self) -> None:
        """Flush the final redacted suffix when task output has stopped."""
        with self._lock:
            self._sender.write(self._stream, self._redact(self._pending))
            self._pending = ""

    def _redact(self, output: str) -> str:
        for secret in self._secrets:
            output = output.replace(secret, "[REDACTED]")
        return output


@contextmanager
def relay_task_output(sender: ProtocolOutputSender, token: str) -> Iterator[None]:
    """Relay task output as redacted, bounded protocol frames."""
    # Keep the exec-side dispatcher lightweight by importing logger state only
    # in the resident process after task authority has been accepted.
    from flwr.supercore.logger import (  # pylint: disable=import-outside-toplevel
        console_handler,
    )

    secrets = (token, FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout = _RelayedTextOutput(sender, "stdout", secrets, original_stdout)
    stderr = _RelayedTextOutput(sender, "stderr", secrets, original_stderr)
    original_log_stream = console_handler.stream
    sys.stdout = cast(TextIO, stdout)
    sys.stderr = cast(TextIO, stderr)
    console_handler.stream = cast(TextIO, stderr)
    try:
        yield
    finally:
        console_handler.stream = original_log_stream
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        stdout.finish()
        stderr.finish()


def send_message(channel: MessageChannel, payload: JSONObject) -> None:
    """Send one bounded newline-delimited JSON protocol message."""
    encoded = encode_message(payload)
    if len(encoded) > MAX_PROTOCOL_MESSAGE_BYTES:
        raise ValueError("Model worker protocol message is too large.")
    offset = 0
    while offset < len(encoded):
        written = channel.write(encoded[offset:])
        if written <= 0:
            raise OSError("Model worker protocol write made no progress.")
        offset += written
    channel.flush()


def encode_message(payload: JSONObject) -> bytes:
    """Encode one JSON-lines protocol message."""
    return (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode() + b"\n"
    )


def read_message(channel: MessageChannel) -> JSONObject:
    """Read one bounded newline-delimited JSON protocol message."""
    encoded = channel.readline(MAX_PROTOCOL_MESSAGE_BYTES + 1)
    if not encoded:
        raise ValueError("Model worker protocol connection closed.")
    if len(encoded) > MAX_PROTOCOL_MESSAGE_BYTES or not encoded.endswith(b"\n"):
        raise ValueError("Model worker protocol message is too large.")
    raw = encoded[:-1]
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ValueError("Model worker protocol message must be valid JSON.") from err
    if not isinstance(payload, dict):
        raise ValueError("Model worker protocol message must be a JSON object.")
    return payload
