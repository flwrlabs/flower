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
"""Contract tests for the versioned local Guardian protocol."""

import json
import threading
import urllib.error
from http.server import ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest

from flwr.supercore.run import Run

from .guardian import GuardianVerificationError, verify_capability
from .guardian_mock import GuardianMockHandler

BINDING = "flwr-capability-binding-v1-sha256:" + "a" * 64


def _run(package: bytes = b"opaque") -> Run:
    run = Run.create_empty(1)
    run.capability_required = True
    run.capability_package = package
    run.capability_binding = BINDING
    return run


def _response(payload: object) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    return response


@patch.dict("os.environ", {}, clear=True)
def test_run_without_capability_keeps_existing_behavior() -> None:
    """Do not require Guardian configuration for ordinary Flower runs."""
    verify_capability(Run.create_empty(1))


def test_guardian_mock_round_trip() -> None:
    """Verify a capability through the real local HTTP mock contract."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), GuardianMockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with patch.dict(
            "os.environ",
            {"FLWR_GUARDIAN_URL": f"http://127.0.0.1:{server.server_port}"},
            clear=True,
        ):
            verify_capability(_run(BINDING.encode("utf-8")))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@patch.dict("os.environ", {"FLWR_GUARDIAN_URL": "http://guardian"}, clear=True)
@patch("urllib.request.urlopen")
def test_guardian_allows_matching_binding(urlopen: MagicMock) -> None:
    """Accept an allowed v1 response containing the expected binding."""
    urlopen.return_value = _response(
        {"version": "v1", "allowed": True, "binding": BINDING}
    )

    verify_capability(_run())


@pytest.mark.parametrize(
    ("package", "payload"),
    [
        (b"", None),
        (b"opaque", {"version": "v1", "allowed": False, "binding": BINDING}),
        (b"opaque", {"version": "v1", "allowed": True, "binding": "wrong"}),
        (b"opaque", {"version": "v2", "allowed": True, "binding": BINDING}),
    ],
)
@patch.dict("os.environ", {"FLWR_GUARDIAN_URL": "http://guardian"}, clear=True)
@patch("urllib.request.urlopen")
def test_guardian_fails_closed(
    urlopen: MagicMock, package: bytes, payload: object
) -> None:
    """Fail closed for missing, denied, mismatched, or malformed outcomes."""
    if payload is not None:
        urlopen.return_value = _response(payload)

    with pytest.raises(GuardianVerificationError):
        verify_capability(_run(package))


@pytest.mark.parametrize(
    "error",
    [TimeoutError("timed out"), urllib.error.URLError("unreachable")],
)
@patch.dict("os.environ", {"FLWR_GUARDIAN_URL": "http://guardian"}, clear=True)
@patch("urllib.request.urlopen")
def test_guardian_fails_closed_on_transport_error(
    urlopen: MagicMock, error: Exception
) -> None:
    """Fail closed when the Guardian times out or is unreachable."""
    urlopen.side_effect = error

    with pytest.raises(GuardianVerificationError):
        verify_capability(_run())


@patch.dict("os.environ", {"FLWR_GUARDIAN_URL": "not-a-url"}, clear=True)
@patch("urllib.request.urlopen")
def test_guardian_fails_closed_on_invalid_endpoint(urlopen: MagicMock) -> None:
    """Reject malformed Guardian endpoint configuration before any request."""
    with pytest.raises(GuardianVerificationError, match=r"HTTP\(S\) URL"):
        verify_capability(_run())

    urlopen.assert_not_called()
