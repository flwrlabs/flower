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
"""Tests for HTTP retry utilities."""

import os
import signal
from unittest.mock import Mock, patch

import httpx
import pytest

from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.run import RunNotRunningException

from .http_retry import make_simple_http_retry_invoker
from .retry_invoker import RetryInvoker


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://api.example")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        "HTTP request failed", request=request, response=response
    )


def _transport_error(
    exception_type: type[httpx.TransportError],
) -> httpx.TransportError:
    return exception_type(
        "Transport failed",
        request=httpx.Request("POST", "http://api.example"),
    )


def _make_test_invoker() -> RetryInvoker:
    invoker = make_simple_http_retry_invoker()
    invoker.max_tries = 2
    invoker.jitter = None
    invoker.wait_function = lambda _: None
    return invoker


@pytest.mark.parametrize(
    "exception",
    [
        _transport_error(httpx.ConnectError),
        _transport_error(httpx.ReadError),
        _transport_error(httpx.WriteError),
        _transport_error(httpx.RemoteProtocolError),
        _transport_error(httpx.ConnectTimeout),
        _transport_error(httpx.ReadTimeout),
        _transport_error(httpx.WriteTimeout),
        _transport_error(httpx.PoolTimeout),
        _http_status_error(httpx.codes.SERVICE_UNAVAILABLE),
        _http_status_error(httpx.codes.GATEWAY_TIMEOUT),
    ],
)
def test_retries_transient_http_failures(exception: Exception) -> None:
    """Retry transport failures and unavailable HTTP responses."""
    target = Mock(side_effect=[exception, "success"])

    assert _make_test_invoker().invoke(target) == "success"
    assert target.call_count == 2


@pytest.mark.parametrize(
    "exception",
    [
        _http_status_error(httpx.codes.INTERNAL_SERVER_ERROR),
        httpx.ConnectError(
            "certificate verify failed",
            request=httpx.Request("POST", "https://api.example"),
        ),
        _transport_error(httpx.UnsupportedProtocol),
        _transport_error(httpx.LocalProtocolError),
    ],
)
def test_does_not_retry_permanent_http_failures(exception: Exception) -> None:
    """Give up immediately for permanent HTTP and TLS failures."""
    target = Mock(side_effect=exception)

    with pytest.raises(type(exception)):
        _make_test_invoker().invoke(target)
    target.assert_called_once_with()


@pytest.mark.parametrize(
    "content",
    [
        "",
        "<html>Forbidden</html>",
        '{"code": 3010}',
        '{"code": "3010", "detail": "Forbidden"}',
        '{"code": 2007, "detail": "Run status does not allow this operation."}',
        '{"code": 3007, "detail": "Task creation is not allowed."}',
    ],
)
def test_other_forbidden_responses_preserve_run_not_running(content: str) -> None:
    """Retain stopped-run handling for other errors and legacy responses."""
    error = _http_status_error(httpx.codes.FORBIDDEN)
    error.response = httpx.Response(403, content=content, request=error.request)
    target = Mock(side_effect=error)

    with pytest.raises(RunNotRunningException):
        _make_test_invoker().invoke(target)
    target.assert_called_once_with()


def test_unavailable_connector_preserves_http_error() -> None:
    """Let callers handle connector authorization without retrying or stopping."""
    error = _http_status_error(httpx.codes.FORBIDDEN)
    error.response = httpx.Response(
        403,
        json=FlowerError(
            ApiErrorCode.RUNTIME_CONNECTOR_NOT_AVAILABLE, "Unavailable connector"
        ).to_json("Connector is not available to this run."),
        request=error.request,
    )
    target = Mock(side_effect=error)

    with pytest.raises(httpx.HTTPStatusError) as raised:
        _make_test_invoker().invoke(target)

    assert raised.value is error
    target.assert_called_once_with()


def test_unauthorized_response_requests_shutdown() -> None:
    """Request process shutdown for an authentication failure."""
    error = _http_status_error(httpx.codes.UNAUTHORIZED)
    target = Mock(side_effect=[error, "success"])
    invoker = _make_test_invoker()

    with (
        patch("flwr.supercore.retry.http_retry.os.kill") as kill,
        patch("flwr.supercore.retry.http_retry.time.sleep"),
    ):
        assert invoker.invoke(target) == "success"

    kill.assert_called_once_with(os.getpid(), signal.SIGINT)
