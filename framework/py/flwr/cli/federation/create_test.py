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
"""Tests for the Flower CLI federation create command."""

import json
from collections.abc import Callable
from unittest.mock import patch

import httpx
import pytest
from click.testing import Result
from typer.testing import CliRunner

from flwr.cli.app import app
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    CreateFederationRequest,
    CreateFederationResponse,
)
from flwr.proto.federation_pb2 import Federation  # pylint: disable=E0611
from flwr.supercore.control import ControlHttpClient
from flwr.supercore.protobuf.constants import (
    CREATE_FEDERATION_NAMESPACE_MEDIA_TYPE,
    PROTOBUF_MEDIA_TYPE,
)

_CREATE_PATH = "/v1/control/create-federation"
_CREATE_MODULE = "flwr.cli.federation.create"


def _invoke(
    args: list[str], handler: Callable[[httpx.Request], httpx.Response]
) -> Result:
    with (
        ControlHttpClient("https://superlink.example") as client,
        httpx.Client(transport=httpx.MockTransport(handler)) as http_client,
        patch.object(client, "_client", http_client),
        patch(f"{_CREATE_MODULE}.migrate"),
        patch(f"{_CREATE_MODULE}.read_superlink_connection"),
        patch(
            f"{_CREATE_MODULE}.init_http_client_from_connection", return_value=client
        ),
    ):
        return CliRunner().invoke(app, ["federation", "create", *args])


def _schema(media_type: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "paths": {
                _CREATE_PATH: {"post": {"requestBody": {"content": {media_type: {}}}}}
            }
        },
    )


@pytest.mark.parametrize("selector", ["demo", "@team_dev/demo"])
def test_create_federation_sends_selected_namespace(
    selector: str, capfd: pytest.CaptureFixture[str]
) -> None:
    """Only explicit selectors require discovery and populate the request handle."""
    explicit = selector.startswith("@")
    expected = "@team_dev/demo" if explicit else "@account/demo"
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "GET":
            return _schema(CREATE_FEDERATION_NAMESPACE_MEDIA_TYPE)
        proto = CreateFederationRequest.FromString(request.content)
        assert proto == CreateFederationRequest(
            federation_name="demo",
            handle="team_dev" if explicit else None,
            description="test federation",
            simulation=True,
        )
        assert proto.HasField("handle") is explicit
        assert request.headers["content-type"] == (
            CREATE_FEDERATION_NAMESPACE_MEDIA_TYPE if explicit else PROTOBUF_MEDIA_TYPE
        )
        response = CreateFederationResponse(federation=Federation(name=expected))
        return httpx.Response(200, content=response.SerializeToString())

    result = _invoke(
        [
            selector,
            "--simulation",
            "--description",
            "test federation",
            "--format",
            "json",
        ],
        handle,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(capfd.readouterr().out) == {
        "success": True,
        "federation-id": expected,
    }
    assert calls == ([("GET", "/openapi.json")] if explicit else []) + [
        ("POST", _CREATE_PATH)
    ]


@pytest.mark.parametrize(
    "selector",
    [
        "@team",
        "team/demo",
        "@/demo",
        "@team/demo/name",
        "@team/2demo",
        "@team/demo_name",
        "@team /demo",
    ],
)
def test_create_federation_rejects_invalid_selector(selector: str) -> None:
    """Reject a malformed explicit selector before contacting the SuperLink."""

    def handle(_request: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid selectors must not contact the SuperLink.")

    result = _invoke([selector], handle)

    assert result.exit_code != 0
    assert "Invalid federation" in result.output


@pytest.mark.parametrize("discovery", ["unsupported", "unavailable", "invalid"])
def test_create_federation_rejects_unconfirmed_support(discovery: str) -> None:
    """Unconfirmed namespace support prevents a create request."""

    def handle(request: httpx.Request) -> httpx.Response:
        assert (request.method, request.url.path) == ("GET", "/openapi.json")
        if discovery == "unsupported":
            return _schema(PROTOBUF_MEDIA_TYPE)
        if discovery == "unavailable":
            return httpx.Response(404)
        return httpx.Response(200, text="<html>SuperLink</html>")

    result = _invoke(["@team/demo"], handle)

    assert result.exit_code != 0
    assert "did not confirm support for explicit federation namespaces" in result.output


def test_create_federation_never_retries_with_legacy_media_type() -> None:
    """Mixed-version SuperLinks can reject the guarded create during an upgrade."""
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "GET":
            return _schema(CREATE_FEDERATION_NAMESPACE_MEDIA_TYPE)
        assert request.headers["content-type"] == CREATE_FEDERATION_NAMESPACE_MEDIA_TYPE
        return httpx.Response(415)

    result = _invoke(["@team/demo"], handle)

    assert result.exit_code != 0
    assert "did not confirm support for explicit federation namespaces" in result.output
    assert calls == [("GET", "/openapi.json"), ("POST", _CREATE_PATH)]


@pytest.mark.parametrize("returned_id", ["@personal/demo", "@team/different", ""])
def test_create_federation_rejects_unexpected_result(returned_id: str) -> None:
    """Only the requested federation ID can be reported as an explicit success."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return _schema(CREATE_FEDERATION_NAMESPACE_MEDIA_TYPE)
        response = CreateFederationResponse(federation=Federation(name=returned_id))
        return httpx.Response(200, content=response.SerializeToString())

    result = _invoke(["@team/demo"], handle)

    assert result.exit_code != 0
    assert "created successfully" not in result.output
