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
"""Tests for provider configuration and Responses validation."""

import pytest

from flwr.supercore.model_provider import (
    DEFAULT_MODEL_API_ENDPOINT,
    model_api_config,
    validate_model_request,
    validate_model_response,
)
from flwr.supercore.typing import JSONObject, JSONValue


@pytest.mark.parametrize(
    "endpoint", ["", DEFAULT_MODEL_API_ENDPOINT, DEFAULT_MODEL_API_ENDPOINT + "/"]
)
def test_default_provider_requires_credentials(
    monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    """Do not send an unauthenticated request to the hosted inference endpoint."""
    monkeypatch.setenv("FLWR_MODEL_API_ENDPOINT", endpoint)
    monkeypatch.delenv("FLWR_MODEL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FLWR_MODEL_API_KEY"):
        model_api_config()


@pytest.mark.parametrize("key", ["", "server-key"])
def test_custom_provider_configuration(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """Keep optional credentials for explicitly configured local providers."""
    monkeypatch.setenv("FLWR_MODEL_API_ENDPOINT", "http://provider/v1/responses/")
    monkeypatch.setenv("FLWR_MODEL_API_KEY", key)
    url, headers = model_api_config()
    assert url == "http://provider/v1/responses"
    assert headers.get("Authorization") == (f"Bearer {key}" if key else None)
    monkeypatch.setenv("FLWR_MODEL_API_ENDPOINT", "http://provider/v1")
    with pytest.raises(RuntimeError, match="/responses"):
        model_api_config()


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", ""),
        ("input", 1),
        ("stream", "true"),
        ("tools", [1]),
        ("reasoning", []),
        ("max_output_tokens", "1"),
    ],
)
def test_invalid_requests(field: str, value: JSONValue) -> None:
    """Reject invalid request shapes before inference starts."""
    payload: JSONObject = {"model": "model", "input": "hello"}
    payload[field] = value
    with pytest.raises(ValueError):
        validate_model_request(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"object": "response", "output": [1]},
        {"object": "response", "error": "failed"},
    ],
)
def test_invalid_responses(payload: JSONObject) -> None:
    """Reject invalid JSON bodies and terminal stream responses."""
    with pytest.raises(ValueError):
        validate_model_response(payload)
