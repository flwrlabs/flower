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
"""Tests for shared connector infrastructure."""


from unittest.mock import Mock, patch

import pytest
import requests

from .errors import ConnectorArgumentError
from .http import ConnectorApiError, request_json_object
from .json_utils import require_int_range


class ExampleApiError(ConnectorApiError):
    """Test connector error."""

    provider = "Example"


def test_json_request_failure_is_secret_safe() -> None:
    """Transport failures should not expose provider secrets."""
    request = Mock(side_effect=requests.RequestException("secret"))

    with (
        patch("flwr.supercore.task_process.connector.http.requests.request", request),
        pytest.raises(ExampleApiError) as exc_info,
    ):
        request_json_object(
            "GET", "https://api.example.com/items", error=ExampleApiError
        )

    assert exc_info.value.code == "request_failed"
    assert exc_info.value.retryable is True
    assert "secret" not in str(exc_info.value)


def test_argument_failure_is_public_and_actionable() -> None:
    """Argument validation should produce a structured model-facing error."""
    with pytest.raises(ConnectorArgumentError) as exc_info:
        require_int_range("many", "Notion", "limit", maximum=100)

    assert exc_info.value.to_json() == {
        "code": "invalid_arguments",
        "message": "Notion limit must be an integer.",
        "retryable": False,
    }
