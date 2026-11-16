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
"""Tests for shared Runtime API dependencies."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request

from flwr.supercore.error import ApiErrorCode, FlowerError

from .runtime import get_runtime_state


def _make_request(app: FastAPI) -> Request:
    """Return a minimal request bound to the FastAPI app."""
    return Request({"type": "http", "app": app})


def test_get_runtime_state_returns_state() -> None:
    """Return state from the configured Runtime state factory."""
    expected_state = Mock()
    factory = Mock()
    factory.state.return_value = expected_state
    app = FastAPI()
    app.state.runtime_state_factory = factory

    assert get_runtime_state(_make_request(app)) is expected_state
    factory.state.assert_called_once_with()


@pytest.mark.parametrize(
    "error_code",
    [
        ApiErrorCode.NODESTATE_NOT_INITIALIZED,
        ApiErrorCode.LINKSTATE_NOT_INITIALIZED,
    ],
)
def test_get_runtime_state_raises_configured_error(error_code: ApiErrorCode) -> None:
    """Raise the error configured by the Runtime API host."""
    app = FastAPI()
    app.state.runtime_state_factory = None
    app.state.runtime_state_factory_error = (error_code, "State is not initialized.")

    with pytest.raises(FlowerError) as exc_info:
        get_runtime_state(_make_request(app))

    assert exc_info.value.code == error_code
    assert exc_info.value.message == "State is not initialized."
