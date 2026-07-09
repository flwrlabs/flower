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
"""Tests for SuperLink FastAPI dependencies."""


from __future__ import annotations

from typing import cast
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException, Request, status
from starlette.datastructures import State

from flwr.server.superlink.linkstate import LinkState, LinkStateFactory

from ..main import create_app
from .dependencies import get_linkstate


def _make_request(app: FastAPI) -> Request[State]:
    """Return a minimal request bound to the FastAPI app."""
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "scheme": "http",
            "app": app,
        }
    )


def test_get_linkstate_yields_linkstate_from_lifespan() -> None:
    """get_linkstate should return the LinkState from the FastAPI app state."""
    expected_linkstate = cast(LinkState, Mock(spec=LinkState))
    state_factory_mock = Mock(spec=LinkStateFactory)
    state_factory_mock.state.return_value = expected_linkstate
    app = create_app(linkstate_factory=cast(LinkStateFactory, state_factory_mock))

    linkstate = get_linkstate(_make_request(app))

    assert app.state.linkstate_factory is state_factory_mock
    assert linkstate is expected_linkstate
    state_factory_mock.state.assert_called_once_with()


@pytest.mark.parametrize(
    "set_linkstate_factory",
    [False, True],
)
def test_get_linkstate_raises_when_linkstate_factory_is_missing(
    set_linkstate_factory: bool,
) -> None:
    """get_linkstate should fail clearly before LinkStateFactory is initialized."""
    app = FastAPI()
    if set_linkstate_factory:
        app.state.linkstate_factory = None

    with pytest.raises(HTTPException) as exc_info:
        get_linkstate(_make_request(app))

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc_info.value.detail == "SuperLink LinkStateFactory is not initialized."
