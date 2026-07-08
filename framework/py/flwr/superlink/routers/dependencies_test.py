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

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException, Request, status
from starlette.datastructures import State

from flwr.server.superlink.linkstate import LinkState, LinkStateFactory

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
    """get_linkstate should return the LinkState from the app lifespan."""
    app = FastAPI()
    expected_linkstate = cast(LinkState, Mock(spec=LinkState))
    state_factory_mock = Mock(spec=LinkStateFactory)
    state_factory_mock.state.return_value = expected_linkstate
    app.state.superlink_lifespan = SimpleNamespace(
        state_factory=cast(LinkStateFactory, state_factory_mock)
    )

    linkstate = get_linkstate(_make_request(app))

    assert linkstate is expected_linkstate
    state_factory_mock.state.assert_called_once_with()


@pytest.mark.parametrize(
    "superlink_lifespan",
    [None, SimpleNamespace(state_factory=None)],
)
def test_get_linkstate_raises_when_lifespan_state_is_missing(
    superlink_lifespan: object | None,
) -> None:
    """get_linkstate should fail clearly before SuperLink state is initialized."""
    app = FastAPI()
    if superlink_lifespan is not None:
        app.state.superlink_lifespan = superlink_lifespan

    with pytest.raises(HTTPException) as exc_info:
        get_linkstate(_make_request(app))

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc_info.value.detail == "SuperLink lifespan state is not initialized."
