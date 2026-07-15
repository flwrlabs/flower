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

import asyncio
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI, Request
from starlette.datastructures import State

from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.error import ApiErrorCode, FlowerError

from ..main import create_app
from .linkstate import get_linkstate


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


async def _get_linkstate_after_lifespan_startup(
    app: FastAPI,
) -> LinkState:
    async with app.router.lifespan_context(app):
        return get_linkstate(_make_request(app))


def _create_app_with_linkstate_factory(
    state_factory_mock: Mock, *, start_legacy_grpc: bool
) -> FastAPI:
    """Create a FastAPI app for either SuperLink HTTP mode."""
    linkstate_factory = cast(LinkStateFactory, state_factory_mock)
    authn_plugin = Mock()
    authz_plugin = Mock()
    if not start_legacy_grpc:
        return create_app(
            linkstate_factory=linkstate_factory,
            authn_plugin=authn_plugin,
            authz_plugin=authz_plugin,
        )

    superlink_lifespan = Mock()
    superlink_lifespan.state_factory = None
    superlink_lifespan.startup.side_effect = lambda: setattr(
        superlink_lifespan, "state_factory", linkstate_factory
    )
    return create_app(
        linkstate_factory=linkstate_factory,
        authn_plugin=authn_plugin,
        authz_plugin=authz_plugin,
        superlink_lifespan=cast(Any, superlink_lifespan),
        start_legacy_grpc=True,
    )


@pytest.mark.parametrize("start_legacy_grpc", [False, True])
def test_get_linkstate_returns_linkstate_after_startup(
    start_legacy_grpc: bool,
) -> None:
    """get_linkstate should return LinkState in both FastAPI HTTP modes."""
    expected_linkstate = cast(LinkState, Mock(spec=LinkState))
    state_factory_mock = Mock(spec=LinkStateFactory)
    state_factory_mock.state.return_value = expected_linkstate
    app = _create_app_with_linkstate_factory(
        state_factory_mock,
        start_legacy_grpc=start_legacy_grpc,
    )

    with patch("flwr.superlink.extensions.get_lifespan_contexts", return_value=()):
        linkstate = asyncio.run(_get_linkstate_after_lifespan_startup(app))

    assert app.state.linkstate_factory is state_factory_mock
    assert linkstate is expected_linkstate


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

    with pytest.raises(FlowerError) as exc_info:
        get_linkstate(_make_request(app))

    assert exc_info.value.code == ApiErrorCode.LINKSTATE_NOT_INITIALIZED
    assert exc_info.value.message == "SuperLink LinkStateFactory is not initialized."
