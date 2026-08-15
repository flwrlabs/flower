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
"""Tests for the SuperNode NodeState dependency."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request

from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supernode.nodestate import NodeState, NodeStateFactory

from .nodestate import get_nodestate


def _make_request(app: FastAPI) -> Request:
    """Return a minimal request bound to the FastAPI app."""
    return Request({"type": "http", "app": app})


def test_get_nodestate_returns_nodestate() -> None:
    """Return NodeState from the configured factory."""
    expected_nodestate = Mock(spec=NodeState)
    nodestate_factory = Mock(spec=NodeStateFactory)
    nodestate_factory.state.return_value = expected_nodestate
    app = FastAPI()
    app.state.nodestate_factory = nodestate_factory

    nodestate = get_nodestate(_make_request(app))

    assert nodestate is expected_nodestate
    nodestate_factory.state.assert_called_once_with()


@pytest.mark.parametrize("set_nodestate_factory", [False, True])
def test_get_nodestate_raises_when_nodestate_factory_is_missing(
    set_nodestate_factory: bool,
) -> None:
    """Fail clearly before NodeStateFactory is initialized."""
    app = FastAPI()
    if set_nodestate_factory:
        app.state.nodestate_factory = None

    with pytest.raises(FlowerError) as exc_info:
        get_nodestate(_make_request(app))

    assert exc_info.value.code == ApiErrorCode.NODESTATE_NOT_INITIALIZED
    assert exc_info.value.message == "SuperNode NodeStateFactory is not initialized."
