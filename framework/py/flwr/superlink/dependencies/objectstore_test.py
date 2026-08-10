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
"""Tests for the SuperLink FastAPI ObjectStore dependency."""

from typing import cast
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request

from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.object_store import ObjectStore, ObjectStoreFactory

from .objectstore import get_objectstore


def _make_request(app: FastAPI) -> Request:
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


def test_get_objectstore_returns_configured_objectstore() -> None:
    """Return the ObjectStore created by the configured factory."""
    expected_objectstore = cast(ObjectStore, Mock(spec=ObjectStore))
    objectstore_factory = Mock(spec=ObjectStoreFactory)
    objectstore_factory.store.return_value = expected_objectstore
    app = FastAPI()
    app.state.objectstore_factory = objectstore_factory

    objectstore = get_objectstore(_make_request(app))

    assert objectstore is expected_objectstore
    objectstore_factory.store.assert_called_once_with()


@pytest.mark.parametrize("set_objectstore_factory", [False, True])
def test_get_objectstore_raises_when_factory_is_missing(
    set_objectstore_factory: bool,
) -> None:
    """Fail clearly when ObjectStoreFactory has not been initialized."""
    app = FastAPI()
    if set_objectstore_factory:
        app.state.objectstore_factory = None

    with pytest.raises(FlowerError) as exc_info:
        get_objectstore(_make_request(app))

    assert exc_info.value.code == ApiErrorCode.OBJECTSTORE_NOT_INITIALIZED
    assert exc_info.value.message == (
        "SuperLink ObjectStoreFactory is not initialized."
    )
