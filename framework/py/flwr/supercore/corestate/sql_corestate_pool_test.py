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
"""Tests for CoreState PostgreSQL pool configuration."""

import os
from unittest.mock import Mock, patch

import pytest

from flwr.server.superlink.linkstate.sql_linkstate import SqlLinkState
from flwr.supercore.sql_mixin import SqlMixin
from flwr.superlink.federation import NoOpFederationManager


class _PostgreSqlLinkState(SqlLinkState):
    """Enable the PostgreSQL dialect without an external test database."""

    allowed_dialects = frozenset({"postgresql"})


def _state() -> _PostgreSqlLinkState:
    """Build a concrete PostgreSQL CoreState without connecting to PostgreSQL."""
    return _PostgreSqlLinkState(
        "postgresql://localhost/flwr", NoOpFederationManager(), Mock()
    )


def test_postgresql_pool_limits() -> None:
    """Pass configured limits to the shared SQL engine."""
    state = _state()
    with (
        patch.dict(
            os.environ,
            {"FLWR_CORESTATE_POOL_SIZE": "20", "FLWR_CORESTATE_MAX_OVERFLOW": "50"},
        ),
        patch.object(SqlMixin, "initialize", return_value=[]) as initialize,
    ):
        state.initialize()
    initialize.assert_called_once_with(False, pool_size=20, max_overflow=50)


def test_postgresql_pool_defaults() -> None:
    """Keep SQLAlchemy's defaults when no limits are configured."""
    state = _state()
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(SqlMixin, "initialize", return_value=[]) as initialize,
    ):
        state.initialize()
    initialize.assert_called_once_with(False, pool_size=None, max_overflow=None)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("FLWR_CORESTATE_POOL_SIZE", "0"),
        ("FLWR_CORESTATE_POOL_SIZE", "invalid"),
        ("FLWR_CORESTATE_MAX_OVERFLOW", "-1"),
    ],
)
def test_postgresql_pool_rejects_invalid_limits(name: str, value: str) -> None:
    """Fail at startup on invalid PostgreSQL pool settings."""
    state = _state()
    with patch.dict(os.environ, {name: value}, clear=True):
        with pytest.raises(ValueError, match=name):
            state.initialize()
