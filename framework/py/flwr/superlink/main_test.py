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
"""Tests for the SuperLink FastAPI app."""

import pytest

from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME

from .main import _FLWR_POSTGRES_ENV_VARS, _get_database_from_env


def _clear_database_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all database environment variables."""
    monkeypatch.delenv("FLWR_DATABASE", raising=False)
    for env_var in _FLWR_POSTGRES_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


def test_get_database_from_env_defaults_to_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Database configuration should default to in-memory state."""
    _clear_database_environment(monkeypatch)

    assert _get_database_from_env() == FLWR_IN_MEMORY_DB_NAME


def test_get_database_from_env_uses_explicit_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit database URL should take precedence over PostgreSQL fields."""
    monkeypatch.setenv("FLWR_DATABASE", "sqlite:///state.db")
    for env_var in _FLWR_POSTGRES_ENV_VARS:
        monkeypatch.setenv(env_var, "ignored")

    assert _get_database_from_env() == "sqlite:///state.db"


def test_get_database_from_env_builds_postgresql_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefixed PostgreSQL environment variables should form a safe URL."""
    _clear_database_environment(monkeypatch)
    values = {
        "FLWR_POSTGRES_HOST": "postgres.example",
        "FLWR_POSTGRES_PORT": "5432",
        "FLWR_POSTGRES_DATABASE": "platform-api",
        "FLWR_POSTGRES_USER": "platform@api",
        "FLWR_POSTGRES_PASSWORD": "p@ss/word",
    }
    for env_var, value in values.items():
        monkeypatch.setenv(env_var, value)

    assert (
        _get_database_from_env() == "postgresql+psycopg://platform%40api:p%40ss%2Fword@"
        "postgres.example:5432/platform-api?sslmode=require"
    )


def test_get_database_from_env_rejects_partial_postgresql_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial PostgreSQL configuration should fail instead of using memory."""
    _clear_database_environment(monkeypatch)
    monkeypatch.setenv("FLWR_POSTGRES_HOST", "postgres.example")

    with pytest.raises(
        ValueError,
        match="FLWR_POSTGRES_PORT.*FLWR_POSTGRES_DATABASE.*FLWR_POSTGRES_USER.*"
        "FLWR_POSTGRES_PASSWORD",
    ):
        _get_database_from_env()
