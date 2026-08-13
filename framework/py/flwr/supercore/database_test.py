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
"""Tests for database configuration helpers."""

import pytest

from .database import postgresql_url_from_environment


def test_postgresql_url_from_environment_preserves_credentials() -> None:
    """Build a URL without loading unrelated application settings."""
    url = postgresql_url_from_environment(
        {
            "FLWR_POSTGRES_USER": "platform-owner",
            "FLWR_POSTGRES_PASSWORD": "p@ss word/%",
            "FLWR_POSTGRES_HOST": "postgres.internal",
            "FLWR_POSTGRES_PORT": "5544",
            "FLWR_POSTGRES_DATABASE": "platform",
        }
    )

    assert url.drivername == "postgresql+psycopg"
    assert url.username == "platform-owner"
    assert url.password == "p@ss word/%"
    assert url.host == "postgres.internal"
    assert url.port == 5544
    assert url.database == "platform"


def test_postgresql_url_from_environment_rejects_missing_values() -> None:
    """Reject incomplete PostgreSQL configuration."""
    with pytest.raises(ValueError, match="FLWR_POSTGRES_PASSWORD"):
        postgresql_url_from_environment(
            {
                "FLWR_POSTGRES_USER": "platform-owner",
                "FLWR_POSTGRES_HOST": "postgres.internal",
                "FLWR_POSTGRES_DATABASE": "platform",
            }
        )
