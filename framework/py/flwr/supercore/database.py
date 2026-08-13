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
"""Database configuration helpers shared by Flower services."""

import os
from collections.abc import Mapping

from sqlalchemy.engine import URL


def postgresql_url_from_environment(
    environ: Mapping[str, str] | None = None,
) -> URL:
    """Build a synchronous PostgreSQL URL from Flower environment variables.

    Parameters
    ----------
    environ : Mapping[str, str] | None
        Environment values to read. If omitted, ``os.environ`` is used.

    Returns
    -------
    sqlalchemy.engine.URL
        A PostgreSQL URL using the psycopg driver.

    Raises
    ------
    ValueError
        If a required value is missing or the configured port is invalid.
    """
    values = os.environ if environ is None else environ
    required = (
        "FLWR_POSTGRES_USER",
        "FLWR_POSTGRES_PASSWORD",
        "FLWR_POSTGRES_HOST",
        "FLWR_POSTGRES_DATABASE",
    )
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise ValueError(
            "Missing required PostgreSQL environment variables: " + ", ".join(missing)
        )

    try:
        port = int(values.get("FLWR_POSTGRES_PORT", "5432"))
    except ValueError as exc:
        raise ValueError("FLWR_POSTGRES_PORT must be an integer") from exc

    return URL.create(
        "postgresql+psycopg",
        username=values["FLWR_POSTGRES_USER"],
        password=values["FLWR_POSTGRES_PASSWORD"],
        host=values["FLWR_POSTGRES_HOST"],
        port=port,
        database=values["FLWR_POSTGRES_DATABASE"],
    )
