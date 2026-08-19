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
"""HTTP server logging configuration tests."""

import logging

import pytest

from .http_logging import (
    LOG_FORMAT,
    HealthCheckAccessFilter,
    UTCFormatter,
    get_uvicorn_log_config,
)


def _access_record(path: str, status_code: object) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:12345", "GET", path, "1.1", status_code),
        exc_info=None,
    )


@pytest.mark.parametrize(
    ("path", "status_code", "expected"),
    [
        ("/health", 200, False),
        ("/health?probe=readiness", "200", False),
        ("/health", 500, True),
        ("/v1/user/profile", 200, True),
    ],
)
def test_health_check_access_filter(
    path: str, status_code: object, expected: bool
) -> None:
    """Filter only successful health-check access records."""
    assert (
        HealthCheckAccessFilter().filter(_access_record(path, status_code)) is expected
    )


def test_health_check_access_filter_keeps_health_checks_at_debug() -> None:
    """Keep successful health checks as debug records in debug mode."""
    record = _access_record("/health", 200)

    assert HealthCheckAccessFilter(debug_enabled=True).filter(record)
    assert record.levelno == logging.DEBUG
    assert record.levelname == "DEBUG"


def test_utc_formatter_uses_expected_http_format() -> None:
    """Format HTTP records with an ISO-8601 UTC timestamp."""
    record = _access_record("/v1/user/profile", 200)
    record.created = 0.123
    record.msecs = 123

    assert UTCFormatter(LOG_FORMAT).format(record) == (
        "1970-01-01T00:00:00.123Z INFO [uvicorn.access] "
        '127.0.0.1:12345 - "GET /v1/user/profile HTTP/1.1" 200'
    )


def test_get_uvicorn_log_config_keeps_configuration_scoped() -> None:
    """Configure Uvicorn and noisy HTTP clients without replacing root logging."""
    config = get_uvicorn_log_config(logging.INFO)

    assert "root" not in config
    assert config["handlers"]["default"]["stream"] == "ext://sys.stdout"
    assert config["loggers"]["uvicorn.access"]["handlers"] == ["access"]
    assert config["loggers"]["httpx"]["level"] == "WARNING"
    assert config["loggers"]["httpcore"]["level"] == "WARNING"
    assert not config["filters"]["health_check_access"]["debug_enabled"]


def test_get_uvicorn_log_config_enables_debug_health_checks() -> None:
    """Pass debug state to the health-check filter."""
    config = get_uvicorn_log_config(logging.DEBUG)

    assert config["filters"]["health_check_access"]["debug_enabled"]
