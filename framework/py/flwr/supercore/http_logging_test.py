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
from io import StringIO

import pytest
from uvicorn.logging import AccessFormatter, DefaultFormatter

from .http_logging import (
    LOG_FORMAT,
    RoutineAccessFilter,
    UTCFormatter,
    configure_uvicorn_logging,
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
        ("/v1/runtime/pull-pending-tasks", 200, False),
        ("/v1/runtime/pull-pending-tasks?timeout=1", "200", False),
        ("/health", 500, True),
        ("/v1/runtime/pull-pending-tasks", 500, True),
        ("/v1/user/profile", 200, True),
        ("//[", 200, True),
    ],
)
def test_routine_access_filter(
    path: str, status_code: object, expected: bool
) -> None:
    """Filter only successful routine access records."""
    assert RoutineAccessFilter().filter(_access_record(path, status_code)) is expected


def test_routine_access_filter_keeps_routine_requests_at_debug() -> None:
    """Keep successful routine requests as debug records in debug mode."""
    record = _access_record("/v1/runtime/pull-pending-tasks", 200)

    assert RoutineAccessFilter(debug_enabled=True).filter(record)
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
    assert not config["filters"]["routine_access"]["debug_enabled"]


def test_get_uvicorn_log_config_enables_debug_routine_requests() -> None:
    """Pass debug state to the routine access filter."""
    config = get_uvicorn_log_config(logging.DEBUG)

    assert config["filters"]["routine_access"]["debug_enabled"]


def test_configure_uvicorn_logging_updates_existing_handlers() -> None:
    """Configure handlers owned by a direct Uvicorn CLI launch."""
    logger_names = ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore")
    loggers = {name: logging.getLogger(name) for name in logger_names}
    original_state = {
        name: (list(logger.handlers), logger.level) for name, logger in loggers.items()
    }
    default_handler = logging.StreamHandler(StringIO())
    access_handler = logging.StreamHandler(StringIO())
    custom_formatter = logging.Formatter("custom: %(message)s")
    default_handler.setFormatter(
        DefaultFormatter(fmt="%(levelprefix)s %(message)s", use_colors=False)
    )
    access_handler.setFormatter(
        AccessFormatter(
            fmt='%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            use_colors=False,
        )
    )

    try:
        loggers["uvicorn"].handlers = [default_handler]
        loggers["uvicorn.error"].handlers = []
        loggers["uvicorn.access"].handlers = [access_handler]
        loggers["uvicorn"].setLevel(logging.ERROR)
        loggers["uvicorn.error"].setLevel(logging.CRITICAL)
        loggers["uvicorn.access"].setLevel(logging.WARNING)
        loggers["httpx"].setLevel(logging.NOTSET)
        loggers["httpcore"].setLevel(logging.NOTSET)

        configure_uvicorn_logging()

        assert loggers["uvicorn"].handlers == [default_handler]
        assert loggers["uvicorn.access"].handlers == [access_handler]
        assert loggers["uvicorn"].level == logging.ERROR
        assert loggers["uvicorn.error"].level == logging.CRITICAL
        assert loggers["uvicorn.access"].level == logging.WARNING
        assert isinstance(default_handler.formatter, UTCFormatter)
        assert isinstance(access_handler.formatter, UTCFormatter)
        routine_filters = [
            log_filter
            for log_filter in access_handler.filters
            if isinstance(log_filter, RoutineAccessFilter)
        ]
        assert len(routine_filters) == 1
        assert not routine_filters[0].debug_enabled
        assert loggers["httpx"].level == logging.WARNING
        assert loggers["httpcore"].level == logging.WARNING

        default_handler.setFormatter(custom_formatter)
        access_handler.setFormatter(custom_formatter)
        access_handler.filters.clear()
        loggers["httpx"].setLevel(logging.INFO)
        loggers["httpcore"].setLevel(logging.DEBUG)
        configure_uvicorn_logging()

        assert default_handler.formatter is custom_formatter
        assert access_handler.formatter is custom_formatter
        assert not access_handler.filters
        assert loggers["httpx"].level == logging.INFO
        assert loggers["httpcore"].level == logging.DEBUG
    finally:
        for name, (handlers, level) in original_state.items():
            loggers[name].handlers = handlers
            loggers[name].setLevel(level)
        default_handler.close()
        access_handler.close()
