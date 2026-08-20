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
"""HTTP server logging configuration."""

import logging
import time
from typing import Any

from uvicorn.logging import AccessFormatter, DefaultFormatter

HEALTH_CHECK_PATH = "/health"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
UVICORN_DEFAULT_LOG_FORMAT = "%(levelprefix)s %(message)s"
UVICORN_DEFAULT_ACCESS_LOG_FORMAT = (
    '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
)


class UTCFormatter(logging.Formatter):
    """Format log timestamps as UTC ISO-8601 values."""

    @staticmethod
    def converter(timestamp: float | None = None) -> time.struct_time:
        """Convert a timestamp to UTC."""
        return time.gmtime(timestamp)

    default_time_format = LOG_DATE_FORMAT
    default_msec_format = "%s.%03dZ"


class HealthCheckAccessFilter(logging.Filter):
    """Hide successful health-check access logs unless debug is enabled."""

    def __init__(self, debug_enabled: bool = False) -> None:
        super().__init__()
        self.debug_enabled = debug_enabled

    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether the log record should be emitted."""
        if not self._is_successful_health_check(record):
            return True
        if not self.debug_enabled:
            return False
        record.levelno = logging.DEBUG
        record.levelname = "DEBUG"
        return True

    @staticmethod
    def _is_successful_health_check(record: logging.LogRecord) -> bool:
        if record.name != "uvicorn.access":
            return False
        if not isinstance(record.args, tuple) or len(record.args) < 5:
            return False

        raw_path = str(record.args[2])
        raw_status_code = record.args[4]
        if isinstance(raw_status_code, int):
            status_code = raw_status_code
        elif isinstance(raw_status_code, str):
            try:
                status_code = int(raw_status_code)
            except ValueError:
                return False
        else:
            return False
        path, _, _ = raw_path.partition("?")
        return path == HEALTH_CHECK_PATH and status_code < 400


def get_uvicorn_log_config(log_level: int) -> dict[str, Any]:
    """Return Uvicorn logging configuration for the SuperLink HTTP API."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "http": {
                "()": UTCFormatter,
                "format": LOG_FORMAT,
            }
        },
        "filters": {
            "health_check_access": {
                "()": HealthCheckAccessFilter,
                "debug_enabled": log_level <= logging.DEBUG,
            }
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "http",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "class": "logging.StreamHandler",
                "filters": ["health_check_access"],
                "formatter": "http",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "httpcore": {
                "level": "WARNING",
                "propagate": True,
            },
            "httpx": {
                "level": "WARNING",
                "propagate": True,
            },
            "uvicorn": {
                "handlers": ["default"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": log_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "level": log_level,
            },
        },
    }


def configure_uvicorn_logging() -> None:
    """Apply HTTP formatting to handlers already created by Uvicorn."""
    formatter = UTCFormatter(LOG_FORMAT)

    for logger_name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            if _is_default_uvicorn_formatter(
                handler.formatter,
                DefaultFormatter,
                UVICORN_DEFAULT_LOG_FORMAT,
            ):
                handler.setFormatter(formatter)

    access_logger = logging.getLogger("uvicorn.access")
    debug_enabled = access_logger.isEnabledFor(logging.DEBUG)
    for handler in access_logger.handlers:
        if _is_default_uvicorn_formatter(
            handler.formatter,
            AccessFormatter,
            UVICORN_DEFAULT_ACCESS_LOG_FORMAT,
        ):
            handler.setFormatter(formatter)
        elif not isinstance(handler.formatter, UTCFormatter):
            continue
        for log_filter in handler.filters:
            if isinstance(log_filter, HealthCheckAccessFilter):
                log_filter.debug_enabled = debug_enabled
                break
        else:
            handler.addFilter(HealthCheckAccessFilter(debug_enabled))

    for logger_name in ("httpcore", "httpx"):
        logger = logging.getLogger(logger_name)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.WARNING)


def _is_default_uvicorn_formatter(
    formatter: logging.Formatter | None,
    formatter_class: type[logging.Formatter],
    log_format: str,
) -> bool:
    """Return whether a formatter matches one of Uvicorn's defaults."""
    return (
        isinstance(formatter, formatter_class)
        and formatter._fmt == log_format  # pylint: disable=protected-access
    )
