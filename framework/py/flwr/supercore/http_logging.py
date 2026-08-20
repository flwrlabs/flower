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

HEALTH_CHECK_PATH = "/health"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


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
