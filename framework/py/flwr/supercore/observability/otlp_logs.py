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
"""Export Flower log records over OTLP when explicitly enabled."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from logging import WARNING, Logger, getLogger
from typing import Any

from flwr.supercore import log
from flwr.supercore.logger import console_handler

_DEFAULT_SERVICE_NAME = "flower"
_DEFAULT_LOGGER_NAMES = ("flwr",)
_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})


class OTLPLogs:
    """Own the OTLP provider and handlers for a Flower process."""

    def __init__(
        self,
        provider: Any,
        handler: Any,
        logger_names: Sequence[str],
    ) -> None:
        self._provider = provider
        self._handler = handler
        self._loggers: tuple[Logger, ...] = tuple(
            getLogger(logger_name) for logger_name in logger_names
        )
        self._started = False
        self._shutdown = False

    def start(self) -> None:
        """Attach the OTLP handler to the configured Flower loggers."""
        if self._started or self._shutdown:
            return
        for logger in self._loggers:
            logger.addHandler(self._handler)
        self._started = True

    def shutdown(self) -> None:
        """Detach the handler and flush the provider's final batch."""
        if self._shutdown:
            return
        self._shutdown = True
        if self._started:
            for logger in self._loggers:
                logger.removeHandler(self._handler)
            self._started = False
        self._provider.shutdown()


def configure_otlp_logs(
    *,
    enabled: bool,
    service_name: str,
    logger_names: Sequence[str] = _DEFAULT_LOGGER_NAMES,
) -> OTLPLogs | None:
    """Create a scoped OTLP exporter, or return ``None`` when disabled.

    OpenTelemetry remains an optional Flower dependency. The imports are deferred
    until logging is enabled so regular Flower installations do not need to carry
    the OTLP packages.
    """
    if not enabled:
        return None

    provider: Any | None = None
    try:
        (
            otlp_log_exporter,
            logger_provider,
            logging_handler,
            batch_log_record_processor,
            resource,
        ) = _load_otlp_dependencies()
        provider = logger_provider(
            resource=resource.create(
                {"service.name": service_name.strip() or _DEFAULT_SERVICE_NAME}
            )
        )
        # Let the exporter read the standard OTEL_EXPORTER_OTLP_LOGS_* settings.
        exporter = otlp_log_exporter()
        provider.add_log_record_processor(batch_log_record_processor(exporter))
        handler = logging_handler(logger_provider=provider)
        handler.setLevel(console_handler.level)
        return OTLPLogs(provider, handler, logger_names)
    except Exception:
        if provider is not None:
            provider.shutdown()
        raise


@contextmanager
def otlp_logs_lifespan(
    *,
    default_service_name: str,
    logger_names: Sequence[str] = _DEFAULT_LOGGER_NAMES,
) -> Iterator[None]:
    """Configure best-effort OTLP logging for one Flower process lifetime."""
    otlp_logs: OTLPLogs | None = None
    if _env_enabled("OTEL_LOGS_ENABLED"):
        try:
            otlp_logs = configure_otlp_logs(
                enabled=True,
                service_name=os.getenv("OTEL_SERVICE_NAME", default_service_name),
                logger_names=logger_names,
            )
            if otlp_logs is not None:
                otlp_logs.start()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if otlp_logs is not None:
                try:
                    otlp_logs.shutdown()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
            log(
                WARNING,
                "OTLP log export setup failed: %s.",
                type(exc).__name__,
            )
            otlp_logs = None

    try:
        yield
    finally:
        if otlp_logs is not None:
            try:
                otlp_logs.shutdown()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                log(
                    WARNING,
                    "OTLP log export shutdown failed: %s.",
                    type(exc).__name__,
                )


def _env_enabled(name: str) -> bool:
    """Return whether a boolean environment variable opts in to a feature."""
    return os.getenv(name, "").strip().lower() in _ENABLED_VALUES


def _load_otlp_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    """Load optional OpenTelemetry dependencies only when OTLP is enabled."""
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.resources import Resource

    return (
        OTLPLogExporter,
        LoggerProvider,
        LoggingHandler,
        BatchLogRecordProcessor,
        Resource,
    )
