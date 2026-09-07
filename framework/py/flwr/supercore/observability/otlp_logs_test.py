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
"""Tests for the optional OTLP log exporter."""

from logging import getLogger
from unittest.mock import MagicMock

from pytest import MonkeyPatch

from flwr.supercore.observability import otlp_logs


def test_configure_otlp_logs_scopes_handler_and_cleans_up(
    monkeypatch: MonkeyPatch,
) -> None:
    """Attach the handler to Flower loggers and clean it up on shutdown."""
    provider = MagicMock()
    exporter = MagicMock()
    processor = MagicMock()
    handler = MagicMock()
    logger_provider = MagicMock(return_value=provider)
    otlp_exporter = MagicMock(return_value=exporter)
    batch_processor = MagicMock(return_value=processor)
    logging_handler = MagicMock(return_value=handler)
    resource = MagicMock()
    resource_instance = MagicMock()
    resource.create.return_value = resource_instance
    monkeypatch.setattr(
        otlp_logs,
        "_load_otlp_dependencies",
        lambda: (
            otlp_exporter,
            logger_provider,
            logging_handler,
            batch_processor,
            resource,
        ),
    )

    logs = otlp_logs.configure_otlp_logs(
        enabled=True,
        service_name=" superexec ",
    )

    assert logs is not None
    otlp_exporter.assert_called_once_with()
    resource.create.assert_called_once_with({"service.name": "superexec"})
    logger_provider.assert_called_once_with(resource=resource_instance)
    provider.add_log_record_processor.assert_called_once_with(processor)

    logs.start()
    logs.start()
    handler_logger = getLogger("flwr")
    assert handler in handler_logger.handlers

    logs.shutdown()
    assert handler not in handler_logger.handlers
    provider.shutdown.assert_called_once_with()


def test_configure_otlp_logs_requires_explicit_opt_in() -> None:
    """Keep log export disabled by default."""
    assert (
        otlp_logs.configure_otlp_logs(
            enabled=False,
            service_name="superexec",
        )
        is None
    )


def test_otlp_logs_lifespan_is_best_effort(
    monkeypatch: MonkeyPatch,
) -> None:
    """An exporter setup failure must not prevent the Flower process from running."""
    monkeypatch.setenv("OTEL_LOGS_ENABLED", "true")
    configure = MagicMock(side_effect=RuntimeError("missing dependency"))
    monkeypatch.setattr(otlp_logs, "configure_otlp_logs", configure)

    with otlp_logs.otlp_logs_lifespan(default_service_name="superexec"):
        pass

    configure.assert_called_once_with(
        enabled=True,
        service_name="superexec",
        logger_names=("flwr",),
    )
