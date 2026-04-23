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
"""Tests for the Flower ConnectorApp process."""


import importlib
from queue import Queue

import pytest

from flwr.common import EventType
from flwr.common.exit import ExitCode

run_connector_module = importlib.import_module("flwr.supercore.agent.run_connector")


def test_run_flwr_connector_exits_with_stub_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Flower ConnectorApp process should fail fast with a clear message."""
    captured: dict[str, object] = {}

    def _flwr_exit(
        code: int,
        message: str | None = None,
        event_type: EventType | None = None,
        event_details: dict[str, object] | None = None,
    ) -> None:
        captured["code"] = code
        captured["message"] = message
        captured["event_type"] = event_type
        captured["event_details"] = event_details
        raise SystemExit(1)

    monkeypatch.setattr(run_connector_module, "flwr_exit", _flwr_exit)

    with pytest.raises(SystemExit):
        run_connector_module.run_connector(
            serverappio_api_address="127.0.0.1:9091",
            log_queue=Queue(),
            token="test-token",
        )

    assert captured == {
        "code": ExitCode.SERVERAPP_EXCEPTION,
        "message": "`flwr-connector` is not implemented yet.",
        "event_type": EventType.FLWR_CONNECTOR_RUN_LEAVE,
        "event_details": {"success": False},
    }
