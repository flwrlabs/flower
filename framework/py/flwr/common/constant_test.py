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
"""Tests for environment-configured Flower constants."""

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 0.5), ("0.1", 0.1), ("0.01", 0.01), ("60", 60.0)],
)
def test_run_events_stream_interval(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: float
) -> None:
    """Preserve the default and accept overrides, including both boundaries."""
    monkeypatch.delenv("FLWR_RUN_EVENTS_STREAM_INTERVAL", raising=False)
    if value is not None:
        monkeypatch.setenv("FLWR_RUN_EVENTS_STREAM_INTERVAL", value)

    constants = runpy.run_path(str(Path(__file__).with_name("constant.py")))

    assert constants["RUN_EVENTS_STREAM_INTERVAL"] == expected


@pytest.mark.parametrize(
    "value", ["", "invalid", "0", "-1", "0.009", "60.1", "nan", "inf", "-inf"]
)
def test_run_events_stream_interval_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Reject invalid configuration before it can reach the polling loop."""
    monkeypatch.setenv("FLWR_RUN_EVENTS_STREAM_INTERVAL", value)

    with pytest.raises(ValueError, match="FLWR_RUN_EVENTS_STREAM_INTERVAL"):
        runpy.run_path(str(Path(__file__).with_name("constant.py")))
