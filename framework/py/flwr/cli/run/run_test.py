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
"""Tests for the `flwr run` command."""


from unittest.mock import Mock, patch

import click
import grpc
import pytest

from flwr.cli.run.run import _wait_for_control_api_channel


def test_wait_for_control_api_channel_retries_until_ready() -> None:
    """Test that Control API readiness waits through transient unavailability."""
    future = Mock()
    future.result.side_effect = [grpc.FutureTimeoutError(), None]

    with patch("flwr.cli.run.run.grpc.channel_ready_future", return_value=future):
        _wait_for_control_api_channel(Mock(), timeout=1, check_interval=0.01)

    assert future.result.call_count == 2


def test_wait_for_control_api_channel_fails_after_timeout() -> None:
    """Test that Control API readiness fails after the timeout expires."""
    future = Mock()
    future.result.side_effect = grpc.FutureTimeoutError()

    with patch("flwr.cli.run.run.grpc.channel_ready_future", return_value=future):
        with pytest.raises(click.ClickException, match="SuperLink is unavailable"):
            _wait_for_control_api_channel(Mock(), timeout=0.01, check_interval=0.01)
