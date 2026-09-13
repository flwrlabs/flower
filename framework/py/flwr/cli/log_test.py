# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Test for Flower command line interface `log` command."""


import unittest
from typing import NoReturn
from unittest.mock import Mock, call, patch

from flwr.proto.control_pb2 import StreamLogsResponse  # pylint: disable=E0611

from .log import _log_with_control_api, print_logs, stream_logs


class InterruptedStreamLogsResponse:
    """Create a StreamLogsResponse object with KeyboardInterrupt."""

    @property
    def log_output(self) -> NoReturn:
        """Raise KeyboardInterrupt to exit logstream test gracefully."""
        raise KeyboardInterrupt

    @property
    def latest_timestamp(self) -> NoReturn:
        """Raise KeyboardInterrupt to exit logstream test gracefully."""
        raise KeyboardInterrupt


class TestFlwrLog(unittest.TestCase):
    """Unit tests for `flwr log` CLI functions."""

    def setUp(self) -> None:
        """Initialize a mock Control API client before each test."""
        self.expected_stream_call = [
            call("log_output_1"),
            call("log_output_2"),
            call("log_output_3"),
        ]
        self.expected_print_call = [call("log_output_1")]
        mock_response_iterator = [
            iter(
                [StreamLogsResponse(log_output=f"log_output_{i}") for i in range(1, 4)]
                + [InterruptedStreamLogsResponse()]
            )
        ]
        self.mock_stub = Mock()
        self.mock_stub.StreamLogs.side_effect = mock_response_iterator

    def test_flwr_log_stream_method(self) -> None:
        """Test stream_logs."""
        with patch("builtins.print") as mock_print:
            with self.assertRaises(KeyboardInterrupt):
                stream_logs(run_id=123, stub=self.mock_stub, after_timestamp=0.0)
                # Assert that mock print was called with the expected arguments
                mock_print.assert_has_calls(self.expected_stream_call)

    def test_flwr_log_print_method(self) -> None:
        """Test print_logs."""
        with patch("builtins.print") as mock_print:
            print_logs(run_id=123, stub=self.mock_stub)
            # Assert that mock print was called with the expected arguments
            mock_print.assert_has_calls(self.expected_print_call)

    def test_log_with_control_api_owns_client_lifecycle(self) -> None:
        """Close the HTTP client at the command boundary."""
        control_client = Mock()
        with (
            patch(
                "flwr.cli.log.init_http_client_from_connection",
                return_value=control_client,
            ),
            patch("flwr.cli.log.start_stream") as start_stream,
        ):
            _log_with_control_api(Mock(), run_id=123, stream=True)

        start_stream.assert_called_once_with(123, control_client)
        control_client.close.assert_called_once_with()
