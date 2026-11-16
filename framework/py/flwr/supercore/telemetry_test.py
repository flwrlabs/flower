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
"""Telemetry tests."""


import os
import time
import unittest
from unittest import mock
from uuid import uuid4

from flwr.supercore.telemetry import EventType, _get_partner_id, _get_source_id, event


class TelemetryTest(unittest.TestCase):
    """Tests for the telemetry module."""

    @mock.patch("flwr.supercore.telemetry.urllib.request.urlopen")
    @mock.patch("flwr.supercore.telemetry.FLWR_TELEMETRY_ENABLED", "1")
    def test_event(self, mock_urlopen: mock.MagicMock) -> None:
        """Test sending a telemetry event."""
        # Prepare
        expected = '{\n    "status": "created"\n}'
        mock_response = mock_urlopen.return_value.__enter__.return_value
        mock_response.read.return_value = expected.encode("utf-8")

        # Execute
        future = event(EventType.PING)
        actual = future.result()

        # Assert
        self.assertEqual(actual, expected)
        mock_urlopen.assert_called_once()

    @mock.patch("flwr.supercore.telemetry.create_event")
    def test_not_blocking(self, mock_create_event: mock.MagicMock) -> None:
        """Test if the code is blocking.

        If the code does not block duration_actual should be less than
        0.005s.
        """

        # Prepare
        def delayed_create_event(*_: object) -> str:
            time.sleep(0.1)
            return "created"

        mock_create_event.side_effect = delayed_create_event
        duration_max = 0.005
        start = time.perf_counter()

        # Execute
        future = event(EventType.PING)
        duration_actual = time.perf_counter() - start
        future.result(timeout=1.0)

        # Assert
        self.assertLess(duration_actual, duration_max)
        mock_create_event.assert_called_once()

    @mock.patch("flwr.supercore.telemetry.FLWR_TELEMETRY_ENABLED", "0")
    def test_telemetry_disabled(self) -> None:
        """Test opt-out."""
        # Prepare
        expected = "disabled"

        # Execute
        future = event(EventType.PING)
        actual = future.result()

        # Assert
        self.assertEqual(actual, expected)

    def test_get_source_id(self) -> None:
        """Test if _get_source_id returns an ID successfully.

        This test might fail if the UNIX user invoking the test has no home directory.
        """
        # Prepare
        # nothing to prepare

        # Execute
        source_id = _get_source_id()

        # Assert
        # source_id should be len 36 as it's a uuid4 in the current
        # implementation
        self.assertIsNotNone(source_id)
        self.assertEqual(len(source_id), 36)

    def test_get_source_id_no_home(self) -> None:
        """Test if _get_source_id returns unavailable without a home dir."""
        # Prepare
        except_value = "unavailable"

        # Execute
        with mock.patch("flwr.supercore.utils.get_flwr_home", side_effect=RuntimeError):
            source_id = _get_source_id()

        # Assert
        self.assertEqual(source_id, except_value)

    def test_get_partner_id(self) -> None:
        """Test if _get_partner_id returns an ID successfully."""
        # Prepare
        generated_id = str(uuid4())
        os.environ["FLWR_TELEMETRY_PARTNER_ID"] = generated_id

        # Execute
        partner_id = _get_partner_id()

        # Assert
        self.assertEqual(partner_id, generated_id)

    def test_get_partner_id_no_env(self) -> None:
        """Test if _get_partner_id returns unavailable without an env variable."""
        # Prepare
        os.environ["FLWR_TELEMETRY_PARTNER_ID"] = ""
        expected_value = "unavailable"

        # Execute
        partner_id = _get_partner_id()

        # Assert
        self.assertEqual(partner_id, expected_value)

    def test_get_partner_id_invalid(self) -> None:
        """Test if _get_partner_id returns invalid with an incorrect env variable."""
        # Prepare
        os.environ["FLWR_TELEMETRY_PARTNER_ID"] = "not a valid ID"
        expected_value = "invalid"

        # Execute
        partner_id = _get_partner_id()

        # Assert
        self.assertEqual(partner_id, expected_value)
