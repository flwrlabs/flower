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
"""Tests for configurable Flower constants."""

import os
import subprocess
import sys
import unittest
from pathlib import Path


class TestHeartbeatEnvironmentConfiguration(unittest.TestCase):
    """Test heartbeat settings read from environment at process startup."""

    def _run_import(self, settings: dict[str, str]) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for name in (
            "FLWR_HEARTBEAT_PROFILE",
            "FLWR_HEARTBEAT_INTERVAL_S",
            "FLWR_HEARTBEAT_RPC_TIMEOUT_S",
            "FLWR_APP_HEARTBEAT_RPC_TIMEOUT_S",
        ):
            env.pop(name, None)
        env.update(settings)
        return subprocess.run(
            [
                sys.executable,
                "-c",
                "from flwr.common.constant import ("
                "HEARTBEAT_DEFAULT_INTERVAL, HEARTBEAT_CALL_TIMEOUT, "
                "APP_HEARTBEAT_CALL_TIMEOUT, HEARTBEAT_PATIENCE, "
                "HEARTBEAT_CLIENTAPP_LEASE); "
                "print(HEARTBEAT_DEFAULT_INTERVAL, HEARTBEAT_CALL_TIMEOUT, "
                "APP_HEARTBEAT_CALL_TIMEOUT, HEARTBEAT_PATIENCE, "
                "HEARTBEAT_CLIENTAPP_LEASE)",
            ],
            cwd=Path(__file__).parents[2],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_default_profile_preserves_current_settings(self) -> None:
        """The new profile selector must not change default deployment behavior."""
        result = self._run_import({})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "30 5 30 2 600")

    def test_slow_profile_selects_coordinated_settings(self) -> None:
        """The slow profile should select its complete coordinated setting bundle."""
        result = self._run_import(
            {
                "FLWR_HEARTBEAT_PROFILE": "slow",
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "60 45 180 20 3600")

    def test_clientapp_timeout_can_be_injected_by_supernode(self) -> None:
        """A SuperNode can pass the centrally negotiated timeout to its worker."""
        result = self._run_import(
            {
                "FLWR_HEARTBEAT_PROFILE": "slow",
                "FLWR_HEARTBEAT_INTERVAL_S": "90",
                "FLWR_HEARTBEAT_RPC_TIMEOUT_S": "30",
                "FLWR_APP_HEARTBEAT_RPC_TIMEOUT_S": "240",
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "90 30 240 20 3600")

    def test_unknown_profile_is_rejected(self) -> None:
        """Reject misspelled profiles to avoid silently using short defaults."""
        result = self._run_import({"FLWR_HEARTBEAT_PROFILE": "slwo"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FLWR_HEARTBEAT_PROFILE must be one of", result.stderr)

    def test_node_heartbeat_timeout_must_be_less_than_interval(self) -> None:
        """Reject invalid heartbeat cadence instead of spinning on a zero wait."""
        result = self._run_import(
            {
                "FLWR_HEARTBEAT_INTERVAL_S": "30",
                "FLWR_HEARTBEAT_RPC_TIMEOUT_S": "30",
            }
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be positive and less than", result.stderr)


if __name__ == "__main__":
    unittest.main()
