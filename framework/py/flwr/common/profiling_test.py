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
"""Unit tests for profiling utilities."""

import unittest

from .profiling import ProfileRecorder


class TestProfileRecorder(unittest.TestCase):
    """Test ProfileRecorder aggregation and derived metrics."""

    def test_summary_and_network(self) -> None:
        recorder = ProfileRecorder(run_id=1)

        recorder.record(
            "server",
            "send_and_receive",
            1,
            None,
            200.0,
            {},
            timestamp_ms=1000.0,
        )
        recorder.record(
            "client",
            "total",
            1,
            10,
            50.0,
            {},
            timestamp_ms=1200.0,
        )
        recorder.record(
            "client",
            "total",
            1,
            10,
            70.0,
            {},
            timestamp_ms=1300.0,
        )

        summary = recorder.summarize()
        entries = {
            (e["scope"], e["task"], e["round"]): e for e in summary["entries"]
        }

        self.assertIn(("server", "send_and_receive", 1), entries)
        self.assertIn(("client", "total", 1), entries)
        self.assertIn(("server", "network", 1), entries)
        self.assertAlmostEqual(entries[("client", "total", 1)]["avg_ms"], 60.0)
        self.assertAlmostEqual(entries[("server", "network", 1)]["avg_ms"], 140.0)
        self.assertAlmostEqual(summary["first_event_ts_ms"], 1000.0)
        self.assertAlmostEqual(summary["last_event_ts_ms"], 1370.0)
        self.assertAlmostEqual(summary["total_execution_ms"], 370.0)
