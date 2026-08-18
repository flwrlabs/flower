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

from .message import Message
from .profiling import (
    ProfileRecorder,
    clear_active_profiler,
    record_network_delivery_metrics_from_messages,
    set_active_profiler,
)
from .record import MetricRecord, RecordDict


class TestProfileRecorder(unittest.TestCase):
    """Test ProfileRecorder aggregation and derived metrics."""

    def test_summary_and_network(self) -> None:
        """Test timing, network byte, and endpoint aggregation."""
        recorder = ProfileRecorder(run_id=1)

        recorder.record(
            "server",
            "network_downstream",
            1,
            None,
            40.0,
            {"network_bytes": 2 * 1024 * 1024},
            timestamp_ms=1000.0,
        )
        recorder.record(
            "server",
            "network_upstream",
            1,
            None,
            100.0,
            {"network_bytes": 3 * 1024 * 1024},
            timestamp_ms=1050.0,
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
        recorder.record(
            "network",
            "upstream",
            1,
            10,
            10.0,
            {
                "network_bytes": 2 * 1024 * 1024,
                "sender_node_id": 10,
                "receiver_node_id": "server",
            },
            timestamp_ms=1400.0,
        )
        recorder.record(
            "network",
            "upstream",
            1,
            10,
            20.0,
            {
                "network_bytes": 1024 * 1024,
                "sender_node_id": 10,
                "receiver_node_id": "server",
            },
            timestamp_ms=1420.0,
        )

        summary = recorder.summarize()
        entries = {
            (e["scope"], e["task"], e["round"], e.get("node_id")): e
            for e in summary["entries"]
        }

        self.assertIn(("server", "network_downstream", 1, None), entries)
        self.assertIn(("client", "total", 1, 10), entries)
        self.assertIn(("server", "network", 1, None), entries)
        self.assertIn(("network", "upstream", 1, 10), entries)
        self.assertAlmostEqual(entries[("client", "total", 1, 10)]["avg_ms"], 60.0)
        self.assertAlmostEqual(entries[("server", "network", 1, None)]["avg_ms"], 140.0)
        self.assertAlmostEqual(
            entries[("server", "network", 1, None)]["total_network_mb"], 5.0
        )
        self.assertAlmostEqual(
            entries[("network", "upstream", 1, 10)]["avg_network_mb"], 1.5
        )
        self.assertAlmostEqual(
            entries[("network", "upstream", 1, 10)]["total_network_mb"], 3.0
        )
        self.assertEqual(entries[("network", "upstream", 1, 10)]["sender_node_id"], 10)
        self.assertEqual(
            entries[("network", "upstream", 1, 10)]["receiver_node_id"], "server"
        )
        self.assertAlmostEqual(summary["first_event_ts_ms"], 1000.0)
        self.assertAlmostEqual(summary["last_event_ts_ms"], 1440.0)
        self.assertAlmostEqual(summary["total_execution_ms"], 440.0)

    def test_derived_network_sums_multiple_events(self) -> None:
        """Derived server network time must not discard repeated transfers."""
        recorder = ProfileRecorder(run_id=1)
        recorder.record(
            "server",
            "network_downstream",
            1,
            None,
            10.0,
            {"network_bytes": 1024},
        )
        recorder.record(
            "server",
            "network_downstream",
            1,
            None,
            20.0,
            {"network_bytes": 2048},
        )
        recorder.record(
            "server",
            "network_upstream",
            1,
            None,
            30.0,
            {"network_bytes": 4096},
        )
        recorder.record(
            "server",
            "network_upstream",
            1,
            None,
            40.0,
            {"network_bytes": 8192},
        )

        entry = next(
            entry
            for entry in recorder.summarize()["entries"]
            if entry["task"] == "network"
        )
        self.assertAlmostEqual(entry["avg_ms"], 100.0)
        self.assertAlmostEqual(entry["total_network_mb"], 15.0 / 1024.0)

    def test_network_delivery_prefers_per_message_timestamp(self) -> None:
        """Upstream timing must not use the end of a multi-reply pull batch."""
        instruction = Message(RecordDict(), dst_node_id=7, message_type="train")
        reply = Message(
            RecordDict({"metrics": MetricRecord()}),
            reply_to=instruction,
        )
        reply.metadata.created_at = 10.0
        reply.metadata.__dict__["_network_delivered_at_ms"] = 10_500.0
        reply.metadata.__dict__["_network_upstream_bytes"] = 1024

        recorder = ProfileRecorder(run_id=1)
        set_active_profiler(recorder)
        try:
            record_network_delivery_metrics_from_messages(
                [reply], delivered_at_ms=50_000.0
            )
        finally:
            clear_active_profiler()

        entry = next(
            entry
            for entry in recorder.summarize()["entries"]
            if entry["scope"] == "network" and entry["task"] == "upstream"
        )
        self.assertAlmostEqual(entry["avg_ms"], 500.0)
