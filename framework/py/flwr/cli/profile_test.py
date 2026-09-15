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
"""Tests for profile output formatting."""

from .profile import (
    _split_entries,
    _summarize_transport_wall_clock,
    _transport_event_round,
    _union_duration_ms,
)


def test_transport_wall_clock_does_not_double_count_overlap() -> None:
    """Overlapping transfer intervals should contribute to wall time once."""
    summary = {
        "events": [
            {
                "scope": "transport",
                "task": "superlink_supernode_downstream",
                "round": 1,
                "node_id": 7,
                "timestamp_ms": 1000.0,
                "duration_ms": 120.0,
                "network_mb": 10.0,
            },
            {
                "scope": "transport",
                "task": "superlink_supernode_upstream",
                "round": 1,
                "node_id": 7,
                "timestamp_ms": 1100.0,
                "duration_ms": 80.0,
                "network_mb": 9.0,
            },
        ]
    }

    rows = _summarize_transport_wall_clock(summary)
    combined = next(entry for entry in rows if entry["node_id"] == 7)

    assert combined["downstream_ms"] == 120.0
    assert combined["upstream_ms"] == 80.0
    assert combined["combined_ms"] == 180.0
    assert combined["downstream_mb"] == 10.0
    assert combined["upstream_mb"] == 9.0
    assert combined["round"] == 1
    assert combined["node_id"] == 7
    round_total = next(entry for entry in rows if entry["node_id"] == "all")
    assert round_total["downstream_ms"] == 120.0
    assert round_total["upstream_ms"] == 80.0
    assert round_total["node_id"] == "all"


def test_interval_union_merges_parallel_transfers() -> None:
    """The interval union should count overlapping clients only once."""
    assert _union_duration_ms([(0.0, 10.0), (2.0, 8.0), (8.0, 15.0)]) == 15.0


def test_transport_round_falls_back_to_numeric_group_id() -> None:
    """Node-side events should retain their ServerApp round correlation."""
    assert _transport_event_round({"group_id": "3"}) == 3
    assert _transport_event_round({"group_id": "evaluation"}) is None
    assert _transport_event_round({"round": 2, "group_id": "3"}) == 2

    rows = _summarize_transport_wall_clock(
        {
            "events": [
                {
                    "scope": "transport",
                    "task": "superlink_supernode_downstream",
                    "group_id": "3",
                    "node_id": 7,
                    "timestamp_ms": 1000.0,
                    "duration_ms": 100.0,
                    "network_mb": 10.0,
                }
            ]
        }
    )

    assert {row["round"] for row in rows} == {3}


def test_transport_round_total_merges_parallel_clients() -> None:
    """The all-client row should report elapsed wall time, not client-time sum."""
    summary = {
        "events": [
            {
                "scope": "transport",
                "task": "superlink_supernode_downstream",
                "round": 1,
                "node_id": node_id,
                "timestamp_ms": timestamp_ms,
                "duration_ms": 100.0,
                "network_mb": 10.0,
            }
            for node_id, timestamp_ms in [(7, 1000.0), (8, 1050.0)]
        ]
    }

    rows = _summarize_transport_wall_clock(summary)
    round_total = next(
        row
        for row in rows
        if row["hop_key"] == "superlink_supernode" and row["node_id"] == "all"
    )

    assert round_total["downstream_ms"] == 150.0
    assert round_total["combined_ms"] == 150.0
    assert round_total["downstream_mb"] == 20.0


def test_full_path_has_per_client_and_all_client_rows() -> None:
    """Full-path rows should retain clients and merge parallel wall time."""
    summary = {
        "events": [
            {
                "scope": "transport",
                "task": task,
                "round": 1,
                "node_id": node_id,
                "timestamp_ms": timestamp_ms,
                "duration_ms": 100.0,
                "network_mb": 10.0,
            }
            for task, node_id, timestamp_ms in [
                ("serverapp_clientapp_downstream", 7, 1000.0),
                ("serverapp_clientapp_upstream", 7, 1100.0),
                ("serverapp_clientapp_downstream", 8, 1050.0),
                ("serverapp_clientapp_upstream", 8, 1150.0),
            ]
        ]
    }

    rows = _summarize_transport_wall_clock(summary)
    combined = [row for row in rows if row["hop_key"] == "serverapp_clientapp"]
    all_clients = next(row for row in combined if row["node_id"] == "all")

    assert {row["node_id"] for row in combined} == {7, 8, "all"}
    assert all_clients["combined_ms"] == 250.0
    assert all_clients["downstream_mb"] == 20.0
    assert all_clients["upstream_mb"] == 20.0


def test_server_network_events_are_labeled_as_serverapp_superlink() -> None:
    """Existing ServerApp RPC measurements should use an explicit hop name."""
    summary = {
        "entries": [
            {
                "scope": "server",
                "task": "network_downstream",
                "round": 1,
                "total_ms": 25.0,
            }
        ]
    }

    regular, transport = _split_entries(summary)

    assert regular == []
    assert transport[0]["task"] == "serverapp_superlink_downstream"
    assert transport[0]["sender_node_id"] == "serverapp"
    assert transport[0]["receiver_node_id"] == "superlink"
