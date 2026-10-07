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
"""Tests for clock-safe interval reconstruction."""

from analyze import intervals, read_records


def _record(
    marker: str, clock: str = "client", scope: str = "request", **extra: object
) -> dict:
    return {
        "schema": 1,
        "marker": marker,
        "clock_domain": clock,
        "scope_id": scope,
        "monotonic_ns": 100,
        **extra,
    }


def test_backfills_identity_and_deduplicates_forwarded_records() -> None:
    """Label pre-run markers using span-end identity and deduplicate log copies."""
    start = _record("client.start_run.started", span_id="s")
    end = _record(
        "client.start_run.finished",
        span_id="s",
        run_id=7,
        monotonic_ns=120,
        duration_ns=15,
    )
    import json

    records = read_records(
        ["DEBUG: runtime_timing " + json.dumps(record) for record in [start, end, end]]
    )
    rows = intervals(records)
    assert len(rows) == 1
    assert rows[0]["run_id"] == 7
    assert rows[0]["duration_ns"] == 15


def test_never_pairs_across_clocks_and_keeps_gaps_unresolved() -> None:
    """Keep missing or cross-process boundaries unresolved."""
    rows = intervals(
        [
            _record("agent.user_code.started", "pod-a", span_id="s", run_id=7),
            _record(
                "agent.user_code.finished",
                "pod-b",
                span_id="s",
                run_id=7,
                duration_ns=10,
            ),
            _record("client.received.first_text", run_id=7),
        ]
    )
    assert len(rows) == 3
    assert all(row["duration_ns"] is None for row in rows)


def test_client_elapsed_and_route_require_confirmed_dispatch() -> None:
    """Measure client receipt locally and avoid treating reservation as acceptance."""
    records = [
        _record("client.start_run.started", span_id="s"),
        _record(
            "client.start_run.finished",
            span_id="s",
            run_id=7,
            monotonic_ns=120,
            duration_ns=15,
        ),
        _record(
            "client.received.first_text", scope="stream", run_id=7, monotonic_ns=200
        ),
        _record(
            "kubernetes.warm_reserved",
            "executor",
            "launch",
            task_id=11,
            route="generic_warm",
        ),
        _record("runtime.task_claimed", "link", "claim", task_id=11, run_id=7),
    ]
    rows = intervals(records)
    elapsed = next(
        row for row in rows if row["stage"] == "client.start_run_to_received.first_text"
    )
    assert elapsed["duration_ns"] == 100
    reserved = next(row for row in rows if row["stage"] == "kubernetes.warm_reserved")
    assert reserved["run_id"] == 7
    assert reserved["route"] is None
    records.append(
        _record(
            "kubernetes.token_ack_result",
            "executor",
            "launch",
            task_id=11,
            success=True,
            route="exact_warm",
        )
    )
    rows = intervals(records)
    assert (
        next(row for row in rows if row["stage"] == "kubernetes.warm_reserved")["route"]
        == "exact_warm"
    )


def test_child_task_keeps_parent_and_fab_correlation() -> None:
    """Enrich Model records with their parent FAB without inheriting its route."""
    records = [
        _record("agent.input_ready", "agent", task_id=11, run_id=7, fab_hash="a" * 64),
        _record(
            "runtime.child_created",
            "link",
            task_id=22,
            parent_task_id=11,
            run_id=7,
            task_type="flwr-model",
        ),
        _record("model.provider.first_text", "model", task_id=22, run_id=7),
    ]
    row = next(
        row for row in intervals(records) if row["stage"] == "model.provider.first_text"
    )
    assert row["parent_task_id"] == 11
    assert row["fab_hash"] == "a" * 64
    assert row["task_type"] == "flwr-model"
    assert row["route"] is None
