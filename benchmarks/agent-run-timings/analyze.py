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
"""Convert metadata-only timing logs into intervals without cross-clock subtraction."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FIELDS = (
    "run_id",
    "task_id",
    "parent_task_id",
    "task_type",
    "fab_hash",
    "route",
    "pod_name",
    "stage",
    "start_marker",
    "end_marker",
    "clock_domain",
    "start_monotonic_ns",
    "end_monotonic_ns",
    "duration_ns",
    "source",
    "success",
    "limitation",
)
IDENTITY_FIELDS = FIELDS[:7]


def read_records(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Extract probe JSON from native or warm-forwarded logs, dropping duplicates."""
    records = []
    seen = set()
    for line in lines:
        _, separator, raw = line.partition("runtime_timing ")
        if not separator:
            continue
        try:
            record, _ = json.JSONDecoder().raw_decode(raw)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("schema") != 1:
            continue
        if not all(
            isinstance(record.get(key), str)
            for key in ("marker", "clock_domain", "scope_id")
        ):
            continue
        if not isinstance(record.get("monotonic_ns"), int):
            continue
        key = (
            record["clock_domain"],
            record["scope_id"],
            record["marker"],
            record["monotonic_ns"],
        )
        if key not in seen:
            records.append(record)
            seen.add(key)
    return records


def intervals(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair spans and client milestones only within their process clock domain."""
    scopes: dict[tuple[str, str], dict[str, Any]] = {}
    tasks: dict[int, dict[str, Any]] = {}
    for record in records:
        scope = scopes.setdefault((record["clock_domain"], record["scope_id"]), {})
        for key in IDENTITY_FIELDS:
            if record.get(key) is not None:
                scope[key] = record[key]
        task_id = record.get("task_id")
        if task_id is not None:
            task = tasks.setdefault(task_id, {})
            for key in IDENTITY_FIELDS:
                if key != "route" and record.get(key) is not None:
                    task[key] = record[key]
            # A reservation alone does not prove which route accepted the task.
            if record.get("success") is True and record["marker"] in (
                "kubernetes.token_ack_result",
                "kubernetes.pod_create.finished",
            ):
                task["route"] = record.get("route")

    # Model/Connector tasks use their requesting AgentApp FAB for correlation.
    # Preserve a child task's own FAB when one is explicitly recorded.
    for task in tasks.values():
        if not task.get("fab_hash"):
            parent = tasks.get(task.get("parent_task_id"), {})
            task["fab_hash"] = parent.get("fab_hash")

    enriched = []
    for record in records:
        identity = scopes[(record["clock_domain"], record["scope_id"])]
        task = tasks.get(identity.get("task_id"), {})
        merged = dict(record)
        for key in IDENTITY_FIELDS:
            merged[key] = identity.get(key, task.get(key))
        # Always use confirmed routing, even on records for an earlier failed attempt.
        merged["route"] = task.get("route")
        enriched.append(merged)

    starts = {
        (record["clock_domain"], record["span_id"]): record
        for record in enriched
        if record.get("span_id") and record["marker"].endswith(".started")
    }
    paired = set()
    rows = []
    for record in enriched:
        marker = record["marker"]
        key = (record["clock_domain"], record.get("span_id"))
        start = starts.get(key)
        if record.get("span_id") and marker.endswith((".finished", ".failed")):
            stage = marker.rsplit(".", 1)[0]
            if start and start["marker"] == stage + ".started":
                paired.add(key)
                rows.append(
                    _row(
                        record,
                        start,
                        stage,
                        "process-local duration; spans may overlap",
                        record.get("duration_ns"),
                    )
                )
            else:
                rows.append(
                    _row(
                        record, None, stage, "missing start marker; interval unresolved"
                    )
                )
        elif not record.get("span_id"):
            rows.append(
                _row(
                    record,
                    None,
                    marker,
                    "boundary only; transport and clock gaps unresolved",
                )
            )
    for key, record in starts.items():
        if key not in paired:
            rows.append(
                _row(
                    record,
                    record,
                    record["marker"].rsplit(".", 1)[0],
                    "missing end marker; interval unresolved",
                )
            )

    # Submission through receipt is measured in the client process, including
    # StartRun overhead. Distinct scopes are joined only by run ID and clock.
    client_starts = {
        (record.get("run_id"), record["clock_domain"]): record
        for record in enriched
        if record["marker"] == "client.start_run.started" and record.get("run_id")
    }
    for record in enriched:
        if record["marker"] not in (
            "client.received.first_event",
            "client.received.first_text",
            "client.first_text_render_requested",
            "client.stream_end",
        ):
            continue
        start = client_starts.get((record.get("run_id"), record["clock_domain"]))
        if start:
            duration = record["monotonic_ns"] - start["monotonic_ns"]
            if duration >= 0:
                rows.append(
                    _row(
                        record,
                        start,
                        "client.start_run_to_"
                        + record["marker"].removeprefix("client."),
                        "same client clock; render request is not UI paint",
                        duration,
                    )
                )
    return rows


def _row(
    end: dict[str, Any],
    start: dict[str, Any] | None,
    stage: str,
    limitation: str,
    duration: int | None = None,
) -> dict[str, Any]:
    source = stage.split(".", 1)[0]
    source = {
        "agent": "worker",
        "model": "worker",
        "kubernetes": "Kubernetes",
        "runtime": "service",
        "control": "service",
        "responses": "service",
        "superexec": "service",
    }.get(source, source)
    return {
        **{key: end.get(key) for key in IDENTITY_FIELDS},
        "stage": stage,
        "start_marker": start["marker"] if start else None,
        "end_marker": end["marker"],
        "clock_domain": end["clock_domain"],
        "start_monotonic_ns": start["monotonic_ns"] if start else None,
        "end_monotonic_ns": end["monotonic_ns"],
        "duration_ns": duration,
        "source": source,
        "success": end.get("success"),
        "limitation": limitation,
    }


def main() -> None:
    """Write a sanitized CSV table for each run present in a captured log set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--run-id", type=int)
    args = parser.parse_args()
    lines = []
    for path in args.logs:
        lines.extend(path.read_text().splitlines())
    rows = intervals(read_records(lines))
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for row in rows:
        if args.run_id is None or row["run_id"] == args.run_id:
            writer.writerow(row)


if __name__ == "__main__":
    main()
