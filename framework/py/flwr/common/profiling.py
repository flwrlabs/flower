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
"""Minimal profiling utilities for Flower runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
import threading
from typing import Any, Callable, Iterable

from .message import Message


@dataclass(frozen=True)
class ProfileEvent:
    """A single profiling event."""

    timestamp_ms: float
    scope: str
    task: str
    round: int | None
    node_id: int | None
    duration_ms: float
    metadata: dict[str, Any]


class ProfileRecorder:
    """Record and summarize profiling events."""

    def __init__(self, run_id: int) -> None:
        self.run_id = run_id
        self._events: list[ProfileEvent] = []
        self._lock = threading.Lock()

    def record(
        self,
        scope: str,
        task: str,
        round: int | None,
        node_id: int | None,
        duration_ms: float,
        metadata: dict[str, Any] | None = None,
        timestamp_ms: float | None = None,
    ) -> None:
        """Record a profiling event."""
        if duration_ms is None:
            return
        event = ProfileEvent(
            timestamp_ms=timestamp_ms if timestamp_ms is not None else time.time() * 1000.0,
            scope=scope,
            task=task,
            round=round,
            node_id=node_id,
            duration_ms=float(duration_ms),
            metadata=metadata or {},
        )
        with self._lock:
            self._events.append(event)

    def summarize(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of all events."""
        with self._lock:
            events = list(self._events)
        total_execution_ms = 0.0
        first_event_ts_ms: float | None = None
        last_event_ts_ms: float | None = None
        if events:
            first_event_ts_ms = min(event.timestamp_ms for event in events)
            last_event_ts_ms = max(
                event.timestamp_ms + max(event.duration_ms, 0.0) for event in events
            )
            total_execution_ms = max(last_event_ts_ms - first_event_ts_ms, 0.0)

        stats: dict[tuple[str, str, int | None, int | None], dict[str, Any]] = {}
        for event in events:
            key = (event.scope, event.task, event.round, event.node_id)
            stat = stats.get(key)
            if stat is None:
                stat = {
                    "scope": event.scope,
                    "task": event.task,
                    "round": event.round,
                    "node_id": event.node_id,
                    "count": 0,
                    "sum_ms": 0.0,
                    "min_ms": None,
                    "max_ms": None,
                    "sum_mem_mb": 0.0,
                    "min_mem_mb": None,
                    "max_mem_mb": None,
                    "mem_count": 0,
                    "sum_mem_delta_mb": 0.0,
                    "min_mem_delta_mb": None,
                    "max_mem_delta_mb": None,
                    "mem_delta_count": 0,
                    "sum_disk_read_mb": 0.0,
                    "sum_disk_write_mb": 0.0,
                    "min_disk_read_mb": None,
                    "max_disk_read_mb": None,
                    "min_disk_write_mb": None,
                    "max_disk_write_mb": None,
                    "disk_read_count": 0,
                    "disk_write_count": 0,
                    "disk_sources": set(),
                }
                stats[key] = stat
            stat["count"] += 1
            stat["sum_ms"] += event.duration_ms
            stat["min_ms"] = (
                event.duration_ms
                if stat["min_ms"] is None
                else min(stat["min_ms"], event.duration_ms)
            )
            stat["max_ms"] = (
                event.duration_ms
                if stat["max_ms"] is None
                else max(stat["max_ms"], event.duration_ms)
            )
            if "memory_mb" in event.metadata and event.metadata["memory_mb"] is not None:
                mem_val = float(event.metadata["memory_mb"])
                stat["sum_mem_mb"] += mem_val
                stat["mem_count"] += 1
                stat["min_mem_mb"] = (
                    mem_val
                    if stat["min_mem_mb"] is None
                    else min(stat["min_mem_mb"], mem_val)
                )
                stat["max_mem_mb"] = (
                    mem_val
                    if stat["max_mem_mb"] is None
                    else max(stat["max_mem_mb"], mem_val)
                )
            if (
                "memory_delta_mb" in event.metadata
                and event.metadata["memory_delta_mb"] is not None
            ):
                mem_delta = float(event.metadata["memory_delta_mb"])
                stat["sum_mem_delta_mb"] += mem_delta
                stat["mem_delta_count"] += 1
                stat["min_mem_delta_mb"] = (
                    mem_delta
                    if stat["min_mem_delta_mb"] is None
                    else min(stat["min_mem_delta_mb"], mem_delta)
                )
                stat["max_mem_delta_mb"] = (
                    mem_delta
                    if stat["max_mem_delta_mb"] is None
                    else max(stat["max_mem_delta_mb"], mem_delta)
                )
            if "disk_read_mb" in event.metadata and event.metadata["disk_read_mb"] is not None:
                disk_read = float(event.metadata["disk_read_mb"])
                stat["sum_disk_read_mb"] += disk_read
                stat["disk_read_count"] += 1
                stat["min_disk_read_mb"] = (
                    disk_read
                    if stat["min_disk_read_mb"] is None
                    else min(stat["min_disk_read_mb"], disk_read)
                )
                stat["max_disk_read_mb"] = (
                    disk_read
                    if stat["max_disk_read_mb"] is None
                    else max(stat["max_disk_read_mb"], disk_read)
                )
            if "disk_write_mb" in event.metadata and event.metadata["disk_write_mb"] is not None:
                disk_write = float(event.metadata["disk_write_mb"])
                stat["sum_disk_write_mb"] += disk_write
                stat["disk_write_count"] += 1
                stat["min_disk_write_mb"] = (
                    disk_write
                    if stat["min_disk_write_mb"] is None
                    else min(stat["min_disk_write_mb"], disk_write)
                )
                stat["max_disk_write_mb"] = (
                    disk_write
                    if stat["max_disk_write_mb"] is None
                    else max(stat["max_disk_write_mb"], disk_write)
                )
            if "disk_source" in event.metadata and event.metadata["disk_source"]:
                stat["disk_sources"].add(event.metadata["disk_source"])

        entries: list[dict[str, Any]] = []
        for stat in stats.values():
            avg_ms = stat["sum_ms"] / stat["count"] if stat["count"] else 0.0
            avg_mem = (
                stat["sum_mem_mb"] / stat["mem_count"]
                if stat["mem_count"]
                else None
            )
            avg_mem_delta = (
                stat["sum_mem_delta_mb"] / stat["mem_delta_count"]
                if stat["mem_delta_count"]
                else None
            )
            avg_disk_read = (
                stat["sum_disk_read_mb"] / stat["disk_read_count"]
                if stat["disk_read_count"]
                else None
            )
            avg_disk_write = (
                stat["sum_disk_write_mb"] / stat["disk_write_count"]
                if stat["disk_write_count"]
                else None
            )
            disk_sources = stat["disk_sources"]
            if len(disk_sources) == 1:
                disk_source = next(iter(disk_sources))
            elif len(disk_sources) > 1:
                disk_source = "mixed"
            else:
                disk_source = None
            entries.append(
                {
                    "scope": stat["scope"],
                    "task": stat["task"],
                    "round": stat["round"],
                    "count": stat["count"],
                    "avg_ms": avg_ms,
                    "min_ms": stat["min_ms"] or 0.0,
                    "max_ms": stat["max_ms"] or 0.0,
                    "avg_mem_mb": avg_mem,
                    "min_mem_mb": stat["min_mem_mb"],
                    "max_mem_mb": stat["max_mem_mb"],
                    "avg_mem_delta_mb": avg_mem_delta,
                    "min_mem_delta_mb": stat["min_mem_delta_mb"],
                    "max_mem_delta_mb": stat["max_mem_delta_mb"],
                    "avg_disk_read_mb": avg_disk_read,
                    "max_disk_read_mb": stat["max_disk_read_mb"],
                    "avg_disk_write_mb": avg_disk_write,
                    "max_disk_write_mb": stat["max_disk_write_mb"],
                    "disk_source": disk_source,
                    "node_id": stat["node_id"],
                }
            )

        # Derive combined network time per round from measured upstream/downstream.
        by_round: dict[int | None, dict[str, float]] = {}
        for entry in entries:
            if entry["round"] is None:
                continue
            round_id = entry["round"]
            if round_id not in by_round:
                by_round[round_id] = {}
            if entry["scope"] == "server" and entry["task"] == "network_upstream":
                by_round[round_id]["upstream_avg"] = entry["avg_ms"]
            if entry["scope"] == "server" and entry["task"] == "network_downstream":
                by_round[round_id]["downstream_avg"] = entry["avg_ms"]

        for round_id, values in by_round.items():
            if "upstream_avg" in values and "downstream_avg" in values:
                network_ms = max(values["upstream_avg"] + values["downstream_avg"], 0.0)
                entries.append(
                    {
                        "scope": "server",
                        "task": "network",
                        "round": round_id,
                        "count": 1,
                        "avg_ms": network_ms,
                        "min_ms": network_ms,
                        "max_ms": network_ms,
                    }
                )

        entries.sort(
            key=lambda e: (
                str(e["scope"]),
                str(e["task"]),
                e["round"] or -1,
                e.get("node_id") or -1,
            )
        )

        event_entries = []
        for event in events:
            metadata = event.metadata or {}
            event_entries.append(
                {
                    "timestamp_ms": event.timestamp_ms,
                    "scope": event.scope,
                    "task": event.task,
                    "round": event.round,
                    "node_id": event.node_id,
                    "duration_ms": event.duration_ms,
                    "memory_start_mb": metadata.get("memory_start_mb"),
                    "memory_end_mb": metadata.get("memory_end_mb"),
                    "memory_delta_mb": metadata.get("memory_delta_mb"),
                    "disk_read_mb": metadata.get("disk_read_mb"),
                    "disk_write_mb": metadata.get("disk_write_mb"),
                    "disk_source": metadata.get("disk_source"),
                }
            )

        return {
            "run_id": self.run_id,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "total_execution_ms": total_execution_ms,
            "first_event_ts_ms": first_event_ts_ms,
            "last_event_ts_ms": last_event_ts_ms,
            "entries": entries,
            "events": event_entries,
        }


_profile_state = threading.local()


def set_active_profiler(profiler: ProfileRecorder | None) -> None:
    """Set the active profiler for the current thread."""
    _profile_state.profiler = profiler


def set_profile_publisher(publisher: Callable[[dict[str, Any]], None] | None) -> None:
    """Set the profile publisher for the current thread."""
    _profile_state.publisher = publisher


def clear_profile_publisher() -> None:
    """Clear the profile publisher for the current thread."""
    if hasattr(_profile_state, "publisher"):
        delattr(_profile_state, "publisher")


def clear_active_profiler() -> None:
    """Clear the active profiler for the current thread."""
    if hasattr(_profile_state, "profiler"):
        delattr(_profile_state, "profiler")


def get_active_profiler() -> ProfileRecorder | None:
    """Return the active profiler for the current thread."""
    return getattr(_profile_state, "profiler", None)


def publish_profile_summary() -> None:
    """Publish the latest profile summary if a publisher is registered."""
    profiler = get_active_profiler()
    publisher = getattr(_profile_state, "publisher", None)
    if profiler is None or publisher is None:
        return
    try:
        publisher(profiler.summarize())
    except Exception:
        # Publishing must not impact normal control flow
        return


def set_current_round(round_id: int | None) -> None:
    """Set the current round for the active thread."""
    _profile_state.current_round = round_id


def get_current_round() -> int | None:
    """Return the current round for the active thread."""
    return getattr(_profile_state, "current_round", None)


def record_profile_metrics_from_messages(messages: Iterable[Message]) -> None:
    """Extract profile metrics from message MetricRecords and record them."""
    profiler = get_active_profiler()
    if profiler is None:
        return

    round_id = get_current_round()
    for msg in messages:
        if msg.has_error() or msg.content is None:
            continue
        try:
            for metric_record in msg.content.metric_records.values():
                durations: dict[str, float] = {}
                mems: dict[str, float] = {}
                mem_starts: dict[str, float] = {}
                mem_ends: dict[str, float] = {}
                mem_deltas: dict[str, float] = {}
                disk_reads: dict[str, float] = {}
                disk_writes: dict[str, float] = {}
                disk_sources: dict[str, str] = {}
                for key, value in metric_record.items():
                    if not key.startswith("profile.client."):
                        continue
                    if key.endswith(".ms"):
                        if not isinstance(value, (int, float)):
                            continue
                        task = key[len("profile.client.") : -3]
                        durations[task] = float(value)
                    elif key.endswith(".mem_mb"):
                        if not isinstance(value, (int, float)):
                            continue
                        task = key[len("profile.client.") : -7]
                        mems[task] = float(value)
                    elif key.endswith(".mem_start_mb"):
                        if not isinstance(value, (int, float)):
                            continue
                        task = key[len("profile.client.") : -13]
                        mem_starts[task] = float(value)
                    elif key.endswith(".mem_end_mb"):
                        if not isinstance(value, (int, float)):
                            continue
                        task = key[len("profile.client.") : -11]
                        mem_ends[task] = float(value)
                    elif key.endswith(".mem_delta_mb"):
                        if not isinstance(value, (int, float)):
                            continue
                        task = key[len("profile.client.") : -13]
                        mem_deltas[task] = float(value)
                    elif key.endswith(".disk_read_mb"):
                        if not isinstance(value, (int, float)):
                            continue
                        task = key[len("profile.client.") : -14]
                        disk_reads[task] = float(value)
                    elif key.endswith(".disk_write_mb"):
                        if not isinstance(value, (int, float)):
                            continue
                        task = key[len("profile.client.") : -15]
                        disk_writes[task] = float(value)
                    elif key.endswith(".disk_source"):
                        if not isinstance(value, str):
                            continue
                        task = key[len("profile.client.") : -12]
                        disk_sources[task] = value
                for task, duration in durations.items():
                    metadata = {}
                    if task in mem_starts:
                        metadata["memory_start_mb"] = mem_starts[task]
                    if task in mem_ends:
                        metadata["memory_end_mb"] = mem_ends[task]
                        metadata["memory_mb"] = mem_ends[task]
                    if task in mems:
                        metadata.setdefault("memory_mb", mems[task])
                    if task in mem_deltas:
                        metadata["memory_delta_mb"] = mem_deltas[task]
                    if task in disk_reads:
                        metadata["disk_read_mb"] = disk_reads[task]
                    if task in disk_writes:
                        metadata["disk_write_mb"] = disk_writes[task]
                    if task in disk_sources:
                        metadata["disk_source"] = disk_sources[task]
                    profiler.record(
                        scope="client",
                        task=task,
                        round=round_id,
                        node_id=msg.metadata.src_node_id,
                        duration_ms=duration,
                        metadata=metadata,
                    )
        except Exception:
            # Profiling should never break normal control flow
            continue


def record_network_delivery_metrics_from_messages(
    messages: Iterable[Message], delivered_at_ms: float
) -> None:
    """Record network delivery timings derived from per-message delivery anchors."""
    profiler = get_active_profiler()
    if profiler is None:
        return

    round_id = get_current_round()
    for msg in messages:
        try:
            if msg.has_error():
                continue
            # Upstream: SuperNode reply enqueue at SuperLink -> ServerApp delivery.
            created_at_ms = float(msg.metadata.created_at) * 1000.0
            upstream_ms = max(delivered_at_ms - created_at_ms, 0.0)
            profiler.record(
                scope="network",
                task="upstream",
                round=round_id,
                node_id=None,
                duration_ms=upstream_ms,
            )

            # Downstream is injected by SuperLink as part of reply payload when
            # both instruction enqueue and ClientApp delivery anchors are available.
            downstream_ms: float | None = None
            downstream_hint = msg.metadata.__dict__.get("_network_downstream_ms")
            if isinstance(downstream_hint, (int, float)):
                downstream_ms = max(float(downstream_hint), 0.0)
            elif msg.has_content():
                metric_record = msg.content.metric_records.get("_flwr_network_delivery")
                if metric_record is not None:
                    value = metric_record.get("downstream_ms")
                    if isinstance(value, (int, float)):
                        downstream_ms = max(float(value), 0.0)

            if downstream_ms is not None:
                profiler.record(
                    scope="network",
                    task="downstream",
                    round=round_id,
                    node_id=None,
                    duration_ms=downstream_ms,
                )
                profiler.record(
                    scope="network",
                    task="combined",
                    round=round_id,
                    node_id=None,
                    duration_ms=downstream_ms + upstream_ms,
                )
        except Exception:
            # Profiling should never break normal control flow
            continue
