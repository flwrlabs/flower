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
"""Inert process for a private idle TaskExecutor slot."""

from __future__ import annotations

import signal
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from types import FrameType
from uuid import uuid4

from flwr.supercore.constant import TaskType

IDLE_SLOT_LABEL = "flower.ai/idle-slot"
IDLE_SLOT_FAB_HASH_ANNOTATION = "flower.ai/idle-slot-fab-hash"
IDLE_SLOT_RUNTIME_IMAGE_ANNOTATION = "flower.ai/idle-slot-runtime-image"
IDLE_SLOT_DEPENDENCY_ENVIRONMENT_ANNOTATION = (
    "flower.ai/idle-slot-dependency-environment"
)
IDLE_SLOT_MODULE = "flwr.supercore.superexec.executor.idle_slot"
IDLE_SLOT_READY_FILE = "/tmp/flwr-taskexecutor-idle-slot-ready"
_TASK_ID_LABEL = "flower.ai/superexec-task-id"
_TASK_TYPE_LABEL = "flower.ai/task-type"


@dataclass(frozen=True)
class _IdleSlotPoolKey:
    """Identify the only task environment compatible with an idle slot."""

    task_type: TaskType
    fab_hash: str
    runtime_image: str
    dependency_environment_version: str

    def __post_init__(self) -> None:
        """Validate values persisted on an idle TaskExecutor Pod."""
        if not isinstance(self.task_type, TaskType):
            raise ValueError("Idle slot pool key requires a TaskType.")
        for field_name in (
            "fab_hash",
            "runtime_image",
            "dependency_environment_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Idle slot pool key requires a non-empty {field_name}."
                )


def _new_idle_slot_id() -> str:
    """Return a DNS-label-safe opaque identifier for one idle slot."""
    return uuid4().hex[:12]


def _run_idle_slot(
    ready_file: Path = Path(IDLE_SLOT_READY_FILE),
    stop_event: Event | None = None,
) -> None:
    """Report readiness, then wait without claiming or executing a task."""
    if stop_event is None:
        stop_event = Event()

        def _request_stop(_signal_number: int, _frame: FrameType | None) -> None:
            stop_event.set()

        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)

    ready_file.parent.mkdir(parents=True, exist_ok=True)
    ready_file.touch()
    try:
        stop_event.wait()
    finally:
        ready_file.unlink(missing_ok=True)


def _is_idle_slot_ready(pod: object, pool_key: _IdleSlotPoolKey) -> bool:
    """Return true for a ready, non-terminating slot with the exact pool key."""
    if not _is_compatible_idle_slot(pod, pool_key):
        return False

    metadata = _object_field(pod, "metadata")
    deletion_timestamp = _object_field(metadata, "deletion_timestamp")
    if deletion_timestamp is None:
        deletion_timestamp = _object_field(metadata, "deletionTimestamp")
    if deletion_timestamp is not None:
        return False

    status = _object_field(pod, "status")
    if _object_field(status, "phase") != "Running":
        return False
    conditions = _object_field(status, "conditions")
    if not isinstance(conditions, Sequence) or isinstance(conditions, str):
        return False
    return any(
        _object_field(condition, "type") == "Ready"
        and _object_field(condition, "status") == "True"
        for condition in conditions
    )


def _is_compatible_idle_slot(pod: object, pool_key: _IdleSlotPoolKey) -> bool:
    """Return true if a Pod is an idle slot for exactly the supplied pool key."""
    metadata = _object_field(pod, "metadata")
    labels = _object_field(metadata, "labels")
    annotations = _object_field(metadata, "annotations")
    expected_fields = (
        (_object_field(labels, IDLE_SLOT_LABEL), "true"),
        (_object_field(labels, _TASK_TYPE_LABEL), pool_key.task_type.value),
        (_object_field(labels, _TASK_ID_LABEL), None),
        (
            _object_field(annotations, IDLE_SLOT_FAB_HASH_ANNOTATION),
            pool_key.fab_hash,
        ),
        (
            _object_field(annotations, IDLE_SLOT_RUNTIME_IMAGE_ANNOTATION),
            pool_key.runtime_image,
        ),
        (
            _object_field(annotations, IDLE_SLOT_DEPENDENCY_ENVIRONMENT_ANNOTATION),
            pool_key.dependency_environment_version,
        ),
    )
    if any(actual != expected for actual, expected in expected_fields):
        return False

    spec = _object_field(pod, "spec")
    containers = _object_field(spec, "containers")
    if not isinstance(containers, Sequence) or isinstance(containers, str):
        return False
    return any(
        _object_field(container, "name") == "taskexecutor"
        and _object_field(container, "image") == pool_key.runtime_image
        for container in containers
    )


def _object_field(value: object, field_name: str) -> object | None:
    """Return a field from a Kubernetes dict or model object."""
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


if __name__ == "__main__":
    _run_idle_slot()
