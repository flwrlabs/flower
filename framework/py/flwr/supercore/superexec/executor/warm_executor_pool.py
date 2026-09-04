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
"""Compatibility and readiness helpers for warm executor pools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from flwr.supercore.constant import TaskType

WARM_EXECUTOR_LABEL = "flower.ai/warm-executor"
WARM_EXECUTOR_FAB_HASH_ANNOTATION = "flower.ai/warm-executor-fab-hash"
WARM_EXECUTOR_RUNTIME_IMAGE_ANNOTATION = "flower.ai/warm-executor-runtime-image"
WARM_EXECUTOR_DEPENDENCY_ENVIRONMENT_ANNOTATION = (
    "flower.ai/warm-executor-dependency-environment"
)
_TASK_ID_LABEL = "flower.ai/superexec-task-id"
_TASK_TYPE_LABEL = "flower.ai/task-type"


@dataclass(frozen=True)
class WarmExecutorPoolKey:
    """Identify the task environment served by a warm executor pool."""

    task_type: TaskType
    fab_hash: str
    runtime_image: str
    dependency_environment_version: str

    def __post_init__(self) -> None:
        """Validate values persisted on a warm TaskExecutor Pod."""
        if not isinstance(self.task_type, TaskType):
            raise ValueError("Warm executor pool key requires a TaskType.")
        for field_name in (
            "fab_hash",
            "runtime_image",
            "dependency_environment_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Warm executor pool key requires a non-empty {field_name}."
                )


def new_warm_executor_id() -> str:
    """Return a DNS-label-safe identifier for one warm TaskExecutor Pod."""
    return uuid4().hex[:12]


def is_warm_executor_ready(pod: object, pool_key: WarmExecutorPoolKey) -> bool:
    """Return true for a ready warm TaskExecutor Pod with the exact pool key."""
    if not _is_compatible_warm_executor(pod, pool_key):
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


def is_warm_executor(pod: object) -> bool:
    """Return true if a Pod is identified as a warm executor."""
    metadata = _object_field(pod, "metadata")
    labels = _object_field(metadata, "labels")
    return _object_field(labels, WARM_EXECUTOR_LABEL) == "true"


def _is_compatible_warm_executor(pod: object, pool_key: WarmExecutorPoolKey) -> bool:
    """Return true if a warm TaskExecutor Pod has the supplied pool key."""
    metadata = _object_field(pod, "metadata")
    labels = _object_field(metadata, "labels")
    annotations = _object_field(metadata, "annotations")
    expected_fields = (
        (_object_field(labels, WARM_EXECUTOR_LABEL), "true"),
        (_object_field(labels, _TASK_TYPE_LABEL), pool_key.task_type.value),
        (_object_field(labels, _TASK_ID_LABEL), None),
        (
            _object_field(annotations, WARM_EXECUTOR_FAB_HASH_ANNOTATION),
            pool_key.fab_hash,
        ),
        (
            _object_field(annotations, WARM_EXECUTOR_RUNTIME_IMAGE_ANNOTATION),
            pool_key.runtime_image,
        ),
        (
            _object_field(
                annotations,
                WARM_EXECUTOR_DEPENDENCY_ENVIRONMENT_ANNOTATION,
            ),
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
