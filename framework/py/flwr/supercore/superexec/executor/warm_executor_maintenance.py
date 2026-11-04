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
"""Synchronous warm-pool capacity and retirement policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from logging import WARNING

from flwr.supercore import log
from flwr.supercore.constant import TaskType

from .kubernetes_resources import WarmExecutorResources
from .types import ExecutionSpec, LaunchResult
from .warm_executor_dispatch import WarmExecutorDispatcher
from .warm_executor_pool import (
    WarmExecutorPoolConfig,
    WarmExecutorPoolKey,
    WarmExecutorPoolState,
    is_compatible_warm_executor,
    is_consumed_warm_executor,
    is_warm_executor_ready,
    new_warm_executor_id,
)

_PRESTARTED_AGENTAPP_CREATION_INTERVAL_SECONDS = 5.0


class WarmExecutorPoolManager:  # pylint: disable=too-many-instance-attributes
    """Maintain warm capacity synchronously with task admission and completion."""

    def __init__(
        self,
        resources: WarmExecutorResources,
        active_pod_count: Callable[[], int],
        *,
        state: WarmExecutorPoolState,
    ) -> None:
        self._resources = resources
        self._config = resources.config
        self._active_pod_count = active_pod_count
        self._pools = {pool.key: pool for pool in self._config.warm_executor_pools}
        self._state = state
        self._lock = state.lock
        self._busy_pods = state.busy_pods
        self._retiring_pods = state.retiring_pods
        self._pool_creation_not_before: dict[WarmExecutorPoolKey, float] = {}
        self._dispatcher = WarmExecutorDispatcher(
            resources, state, self._replenish_pool
        )

    def launch(
        self, spec: ExecutionSpec, runtime_root_certificates: str | None
    ) -> LaunchResult | None:
        """Dispatch a compatible task to a ready Pod or use the cold fallback."""
        return self._dispatcher.launch(spec, runtime_root_certificates)

    def has_ready_pod(self, task_type: TaskType, fab_hash: str | None = None) -> bool:
        """Return whether a matching warm Pod can take a task without new capacity."""
        return self._dispatcher.has_ready_pod(task_type, fab_hash)

    def ready_pool_keys(self) -> set[WarmExecutorPoolKey]:
        """Return compatible pool keys with a Pod ready for dispatch."""
        return self._dispatcher.ready_pool_keys()

    def sweep_completed_pods(self, sweep: Callable[[], None]) -> None:
        """Run completed-Pod cleanup without racing warm Pod creation."""
        self._resources.sweep(sweep)

    def _replenish_pool(
        self, key: WarmExecutorPoolKey, reserved_pod_capacity: int
    ) -> None:
        """Replenish one pool while the dispatcher holds the shared state lock."""
        self._ensure_pool_capacity(self._pools[key], reserved_pod_capacity)

    def ensure_capacity(self, reserved_pod_capacity: int = 0) -> None:
        """Create missing idle Pods for every configured compatible pool."""
        with self._lock:
            if self._state.closed:
                return
            if not self._reconcile_owned_pods():
                return
            for pool in self._pools.values():
                self._ensure_pool_capacity(pool, reserved_pod_capacity)

    def retry_retiring_pods(self) -> None:
        """Retry cleanup, including surviving tasks, without refilling pools."""
        with self._lock:
            if self._state.closed:
                return
            self._reconcile_owned_pods()

    def close(self) -> None:
        """Stop dispatch and delete only idle Pods owned by this SuperExec."""
        with self._lock:
            if self._state.closed:
                return
            self._state.closed = True
            busy_pods = self._busy_pods.copy()
        pods = self._resources.list_owned()
        if pods is None:
            return
        for pod in pods:
            pod_name = _object_name(pod)
            if (
                pod_name is not None
                and pod_name not in busy_pods
                and (
                    not is_consumed_warm_executor(pod)
                    or self._resources.consumed_has_finished(pod)
                )
            ):
                self._resources.delete(pod_name)

    def _ensure_pool_capacity(
        self, pool: WarmExecutorPoolConfig, reserved_pod_capacity: int = 0
    ) -> None:
        if self._state.closed or pool.key in self._retiring_pods.values():
            return
        creation_not_before = self._pool_creation_not_before.get(pool.key)
        if creation_not_before is not None:
            if self._config.monotonic() < creation_not_before:
                return
            self._pool_creation_not_before.pop(pool.key, None)
        pods = self._resources.list_owned()
        if pods is None:
            return
        compatible_count = sum(
            1
            for pod in pods
            if self._resources.is_active(pod, pool.key)
            and _object_name(pod) not in self._busy_pods
            and not is_consumed_warm_executor(pod)
        )
        pods_to_create = max(pool.size - compatible_count, 0)
        if self._config.active_pod_budget is not None:
            try:
                available_pod_capacity = (
                    self._config.active_pod_budget
                    - self._active_pod_count()
                    - reserved_pod_capacity
                )
            except Exception:  # pylint: disable=broad-exception-caught
                log(
                    WARNING,
                    "Warm executor capacity check failed; "
                    "not creating replacement Pods.",
                    exc_info=True,
                )
                return
            pods_to_create = min(pods_to_create, max(available_pod_capacity, 0))
        if pool.key.fab_hash is not None and pods_to_create:
            self._delay_pool_creation(pool.key)
        for _ in range(pods_to_create):
            try:
                self._resources.create(pool.key, new_warm_executor_id())
            except Exception:  # pylint: disable=broad-exception-caught
                log(WARNING, "Failed to create a warm TaskExecutor Pod.", exc_info=True)
                return

    def _delay_pool_creation(self, key: WarmExecutorPoolKey) -> None:
        """Rate-limit creation for one FAB-specific pool."""
        self._pool_creation_not_before[key] = max(
            self._pool_creation_not_before.get(key, float("-inf")),
            self._config.monotonic() + _PRESTARTED_AGENTAPP_CREATION_INTERVAL_SECONDS,
        )

    def _reconcile_owned_pods(self) -> bool:
        """Delete owned Pods that are obsolete or exceed configured capacity."""
        pods = self._resources.list_owned()
        if pods is None:
            return False
        self._retry_retiring_pods(pods)

        compatible_pods: dict[WarmExecutorPoolKey, list[object]] = {
            pool.key: [] for pool in self._pools.values()
        }
        for pod in pods:
            pod_name = _object_name(pod)
            retiring_pool = next(
                (
                    candidate
                    for candidate in self._pools.values()
                    if is_compatible_warm_executor(pod, candidate.key)
                ),
                None,
            )
            pool = next(
                (
                    candidate
                    for candidate in self._pools.values()
                    if self._resources.is_active(pod, candidate.key)
                ),
                None,
            )
            if (
                pod_name is not None
                and pod_name not in self._busy_pods
                and is_consumed_warm_executor(pod)
            ):
                if self._resources.consumed_has_finished(pod):
                    self._retire_pod(
                        pod_name,
                        retiring_pool.key if retiring_pool is not None else None,
                    )
                continue
            if pod_name in self._retiring_pods:
                continue
            if pool is None:
                if pod_name is not None and pod_name not in self._busy_pods:
                    self._retire_pod(
                        pod_name,
                        retiring_pool.key if retiring_pool is not None else None,
                    )
                continue
            compatible_pods[pool.key].append(pod)

        for pool in self._pools.values():
            idle_pods = [
                pod
                for pod in compatible_pods[pool.key]
                if _object_name(pod) not in self._busy_pods
            ]
            # Pylint cannot infer that list.sort invokes its key synchronously.
            # pylint: disable=cell-var-from-loop
            idle_pods.sort(key=lambda pod: not is_warm_executor_ready(pod, pool.key))
            # pylint: enable=cell-var-from-loop
            for pod in idle_pods[pool.size :]:
                pod_name = _object_name(pod)
                if pod_name is not None:
                    self._retire_pod(pod_name, pool.key)
        return True

    def _retry_retiring_pods(self, pods: list[object]) -> None:
        """Retry deletion of consumed Pods without making them dispatchable."""
        pod_names = {
            pod_name for pod in pods if (pod_name := _object_name(pod)) is not None
        }
        for pod_name in self._retiring_pods.keys() - pod_names:
            self._release_pod(pod_name)
        for pod in pods:
            pod_name = _object_name(pod)
            if pod_name is not None and pod_name in self._retiring_pods:
                if self._resources.delete(pod_name):
                    self._release_pod(pod_name)

    def _release_pod(self, pod_name: str) -> None:
        with self._lock:
            self._busy_pods.discard(pod_name)
            self._retiring_pods.pop(pod_name, None)

    def _retire_pod(
        self, pod_name: str, key: WarmExecutorPoolKey | None = None
    ) -> bool:
        with self._lock:
            self._retiring_pods[pod_name] = key
        if self._resources.delete(pod_name):
            self._release_pod(pod_name)
            return True
        return False


def _object_name(value: object) -> str | None:
    """Return an object's metadata name."""
    metadata = _object_field(value, "metadata")
    name = _object_field(metadata, "name")
    if isinstance(name, str) and name.strip():
        return name
    return None


def _object_field(value: object, field_name: str) -> object | None:
    """Return a field from a Kubernetes dict or model object."""
    if isinstance(value, Mapping):
        return value.get(field_name)
    return getattr(value, field_name, None)
