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
"""Warm-pool lifecycle policy and its in-process maintenance worker."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from logging import WARNING

from flwr.supercore import log
from flwr.supercore.constant import TaskType

from .kubernetes_resources import WarmExecutorResources
from .types import ExecutionSpec, LaunchResult
from .warm_executor_dispatch import WarmExecutorDispatcher
from .warm_executor_pool import (
    PendingWarmExecutor,
    WarmExecutorPoolConfig,
    WarmExecutorPoolKey,
    WarmExecutorPoolState,
    is_compatible_warm_executor,
    is_consumed_warm_executor,
    is_warm_executor_ready,
    new_warm_executor_id,
)

_WARM_MAINTENANCE_MIN_INTERVAL_SECONDS = 0.1
_WARM_MAINTENANCE_RETRY_INTERVAL_SECONDS = 5.0
_WARM_MAINTENANCE_STOP_TIMEOUT_SECONDS = 5.0
_PRESTARTED_AGENTAPP_CREATION_INTERVAL_SECONDS = 5.0
_WARM_EXECUTOR_CREATION_RETRY_SECONDS = 5.0
_WARM_EXECUTOR_CREATION_CONFIRM_SECONDS = 30.0


class WarmExecutorPoolManager:  # pylint: disable=too-many-instance-attributes
    """Manage warm capacity while keeping maintenance off task dispatch."""

    def __init__(
        self,
        resources: WarmExecutorResources,
        active_pod_count: Callable[[set[str]], int],
        *,
        request_maintenance: Callable[[], None] | None = None,
        sweep_completed_pods: Callable[[], None] | None = None,
    ) -> None:
        self._resources = resources
        config = resources.config
        self._config = config
        self._active_pod_count = active_pod_count
        self._pools = {pool.key: pool for pool in config.warm_executor_pools}
        self._state = WarmExecutorPoolState()
        self._lock = self._state.lock
        self._busy_pods = self._state.busy_pods
        self._retiring_pods = self._state.retiring_pods
        self._pool_creation_not_before: dict[WarmExecutorPoolKey, float] = {}
        self._pending_creations = self._state.pending_creations
        self._cleanup_started = False
        self._maintenance_started_at = config.monotonic()
        self._maintenance_lock = threading.Lock()
        self._maintenance_requested = threading.Event()
        self._maintenance_stopped = threading.Event()
        self._maintenance_start_lock = threading.Lock()
        self._maintenance_thread: threading.Thread | None = None
        self._sweep_completed_pods = sweep_completed_pods or (lambda: None)
        self._dispatcher = WarmExecutorDispatcher(
            self._resources,
            self._state,
            request_maintenance or self.request_maintenance,
        )

    def launch(
        self, spec: ExecutionSpec, runtime_root_certificates: str | None
    ) -> LaunchResult | None:
        """Try a warm handoff, preserving uncertain delivery as a launch result."""
        return self._dispatcher.launch(spec, runtime_root_certificates)

    def has_ready_pod(self, task_type: TaskType, fab_hash: str | None = None) -> bool:
        """Return whether compatible unclaimed warm capacity is available."""
        return self._dispatcher.has_ready_pod(task_type, fab_hash)

    def eligible_capacity(
        self, supported_task_types: set[TaskType]
    ) -> tuple[set[TaskType], set[str]]:
        """Return task types and FAB hashes served by compatible ready workers."""
        ready_keys = self._dispatcher.ready_pool_keys()
        ready_types = {
            key.task_type
            for key in ready_keys
            if key.fab_hash is None and key.task_type in supported_task_types
        }
        ready_fabs = {
            key.fab_hash
            for key in ready_keys
            if key.fab_hash is not None
            and key.task_type == TaskType.AGENT_APP
            and key.task_type in supported_task_types
        }
        return ready_types, ready_fabs

    def sweep_completed_pods(self, sweep: Callable[[], None]) -> None:
        """Coordinate resource cleanup with in-flight creation."""
        self._resources.sweep(sweep)

    def ensure_capacity(self, reserved_pod_capacity: int = 0) -> None:
        """Create missing idle Pods for every configured compatible pool."""
        with self._maintenance_lock:
            with self._lock:
                self._maintenance_started_at = self._config.monotonic()
                if self._state.closed:
                    return
            if not self._reconcile_owned_pods():
                return
            for pool in self._pools.values():
                self._ensure_pool_capacity(pool, reserved_pod_capacity)

    def next_maintenance_delay(self, maximum: float) -> float:
        """Wake for creation pacing and retries without polling idle pools faster."""
        with self._lock:
            now = self._config.monotonic()
            deadlines = [
                *self._pool_creation_not_before.values(),
                *(pending.retry_at for pending in self._pending_creations.values()),
            ]
            # Already-attempted deadlines must not spin when capacity stays full.
            delays = [
                deadline - now
                for deadline in deadlines
                if deadline > self._maintenance_started_at
            ]
        return max(0.0, min([maximum, *delays]))

    def pending_pod_names(self) -> set[str]:
        """Count unobserved creations against foreground admission capacity."""
        with self._lock:
            return set(self._pending_creations)

    def stop(self) -> None:
        """Stop new dispatch and creation without waiting for Kubernetes I/O."""
        with self._lock:
            self._state.closed = True

    def _close_idle_pods(self) -> None:
        """Stop dispatch and delete only idle Pods owned by this SuperExec."""
        with self._lock:
            if self._cleanup_started:
                return
            self._state.closed = True
            self._cleanup_started = True
            busy_pods = self._busy_pods.copy()
            consumed_pods = self._state.consumed_pods.copy()
        with self._maintenance_lock:
            pods = self._resources.list_owned()
            pending_names = self.pending_pod_names()
        if pods is None:
            pods = []
        observed_names = {
            name for pod in pods if (name := _object_name(pod)) is not None
        }
        for pod in pods:
            pod_name = _object_name(pod)
            if (
                pod_name is not None
                and pod_name not in busy_pods
                and (
                    (
                        not is_consumed_warm_executor(pod)
                        and pod_name not in consumed_pods
                    )
                    or self._resources.consumed_has_finished(pod)
                )
            ):
                self._resources.delete(pod_name)
        for pod_name in pending_names - observed_names - busy_pods:
            self._resources.delete(pod_name)

    def _ensure_pool_capacity(  # pylint: disable=too-many-branches,too-many-return-statements
        self, pool: WarmExecutorPoolConfig, reserved_pod_capacity: int = 0
    ) -> None:
        with self._lock:
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
        observed_names = {
            name for pod in pods if (name := _object_name(pod)) is not None
        }
        with self._lock:
            for pod_name in observed_names:
                self._pending_creations.pop(pod_name, None)
            pending = [
                (pod_name, creation)
                for pod_name, creation in self._pending_creations.items()
                if creation.key == pool.key
            ]
        for pod_name, creation in pending:
            if self._config.monotonic() >= creation.retry_at:
                if not self._has_creation_capacity(
                    reserved_pod_capacity, reusing_pending=True
                ):
                    continue
                self._create_pending_pod(pod_name)
        with self._lock:
            if self._state.closed or pool.key in self._retiring_pods.values():
                return
            compatible_count = sum(
                1
                for pod in pods
                if self._resources.is_active(pod, pool.key)
                and _object_name(pod) not in self._busy_pods
                and _object_name(pod) not in self._state.consumed_pods
                and not is_consumed_warm_executor(pod)
            )
            pending_count = sum(
                creation.key == pool.key
                for creation in self._pending_creations.values()
            )
        pods_to_create = max(pool.size - compatible_count - pending_count, 0)
        for index in range(pods_to_create):
            # Foreground cold launches may consume headroom during a slow create.
            if not self._has_creation_capacity(reserved_pod_capacity):
                return
            executor_id = new_warm_executor_id()
            with self._lock:
                if self._state.closed:
                    return
                if index == 0 and pool.key.fab_hash is not None:
                    self._delay_pool_creation(pool.key)
                pod_name = self._resources.pod_name(executor_id)
                self._pending_creations[pod_name] = PendingWarmExecutor(
                    pool.key, executor_id
                )
            if not self._create_pending_pod(pod_name):
                return

    def _has_creation_capacity(
        self, reserved_pod_capacity: int, *, reusing_pending: bool = False
    ) -> bool:
        """Keep cold-launch headroom and account for unobserved warm creations."""
        with self._lock:
            if self._state.closed:
                return False
            pending_names = set(self._pending_creations)
        if self._config.active_pod_budget is None:
            return True
        try:
            required = self._active_pod_count(pending_names) + reserved_pod_capacity
            return required + int(not reusing_pending) <= self._config.active_pod_budget
        except Exception:  # pylint: disable=broad-exception-caught
            log(
                WARNING,
                "Warm executor capacity check failed; retrying later.",
                exc_info=True,
            )
            return False

    def _create_pending_pod(self, pod_name: str) -> bool:
        """Retry possibly accepted creations without resurrecting claimed workers."""
        with self._lock:
            pending = self._pending_creations.get(pod_name)
            if pending is None or self._state.closed:
                return False
        try:
            self._resources.create(pending.key, pending.executor_id)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            with self._lock:
                if self._pending_creations.get(pod_name) is pending:
                    retry_at = (
                        self._config.monotonic() + _WARM_EXECUTOR_CREATION_RETRY_SECONDS
                    )
                    if (
                        self._resources.is_definite_rejection(exc)
                        and not pending.may_exist
                    ):
                        self._pending_creations.pop(pod_name)
                        self._pool_creation_not_before[pending.key] = max(
                            self._pool_creation_not_before.get(pending.key, retry_at),
                            retry_at,
                        )
                    else:
                        pending.may_exist = True
                        pending.retry_at = retry_at
            log(WARNING, "Failed to create a warm TaskExecutor Pod.", exc_info=True)
            return False
        with self._lock:
            if self._pending_creations.get(pod_name) is pending:
                pending.may_exist = True
                pending.retry_at = (
                    self._config.monotonic() + _WARM_EXECUTOR_CREATION_CONFIRM_SECONDS
                )
        return True

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
        with self._lock:
            retiring_before = set(self._retiring_pods)
        self._retry_retiring_pods(pods)

        compatible_pods: dict[WarmExecutorPoolKey, list[object]] = {
            pool.key: [] for pool in self._pools.values()
        }
        for pod in pods:
            pod_name = _object_name(pod)
            if pod_name in retiring_before:
                continue
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
            with self._lock:
                consumed = pod_name in self._state.consumed_pods
            if pod_name is not None and (is_consumed_warm_executor(pod) or consumed):
                with self._lock:
                    busy = pod_name in self._busy_pods
                if not busy and self._resources.consumed_has_finished(pod):
                    self._retire_idle_pod(
                        pod_name,
                        retiring_pool.key if retiring_pool is not None else None,
                        confirmed_finished=True,
                    )
                continue
            with self._lock:
                if pod_name in self._retiring_pods:
                    continue
            if pool is None:
                if pod_name is not None:
                    self._retire_idle_pod(
                        pod_name,
                        retiring_pool.key if retiring_pool is not None else None,
                    )
                continue
            compatible_pods[pool.key].append(pod)

        for pool in self._pools.values():
            with self._lock:
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
                    self._retire_idle_pod(pod_name, pool.key)
        return True

    def _retry_retiring_pods(self, pods: list[object]) -> None:
        """Retry deletion of consumed Pods without making them dispatchable."""
        pod_names = {
            pod_name for pod in pods if (pod_name := _object_name(pod)) is not None
        }
        with self._lock:
            retiring_names = set(self._retiring_pods)
        for pod_name in retiring_names - pod_names:
            self._release_pod(pod_name)
        for pod in pods:
            pod_name = _object_name(pod)
            if pod_name is not None and pod_name in retiring_names:
                if self._resources.delete(pod_name):
                    self._release_pod(pod_name)

    def _release_pod(self, pod_name: str) -> None:
        with self._lock:
            self._busy_pods.discard(pod_name)
            self._state.consumed_pods.discard(pod_name)
            self._pending_creations.pop(pod_name, None)
            self._retiring_pods.pop(pod_name, None)

    def _retire_idle_pod(
        self,
        pod_name: str,
        key: WarmExecutorPoolKey | None = None,
        *,
        confirmed_finished: bool = False,
    ) -> None:
        """Retire an idle snapshot only if it cannot hide an unconfirmed task."""
        with self._lock:
            if (
                self._state.closed
                or pod_name in self._busy_pods
                or pod_name in self._retiring_pods
                or (pod_name in self._state.consumed_pods and not confirmed_finished)
            ):
                return
            self._retiring_pods[pod_name] = key
        if self._resources.delete(pod_name):
            self._release_pod(pod_name)

    def request_maintenance(self) -> None:
        """Coalesce warm-pool maintenance requests onto one worker."""
        with self._maintenance_start_lock:
            if self._maintenance_stopped.is_set():
                return
            if self._maintenance_thread is None:
                self._maintenance_thread = threading.Thread(
                    target=self._maintain_pools,
                    name="flwr-warm-pool-maintenance",
                    daemon=True,
                )
                self._maintenance_thread.start()
        self._maintenance_requested.set()

    def _maintain_pools(self) -> None:
        """Fill and clean warm pools on wakeups and during idle periods."""
        last_run_at: float | None = None
        while not self._maintenance_stopped.is_set():
            self._maintenance_requested.wait(
                self.next_maintenance_delay(_WARM_MAINTENANCE_RETRY_INTERVAL_SECONDS)
            )
            self._maintenance_requested.clear()
            if self._maintenance_stopped.is_set():
                break
            if last_run_at is not None:
                delay = max(
                    last_run_at
                    + _WARM_MAINTENANCE_MIN_INTERVAL_SECONDS
                    - time.monotonic(),
                    0.0,
                )
                if self._maintenance_stopped.wait(delay):
                    break
            last_run_at = time.monotonic()
            try:
                self._sweep_completed_pods()
                self.ensure_capacity(reserved_pod_capacity=1)
            except Exception:  # pylint: disable=broad-exception-caught
                log(
                    WARNING,
                    "Warm executor maintenance failed; retrying.",
                    exc_info=True,
                )
        self._close_idle_pods()

    def close(self) -> None:
        """Stop maintenance with a bounded wait, then clean idle owned workers."""
        self._maintenance_stopped.set()
        self.stop()
        self._maintenance_requested.set()
        with self._maintenance_start_lock:
            thread = self._maintenance_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_WARM_MAINTENANCE_STOP_TIMEOUT_SECONDS)
            if thread.is_alive():
                log(WARNING, "Warm pool maintenance is still stopping.")
                return
        self._close_idle_pods()


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
