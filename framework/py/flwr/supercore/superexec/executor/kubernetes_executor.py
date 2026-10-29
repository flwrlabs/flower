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
"""Kubernetes admission and cold task launches with optional warm dispatch."""

from logging import INFO, WARNING

from flwr.supercore import log
from flwr.supercore.constant import TaskType

from .kubernetes_resources import (
    COMPLETED_POD_SWEEP_INTERVAL_SECONDS,
    KubernetesClient,
    KubernetesExecutorConfig,
    KubernetesTaskResources,
    WarmExecutorResources,
    launch_result_from_exception,
)
from .types import ExecutionSpec, LaunchResult
from .warm_executor_maintenance import WarmExecutorPoolManager


class KubernetesExecutor:  # pylint: disable=too-many-instance-attributes
    """Submit TaskExecutor Pods to Kubernetes."""

    def __init__(
        self,
        *,
        client: KubernetesClient,
        config: KubernetesExecutorConfig,
        exec_client: KubernetesClient | None = None,
        probe_client: KubernetesClient | None = None,
    ) -> None:
        self._resources = KubernetesTaskResources(client, config)
        self._config = config
        self._last_completed_pod_sweep_at: float | None = None
        self._last_capacity_log_at: float | None = None
        self._warm_executor_pool_manager = (
            WarmExecutorPoolManager(
                WarmExecutorResources(client, config, exec_client, probe_client),
                lambda: self._active_pod_count(include_pending=False),
                sweep_completed_pods=self._sweep_completed_pods_if_due,
            )
            if config.warm_executor_owner
            else None
        )

    def wait_for_capacity(
        self,
        task_type: TaskType | None = None,
        *,
        fab_hash: str | None = None,
        insecure: bool = False,
        root_certificates_path: str | None = None,
    ) -> None:
        """Wait until the configured resource pool is below its active Pod budget."""
        self._wait_for_capacity(
            task_type,
            fab_hash=fab_hash,
            allow_warm_dispatch=self._can_dispatch_warm(
                insecure, root_certificates_path
            ),
            reconcile_warm_pools=True,
        )

    def get_eligible_capacity(
        self,
        supported_task_types: set[TaskType],
        *,
        insecure: bool = False,
        root_certificates_path: str | None = None,
    ) -> tuple[set[TaskType], set[str]]:
        """Return task types and FABs with a cold slot or ready warm Pod."""
        manager = self._warm_executor_pool_manager
        if manager is not None:
            manager.request_maintenance()
        else:
            self._sweep_completed_pods_if_due()
        if self._config.active_pod_budget is None:
            return supported_task_types, set()
        try:
            active_pod_count = self._active_pod_count()
        except Exception:  # pylint: disable=broad-exception-caught
            log(
                WARNING,
                "Kubernetes capacity check failed; proceeding without waiting. "
                "selector=%s",
                self._resources.pool_selector,
                exc_info=True,
            )
            return supported_task_types, set()
        if active_pod_count < self._config.active_pod_budget:
            return supported_task_types, set()
        if manager is not None and self._can_dispatch_warm(
            insecure, root_certificates_path
        ):
            ready_types, ready_fabs = manager.eligible_capacity(supported_task_types)
            if ready_types or ready_fabs:
                return ready_types, ready_fabs

        if self._config.capacity_log_interval is not None:
            now = self._config.monotonic()
            if (
                self._last_capacity_log_at is None
                or now - self._last_capacity_log_at
                >= self._config.capacity_log_interval
            ):
                log(
                    INFO,
                    "Waiting for Kubernetes TaskExecutor capacity: "
                    "%s active Pods, budget %s, selector %s",
                    active_pod_count,
                    self._config.active_pod_budget,
                    self._resources.pool_selector,
                )
                self._last_capacity_log_at = now
        return set(), set()

    def reconcile(self) -> None:
        """Maintain warm capacity even when there are no pending tasks."""
        if self._warm_executor_pool_manager is not None:
            self._warm_executor_pool_manager.request_maintenance()

    def _wait_for_capacity(  # pylint: disable=too-many-branches
        self,
        task_type: TaskType | None,
        *,
        fab_hash: str | None = None,
        allow_warm_dispatch: bool,
        reconcile_warm_pools: bool,
    ) -> bool:
        """Wait for capacity and return True for warm dispatch or False for cold."""
        if self._warm_executor_pool_manager is None:
            self._sweep_completed_pods_if_due()
        if reconcile_warm_pools and self._warm_executor_pool_manager is not None:
            self._warm_executor_pool_manager.request_maintenance()
            has_ready_warm_pod = (
                allow_warm_dispatch
                and task_type is not None
                and self._warm_executor_pool_manager.has_ready_pod(task_type, fab_hash)
            )
            if has_ready_warm_pod:
                return True
        last_log_at: float | None = None
        waited_for_capacity = False
        while True:
            if self._config.active_pod_budget is None:
                return False
            if (
                allow_warm_dispatch
                and self._warm_executor_pool_manager is not None
                and task_type is not None
                and self._warm_executor_pool_manager.has_ready_pod(task_type, fab_hash)
            ):
                return True
            try:
                active_pod_count = self._active_pod_count()
            except Exception:  # pylint: disable=broad-exception-caught
                log(
                    WARNING,
                    "Kubernetes capacity check failed; proceeding without waiting. "
                    "selector=%s",
                    self._resources.pool_selector,
                    exc_info=True,
                )
                return False
            if active_pod_count < self._config.active_pod_budget:
                if waited_for_capacity:
                    if self._warm_executor_pool_manager is None:
                        self._last_completed_pod_sweep_at = self._config.monotonic()
                        self._sweep_completed_pods()
                    else:
                        self._warm_executor_pool_manager.request_maintenance()
                return False

            if self._config.capacity_log_interval is not None:
                now = self._config.monotonic()
                if (
                    last_log_at is None
                    or now - last_log_at >= self._config.capacity_log_interval
                ):
                    log(
                        INFO,
                        "Waiting for Kubernetes TaskExecutor capacity: "
                        "%s active Pods, budget %s, selector %s",
                        active_pod_count,
                        self._config.active_pod_budget,
                        self._resources.pool_selector,
                    )
                    last_log_at = now

            waited_for_capacity = True
            self._config.sleep(self._config.capacity_poll_interval)

    def _sweep_completed_pods_if_due(self) -> None:
        """Run best-effort completed Pod cleanup if the internal throttle allows it."""
        now = self._config.monotonic()
        if (
            self._last_completed_pod_sweep_at is not None
            and now - self._last_completed_pod_sweep_at
            < COMPLETED_POD_SWEEP_INTERVAL_SECONDS
        ):
            return

        self._last_completed_pod_sweep_at = now
        self._sweep_completed_pods()

    def _sweep_completed_pods(self) -> None:
        """Run best-effort completed Pod cleanup."""
        try:
            if self._warm_executor_pool_manager is None:
                self._resources.sweep_completed_pods()
            else:
                self._warm_executor_pool_manager.sweep_completed_pods(
                    self._resources.sweep_completed_pods
                )
        except Exception:  # pylint: disable=broad-exception-caught
            log(
                WARNING,
                "Kubernetes completed Pod sweep failed; proceeding. selector=%s",
                self._resources.pool_selector,
                exc_info=True,
            )

    def launch(self, spec: ExecutionSpec) -> LaunchResult:
        """Try a warm worker, then launch a cold task if no handoff occurred."""
        try:
            runtime_root_certificates = self._resources.root_certificates(spec)
            if self._warm_executor_pool_manager is not None:
                allow_warm_dispatch = self._can_dispatch_warm(
                    spec.insecure, spec.root_certificates_path
                )
                while True:
                    if allow_warm_dispatch:
                        warm_result = self._warm_executor_pool_manager.launch(
                            spec, runtime_root_certificates
                        )
                        if warm_result is not None:
                            return warm_result
                    # Before token delivery, a warm Pod may recover while cold
                    # capacity is full. Retry reservation when that happens.
                    if not self._wait_for_capacity(
                        spec.task_type,
                        fab_hash=spec.fab_hash,
                        allow_warm_dispatch=allow_warm_dispatch,
                        reconcile_warm_pools=False,
                    ):
                        break
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return launch_result_from_exception(exc)
        return self._resources.launch_cold(spec, runtime_root_certificates)

    def close(self) -> None:
        """Stop warm-pool maintenance and clean idle owned workers."""
        if self._warm_executor_pool_manager is not None:
            self._warm_executor_pool_manager.close()

    def _can_dispatch_warm(
        self, insecure: bool, root_certificates_path: str | None
    ) -> bool:
        """Return whether warm dispatch can use the task's Runtime transport."""
        return (
            insecure
            or self._config.runtime_root_certificates is not None
            or root_certificates_path is None
        )

    def _active_pod_count(self, *, include_pending: bool = True) -> int:
        """Count observed Pods and any unobserved warm creation reservations."""
        pending = (
            self._warm_executor_pool_manager.pending_pod_names()
            if include_pending and self._warm_executor_pool_manager is not None
            else set()
        )
        return self._resources.active_pod_count(pending)
