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
"""Kubernetes executor for SuperExec TaskExecutor processes."""

# pylint: disable=too-many-lines

import importlib
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from logging import INFO, WARNING
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

from flwr.supercore import log
from flwr.supercore.constant import (
    TASK_TYPE_TO_APPIO_API_ADDRESS_ARG,
    TASK_TYPE_TO_COMMAND,
    TaskType,
)
from flwr.supercore.typing import JSONObject

from .types import ExecutionSpec, LaunchResult
from .warm_executor import (
    WARM_EXECUTOR_MODULE,
    WARM_EXECUTOR_READINESS_COMMAND,
    WARM_EXECUTOR_READY_DIRECTORY,
    WARM_EXECUTOR_READY_FILE,
)
from .warm_executor_pool import (
    WARM_EXECUTOR_DEPENDENCY_ENVIRONMENT_ANNOTATION,
    WARM_EXECUTOR_FAB_HASH_ANNOTATION,
    WARM_EXECUTOR_LABEL,
    WARM_EXECUTOR_RUNTIME_IMAGE_ANNOTATION,
    WarmExecutorPoolConfig,
    WarmExecutorPoolKey,
    is_compatible_warm_executor,
    is_warm_executor,
    is_warm_executor_ready,
    new_warm_executor_id,
)

APPIO_CREDENTIALS_MOUNT_PATH = "/run/flwr/appio"
APPIO_TOKEN_FILE_PATH = f"{APPIO_CREDENTIALS_MOUNT_PATH}/token"
APPIO_ROOT_CERTIFICATES_FILE_PATH = f"{APPIO_CREDENTIALS_MOUNT_PATH}/ca.crt"
LAUNCH_ATTEMPT_LABEL = "flower.ai/launch-attempt"
_TASK_ID_LABEL = "flower.ai/superexec-task-id"
_NAME_LABEL = "app.kubernetes.io/name"
_COMPONENT_LABEL = "app.kubernetes.io/component"
_TASK_TYPE_LABEL = "flower.ai/task-type"
_RESOURCE_POOL_LABEL = "flower.ai/resource-pool"
_WARM_EXECUTOR_OWNER_LABEL = "flower.ai/warm-executor-owner"
_EXECUTOR_OWNED_LABELS = frozenset(
    {
        _NAME_LABEL,
        _COMPONENT_LABEL,
        _TASK_ID_LABEL,
        _TASK_TYPE_LABEL,
        LAUNCH_ATTEMPT_LABEL,
        _RESOURCE_POOL_LABEL,
        WARM_EXECUTOR_LABEL,
        _WARM_EXECUTOR_OWNER_LABEL,
    }
)
_APPIO_CREDENTIAL_SECRET_SUFFIX = "-appio"
_WARM_EXECUTOR_READY_VOLUME_NAME = "warm-executor-ready"
_RESERVED_TASKEXECUTOR_VOLUME_NAMES = frozenset(
    {"appio-credentials", _WARM_EXECUTOR_READY_VOLUME_NAME}
)
_RESERVED_TASKEXECUTOR_VOLUME_MOUNT_PATHS = frozenset(
    {
        APPIO_CREDENTIALS_MOUNT_PATH,
        WARM_EXECUTOR_READY_DIRECTORY,
        WARM_EXECUTOR_READY_FILE,
    }
)
_COMPLETED_POD_SWEEP_INTERVAL_SECONDS = 60.0
_FORBIDDEN_TASKEXECUTOR_ENV_NAMES = frozenset(
    {
        "FLWR_MODEL_API_KEY",
        "BRAVE_API_KEY",
        "TAVILY_API_KEY",
        "EXA_API_KEY",
    }
)
_KUBERNETES_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DNS_LABEL_PATTERN = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_WARM_EXECUTOR_ACK_TIMEOUT_SECONDS = 5.0
_AGENTAPP_TOKEN_STDIN_ACKNOWLEDGEMENT = "FLWR_AGENTAPP_TOKEN_ACCEPTED"


class KubernetesList(Protocol):
    """Subset of Kubernetes list responses used by executor list helpers."""

    items: Sequence[object]


class KubernetesClient(Protocol):
    """Subset of Kubernetes CoreV1Api used by the executor."""

    def create_namespaced_secret(self, namespace: str, body: JSONObject) -> object:
        """Create a Kubernetes Secret in the selected namespace."""

    def create_namespaced_pod(self, namespace: str, body: JSONObject) -> object:
        """Create a Kubernetes Pod in the selected namespace."""

    def delete_namespaced_secret(self, name: str, namespace: str) -> object:
        """Delete a Kubernetes Secret from the selected namespace."""

    def list_namespaced_secret(
        self, namespace: str, label_selector: str
    ) -> KubernetesList:
        """List Kubernetes Secrets in the selected namespace."""

    def delete_namespaced_pod(
        self, name: str, namespace: str, grace_period_seconds: int = 0
    ) -> object:
        """Delete a Kubernetes Pod in the selected namespace."""

    def list_namespaced_pod(
        self, namespace: str, label_selector: str
    ) -> KubernetesList:
        """List Kubernetes Pods in the selected namespace."""

    def connect_get_namespaced_pod_exec(
        self, *args: object, **kwargs: object
    ) -> object:
        """Open a Pod exec connection used for one warm task handoff."""


def create_incluster_kubernetes_client() -> KubernetesClient:
    """Create a KubernetesClient backed by in-cluster ServiceAccount auth."""
    try:
        kubernetes_client = importlib.import_module("kubernetes.client")
        kubernetes_config = importlib.import_module("kubernetes.config")
    except ModuleNotFoundError as exc:
        missing_module = exc.name
        if missing_module in {"kubernetes", "kubernetes.client", "kubernetes.config"}:
            raise RuntimeError(
                "Kubernetes Python client package is required for the Kubernetes "
                "executor. Install the official 'kubernetes' package in the "
                "SuperExec environment."
            ) from exc
        raise

    try:
        kubernetes_config.load_incluster_config()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise RuntimeError(
            "Failed to load in-cluster Kubernetes configuration for the Kubernetes "
            "executor. Run SuperExec in a Kubernetes Pod with ServiceAccount "
            "credentials."
        ) from exc

    client: KubernetesClient = kubernetes_client.CoreV1Api()
    return client


@dataclass
class KubernetesExecutorConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration needed to build one TaskExecutor Pod and Secret.

    Parameters
    ----------
    namespace : str
        Kubernetes namespace for TaskExecutor Pods and credential Secrets.
    image : str
        Container image used for TaskExecutor Pods.
    runtime_root_certificates : str | None
        Optional PEM data mounted as ca.crt. If unset, launch uses
        ExecutionSpec.root_certificates_path when provided.
    image_pull_policy : str | None
        Optional Kubernetes imagePullPolicy for the TaskExecutor container.
    labels : dict[str, str] | None
        Extra labels added to generated Pods and Secrets.
    annotations : dict[str, str] | None
        Extra annotations added to generated Pods and Secrets.
    resource_pool : str | None
        Optional Flower resource-pool label value.
    resources : JSONObject | None
        Optional Kubernetes container resource requests and limits.
    env : list[JSONObject] | None
        Optional explicit TaskExecutor container environment. Only literal
        name/value entries are supported.
    volumes : list[JSONObject] | None
        Optional Kubernetes Pod volumes.
    volume_mounts : list[JSONObject] | None
        Optional Kubernetes TaskExecutor container volume mounts.
    node_selector : dict[str, str] | None
        Optional Kubernetes nodeSelector.
    tolerations : list[JSONObject] | None
        Optional Kubernetes tolerations.
    affinity : JSONObject | None
        Optional Kubernetes affinity.
    priority_class_name : str | None
        Optional Kubernetes priorityClassName.
    pod_security_context : JSONObject | None
        Optional Kubernetes Pod securityContext.
    container_security_context : JSONObject | None
        Optional TaskExecutor container securityContext.
    service_account_name : str | None
        Optional Kubernetes serviceAccountName. Service account policy/RBAC is
        decided outside this executor.
    """

    namespace: str
    image: str
    runtime_root_certificates: str | None = None
    image_pull_policy: str | None = None
    labels: dict[str, str] | None = None
    annotations: dict[str, str] | None = None
    resource_pool: str | None = None
    resources: JSONObject | None = None
    env: list[JSONObject] | None = None
    volumes: list[JSONObject] | None = None
    volume_mounts: list[JSONObject] | None = None
    node_selector: dict[str, str] | None = None
    tolerations: list[JSONObject] | None = None
    affinity: JSONObject | None = None
    priority_class_name: str | None = None
    pod_security_context: JSONObject | None = None
    container_security_context: JSONObject | None = None
    # Optional Pod field only; service account policy/RBAC is decided elsewhere.
    service_account_name: str | None = None
    active_pod_budget: int | None = None
    capacity_poll_interval: float = 1.0
    capacity_log_interval: float | None = None
    # Each static pool needs one active owner. Deployments with warm pools must
    # use one SuperExec replica per owner value unless they add leader election.
    warm_executor_owner: str | None = None
    warm_executor_pools: tuple[WarmExecutorPoolConfig, ...] = ()
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        """Validate config values used to build TaskExecutor Pods."""
        if self.env is not None:
            self.env = _taskexecutor_env(self.env)
        if self.volumes is not None:
            self.volumes = _taskexecutor_volumes(self.volumes)
        if self.volume_mounts is not None:
            self.volume_mounts = _taskexecutor_volume_mounts(self.volume_mounts)
        if self.warm_executor_pools and not self.warm_executor_owner:
            raise ValueError(
                "warm_executor_owner is required when warm_executor_pools are set."
            )
        if any(
            pool.key.task_type != TaskType.AGENT_APP
            for pool in self.warm_executor_pools
        ):
            raise ValueError("warm executor pools support only AgentApp tasks.")
        if self.warm_executor_owner and not _is_dns_label(self.warm_executor_owner):
            raise ValueError("warm_executor_owner must be a DNS label.")
        identities = [
            (pool.key.task_type, pool.key.fab_hash, pool.key.runtime_image)
            for pool in self.warm_executor_pools
        ]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "warm executor pools must not repeat a task type, FAB, and image."
            )
        warm_pod_count = sum(pool.size for pool in self.warm_executor_pools)
        if (
            self.active_pod_budget is not None
            and warm_pod_count >= self.active_pod_budget
        ):
            raise ValueError(
                "active_pod_budget must exceed the configured warm executor capacity."
            )


class _WarmExecutorUnavailable(RuntimeError):
    """Raised before a task token is sent to a warm executor Pod."""


class _KubernetesWarmExecutorDispatch:
    """Interact with one Kubernetes exec stream without logging task authority."""

    def __init__(self, response: object) -> None:
        self._response = response

    def send_token(self, token: str) -> None:
        """Send one token over stdin without retaining it in Pod metadata."""
        write_stdin = getattr(self._response, "write_stdin", None)
        if not callable(write_stdin):
            raise _WarmExecutorUnavailable(
                "Kubernetes exec stream does not support standard input."
            )
        write_stdin(f"{token}\n")

    def wait_for_acceptance(self, timeout: float) -> bool:
        """Return whether the task child acknowledged consuming the token."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _AGENTAPP_TOKEN_STDIN_ACKNOWLEDGEMENT in self._read_stdout():
                return True
            self._read_stderr()
            if not self._is_open():
                return False
            self._update(min(0.5, deadline - time.monotonic()))
        return False

    def wait_for_close(self) -> None:
        """Wait for the one-task child to exit or the exec stream to close."""
        while self._is_open():
            self._update(1.0)
            self._read_stdout()
            self._read_stderr()

    def close(self) -> None:
        """Close the Kubernetes exec stream best-effort."""
        close = getattr(self._response, "close", None)
        if callable(close):
            close()

    def _is_open(self) -> bool:
        is_open = getattr(self._response, "is_open", None)
        return bool(is_open()) if callable(is_open) else False

    def _update(self, timeout: float) -> None:
        update = getattr(self._response, "update", None)
        if callable(update):
            update(timeout=max(timeout, 0.0))

    def _read_stdout(self) -> str:
        peek_stdout = getattr(self._response, "peek_stdout", None)
        read_stdout = getattr(self._response, "read_stdout", None)
        if not callable(read_stdout) or (callable(peek_stdout) and not peek_stdout()):
            return ""
        stdout = read_stdout()
        return stdout if isinstance(stdout, str) else ""

    def _read_stderr(self) -> None:
        peek_stderr = getattr(self._response, "peek_stderr", None)
        read_stderr = getattr(self._response, "read_stderr", None)
        if callable(read_stderr) and (not callable(peek_stderr) or peek_stderr()):
            read_stderr()


class _WarmExecutorPoolManager:
    """Own and dispatch a fixed set of compatible, one-task warm Pods."""

    def __init__(
        self,
        client: KubernetesClient,
        config: KubernetesExecutorConfig,
        active_pod_count: Callable[[], int],
    ) -> None:
        self._client = client
        self._config = config
        self._active_pod_count = active_pod_count
        self._pools = {pool.key: pool for pool in config.warm_executor_pools}
        self._busy_pods: set[str] = set()
        self._closed = False
        self._lock = threading.Lock()
        self.ensure_capacity()

    # pylint: disable-next=too-many-return-statements
    def launch(
        self, spec: ExecutionSpec, runtime_root_certificates: str | None
    ) -> LaunchResult | None:
        """Dispatch a compatible task to a ready Pod or use the cold fallback."""
        pool = self._pool_for_spec(spec)
        if pool is None:
            return None

        with self._lock:
            if self._closed:
                return None
            try:
                pod_name = self._take_ready_pod(pool.key)
            except _WarmExecutorUnavailable:
                return None
            if pod_name is None:
                self._ensure_pool_capacity(pool, reserved_pod_capacity=1)
                return None

        try:
            dispatch = self._open_dispatch(
                pod_name=pod_name,
                spec=spec,
                runtime_root_certificates=runtime_root_certificates,
            )
        except _WarmExecutorUnavailable:
            self._retire_unavailable_pod(pod_name)
            return None

        try:
            dispatch.send_token(spec.token)
        except Exception:  # pylint: disable=broad-exception-caught
            self._retire_after_dispatch(pod_name, pool.key, dispatch)
            return LaunchResult.unknown(
                "Warm executor token delivery outcome was unknown."
            )

        try:
            accepted = dispatch.wait_for_acceptance(_WARM_EXECUTOR_ACK_TIMEOUT_SECONDS)
        except Exception:  # pylint: disable=broad-exception-caught
            self._retire_after_dispatch(pod_name, pool.key, dispatch)
            return LaunchResult.unknown(
                "Warm executor token acknowledgement outcome was unknown."
            )
        self._retire_after_dispatch(pod_name, pool.key, dispatch)
        if accepted:
            return LaunchResult.accepted()
        return LaunchResult.unknown(
            "Warm executor did not acknowledge task token delivery."
        )

    def ensure_capacity(self, reserved_pod_capacity: int = 0) -> None:
        """Create missing idle Pods for every configured compatible pool."""
        with self._lock:
            if self._closed:
                return
            self._reconcile_owned_pods()
            for pool in self._pools.values():
                self._ensure_pool_capacity(pool, reserved_pod_capacity)

    def has_ready_pod(self, task_type: TaskType, fab_hash: str | None) -> bool:
        """Return whether a matching warm Pod can take a task without new capacity."""
        pool = self._pool_for_task(task_type, fab_hash)
        if pool is None:
            return False
        with self._lock:
            if self._closed:
                return False
            pods = self._owned_warm_pods()
            if pods is None:
                return False
            return any(
                (_object_name(pod) is not None)
                and (_object_name(pod) not in self._busy_pods)
                and is_warm_executor_ready(pod, pool.key)
                for pod in pods
            )

    def close(self) -> None:
        """Delete all Pods owned by this SuperExec instance."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        pods = self._owned_warm_pods()
        if pods is None:
            return
        for pod in pods:
            pod_name = _object_name(pod)
            if pod_name is not None:
                self._delete_pod(pod_name)

    def _pool_for_spec(self, spec: ExecutionSpec) -> WarmExecutorPoolConfig | None:
        return self._pool_for_task(spec.task_type, spec.fab_hash)

    def _pool_for_task(
        self, task_type: TaskType, fab_hash: str | None
    ) -> WarmExecutorPoolConfig | None:
        for pool in self._pools.values():
            if (
                pool.key.task_type == task_type
                and pool.key.fab_hash == fab_hash
                and pool.key.runtime_image == self._config.image
            ):
                return pool
        return None

    def _take_ready_pod(self, key: WarmExecutorPoolKey) -> str | None:
        pods = self._owned_warm_pods()
        if pods is None:
            raise _WarmExecutorUnavailable("Warm executor Pods could not be listed.")
        for pod in pods:
            pod_name = _object_name(pod)
            if (
                pod_name is not None
                and pod_name not in self._busy_pods
                and is_warm_executor_ready(pod, key)
            ):
                self._busy_pods.add(pod_name)
                return pod_name
        return None

    def _ensure_pool_capacity(
        self, pool: WarmExecutorPoolConfig, reserved_pod_capacity: int = 0
    ) -> None:
        pods = self._owned_warm_pods()
        if pods is None:
            return
        compatible_count = sum(
            1 for pod in pods if _is_active_warm_executor(pod, pool.key)
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
        for _ in range(pods_to_create):
            try:
                pod = _build_warm_executor_pod(
                    pool.key, self._config, new_warm_executor_id()
                )
                self._client.create_namespaced_pod(self._config.namespace, pod)
            except Exception:  # pylint: disable=broad-exception-caught
                log(WARNING, "Failed to create a warm TaskExecutor Pod.", exc_info=True)
                return

    def _reconcile_owned_pods(self) -> None:
        """Delete owned Pods that are obsolete or exceed configured capacity."""
        pods = self._owned_warm_pods()
        if pods is None:
            return

        compatible_pods: dict[WarmExecutorPoolKey, list[object]] = {
            pool.key: [] for pool in self._pools.values()
        }
        for pod in pods:
            pod_name = _object_name(pod)
            pool = next(
                (
                    candidate
                    for candidate in self._pools.values()
                    if _is_active_warm_executor(pod, candidate.key)
                ),
                None,
            )
            if pool is None:
                if pod_name is not None and pod_name not in self._busy_pods:
                    self._delete_pod(pod_name)
                continue
            compatible_pods[pool.key].append(pod)

        for pool in self._pools.values():
            for pod in compatible_pods[pool.key][pool.size :]:
                pod_name = _object_name(pod)
                if pod_name is not None and pod_name not in self._busy_pods:
                    self._delete_pod(pod_name)

    def _owned_warm_pods(self) -> list[object] | None:
        try:
            pod_list = self._client.list_namespaced_pod(
                self._config.namespace,
                label_selector=_warm_executor_owner_label_selector(self._config),
            )
        except Exception:  # pylint: disable=broad-exception-caught
            log(WARNING, "Failed to list warm TaskExecutor Pods.", exc_info=True)
            return None
        return _pod_items(pod_list)

    def _open_dispatch(
        self,
        *,
        pod_name: str,
        spec: ExecutionSpec,
        runtime_root_certificates: str | None,
    ) -> _KubernetesWarmExecutorDispatch:
        try:
            stream = importlib.import_module("kubernetes.stream").stream
            response = stream(
                self._client.connect_get_namespaced_pod_exec,
                pod_name,
                self._config.namespace,
                command=_warm_taskexecutor_command(spec, runtime_root_certificates),
                stderr=True,
                stdin=True,
                stdout=True,
                tty=False,
                _preload_content=False,
            )
        except Exception as err:  # pylint: disable=broad-exception-caught
            raise _WarmExecutorUnavailable(
                "Warm TaskExecutor Pod is unavailable for dispatch."
            ) from err
        return _KubernetesWarmExecutorDispatch(response)

    def _retire_after_dispatch(
        self,
        pod_name: str,
        key: WarmExecutorPoolKey,
        dispatch: _KubernetesWarmExecutorDispatch,
    ) -> None:
        threading.Thread(
            target=self._wait_for_task_and_replace,
            args=(pod_name, key, dispatch),
            daemon=True,
        ).start()

    def _wait_for_task_and_replace(
        self,
        pod_name: str,
        key: WarmExecutorPoolKey,
        dispatch: _KubernetesWarmExecutorDispatch,
    ) -> None:
        try:
            dispatch.wait_for_close()
        finally:
            dispatch.close()
            if self._delete_pod(pod_name):
                with self._lock:
                    self._busy_pods.discard(pod_name)
                    if not self._closed:
                        self._ensure_pool_capacity(self._pools[key])

    def _release_pod(self, pod_name: str) -> None:
        with self._lock:
            self._busy_pods.discard(pod_name)

    def _retire_unavailable_pod(self, pod_name: str) -> None:
        """Delete a Pod that failed before task authority was delivered."""
        if self._delete_pod(pod_name):
            self._release_pod(pod_name)

    def _delete_pod(self, pod_name: str) -> bool:
        try:
            self._client.delete_namespaced_pod(
                name=pod_name,
                namespace=self._config.namespace,
                grace_period_seconds=0,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            log(WARNING, "Failed to delete warm TaskExecutor Pod %s.", pod_name)
            return False
        return True


class KubernetesExecutor:
    """Submit TaskExecutor Pods to Kubernetes."""

    def __init__(
        self,
        *,
        client: KubernetesClient,
        config: KubernetesExecutorConfig,
    ) -> None:
        self._client = client
        self._config = config
        self._completed_pod_sweeper = CompletedPodSweeper(client=client, config=config)
        self._last_completed_pod_sweep_at: float | None = None
        self._warm_executor_pool_manager = (
            _WarmExecutorPoolManager(client, config, self._active_pod_count)
            if config.warm_executor_pools
            else None
        )

    def wait_for_capacity(
        self,
        task_type: TaskType | None = None,
        fab_hash: str | None = None,
        *,
        insecure: bool = False,
        root_certificates_path: str | None = None,
    ) -> None:
        """Wait until the configured resource pool is below its active Pod budget."""
        self._wait_for_capacity(
            task_type,
            fab_hash,
            allow_warm_dispatch=self._can_dispatch_warm(
                insecure, root_certificates_path
            ),
            reconcile_warm_pools=True,
        )

    def _wait_for_capacity(
        self,
        task_type: TaskType | None,
        fab_hash: str | None,
        *,
        allow_warm_dispatch: bool,
        reconcile_warm_pools: bool,
    ) -> None:
        """Wait for cold capacity, or allow an already-ready warm dispatch."""
        self._sweep_completed_pods_if_due()
        if reconcile_warm_pools and self._warm_executor_pool_manager is not None:
            has_ready_warm_pod = (
                allow_warm_dispatch
                and task_type is not None
                and self._warm_executor_pool_manager.has_ready_pod(task_type, fab_hash)
            )
            self._warm_executor_pool_manager.ensure_capacity(
                reserved_pod_capacity=0 if has_ready_warm_pod else 1
            )
            if has_ready_warm_pod:
                return
        if self._config.active_pod_budget is None:
            return

        last_log_at: float | None = None
        waited_for_capacity = False
        while True:
            if (
                allow_warm_dispatch
                and self._warm_executor_pool_manager is not None
                and task_type is not None
                and self._warm_executor_pool_manager.has_ready_pod(task_type, fab_hash)
            ):
                return
            try:
                active_pod_count = self._active_pod_count()
            except Exception:  # pylint: disable=broad-exception-caught
                log(
                    WARNING,
                    "Kubernetes capacity check failed; proceeding without waiting. "
                    "selector=%s",
                    _capacity_label_selector(self._config),
                    exc_info=True,
                )
                return
            if active_pod_count < self._config.active_pod_budget:
                if waited_for_capacity:
                    self._last_completed_pod_sweep_at = self._config.monotonic()
                    self._sweep_completed_pods()
                return

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
                        _capacity_label_selector(self._config),
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
            < _COMPLETED_POD_SWEEP_INTERVAL_SECONDS
        ):
            return

        self._last_completed_pod_sweep_at = now
        self._sweep_completed_pods()

    def _sweep_completed_pods(self) -> None:
        """Run best-effort completed Pod cleanup."""
        try:
            self._completed_pod_sweeper.sweep()
        except Exception:  # pylint: disable=broad-exception-caught
            log(
                WARNING,
                "Kubernetes completed Pod sweep failed; proceeding. selector=%s",
                _taskexecutor_pool_label_selector(self._config),
                exc_info=True,
            )

    def launch(self, spec: ExecutionSpec) -> LaunchResult:
        """Submit the TaskExecutor Pod and credential Secret."""
        try:
            runtime_root_certificates = _get_runtime_root_certificates(
                spec, self._config
            )
            if self._warm_executor_pool_manager is not None:
                warm_result = self._warm_executor_pool_manager.launch(
                    spec, runtime_root_certificates
                )
                if warm_result is not None:
                    return warm_result
                self._wait_for_capacity(
                    None,
                    None,
                    allow_warm_dispatch=False,
                    reconcile_warm_pools=False,
                )
            launch_attempt_id = _new_launch_attempt_id()
            secret_name = _credential_secret_name(spec, launch_attempt_id)
            secret = _build_appio_credentials_secret(
                spec, self._config, runtime_root_certificates, launch_attempt_id
            )
            pod = _build_taskexecutor_pod(
                spec, self._config, runtime_root_certificates, launch_attempt_id
            )
            self._client.create_namespaced_secret(self._config.namespace, secret)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return _launch_result_from_exception(exc)

        try:
            self._client.create_namespaced_pod(self._config.namespace, pod)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result = _launch_result_from_exception(exc)
            if _is_definite_pod_rejection(exc):
                _delete_secret_best_effort(
                    self._client, self._config.namespace, secret_name
                )
            return result

        return LaunchResult.accepted()

    def close(self) -> None:
        """Delete idle warm Pods owned by this SuperExec instance."""
        if self._warm_executor_pool_manager is not None:
            self._warm_executor_pool_manager.close()

    def _can_dispatch_warm(
        self, insecure: bool, root_certificates_path: str | None
    ) -> bool:
        """Return whether warm dispatch can use the task's Runtime transport."""
        return insecure or (
            self._config.runtime_root_certificates is None
            and root_certificates_path is None
        )

    def _launch_warm_executor(self, pool_key: WarmExecutorPoolKey) -> LaunchResult:
        """Submit one warm TaskExecutor Pod for a fixed compatibility key."""
        try:
            pod = _build_warm_executor_pod(
                pool_key, self._config, new_warm_executor_id()
            )
            self._client.create_namespaced_pod(self._config.namespace, pod)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return _launch_result_from_exception(exc)
        return LaunchResult.accepted()

    def _active_pod_count(self) -> int:
        """Return the active TaskExecutor Pod count for the configured pool."""
        pod_list = self._client.list_namespaced_pod(
            self._config.namespace,
            label_selector=_capacity_label_selector(self._config),
        )
        return sum(1 for pod in _pod_items(pod_list) if _is_active_pod(pod))


class CompletedPodSweeper:
    """Delete terminal TaskExecutor Pods and orphaned credential Secrets."""

    def __init__(
        self,
        *,
        client: KubernetesClient,
        config: KubernetesExecutorConfig,
    ) -> None:
        self._client = client
        self._config = config

    def sweep(self) -> None:
        """Delete terminal Pods and orphaned credential Secrets."""
        selector = _taskexecutor_pool_label_selector(self._config)
        pods = _pod_items(
            self._client.list_namespaced_pod(
                self._config.namespace, label_selector=selector
            )
        )
        secrets = _secret_items(
            self._client.list_namespaced_secret(
                self._config.namespace, label_selector=selector
            )
        )
        pod_names = {name for pod in pods if (name := _object_name(pod)) is not None}
        task_secret_names = {
            name
            for secret in secrets
            if (name := _object_name(secret)) is not None and _has_task_id_label(secret)
        }

        for pod in pods:
            pod_name = _object_name(pod)
            if (
                pod_name is None
                or not (_has_task_id_label(pod) or is_warm_executor(pod))
                or not _is_terminal_pod(pod)
            ):
                continue
            self._delete_pod(pod_name)
            credential_secret_name = _credential_secret_name_from_pod_name(pod_name)
            if credential_secret_name in task_secret_names:
                self._delete_secret(credential_secret_name)

        # Delete credential Secrets whose owner Pod is no longer listed.
        for secret in secrets:
            secret_name = _object_name(secret)
            if secret_name is None or not _has_task_id_label(secret):
                continue
            pod_name = _pod_name_from_credential_secret_name(secret_name)
            if pod_name is None or pod_name in pod_names:
                continue
            self._delete_secret(secret_name)

    def _delete_pod(self, name: str) -> None:
        """Delete a Pod, tolerating already-deleted objects."""
        try:
            self._client.delete_namespaced_pod(
                name=name,
                namespace=self._config.namespace,
                grace_period_seconds=0,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _raise_unless_not_found(exc)

    def _delete_secret(self, name: str) -> None:
        """Delete a Secret, tolerating already-deleted objects."""
        try:
            self._client.delete_namespaced_secret(
                name=name, namespace=self._config.namespace
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _raise_unless_not_found(exc)


def _build_appio_credentials_secret(
    spec: ExecutionSpec,
    config: KubernetesExecutorConfig,
    runtime_root_certificates: str | None,
    launch_attempt_id: str,
) -> JSONObject:
    """Build the AppIo credential Secret for a TaskExecutor Pod."""
    data: JSONObject = {"token": spec.token}
    if runtime_root_certificates is not None:
        data["ca.crt"] = runtime_root_certificates

    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": _metadata(
            _credential_secret_name(spec, launch_attempt_id),
            spec,
            config,
            launch_attempt_id,
        ),
        "type": "Opaque",
        "stringData": data,
    }


def _build_taskexecutor_pod(
    spec: ExecutionSpec,
    config: KubernetesExecutorConfig,
    runtime_root_certificates: str | None,
    launch_attempt_id: str,
) -> JSONObject:
    """Build the TaskExecutor Pod for a claimed SuperExec task."""
    volume_mounts: list[JSONObject] = [
        {
            "name": "appio-credentials",
            "mountPath": APPIO_CREDENTIALS_MOUNT_PATH,
            "readOnly": True,
        }
    ]
    if config.volume_mounts is not None:
        volume_mounts.extend(config.volume_mounts)

    container: JSONObject = {
        "name": "taskexecutor",
        "image": config.image,
        "command": [TASK_TYPE_TO_COMMAND[spec.task_type]],
        "args": _taskexecutor_args(spec, runtime_root_certificates),
        "volumeMounts": volume_mounts,
    }
    _apply_taskexecutor_container_config(container, config)

    volumes: list[JSONObject] = [
        {
            "name": "appio-credentials",
            "secret": {
                "secretName": _credential_secret_name(spec, launch_attempt_id),
                "defaultMode": 0o444,
            },
        }
    ]
    if config.volumes is not None:
        volumes.extend(config.volumes)

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": _metadata(
            _pod_name(spec, launch_attempt_id), spec, config, launch_attempt_id
        ),
        "spec": _taskexecutor_pod_spec(container, volumes, config),
    }


def _build_warm_executor_pod(
    pool_key: WarmExecutorPoolKey,
    config: KubernetesExecutorConfig,
    executor_id: str,
) -> JSONObject:
    """Build a warm TaskExecutor Pod without task authority or credentials."""
    container: JSONObject = {
        "name": "taskexecutor",
        "image": pool_key.runtime_image,
        "command": ["python", "-m", WARM_EXECUTOR_MODULE],
        "volumeMounts": [
            {
                "name": _WARM_EXECUTOR_READY_VOLUME_NAME,
                "mountPath": WARM_EXECUTOR_READY_DIRECTORY,
            },
            *(config.volume_mounts or []),
        ],
        "readinessProbe": {
            "exec": {"command": list(WARM_EXECUTOR_READINESS_COMMAND)},
            "periodSeconds": 1,
        },
    }
    _apply_taskexecutor_container_config(container, config)

    volumes: list[JSONObject] = [
        {"name": _WARM_EXECUTOR_READY_VOLUME_NAME, "emptyDir": {}},
        *(config.volumes or []),
    ]
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": _warm_executor_metadata(
            _warm_executor_pod_name(executor_id), pool_key, config
        ),
        "spec": _taskexecutor_pod_spec(container, volumes, config),
    }


def _apply_taskexecutor_container_config(
    container: JSONObject, config: KubernetesExecutorConfig
) -> None:
    """Apply shared optional configuration to a TaskExecutor container."""
    if config.image_pull_policy is not None:
        container["imagePullPolicy"] = config.image_pull_policy
    if config.resources is not None:
        container["resources"] = config.resources
    if config.env is not None:
        container["env"] = config.env
    if config.container_security_context is not None:
        container["securityContext"] = config.container_security_context


def _taskexecutor_pod_spec(
    container: JSONObject,
    volumes: list[JSONObject],
    config: KubernetesExecutorConfig,
) -> JSONObject:
    """Build the shared TaskExecutor Pod lifecycle and placement fields."""
    pod_spec: JSONObject = {
        "automountServiceAccountToken": False,
        "restartPolicy": "Never",
        "containers": [container],
    }
    if volumes:
        pod_spec["volumes"] = volumes
    if config.service_account_name is not None:
        pod_spec["serviceAccountName"] = config.service_account_name
    if config.node_selector is not None:
        pod_spec["nodeSelector"] = cast(JSONObject, config.node_selector)
    if config.tolerations is not None:
        pod_spec["tolerations"] = config.tolerations
    if config.affinity is not None:
        pod_spec["affinity"] = config.affinity
    if config.priority_class_name is not None:
        pod_spec["priorityClassName"] = config.priority_class_name
    if config.pod_security_context is not None:
        pod_spec["securityContext"] = config.pod_security_context

    return pod_spec


def _taskexecutor_args(
    spec: ExecutionSpec, runtime_root_certificates: str | None
) -> list[str]:
    """Build TaskExecutor arguments with file-based credential delivery."""
    args = [
        TASK_TYPE_TO_APPIO_API_ADDRESS_ARG[spec.task_type],
        spec.runtime_api_address,
        "--token-file",
        APPIO_TOKEN_FILE_PATH,
    ]

    if spec.insecure:
        args.append("--insecure")
    elif runtime_root_certificates is not None:
        args.extend(["--root-certificates", APPIO_ROOT_CERTIFICATES_FILE_PATH])

    if spec.runtime_dependency_install:
        args.append("--allow-runtime-dependency-installation")

    return args


def _warm_taskexecutor_command(
    spec: ExecutionSpec, runtime_root_certificates: str | None
) -> list[str]:
    """Build a one-task child command that receives authority on standard input."""
    command = [
        TASK_TYPE_TO_COMMAND[spec.task_type],
        TASK_TYPE_TO_APPIO_API_ADDRESS_ARG[spec.task_type],
        spec.runtime_api_address,
        "--token-stdin",
    ]
    if spec.insecure:
        command.append("--insecure")
    elif runtime_root_certificates is not None:
        raise _WarmExecutorUnavailable(
            "Warm executor dispatch cannot safely deliver Runtime API certificates."
        )
    if spec.runtime_dependency_install:
        command.append("--allow-runtime-dependency-installation")
    return command


def _taskexecutor_env(env: list[JSONObject]) -> list[JSONObject]:
    """Build validated TaskExecutor container environment entries."""
    if not isinstance(env, list):
        raise ValueError("TaskExecutor env must be a list of mappings.")
    entries: list[JSONObject] = []
    for entry in env:
        if not isinstance(entry, dict):
            raise ValueError("TaskExecutor env entries must be mappings.")
        # Keep this path limited to non-secret literal config. Design secret
        # references separately before allowing them into TaskExecutor Pods.
        if "valueFrom" in entry:
            raise ValueError(
                "TaskExecutor env entries support literal 'value' only; "
                "'valueFrom' is not supported."
            )
        if set(entry) != {"name", "value"}:
            raise ValueError(
                "TaskExecutor env entries must contain exactly 'name' and 'value'."
            )
        name = entry["name"]
        value = entry["value"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("TaskExecutor env names must be non-empty strings.")
        # Validate env name locally so invalid executor config fails before Pod creation
        if not _KUBERNETES_ENV_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"TaskExecutor env name {name!r} must be a valid Kubernetes "
                "environment variable name."
            )
        if not isinstance(value, str):
            raise ValueError(f"TaskExecutor env value for {name!r} must be a string.")
        # Reject task-visible provider API key env names before Pod construction
        if name in _FORBIDDEN_TASKEXECUTOR_ENV_NAMES:
            raise ValueError(
                f"TaskExecutor env name {name!r} is not allowed because it is a "
                "provider API key."
            )
        # Copy only the validated Kubernetes env shape into the generated Pod spec.
        entries.append({"name": name, "value": value})
    return entries


def _taskexecutor_volumes(volumes: list[JSONObject]) -> list[JSONObject]:
    """Build validated TaskExecutor Pod volumes."""
    if not isinstance(volumes, list):
        raise ValueError("TaskExecutor volumes must be a list of mappings.")
    entries: list[JSONObject] = []
    for entry in volumes:
        if not isinstance(entry, dict):
            raise ValueError("TaskExecutor volume entries must be mappings.")
        volume_name = entry.get("name")
        if volume_name in _RESERVED_TASKEXECUTOR_VOLUME_NAMES:
            raise ValueError(f"TaskExecutor volume name {volume_name!r} is reserved.")
        if "secret" in entry:
            raise ValueError("TaskExecutor secret volumes are not supported.")
        if _has_rejected_projected_source(entry):
            raise ValueError(
                "TaskExecutor projected secret and serviceAccountToken volumes "
                "are not supported."
            )
        entries.append(entry)
    return entries


def _taskexecutor_volume_mounts(volume_mounts: list[JSONObject]) -> list[JSONObject]:
    """Build validated TaskExecutor container volume mounts."""
    if not isinstance(volume_mounts, list):
        raise ValueError("TaskExecutor volume mounts must be a list of mappings.")
    entries: list[JSONObject] = []
    for entry in volume_mounts:
        if not isinstance(entry, dict):
            raise ValueError("TaskExecutor volume mount entries must be mappings.")
        volume_name = entry.get("name")
        if volume_name in _RESERVED_TASKEXECUTOR_VOLUME_NAMES:
            raise ValueError(
                f"TaskExecutor volume mount name {volume_name!r} is reserved."
            )
        mount_path = entry.get("mountPath")
        if mount_path in _RESERVED_TASKEXECUTOR_VOLUME_MOUNT_PATHS:
            raise ValueError(
                f"TaskExecutor volume mount path {mount_path!r} is reserved."
            )
        entries.append(entry)
    return entries


def _has_rejected_projected_source(volume: JSONObject) -> bool:
    """Return true if a projected volume source exposes credentials."""
    projected = volume.get("projected")
    if not isinstance(projected, dict):
        return False
    sources = projected.get("sources")
    if not isinstance(sources, list):
        return False
    return any(
        isinstance(source, dict)
        and ("secret" in source or "serviceAccountToken" in source)
        for source in sources
    )


def _get_runtime_root_certificates(
    spec: ExecutionSpec, config: KubernetesExecutorConfig
) -> str | None:
    """Return PEM data for Runtime API root certificates, if configured."""
    if config.runtime_root_certificates is not None:
        return config.runtime_root_certificates
    if spec.root_certificates_path is not None:
        return (
            Path(spec.root_certificates_path).expanduser().read_text(encoding="utf-8")
        )
    return None


def _new_launch_attempt_id() -> str:
    """Return a DNS-label-safe opaque identifier for one local launch call."""
    return uuid4().hex[:12]


def _pod_name(spec: ExecutionSpec, launch_attempt_id: str) -> str:
    """Return the TaskExecutor Pod name."""
    return f"flwr-taskexecutor-{spec.task_id}-{launch_attempt_id}"


def _warm_executor_pod_name(executor_id: str) -> str:
    """Return the name of a warm TaskExecutor Pod."""
    return f"flwr-taskexecutor-warm-{executor_id}"


def _credential_secret_name(spec: ExecutionSpec, launch_attempt_id: str) -> str:
    """Return the AppIo credential Secret name."""
    return _credential_secret_name_from_pod_name(_pod_name(spec, launch_attempt_id))


def _credential_secret_name_from_pod_name(pod_name: str) -> str:
    """Return the AppIo credential Secret name for a TaskExecutor Pod name."""
    return f"{pod_name}{_APPIO_CREDENTIAL_SECRET_SUFFIX}"


def _pod_name_from_credential_secret_name(secret_name: str) -> str | None:
    """Return the owner Pod name encoded in a credential Secret name."""
    if not secret_name.endswith(_APPIO_CREDENTIAL_SECRET_SUFFIX):
        return None
    pod_name = secret_name[: -len(_APPIO_CREDENTIAL_SECRET_SUFFIX)]
    if not pod_name:
        return None
    return pod_name


def _metadata(
    name: str,
    spec: ExecutionSpec,
    config: KubernetesExecutorConfig,
    launch_attempt_id: str,
) -> JSONObject:
    """Return Kubernetes object metadata."""
    metadata: JSONObject = {
        "name": name,
        "namespace": config.namespace,
        "labels": _labels(spec, config, launch_attempt_id),
    }
    if config.annotations is not None:
        metadata["annotations"] = cast(JSONObject, config.annotations)
    return metadata


def _warm_executor_metadata(
    name: str,
    pool_key: WarmExecutorPoolKey,
    config: KubernetesExecutorConfig,
) -> JSONObject:
    """Return metadata identifying one compatible warm TaskExecutor Pod."""
    labels: JSONObject = {}
    labels.update(_caller_labels(config))
    labels.update(
        {
            _NAME_LABEL: "flower",
            _COMPONENT_LABEL: "taskexecutor",
            _TASK_TYPE_LABEL: pool_key.task_type.value,
            WARM_EXECUTOR_LABEL: "true",
        }
    )
    if config.resource_pool is not None:
        labels[_RESOURCE_POOL_LABEL] = config.resource_pool
    if config.warm_executor_owner is not None:
        labels[_WARM_EXECUTOR_OWNER_LABEL] = config.warm_executor_owner

    annotations: JSONObject = {}
    annotations.update(config.annotations or {})
    annotations.update(
        {
            WARM_EXECUTOR_RUNTIME_IMAGE_ANNOTATION: pool_key.runtime_image,
            WARM_EXECUTOR_DEPENDENCY_ENVIRONMENT_ANNOTATION: (
                pool_key.dependency_environment_version
            ),
        }
    )
    if pool_key.fab_hash is None:
        annotations.pop(WARM_EXECUTOR_FAB_HASH_ANNOTATION, None)
    else:
        annotations[WARM_EXECUTOR_FAB_HASH_ANNOTATION] = pool_key.fab_hash
    return {
        "name": name,
        "namespace": config.namespace,
        "labels": labels,
        "annotations": annotations,
    }


def _labels(
    spec: ExecutionSpec, config: KubernetesExecutorConfig, launch_attempt_id: str
) -> JSONObject:
    """Return stable labels for Kubernetes objects."""
    labels: JSONObject = {}
    labels.update(_caller_labels(config))
    # Apply executor-owned labels last; selectors and cleanup rely on them.
    labels.update(
        {
            _NAME_LABEL: "flower",
            _COMPONENT_LABEL: "taskexecutor",
            _TASK_ID_LABEL: str(spec.task_id),
            _TASK_TYPE_LABEL: spec.task_type.value,
            LAUNCH_ATTEMPT_LABEL: launch_attempt_id,
        }
    )
    if config.resource_pool is not None:
        labels[_RESOURCE_POOL_LABEL] = config.resource_pool
    return labels


def _capacity_label_selector(config: KubernetesExecutorConfig) -> str:
    """Return the label selector used for resource-pool capacity checks."""
    return _taskexecutor_pool_label_selector(config)


def _taskexecutor_pool_label_selector(config: KubernetesExecutorConfig) -> str:
    """Return the label selector for TaskExecutor pool-scoped operations."""
    return _label_selector(_taskexecutor_pool_labels(config))


def _taskexecutor_pool_labels(config: KubernetesExecutorConfig) -> dict[str, str]:
    """Return labels identifying a scoped TaskExecutor pool."""
    labels = _caller_labels(config)
    labels.update(
        {
            _NAME_LABEL: "flower",
            _COMPONENT_LABEL: "taskexecutor",
        }
    )
    if config.resource_pool is not None:
        labels[_RESOURCE_POOL_LABEL] = config.resource_pool
    return labels


def _warm_executor_owner_label_selector(config: KubernetesExecutorConfig) -> str:
    """Return a selector limited to warm Pods owned by this SuperExec instance."""
    assert config.warm_executor_owner is not None
    labels = {
        _NAME_LABEL: "flower",
        _COMPONENT_LABEL: "taskexecutor",
        WARM_EXECUTOR_LABEL: "true",
        _WARM_EXECUTOR_OWNER_LABEL: config.warm_executor_owner,
    }
    return _label_selector(labels)


def _caller_labels(config: KubernetesExecutorConfig) -> dict[str, str]:
    """Return caller-provided labels that are not owned by the executor."""
    return {
        key: value
        for key, value in (config.labels or {}).items()
        if key not in _EXECUTOR_OWNED_LABELS
    }


def _label_selector(labels: dict[str, str]) -> str:
    """Return a Kubernetes equality label selector."""
    return ",".join(f"{key}={value}" for key, value in sorted(labels.items()))


def _pod_items(pod_list: KubernetesList | Mapping[str, object]) -> list[object]:
    """Return Pod items from a Kubernetes list response."""
    items = _object_field(pod_list, "items")
    if isinstance(items, Sequence) and not isinstance(items, str):
        return list(items)
    return []


def _secret_items(secret_list: KubernetesList | Mapping[str, object]) -> list[object]:
    """Return Secret items from a Kubernetes list response."""
    items = _object_field(secret_list, "items")
    if isinstance(items, Sequence) and not isinstance(items, str):
        return list(items)
    return []


def _is_active_pod(pod: object) -> bool:
    """Return true if a Pod counts against best-effort launch capacity."""
    status = _object_field(pod, "status")
    if _object_field(status, "phase") in {"Succeeded", "Failed"}:
        return False

    metadata = _object_field(pod, "metadata")
    deletion_timestamp = _object_field(metadata, "deletion_timestamp")
    if deletion_timestamp is None:
        deletion_timestamp = _object_field(metadata, "deletionTimestamp")
    if deletion_timestamp is not None:
        return True

    return _object_field(status, "phase") not in {"Succeeded", "Failed"}


def _is_active_warm_executor(pod: object, pool_key: WarmExecutorPoolKey) -> bool:
    """Return true when a compatible warm Pod still occupies its pool slot."""
    if not is_compatible_warm_executor(pod, pool_key):
        return False
    metadata = _object_field(pod, "metadata")
    deletion_timestamp = _object_field(metadata, "deletion_timestamp")
    if deletion_timestamp is None:
        deletion_timestamp = _object_field(metadata, "deletionTimestamp")
    if deletion_timestamp is not None:
        return False
    return not _is_terminal_pod(pod)


def _is_terminal_pod(pod: object) -> bool:
    """Return true if a Pod reached a terminal phase."""
    status = _object_field(pod, "status")
    return _object_field(status, "phase") in {"Succeeded", "Failed"}


def _object_name(value: object) -> str | None:
    """Return an object's metadata name."""
    metadata = _object_field(value, "metadata")
    name = _object_field(metadata, "name")
    if isinstance(name, str) and name.strip():
        return name
    return None


def _has_task_id_label(value: object) -> bool:
    """Return true if an object carries a TaskExecutor task-id label."""
    metadata = _object_field(value, "metadata")
    labels = _object_field(metadata, "labels")
    task_id = _object_field(labels, _TASK_ID_LABEL)
    return isinstance(task_id, str) and bool(task_id.strip())


def _object_field(value: object, field_name: str) -> object | None:
    """Return a field from a Kubernetes dict or model object."""
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _launch_result_from_exception(exc: Exception) -> LaunchResult:
    """Map immediate Kubernetes API exceptions to launch results."""
    message = f"{type(exc).__name__}: {exc}"
    status = _exception_status(exc)
    lower_message = message.lower()

    if isinstance(exc, (ConnectionError, TimeoutError)):
        return LaunchResult.unknown(message)

    if status == 429 or _is_capacity_message(lower_message):
        return LaunchResult.capacity_rejected(message)

    if status is not None and (status == 408 or status >= 500):
        return LaunchResult.unknown(message)

    return LaunchResult.failed(message)


def _is_definite_pod_rejection(exc: Exception) -> bool:
    """Return true when Pod submission definitely failed before acceptance."""
    status = _exception_status(exc)
    if status is None:
        return False

    # Cleanup only relies on the Kubernetes API status contract. Message matching
    # remains useful for LaunchResult mapping, but it is too heuristic to decide
    # whether deleting the just-created Secret is safe.
    return 400 <= status < 500 and status != 408


def _delete_secret_best_effort(
    client: KubernetesClient, namespace: str, secret_name: str
) -> None:
    """Best-effort cleanup for a Secret whose Pod was definitely rejected."""
    try:
        client.delete_namespaced_secret(secret_name, namespace)
    except Exception:  # pylint: disable=broad-exception-caught
        log(
            WARNING,
            "Failed to delete Kubernetes credential Secret %r in namespace %r",
            secret_name,
            namespace,
            exc_info=True,
        )


def _exception_status(exc: Exception) -> int | None:
    """Return an HTTP-like status from Kubernetes client exceptions."""
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    if isinstance(status, str) and status.isdigit():
        return int(status)
    return None


def _raise_unless_not_found(exc: Exception) -> None:
    """Raise Kubernetes client exceptions except already-deleted objects."""
    if _exception_status(exc) == 404:
        return
    raise exc


def _is_capacity_message(message: str) -> bool:
    """Return true for quota/admission capacity rejection messages."""
    # Status codes are preferred above; this intentionally brittle fallback covers
    # common Kubernetes quota/scheduler wording when clients only expose messages.
    capacity_markers = (
        "exceeded quota",
        "resourcequota",
        "quota exceeded",
        "too many requests",
        "rate limit",
        "insufficient cpu",
        "insufficient memory",
        "insufficient pods",
    )
    return any(marker in message for marker in capacity_markers)


def _is_dns_label(value: str) -> bool:
    """Return whether value can be used as a Kubernetes label value here."""
    return len(value) <= 63 and _DNS_LABEL_PATTERN.fullmatch(value) is not None
