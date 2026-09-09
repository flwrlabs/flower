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
"""Kubernetes dispatch and pool lifecycle for one-task warm AgentApp executors."""

from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from logging import WARNING
from typing import TYPE_CHECKING

from flwr.supercore import log
from flwr.supercore.constant import (
    TASK_TYPE_TO_APPIO_API_ADDRESS_ARG,
    TASK_TYPE_TO_COMMAND,
    TaskType,
)
from flwr.supercore.typing import JSONObject

from .types import ExecutionSpec, LaunchResult
from .warm_executor_pool import (
    WarmExecutorPoolConfig,
    WarmExecutorPoolKey,
    is_warm_executor_ready,
    new_warm_executor_id,
)

if TYPE_CHECKING:
    from .kubernetes_executor import KubernetesClient, KubernetesExecutorConfig

_TOKEN_STDIN_ACKNOWLEDGEMENT = "FLWR_AGENTAPP_TOKEN_ACCEPTED"
WARM_EXECUTOR_CONSUMED_ANNOTATION = "flower.ai/warm-executor-consumed"
_WARM_EXECUTOR_ACK_TIMEOUT_SECONDS = 5.0
# A surviving consumed Pod is safe to retire only after all task processes exit.
# Ignore PID 1 (the idle parent), this probe, and zombies. A concurrent readiness
# probe can delay retirement, but cannot make a running task appear finished.
_WARM_EXECUTOR_IDLE_CHECK = """\
import os
from pathlib import Path
for process in Path('/proc').iterdir():
    if not process.name.isdigit() or int(process.name) in (1, os.getpid()):
        continue
    try:
        state = (process / 'stat').read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        break
    if state != 'Z':
        break
else:
    print('FLWR_WARM_EXECUTOR_IDLE')
"""


class WarmAgentAppUnavailable(RuntimeError):
    """Raised before a task token is sent to a warm AgentApp executor Pod."""


class KubernetesWarmAgentAppDispatch:
    """Interact with one Kubernetes exec stream without logging task authority."""

    def __init__(self, response: object) -> None:
        self._response = response

    def send_token(self, token: str) -> None:
        """Send one token over stdin without retaining it in Pod metadata."""
        write_stdin = getattr(self._response, "write_stdin", None)
        if not callable(write_stdin):
            raise WarmAgentAppUnavailable(
                "Kubernetes exec stream does not support standard input."
            )
        write_stdin(f"{token}\n")

    def wait_for_acceptance(self, timeout: float) -> bool:
        """Return whether the task child acknowledged consuming the token."""
        deadline = time.monotonic() + timeout
        stdout = ""
        while time.monotonic() < deadline:
            stdout += self._read_stdout()
            self._read_stderr()
            self._discard_combined_output()
            if _TOKEN_STDIN_ACKNOWLEDGEMENT in stdout:
                return True
            stdout = stdout[-len(_TOKEN_STDIN_ACKNOWLEDGEMENT) :]
            if not self._is_open():
                return False
            self._update(min(0.5, deadline - time.monotonic()))
        return False

    def wait_for_close(self) -> bool:
        """Return whether stream closure confirms that the child exited."""
        while self._is_open():
            self._update(1.0)
            self._read_stdout()
            self._read_stderr()
            self._discard_combined_output()
        # A disconnected socket is not proof of process exit. Kubernetes sends
        # the exit status on a separate channel, which must not be discarded.
        return isinstance(getattr(self._response, "returncode", None), int)

    def close(self) -> None:
        """Close the Kubernetes exec stream best-effort."""
        close = getattr(self._response, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pylint: disable=broad-exception-caught
                log(WARNING, "Failed to close warm TaskExecutor exec stream.")

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

    def _discard_combined_output(self) -> None:
        # WSClient.read_all() also clears unread channels, including the exit
        # status and stdout received while peeking stderr. Its capture_all option
        # is not accepted by CoreV1Api, so clear only the combined buffer.
        combined_output = getattr(self._response, "_all", None)
        if combined_output is not None:
            combined_output.seek(0)
            combined_output.truncate()


def warm_agentapp_command(
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
        raise WarmAgentAppUnavailable(
            "Warm executor dispatch cannot safely deliver Runtime API certificates."
        )
    if spec.runtime_dependency_install:
        command.append("--allow-runtime-dependency-installation")
    return command


class WarmAgentAppPoolManager:  # pylint: disable=too-many-instance-attributes,too-many-arguments
    """Own and dispatch a fixed set of compatible, one-task warm AgentApp Pods."""

    def __init__(
        self,
        client: KubernetesClient,
        config: KubernetesExecutorConfig,
        active_pod_count: Callable[[], int],
        exec_client: KubernetesClient | None = None,
        *,
        build_warm_executor_pod: Callable[
            [WarmExecutorPoolKey, KubernetesExecutorConfig, str], JSONObject
        ],
        has_warm_executor_configuration: Callable[
            [object, KubernetesExecutorConfig], bool
        ],
        is_active_warm_executor: Callable[
            [object, WarmExecutorPoolKey, KubernetesExecutorConfig], bool
        ],
        warm_executor_owner_label_selector: Callable[[KubernetesExecutorConfig], str],
    ) -> None:
        self._client = client
        self._exec_client = exec_client or client
        self._config = config
        self._active_pod_count = active_pod_count
        self._build_warm_executor_pod = build_warm_executor_pod
        self._has_warm_executor_configuration = has_warm_executor_configuration
        self._is_active_warm_executor = is_active_warm_executor
        self._warm_executor_owner_label_selector = warm_executor_owner_label_selector
        self._pools = {pool.key: pool for pool in config.warm_executor_pools}
        self._busy_pods: set[str] = set()
        self._retiring_pods: set[str] = set()
        self._closed = False
        self._lock = threading.RLock()

    # pylint: disable-next=too-many-return-statements
    def launch(
        self, spec: ExecutionSpec, runtime_root_certificates: str | None
    ) -> LaunchResult | None:
        """Dispatch a compatible task to a ready Pod or use the cold fallback."""
        pool = self._pool_for_task(spec.task_type)
        if pool is None:
            return None

        with self._lock:
            if self._closed:
                return None
            try:
                pod_name = self._take_ready_pod(pool.key)
            except WarmAgentAppUnavailable:
                return None
            if pod_name is None:
                self._ensure_pool_capacity(pool, reserved_pod_capacity=1)
                return None
            self._ensure_pool_capacity(pool)

        try:
            dispatch = self._open_dispatch(
                pod_name=pod_name,
                spec=spec,
                runtime_root_certificates=runtime_root_certificates,
            )
        except WarmAgentAppUnavailable:
            self._retire_unavailable_pod(pod_name)
            return None

        try:
            dispatch.send_token(spec.token)
        except Exception:  # pylint: disable=broad-exception-caught
            # End stdin if delivery stopped mid-frame. Reconciliation will
            # preserve the Pod if a task nevertheless started.
            dispatch.close()
            self._retire_after_dispatch(pod_name, pool.key, dispatch)
            return LaunchResult.unknown(
                "Warm executor token delivery outcome was unknown."
            )

        try:
            accepted = dispatch.wait_for_acceptance(_WARM_EXECUTOR_ACK_TIMEOUT_SECONDS)
        except Exception:  # pylint: disable=broad-exception-caught
            dispatch.close()
            self._retire_after_dispatch(pod_name, pool.key, dispatch)
            return LaunchResult.unknown(
                "Warm executor token acknowledgement outcome was unknown."
            )
        if not accepted:
            dispatch.close()
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
            if not self._reconcile_owned_pods():
                return
            for pool in self._pools.values():
                self._ensure_pool_capacity(pool, reserved_pod_capacity)

    def retry_retiring_pods(self) -> None:
        """Retry cleanup, including surviving tasks, without refilling pools."""
        with self._lock:
            if self._closed:
                return
            self._reconcile_owned_pods()

    def has_ready_pod(self, task_type: TaskType) -> bool:
        """Return whether a matching warm Pod can take a task without new capacity."""
        pool = self._pool_for_task(task_type)
        if pool is None:
            return False
        with self._lock:
            if self._closed:
                return False
            pods = self._owned_warm_pods()
            if pods is None:
                return False
            return any(self._is_dispatchable(pod, pool.key) for pod in pods)

    def close(self) -> None:
        """Stop dispatch and delete only idle Pods owned by this SuperExec."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            busy_pods = self._busy_pods.copy()
        pods = self._owned_warm_pods()
        if pods is None:
            return
        for pod in pods:
            pod_name = _object_name(pod)
            if (
                pod_name is not None
                and pod_name not in busy_pods
                and (
                    not _is_consumed_warm_executor(pod)
                    or self._consumed_pod_has_finished(pod)
                )
            ):
                self._delete_pod(pod_name)

    def _pool_for_task(self, task_type: TaskType) -> WarmExecutorPoolConfig | None:
        for pool in self._pools.values():
            if (
                pool.key.task_type == task_type
                and pool.key.runtime_image == self._config.image
            ):
                return pool
        return None

    def _is_dispatchable(self, pod: object, key: WarmExecutorPoolKey) -> bool:
        """Use the same availability check for admission and reservation."""
        pod_name = _object_name(pod)
        return (
            pod_name is not None
            and pod_name not in self._busy_pods
            and pod_name not in self._retiring_pods
            and not _is_consumed_warm_executor(pod)
            and self._has_warm_executor_configuration(pod, self._config)
            and is_warm_executor_ready(pod, key)
        )

    def _take_ready_pod(self, key: WarmExecutorPoolKey) -> str | None:
        pods = self._owned_warm_pods()
        if pods is None:
            raise WarmAgentAppUnavailable("Warm executor Pods could not be listed.")
        for pod in pods:
            pod_name = _object_name(pod)
            if pod_name is not None and self._is_dispatchable(pod, key):
                try:
                    self._mark_pod_consumed(pod_name)
                except WarmAgentAppUnavailable:
                    self._retire_unavailable_pod(pod_name)
                    raise
                self._busy_pods.add(pod_name)
                return pod_name
        return None

    def _ensure_pool_capacity(
        self, pool: WarmExecutorPoolConfig, reserved_pod_capacity: int = 0
    ) -> None:
        if self._closed or self._retiring_pods:
            return
        pods = self._owned_warm_pods()
        if pods is None:
            return
        compatible_count = sum(
            1
            for pod in pods
            if self._is_active_warm_executor(pod, pool.key, self._config)
            and _object_name(pod) not in self._busy_pods
            and not _is_consumed_warm_executor(pod)
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
                pod = self._build_warm_executor_pod(
                    pool.key, self._config, new_warm_executor_id()
                )
                self._client.create_namespaced_pod(self._config.namespace, pod)
            except Exception:  # pylint: disable=broad-exception-caught
                log(WARNING, "Failed to create a warm TaskExecutor Pod.", exc_info=True)
                return

    def _reconcile_owned_pods(self) -> bool:
        """Delete owned Pods that are obsolete or exceed configured capacity."""
        pods = self._owned_warm_pods()
        if pods is None:
            return False
        self._retry_retiring_pods(pods)

        compatible_pods: dict[WarmExecutorPoolKey, list[object]] = {
            pool.key: [] for pool in self._pools.values()
        }
        for pod in pods:
            pod_name = _object_name(pod)
            if (
                pod_name is not None
                and pod_name not in self._busy_pods
                and _is_consumed_warm_executor(pod)
            ):
                if self._consumed_pod_has_finished(pod):
                    self._retire_pod(pod_name)
                continue
            if pod_name in self._retiring_pods:
                continue
            pool = next(
                (
                    candidate
                    for candidate in self._pools.values()
                    if self._is_active_warm_executor(pod, candidate.key, self._config)
                ),
                None,
            )
            if pool is None:
                if pod_name is not None and pod_name not in self._busy_pods:
                    self._retire_pod(pod_name)
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
                    self._retire_pod(pod_name)
        return True

    def _retry_retiring_pods(self, pods: list[object]) -> None:
        """Retry deletion of consumed Pods without making them dispatchable."""
        pod_names = {
            pod_name for pod in pods if (pod_name := _object_name(pod)) is not None
        }
        for pod_name in self._retiring_pods - pod_names:
            self._release_pod(pod_name)
        for pod in pods:
            pod_name = _object_name(pod)
            if pod_name is not None and pod_name in self._retiring_pods:
                if self._delete_pod(pod_name):
                    self._release_pod(pod_name)

    def _owned_warm_pods(self) -> list[object] | None:
        try:
            pod_list = self._client.list_namespaced_pod(
                self._config.namespace,
                label_selector=self._warm_executor_owner_label_selector(self._config),
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
    ) -> KubernetesWarmAgentAppDispatch:
        try:
            stream = importlib.import_module("kubernetes.stream").stream
            # Kubernetes temporarily changes the exec client's transport during
            # the handshake. Serialize it with orphan-process probes.
            with self._lock:
                response = stream(
                    self._exec_client.connect_get_namespaced_pod_exec,
                    pod_name,
                    self._config.namespace,
                    container="taskexecutor",
                    command=warm_agentapp_command(spec, runtime_root_certificates),
                    stderr=True,
                    stdin=True,
                    stdout=True,
                    tty=False,
                    _preload_content=False,
                    _request_timeout=_WARM_EXECUTOR_ACK_TIMEOUT_SECONDS,
                )
        except Exception as err:  # pylint: disable=broad-exception-caught
            raise WarmAgentAppUnavailable(
                "Warm TaskExecutor Pod is unavailable for dispatch."
            ) from err
        return KubernetesWarmAgentAppDispatch(response)

    def _consumed_pod_has_finished(self, pod: object) -> bool:
        """Preserve surviving tasks unless their exit can be established."""
        if _object_field(_object_field(pod, "status"), "phase") in {
            "Succeeded",
            "Failed",
        }:
            return True
        response = None
        try:
            stream = importlib.import_module("kubernetes.stream").stream
            with self._lock:
                response = stream(
                    self._exec_client.connect_get_namespaced_pod_exec,
                    _object_name(pod),
                    self._config.namespace,
                    container="taskexecutor",
                    command=["python", "-c", _WARM_EXECUTOR_IDLE_CHECK],
                    stdin=False,
                    stdout=True,
                    stderr=True,
                    tty=False,
                    _preload_content=False,
                    _request_timeout=_WARM_EXECUTOR_ACK_TIMEOUT_SECONDS,
                )
            response.run_forever(timeout=_WARM_EXECUTOR_ACK_TIMEOUT_SECONDS)
            output = response.read_all()
            return (
                isinstance(output, str) and output.strip() == "FLWR_WARM_EXECUTOR_IDLE"
            )
        except Exception:  # pylint: disable=broad-exception-caught
            log(WARNING, "Could not establish whether a consumed warm Pod is idle.")
            return False
        finally:
            if response is not None:
                KubernetesWarmAgentAppDispatch(response).close()

    def _mark_pod_consumed(self, pod_name: str) -> None:
        try:
            self._client.patch_namespaced_pod(
                name=pod_name,
                namespace=self._config.namespace,
                body={
                    "metadata": {
                        "annotations": {WARM_EXECUTOR_CONSUMED_ANNOTATION: "true"}
                    }
                },
            )
        except Exception as err:  # pylint: disable=broad-exception-caught
            raise WarmAgentAppUnavailable(
                "Failed to persist warm TaskExecutor Pod consumption."
            ) from err

    def _retire_after_dispatch(
        self,
        pod_name: str,
        key: WarmExecutorPoolKey,
        dispatch: KubernetesWarmAgentAppDispatch,
    ) -> None:
        try:
            cleanup_thread = threading.Thread(
                target=self._wait_for_task_and_replace,
                args=(pod_name, key, dispatch),
                daemon=True,
            )
            cleanup_thread.start()
        except Exception:  # pylint: disable=broad-exception-caught
            # Reconciliation can check the persisted consumption marker later.
            # Never block a successful launch for the lifetime of its task.
            log(
                WARNING,
                "Warm TaskExecutor cleanup will be retried during reconciliation.",
            )
            dispatch.close()
            with self._lock:
                self._busy_pods.discard(pod_name)

    def _wait_for_task_and_replace(
        self,
        pod_name: str,
        key: WarmExecutorPoolKey,
        dispatch: KubernetesWarmAgentAppDispatch,
    ) -> None:
        completed = False
        try:
            completed = dispatch.wait_for_close()
        except Exception:  # pylint: disable=broad-exception-caught
            log(WARNING, "Warm TaskExecutor exec stream ended without an exit status.")
        finally:
            dispatch.close()
            with self._lock:
                if completed:
                    self._retire_pod(pod_name)
                else:
                    self._busy_pods.discard(pod_name)
                if not self._closed:
                    self._ensure_pool_capacity(self._pools[key])

    def _release_pod(self, pod_name: str) -> None:
        with self._lock:
            self._busy_pods.discard(pod_name)
            self._retiring_pods.discard(pod_name)

    def _retire_unavailable_pod(self, pod_name: str) -> None:
        """Delete a Pod that failed before task authority was delivered."""
        self._retire_pod(pod_name)

    def _retire_pod(self, pod_name: str) -> bool:
        with self._lock:
            self._retiring_pods.add(pod_name)
        if self._delete_pod(pod_name):
            self._release_pod(pod_name)
            return True
        return False

    def _delete_pod(self, pod_name: str) -> bool:
        try:
            self._client.delete_namespaced_pod(
                name=pod_name,
                namespace=self._config.namespace,
                grace_period_seconds=0,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if _exception_status(exc) == 404:
                return True
            log(WARNING, "Failed to delete warm TaskExecutor Pod %s.", pod_name)
            return False
        return True


def _is_consumed_warm_executor(pod: object) -> bool:
    """Return true when a Pod was reserved for a task before this process started."""
    metadata = _object_field(pod, "metadata")
    annotations = _object_field(metadata, "annotations")
    return _object_field(annotations, WARM_EXECUTOR_CONSUMED_ANNOTATION) == "true"


def _pod_items(pod_list: object) -> list[object]:
    """Return Pod items from a Kubernetes list response."""
    items = _object_field(pod_list, "items")
    if isinstance(items, Sequence) and not isinstance(items, str):
        return list(items)
    return []


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


def _exception_status(exc: Exception) -> int | None:
    """Return a Kubernetes API status code when exposed by an exception."""
    status = getattr(exc, "status", None)
    return status if isinstance(status, int) else None
