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
"""Worker reservation and Kubernetes task handoff for warm executors."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from logging import INFO, WARNING

from flwr.common.constant import (
    FLWR_AGENTAPP_TOKEN_STDIN_ACKNOWLEDGEMENT,
    FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT,
)
from flwr.supercore import log
from flwr.supercore.constant import (
    TASK_TYPE_TO_APPIO_API_ADDRESS_ARG,
    TASK_TYPE_TO_COMMAND,
    TaskType,
)
from flwr.supercore.tracing import trace_span
from flwr.supercore.warm_executor_constants import (
    WARM_AGENTAPP_EXECUTOR_MODULE,
    WARM_CONNECTOR_EXECUTOR_MODULE,
)

from .kubernetes_resources import WarmExecutorResources
from .types import ExecutionSpec, LaunchResult
from .warm_executor_pool import (
    WARM_EXECUTOR_ACK_TIMEOUT_SECONDS,
    WARM_EXECUTOR_ROOT_CERTIFICATES_FILE_PATH,
    WarmExecutorPoolConfig,
    WarmExecutorPoolKey,
    WarmExecutorPoolState,
    WarmExecutorUnavailable,
    is_consumed_warm_executor,
    is_warm_executor_ready,
)

_TOKEN_STDIN_ACKNOWLEDGEMENTS = (
    FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT,
    FLWR_AGENTAPP_TOKEN_STDIN_ACKNOWLEDGEMENT,
)
_TOKEN_STDIN_ACKNOWLEDGEMENT_BUFFER_SIZE = max(map(len, _TOKEN_STDIN_ACKNOWLEDGEMENTS))
_MAX_OUTPUT_LINE_BUFFER_SIZE = 16_384


class KubernetesWarmExecutorDispatch:
    """Interact with one Kubernetes exec stream without logging task authority."""

    def __init__(
        self,
        response: object,
        *,
        pod_name: str | None = None,
        task_id: int | None = None,
        log_output: bool = False,
    ) -> None:
        self._response = response
        self._pod_name = pod_name
        self._task_id = task_id
        self._log_output = log_output
        self._output_after_acceptance: list[tuple[str, str]] = []
        self._output_buffers = {"stdout": "", "stderr": ""}

    def send_token(self, token: str) -> None:
        """Send one token over stdin without retaining it in Pod metadata."""
        write_stdin = getattr(self._response, "write_stdin", None)
        if not callable(write_stdin):
            raise WarmExecutorUnavailable(
                "Kubernetes exec stream does not support standard input."
            )
        write_stdin(f"{token}\n")

    def wait_for_acceptance(self, timeout: float) -> bool:
        """Return whether the task child acknowledged consuming the token."""
        deadline = time.monotonic() + timeout
        stdout = ""
        stdout_before_acceptance: list[str] = []
        stderr_before_acceptance: list[str] = []
        while time.monotonic() < deadline:
            stdout += self._read_stdout()
            stderr = self._read_stderr()
            if stderr:
                stderr_before_acceptance.append(stderr)
            self._discard_combined_output()
            acknowledgement_span = self._acknowledgement_span(stdout)
            if acknowledgement_span is not None:
                acknowledgement_start, acknowledgement_end = acknowledgement_span
                self._output_after_acceptance.extend(
                    ("stdout", output)
                    for output in (
                        "".join(
                            (*stdout_before_acceptance, stdout[:acknowledgement_start])
                        ),
                        stdout[acknowledgement_end:].removeprefix("\n"),
                    )
                    if output
                )
                self._output_after_acceptance.extend(
                    ("stderr", output) for output in stderr_before_acceptance
                )
                return True
            stdout_before_acceptance.append(
                stdout[:-_TOKEN_STDIN_ACKNOWLEDGEMENT_BUFFER_SIZE]
            )
            stdout = stdout[-_TOKEN_STDIN_ACKNOWLEDGEMENT_BUFFER_SIZE:]
            if not self._is_open():
                self._log_unacknowledged_output(
                    stdout_before_acceptance, stdout, stderr_before_acceptance
                )
                return False
            self._update(min(0.5, deadline - time.monotonic()))
        self._log_unacknowledged_output(
            stdout_before_acceptance, stdout, stderr_before_acceptance
        )
        return False

    def _log_unacknowledged_output(
        self,
        stdout_before_acceptance: list[str],
        stdout: str,
        stderr_before_acceptance: list[str],
    ) -> None:
        """Log output observed before a child failed to acknowledge its token."""
        log_output = self._log_output
        # A later cleanup drain must not log a queued token acknowledgement.
        self._log_output = False
        if not log_output:
            return
        self._log_stream_output("stdout", "".join((*stdout_before_acceptance, stdout)))
        self._log_stream_output("stderr", "".join(stderr_before_acceptance))
        self._flush_output_buffers()

    def wait_for_close(self, *, forward_output: bool = False) -> bool:
        """Return whether stream closure confirms that the child exited.

        Forward child output only after token acceptance when the task opted in.
        This keeps the token acknowledgement out of the SuperExec logs.
        """
        self._drain_output(forward_output)
        while self._is_open():
            self._update(1.0)
            self._drain_output(forward_output)
            self._discard_combined_output()
        self._flush_output_buffers()
        # A disconnected socket is not proof of process exit. Kubernetes sends
        # the exit status on a separate channel, which must not be discarded.
        return isinstance(getattr(self._response, "returncode", None), int)

    def close(self) -> None:
        """Close the Kubernetes exec stream best-effort."""
        # Closing before acceptance also disables logging in the cleanup drain.
        self._log_output = False
        close = getattr(self._response, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pylint: disable=broad-exception-caught
                log(WARNING, "Failed to close warm TaskExecutor exec stream.")
        self._flush_output_buffers()

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

    def _read_stderr(self) -> str:
        peek_stderr = getattr(self._response, "peek_stderr", None)
        read_stderr = getattr(self._response, "read_stderr", None)
        if callable(read_stderr) and (not callable(peek_stderr) or peek_stderr()):
            stderr = read_stderr()
            return stderr if isinstance(stderr, str) else ""
        return ""

    @staticmethod
    def _acknowledgement_span(stdout: str) -> tuple[int, int] | None:
        """Return the span of the first complete token acknowledgement."""
        acknowledgement_start: int | None = None
        acknowledgement_end: int | None = None
        for acknowledgement in _TOKEN_STDIN_ACKNOWLEDGEMENTS:
            start = stdout.find(acknowledgement)
            if start >= 0 and (
                acknowledgement_start is None or start < acknowledgement_start
            ):
                acknowledgement_start = start
                acknowledgement_end = start + len(acknowledgement)
        if acknowledgement_start is None or acknowledgement_end is None:
            return None
        return acknowledgement_start, acknowledgement_end

    def _drain_output(self, forward_output: bool) -> None:
        """Consume exec output and optionally mirror it to SuperExec output."""
        output_after_acceptance = self._output_after_acceptance
        self._output_after_acceptance = []
        for stream, output in (
            *output_after_acceptance,
            ("stdout", self._read_stdout()),
            ("stderr", self._read_stderr()),
        ):
            if not output:
                continue
            output_stream = sys.stdout if stream == "stdout" else sys.stderr
            if forward_output:
                output_stream.write(output)
                output_stream.flush()
            if self._log_output:
                self._log_stream_output(stream, output)

    def _log_stream_output(self, stream: str, output: str) -> None:
        """Log complete child-output lines while retaining a partial suffix."""
        buffered = self._output_buffers[stream] + output
        trailing_carriage_return = buffered.endswith("\r")
        if trailing_carriage_return:
            buffered = buffered[:-1]
        lines = buffered.splitlines(keepends=True)
        self._output_buffers[stream] = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._output_buffers[stream] = lines.pop()
        if trailing_carriage_return:
            self._output_buffers[stream] += "\r"
        for line in lines:
            self._log_output_line(stream, line.rstrip("\r\n"))
        while (
            len(self._output_buffers[stream].removesuffix("\r"))
            > _MAX_OUTPUT_LINE_BUFFER_SIZE
        ):
            chunk = self._output_buffers[stream][:_MAX_OUTPUT_LINE_BUFFER_SIZE]
            self._output_buffers[stream] = self._output_buffers[stream][
                _MAX_OUTPUT_LINE_BUFFER_SIZE:
            ]
            self._log_output_line(stream, chunk)

    def _flush_output_buffers(self) -> None:
        """Log any final partial child-output lines."""
        for stream, output in self._output_buffers.items():
            if output:
                self._log_output_line(stream, output.removesuffix("\r"))
        self._output_buffers = {"stdout": "", "stderr": ""}

    def _log_output_line(self, stream: str, output: str) -> None:
        """Log one child-output line with enough dispatch context to correlate it."""
        filtered_output = output
        for acknowledgement in _TOKEN_STDIN_ACKNOWLEDGEMENTS:
            filtered_output = filtered_output.replace(acknowledgement, "")
        if output and not filtered_output:
            return
        log(
            INFO,
            "Warm TaskExecutor output pod=%s task_id=%s stream=%s: %s",
            self._pod_name or "unknown",
            self._task_id if self._task_id is not None else "unknown",
            stream,
            filtered_output,
        )

    def _discard_combined_output(self) -> None:
        # WSClient.read_all() also clears unread channels, including the exit
        # status and stdout received while peeking stderr. Its capture_all option
        # is not accepted by CoreV1Api, so clear only the combined buffer.
        combined_output = getattr(self._response, "_all", None)
        if combined_output is not None:
            combined_output.seek(0)
            combined_output.truncate()


def warm_executor_command(
    spec: ExecutionSpec,
    runtime_root_certificates: str | None,
    pool_key: WarmExecutorPoolKey | None = None,
) -> list[str]:
    """Build a one-task child command that receives authority on standard input."""
    prestarted_agentapp = (
        spec.task_type == TaskType.AGENT_APP
        and pool_key is not None
        and pool_key.fab_hash is not None
    )
    worker_module = None
    if prestarted_agentapp:
        worker_module = WARM_AGENTAPP_EXECUTOR_MODULE
    elif spec.task_type == TaskType.CONNECTOR:
        worker_module = WARM_CONNECTOR_EXECUTOR_MODULE
    if worker_module is not None:
        command = [
            "python",
            "-m",
            worker_module,
            "dispatch",
            "--runtime-api-address",
            spec.runtime_api_address,
            "--token-stdin",
        ]
        if prestarted_agentapp:
            assert pool_key is not None
            assert pool_key.fab_hash is not None
            assert spec.fab_hash is not None
            if spec.fab_hash != pool_key.fab_hash:
                raise ValueError("AgentApp task does not match its warm executor pool.")
            command.extend(["--fab-hash", spec.fab_hash])
    else:
        command = [
            TASK_TYPE_TO_COMMAND[spec.task_type],
            TASK_TYPE_TO_APPIO_API_ADDRESS_ARG[spec.task_type],
            spec.runtime_api_address,
            "--token-stdin",
        ]
    if spec.insecure:
        command.append("--insecure")
    elif runtime_root_certificates is not None:
        command.extend(
            [
                "--root-certificates",
                WARM_EXECUTOR_ROOT_CERTIFICATES_FILE_PATH,
            ]
        )
    if spec.runtime_dependency_install and not prestarted_agentapp:
        command.append("--allow-runtime-dependency-installation")
    return command


class WarmExecutorDispatcher:
    """Reserve compatible workers and deliver one task invocation."""

    def __init__(
        self,
        resources: WarmExecutorResources,
        state: WarmExecutorPoolState,
        request_maintenance: Callable[[], None] | None = None,
    ) -> None:
        self._resources = resources
        self._state = state
        self._lock = state.lock
        self._busy_pods = state.busy_pods
        self._retiring_pods = state.retiring_pods
        self._pools = {
            pool.key: pool for pool in self._resources.config.warm_executor_pools
        }
        self._request_maintenance = request_maintenance

    # pylint: disable-next=too-many-return-statements
    def launch(
        self, spec: ExecutionSpec, runtime_root_certificates: str | None
    ) -> LaunchResult | None:
        """Dispatch a compatible task to a ready Pod or use the cold fallback."""
        candidates = self._candidate_pools(spec.task_type, spec.fab_hash)
        if not candidates:
            return None
        primary_pool = candidates[0]

        with self._lock:
            if self._state.closed:
                self._log_dispatch(primary_pool, "setup_unavailable")
                return None
            try:
                with trace_span("executor.reserve"):
                    claimed = self._take_ready_pod(candidates)
            except WarmExecutorUnavailable:
                self._log_dispatch(primary_pool, "setup_unavailable")
                return None
            if claimed is None:
                if self._request_maintenance is not None:
                    self._request_maintenance()
                self._log_dispatch(primary_pool, "capacity_unavailable")
                return None
            pool, pod_name = claimed
            # Reconciliation replenishes consumed capacity after dispatch.

        try:
            with trace_span("executor.dispatch_open"):
                dispatch = self._open_dispatch(
                    pod_name=pod_name,
                    spec=spec,
                    pool_key=pool.key,
                    runtime_root_certificates=runtime_root_certificates,
                )
        except WarmExecutorUnavailable:
            self._retire_unavailable_pod(pod_name, pool.key)
            self._log_dispatch(pool, "setup_unavailable")
            return None

        try:
            with trace_span("executor.token_send"):
                dispatch.send_token(spec.token)
        except Exception:  # pylint: disable=broad-exception-caught
            # End stdin if delivery stopped mid-frame. Reconciliation will
            # preserve the Pod if a task nevertheless started.
            dispatch.close()
            self._retire_after_dispatch(pod_name, pool.key, dispatch)
            self._log_dispatch(pool, "unknown")
            return LaunchResult.unknown(
                "Warm executor token delivery outcome was unknown."
            )

        try:
            with trace_span("executor.accept_wait") as acceptance:
                accepted = dispatch.wait_for_acceptance(
                    WARM_EXECUTOR_ACK_TIMEOUT_SECONDS
                )
                acceptance.set_attribute("flwr.executor.accepted", accepted)
        except Exception:  # pylint: disable=broad-exception-caught
            dispatch.close()
            self._retire_after_dispatch(pod_name, pool.key, dispatch)
            self._log_dispatch(pool, "unknown")
            return LaunchResult.unknown(
                "Warm executor token acknowledgement outcome was unknown."
            )
        if not accepted:
            dispatch.close()
        self._retire_after_dispatch(
            pod_name,
            pool.key,
            dispatch,
            forward_output=accepted and not spec.suppress_output,
        )
        if accepted:
            self._log_dispatch(pool, "warm_claimed")
            return LaunchResult.accepted()
        self._log_dispatch(pool, "unknown")
        return LaunchResult.unknown(
            "Warm executor did not acknowledge task token delivery."
        )

    @staticmethod
    def _log_dispatch(pool: WarmExecutorPoolConfig, outcome: str) -> None:
        """Log one configured warm-pool dispatch decision."""
        log(
            INFO,
            "warm_executor_dispatch task_type=%s pool_size=%s outcome=%s",
            pool.key.task_type.value,
            pool.size,
            outcome,
        )

    def has_ready_pod(self, task_type: TaskType, fab_hash: str | None = None) -> bool:
        """Return whether a matching warm Pod can take a task without new capacity."""
        candidates = self._candidate_pools(task_type, fab_hash)
        if not candidates:
            return False
        with self._lock:
            if self._state.closed:
                return False
            pods = self._resources.list_owned()
            if pods is None:
                return False
            return any(
                self._is_dispatchable(pod, pool.key)
                for pool in candidates
                for pod in pods
            )

    def ready_pool_keys(self) -> set[WarmExecutorPoolKey]:
        """Return compatible pool keys with a Pod ready for dispatch."""
        with self._lock:
            if self._state.closed:
                return set()
            pods = self._resources.list_owned()
            if pods is None:
                return set()
            return {
                pool.key
                for pool in self._pools.values()
                if pool.key.runtime_image == self._resources.config.image
                and any(self._is_dispatchable(pod, pool.key) for pod in pods)
            }

    def _candidate_pools(
        self, task_type: TaskType, fab_hash: str | None
    ) -> tuple[WarmExecutorPoolConfig, ...]:
        """Return exact then generic pools used by admission and reservation."""
        pools = tuple(
            pool
            for pool in self._pools.values()
            if pool.key.task_type == task_type
            and pool.key.runtime_image == self._resources.config.image
        )
        exact = tuple(
            pool
            for pool in pools
            if fab_hash is not None and pool.key.fab_hash == fab_hash
        )
        generic = tuple(pool for pool in pools if pool.key.fab_hash is None)
        return (*exact, *generic)

    def _is_dispatchable(self, pod: object, key: WarmExecutorPoolKey) -> bool:
        """Use the same availability check for admission and reservation."""
        pod_name = _object_name(pod)
        return (
            pod_name is not None
            and pod_name not in self._busy_pods
            and pod_name not in self._state.consumed_pods
            and pod_name not in self._retiring_pods
            and not is_consumed_warm_executor(pod)
            and self._resources.has_configuration(pod, key)
            and is_warm_executor_ready(pod, key)
        )

    def _take_ready_pod(
        self, candidates: Sequence[WarmExecutorPoolConfig]
    ) -> tuple[WarmExecutorPoolConfig, str] | None:
        pods = self._resources.list_owned()
        if pods is None:
            raise WarmExecutorUnavailable("Warm executor Pods could not be listed.")
        for pool in candidates:
            for pod in pods:
                pod_name = _object_name(pod)
                if pod_name is not None and self._is_dispatchable(pod, pool.key):
                    try:
                        self._resources.mark_consumed(pod_name)
                    except WarmExecutorUnavailable:
                        self._retire_unavailable_pod(pod_name, pool.key)
                        raise
                    self._state.consumed_pods.add(pod_name)
                    self._state.pending_creations.pop(pod_name, None)
                    self._busy_pods.add(pod_name)
                    return pool, pod_name
        return None

    def _retire_after_dispatch(
        self,
        pod_name: str,
        key: WarmExecutorPoolKey,
        dispatch: KubernetesWarmExecutorDispatch,
        *,
        forward_output: bool = False,
    ) -> None:
        try:
            cleanup_thread = threading.Thread(
                target=self._wait_for_task_and_replace,
                args=(pod_name, key, dispatch, forward_output),
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
        dispatch: KubernetesWarmExecutorDispatch,
        forward_output: bool = False,
    ) -> None:
        completed = False
        try:
            completed = dispatch.wait_for_close(forward_output=forward_output)
        except Exception:  # pylint: disable=broad-exception-caught
            log(WARNING, "Warm TaskExecutor exec stream ended without an exit status.")
        finally:
            dispatch.close()
            if completed:
                self._retire_pod(pod_name, key)
            else:
                with self._lock:
                    self._busy_pods.discard(pod_name)
            if self._request_maintenance is not None:
                self._request_maintenance()

    def _release_pod(self, pod_name: str) -> None:
        with self._lock:
            self._busy_pods.discard(pod_name)
            self._state.consumed_pods.discard(pod_name)
            self._state.pending_creations.pop(pod_name, None)
            self._retiring_pods.pop(pod_name, None)

    def _retire_unavailable_pod(self, pod_name: str, key: WarmExecutorPoolKey) -> None:
        """Retire a Pod that failed before task authority was delivered."""
        if self._request_maintenance is None:
            self._retire_pod(pod_name, key)
            return
        with self._lock:
            self._retiring_pods[pod_name] = key
        self._request_maintenance()

    def _retire_pod(
        self, pod_name: str, key: WarmExecutorPoolKey | None = None
    ) -> bool:
        with self._lock:
            self._retiring_pods[pod_name] = key
        if self._resources.delete(pod_name):
            self._release_pod(pod_name)
            return True
        return False

    def _open_dispatch(
        self,
        *,
        pod_name: str,
        spec: ExecutionSpec,
        pool_key: WarmExecutorPoolKey,
        runtime_root_certificates: str | None,
    ) -> KubernetesWarmExecutorDispatch:
        response = self._resources.open_exec(
            pod_name=pod_name,
            command=warm_executor_command(spec, runtime_root_certificates, pool_key),
        )
        return KubernetesWarmExecutorDispatch(
            response,
            pod_name=pod_name,
            task_id=spec.task_id,
            log_output=self._resources.config.log_warm_executor_output
            and spec.suppress_output,
        )


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
