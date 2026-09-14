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
"""Tests for SuperExec Kubernetes executor."""

# pylint: disable=too-many-lines

import importlib
import subprocess
import sys
import threading
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, call

import pytest

from flwr.common.constant import (
    FLWR_AGENTAPP_TOKEN_STDIN_ACKNOWLEDGEMENT,
    FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT,
)
from flwr.supercore.constant import TaskType

from . import kubernetes_executor as kube
from . import warm_agentapp_executor
from .kubernetes_executor import (
    _COMPLETED_POD_SWEEP_INTERVAL_SECONDS,
    _TASK_ID_LABEL,
    _WARM_EXECUTOR_CONSUMED_ANNOTATION,
    APPIO_CREDENTIALS_MOUNT_PATH,
    APPIO_ROOT_CERTIFICATES_FILE_PATH,
    APPIO_TOKEN_FILE_PATH,
    LAUNCH_ATTEMPT_LABEL,
    CompletedPodSweeper,
    KubernetesExecutor,
    KubernetesExecutorConfig,
    _build_appio_credentials_secret,
    _build_taskexecutor_pod,
    _get_runtime_root_certificates,
)
from .types import ExecutionSpec, LaunchResultStatus
from .warm_executor import (
    WARM_EXECUTOR_MODULE,
    WARM_EXECUTOR_READINESS_COMMAND,
    WARM_EXECUTOR_READY_DIRECTORY,
    WARM_EXECUTOR_READY_FILE,
)
from .warm_executor_pool import (
    WARM_EXECUTOR_CONFIGURATION_ANNOTATION,
    WARM_EXECUTOR_LABEL,
    WARM_EXECUTOR_RUNTIME_IMAGE_ANNOTATION,
    WarmExecutorPoolConfig,
    WarmExecutorPoolKey,
    is_warm_executor_ready,
)


class _KubernetesApiError(Exception):
    """Minimal Kubernetes client error used by executor tests."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status


_LAUNCH_ATTEMPT_ID = "abc123def456"
_NEXT_LAUNCH_ATTEMPT_ID = "def456abc123"
_POD_NAME = f"flwr-taskexecutor-123-{_LAUNCH_ATTEMPT_ID}"
_NEXT_POD_NAME = f"flwr-taskexecutor-123-{_NEXT_LAUNCH_ATTEMPT_ID}"
_SECRET_NAME = f"{_POD_NAME}-appio"
_NEXT_SECRET_NAME = f"{_NEXT_POD_NAME}-appio"


def _execution_spec(**overrides: Any) -> ExecutionSpec:
    base: dict[str, Any] = {
        "task_type": TaskType.SERVER_APP,
        "runtime_api_address": "appio.example.com:9092",
        "token": "task-token",
        "insecure": False,
        "root_certificates_path": None,
        "runtime_dependency_install": False,
        "parent_pid": None,
        "suppress_output": True,
        "task_id": 123,
    }
    base.update(overrides)
    return ExecutionSpec(**base)


def _executor_config(**overrides: Any) -> KubernetesExecutorConfig:
    base: dict[str, Any] = {
        "namespace": "flower-system",
        "image": "ghcr.io/flwrlabs/taskexecutor:dev",
        "runtime_root_certificates": "root-ca",
    }
    base.update(overrides)
    return KubernetesExecutorConfig(**base)


def _warm_executor_pool_key(**overrides: Any) -> WarmExecutorPoolKey:
    base: dict[str, Any] = {
        "task_type": TaskType.AGENT_APP,
        "runtime_image": "ghcr.io/flwrlabs/taskexecutor:warm",
    }
    base.update(overrides)
    return WarmExecutorPoolKey(**base)


def _ready_warm_pod(
    pool_key: WarmExecutorPoolKey,
    config: KubernetesExecutorConfig,
    name: str = "flwr-taskexecutor-warm-ready",
) -> dict[str, Any]:
    """Build one compatible Ready Pod returned by the Kubernetes API."""
    pod = _as_dict(
        kube._build_warm_executor_pod(  # pylint: disable=protected-access
            pool_key, config, "ready"
        )
    )
    pod["metadata"]["name"] = name
    pod["status"] = {
        "phase": "Running",
        "conditions": [{"type": "Ready", "status": "True"}],
    }
    return pod


class _WarmExecResponse:
    """Minimal Kubernetes exec response with one token acknowledgement."""

    def __init__(self, acknowledge: bool = True, stderr: str = "") -> None:
        self.written: list[str] = []
        self._all = StringIO()
        self.read_stderr_calls = 0
        self._acknowledge = acknowledge
        self._stderr = stderr
        self._stdout = ""
        self._open = True

    def write_stdin(self, data: str) -> None:
        """Receive the private task token and make the child acknowledge it."""
        self.written.append(data)
        if self._acknowledge:
            self._stdout = "FLWR_AGENTAPP_TOKEN_ACCEPTED\n"

    def is_open(self) -> bool:
        """Return whether the child exec process is still running."""
        return self._open

    @property
    def returncode(self) -> int | None:
        """Report process exit separately from acknowledgement."""
        return None if self._open else 0

    def update(self, timeout: float) -> None:
        """Advance the response so an unacknowledged child exits."""
        del timeout
        if not self._acknowledge:
            self._open = False

    def peek_stdout(self) -> bool:
        """Return whether standard output is available."""
        return bool(self._stdout)

    def read_stdout(self) -> str:
        """Consume standard output."""
        stdout = self._stdout
        self._stdout = ""
        return stdout

    def peek_stderr(self) -> bool:
        """Return whether standard error is available."""
        return bool(self._stderr)

    def read_stderr(self) -> str:
        """Discard standard error without retaining task output."""
        self.read_stderr_calls += 1
        stderr = self._stderr
        self._stderr = ""
        return stderr

    def close(self) -> None:
        """Close the response after one task."""
        self._open = False


def _as_dict(value: object) -> dict[str, Any]:
    """Return a typed dict for nested JSON assertions."""
    return cast(dict[str, Any], value)


def _appio_root_certificates(
    spec: ExecutionSpec, config: KubernetesExecutorConfig
) -> str | None:
    """Return AppIo root certificates for object-building tests."""
    return _get_runtime_root_certificates(spec, config)


def _task_labels(task_id: int) -> dict[str, str]:
    labels = _taskexecutor_labels()
    labels[_TASK_ID_LABEL] = str(task_id)
    return labels


def _taskexecutor_labels() -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "flower",
        "app.kubernetes.io/component": "taskexecutor",
        "flower.ai/task-type": "flwr-serverapp",
    }


def _pod(
    phase: str,
    deletion_timestamp: str | None = None,
    *,
    name: str = _POD_NAME,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"name": name}
    if deletion_timestamp is not None:
        metadata["deletionTimestamp"] = deletion_timestamp
    if labels is not None:
        metadata["labels"] = labels

    status: dict[str, Any] = {"phase": phase}
    return {"metadata": metadata, "status": status}


def _secret(name: str, labels: dict[str, str] | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"name": name}
    if labels is not None:
        metadata["labels"] = labels
    return {"metadata": metadata}


def test_build_appio_credentials_secret_contains_token_and_ca() -> None:
    """Test building the AppIo credential Secret."""
    spec = _execution_spec()
    config = _executor_config()

    secret = _as_dict(
        _build_appio_credentials_secret(
            spec, config, _appio_root_certificates(spec, config), _LAUNCH_ATTEMPT_ID
        )
    )

    assert secret == {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": _SECRET_NAME,
            "namespace": "flower-system",
            "labels": {
                "app.kubernetes.io/name": "flower",
                "app.kubernetes.io/component": "taskexecutor",
                "flower.ai/superexec-task-id": "123",
                "flower.ai/task-type": "flwr-serverapp",
                LAUNCH_ATTEMPT_LABEL: _LAUNCH_ATTEMPT_ID,
            },
        },
        "type": "Opaque",
        "stringData": {"token": "task-token", "ca.crt": "root-ca"},
    }


def test_build_taskexecutor_pod_uses_secret_files_for_credentials() -> None:
    """Test Pod construction uses mounted files instead of credential args."""
    spec = _execution_spec()
    config = _executor_config()

    pod = _as_dict(
        _build_taskexecutor_pod(
            spec, config, _appio_root_certificates(spec, config), _LAUNCH_ATTEMPT_ID
        )
    )
    container = pod["spec"]["containers"][0]

    assert APPIO_CREDENTIALS_MOUNT_PATH == "/run/flwr/appio"
    assert pod["metadata"]["name"] == _POD_NAME
    assert pod["metadata"]["namespace"] == "flower-system"
    assert container["image"] == "ghcr.io/flwrlabs/taskexecutor:dev"
    assert container["command"] == ["flwr-serverapp"]
    assert "env" not in container
    assert container["args"] == [
        "--runtime-api-address",
        "appio.example.com:9092",
        "--token-file",
        APPIO_TOKEN_FILE_PATH,
        "--root-certificates",
        APPIO_ROOT_CERTIFICATES_FILE_PATH,
    ]
    assert "task-token" not in container["command"]
    assert "task-token" not in container["args"]
    assert container["volumeMounts"] == [
        {
            "name": "appio-credentials",
            "mountPath": APPIO_CREDENTIALS_MOUNT_PATH,
            "readOnly": True,
        }
    ]
    assert pod["spec"]["volumes"] == [
        {
            "name": "appio-credentials",
            "secret": {
                "secretName": _SECRET_NAME,
                "defaultMode": 0o444,
            },
        }
    ]
    assert pod["spec"]["automountServiceAccountToken"] is False
    assert pod["spec"]["restartPolicy"] == "Never"


def test_launch_warm_executor_is_inert_and_becomes_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a warm TaskExecutor is inert, reports readiness, and is swept."""
    client = Mock()
    monkeypatch.setattr(
        kube,
        "new_warm_executor_id",
        Mock(side_effect=["executor123", "executor456"]),
    )
    config = _executor_config(
        labels={WARM_EXECUTOR_LABEL: "false"},
        warm_executor_owner="superexec-a",
        annotations={
            "example.com/setting": "configured",
            _WARM_EXECUTOR_CONSUMED_ANNOTATION: "true",
        },
        container_security_context={"readOnlyRootFilesystem": True},
    )
    executor = KubernetesExecutor(client=client, config=config)
    pool_key = _warm_executor_pool_key()

    result = executor._launch_warm_executor(  # pylint: disable=protected-access
        pool_key
    )

    assert result.status == LaunchResultStatus.ACCEPTED
    client.create_namespaced_pod.assert_called_once()
    client.create_namespaced_secret.assert_called_once()
    pod = _as_dict(client.create_namespaced_pod.call_args.args[1])
    root_certificates_secret = _as_dict(
        client.create_namespaced_secret.call_args.args[1]
    )
    metadata = pod["metadata"]
    container = pod["spec"]["containers"][0]

    assert metadata["labels"] == {
        "app.kubernetes.io/name": "flower",
        "app.kubernetes.io/component": "taskexecutor",
        "flower.ai/task-type": "flwr-agentapp",
        WARM_EXECUTOR_LABEL: "true",
        "flower.ai/warm-executor-owner": "superexec-a",
    }
    annotations = metadata["annotations"]
    assert annotations["example.com/setting"] == "configured"
    assert annotations[WARM_EXECUTOR_RUNTIME_IMAGE_ANNOTATION] == (
        "ghcr.io/flwrlabs/taskexecutor:warm"
    )
    assert len(annotations[WARM_EXECUTOR_CONFIGURATION_ANNOTATION]) == 64
    assert len(annotations) == 3
    assert _WARM_EXECUTOR_CONSUMED_ANNOTATION not in annotations
    assert _TASK_ID_LABEL not in metadata["labels"]
    assert LAUNCH_ATTEMPT_LABEL not in metadata["labels"]
    assert root_certificates_secret["stringData"] == {"ca.crt": "root-ca"}
    assert root_certificates_secret["metadata"]["labels"] == metadata["labels"]
    assert "task-token" not in repr(pod)
    assert "task-token" not in repr(root_certificates_secret)
    assert container == {
        "name": "taskexecutor",
        "image": "ghcr.io/flwrlabs/taskexecutor:warm",
        "command": ["python", "-m", WARM_EXECUTOR_MODULE],
        "volumeMounts": [
            {
                "name": "warm-executor-ready",
                "mountPath": WARM_EXECUTOR_READY_DIRECTORY,
            },
            {
                "name": "warm-executor-root-certificates",
                "mountPath": "/run/flwr/runtime-ca",
                "readOnly": True,
            },
        ],
        "readinessProbe": {
            "exec": {"command": list(WARM_EXECUTOR_READINESS_COMMAND)},
            "periodSeconds": 1,
        },
        "securityContext": {"readOnlyRootFilesystem": True},
    }
    assert pod["spec"]["volumes"] == [
        {"name": "warm-executor-ready", "emptyDir": {}},
        {
            "name": "warm-executor-root-certificates",
            "secret": {
                "secretName": "flwr-taskexecutor-warm-executor123-runtime-ca",
                "defaultMode": 0o444,
            },
        },
    ]
    assert pod["spec"]["automountServiceAccountToken"] is False

    pod["status"] = {
        "phase": "Running",
        "conditions": [{"type": "Ready", "status": "False"}],
    }
    assert not is_warm_executor_ready(pod, pool_key)
    pod["status"]["conditions"][0]["status"] = "True"
    assert is_warm_executor_ready(pod, pool_key)

    incompatible_keys = (
        _warm_executor_pool_key(task_type=TaskType.MODEL),
        _warm_executor_pool_key(runtime_image="taskexecutor:other"),
    )
    assert not any(is_warm_executor_ready(pod, key) for key in incompatible_keys)

    pod["status"]["phase"] = "Succeeded"
    client.list_namespaced_pod.return_value = {"items": [pod]}
    client.list_namespaced_secret.return_value = {"items": []}

    CompletedPodSweeper(client=client, config=config).sweep()

    client.delete_namespaced_pod.assert_called_once_with(
        name="flwr-taskexecutor-warm-executor123",
        namespace="flower-system",
        grace_period_seconds=0,
    )
    client.delete_namespaced_secret.assert_not_called()
    client.list_namespaced_pod.assert_called_with(
        "flower-system",
        label_selector=(
            "app.kubernetes.io/component=taskexecutor,app.kubernetes.io/name=flower,"
            "flower.ai/warm-executor=true,flower.ai/warm-executor-owner=superexec-a"
        ),
    )


@pytest.mark.parametrize(
    ("insecure", "transport_args"),
    [
        (True, ["--insecure"]),
        (False, ["--root-certificates", "/run/flwr/runtime-ca/ca.crt"]),
    ],
)
def test_launch_dispatches_compatible_ready_pod_and_replenishes_idle_capacity(
    monkeypatch: pytest.MonkeyPatch,
    insecure: bool,
    transport_args: list[str],
) -> None:
    """A dispatched warm Pod should be replaced before its child exits."""
    client = Mock()
    exec_client = Mock()
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        runtime_root_certificates=None if insecure else "root-ca",
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
    )
    client.list_namespaced_pod.return_value = {
        "items": [_ready_warm_pod(pool_key, config)]
    }
    response = _WarmExecResponse()
    stream = Mock(return_value=response)
    monkeypatch.setattr(
        importlib,
        "import_module",
        Mock(return_value=SimpleNamespace(stream=stream)),
    )
    started: list[tuple[object, tuple[object, ...]]] = []

    def _thread(*, target: object, args: tuple[object, ...], daemon: bool) -> object:
        assert daemon
        return SimpleNamespace(start=lambda: started.append((target, args)))

    monkeypatch.setattr(threading, "Thread", _thread)
    executor = KubernetesExecutor(
        client=client,
        config=config,
        exec_client=exec_client,
    )

    result = executor.launch(
        _execution_spec(task_type=TaskType.AGENT_APP, insecure=insecure)
    )

    assert result.status == LaunchResultStatus.ACCEPTED
    assert response.written == ["task-token\n"]
    assert response.is_open()
    if insecure:
        client.create_namespaced_secret.assert_not_called()
    else:
        assert _as_dict(client.create_namespaced_secret.call_args.args[1])[
            "stringData"
        ] == {"ca.crt": "root-ca"}
    client.patch_namespaced_pod.assert_called_once_with(
        name="flwr-taskexecutor-warm-ready",
        namespace="flower-system",
        body={
            "metadata": {
                "annotations": {
                    _WARM_EXECUTOR_CONSUMED_ANNOTATION: "true",
                }
            }
        },
    )
    client.create_namespaced_pod.assert_called_once()
    replacement_pod = _as_dict(client.create_namespaced_pod.call_args.args[1])
    assert replacement_pod["spec"]["containers"][0]["command"] == [
        "python",
        "-m",
        WARM_EXECUTOR_MODULE,
    ]
    command = stream.call_args.kwargs["command"]
    assert stream.call_args.args[0] is exec_client.connect_get_namespaced_pod_exec
    assert stream.call_args.kwargs["container"] == "taskexecutor"
    assert command == [
        "flwr-agentapp",
        "--runtime-api-address",
        "appio.example.com:9092",
        "--token-stdin",
        *transport_args,
    ]
    assert "task-token" not in command
    assert len(started) == 1


def test_launch_falls_back_to_cold_pod_when_no_ready_warm_pod_exists() -> None:
    """No token should be delivered to a missing Pod before cold fallback."""
    client = Mock()
    client.list_namespaced_pod.return_value = {"items": []}
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        runtime_root_certificates=None,
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
    )
    executor = KubernetesExecutor(client=client, config=config)

    result = executor.launch(
        _execution_spec(task_type=TaskType.AGENT_APP, insecure=True)
    )

    assert result.status == LaunchResultStatus.ACCEPTED
    client.create_namespaced_secret.assert_called_once()
    cold_pod = _as_dict(client.create_namespaced_pod.call_args.args[1])
    assert cold_pod["spec"]["containers"][0]["command"] == ["flwr-agentapp"]


def test_launch_falls_back_to_cold_pod_when_warm_pods_cannot_be_listed() -> None:
    """A failed warm Pod list must not be treated as missing pool capacity."""
    client = Mock()
    client.list_namespaced_pod.side_effect = _KubernetesApiError(403, "forbidden")
    client.list_namespaced_secret.return_value = {"items": []}
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        runtime_root_certificates=None,
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
    )

    result = KubernetesExecutor(client=client, config=config).launch(
        _execution_spec(task_type=TaskType.AGENT_APP, insecure=True)
    )

    assert result.status == LaunchResultStatus.ACCEPTED
    cold_pod = _as_dict(client.create_namespaced_pod.call_args.args[1])
    assert cold_pod["spec"]["containers"][0]["command"] == ["flwr-agentapp"]


@pytest.mark.parametrize("delete_fails", [False, True])
def test_launch_retires_warm_pod_when_consumption_cannot_be_persisted(
    delete_fails: bool,
) -> None:
    """A Pod without a persisted reservation must not remain dispatchable."""
    client = Mock()
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        runtime_root_certificates=None,
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
    )
    warm_pod = _ready_warm_pod(pool_key, config)
    client.list_namespaced_pod.return_value = {"items": [warm_pod]}
    client.patch_namespaced_pod.side_effect = _KubernetesApiError(403, "forbidden")
    if delete_fails:
        client.delete_namespaced_pod.side_effect = _KubernetesApiError(500, "error")
    executor = KubernetesExecutor(client=client, config=config)

    result = executor.launch(
        _execution_spec(task_type=TaskType.AGENT_APP, insecure=True)
    )

    assert result.status == LaunchResultStatus.ACCEPTED
    client.delete_namespaced_pod.assert_called_with(
        name=warm_pod["metadata"]["name"],
        namespace="flower-system",
        grace_period_seconds=0,
    )
    if delete_fails:
        # pylint: disable-next=protected-access
        manager = executor._warm_executor_pool_manager
        assert manager is not None
        assert not manager.has_ready_pod(TaskType.AGENT_APP)
        # pylint: disable-next=protected-access
        assert manager._take_ready_pod(pool_key) is None
    cold_pod = _as_dict(client.create_namespaced_pod.call_args.args[1])
    assert cold_pod["spec"]["containers"][0]["command"] == ["flwr-agentapp"]


def test_launch_retires_warm_pod_when_dispatch_cannot_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable warm Pod must not remain reusable before cold fallback."""
    client = Mock()
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        runtime_root_certificates=None,
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
    )
    warm_pod = _ready_warm_pod(pool_key, config)
    client.list_namespaced_pod.return_value = {"items": [warm_pod]}
    monkeypatch.setattr(
        importlib,
        "import_module",
        Mock(return_value=SimpleNamespace(stream=Mock(side_effect=RuntimeError))),
    )
    executor = KubernetesExecutor(client=client, config=config)

    result = executor.launch(
        _execution_spec(task_type=TaskType.AGENT_APP, insecure=True)
    )

    assert result.status == LaunchResultStatus.ACCEPTED
    client.delete_namespaced_pod.assert_called_once_with(
        name=warm_pod["metadata"]["name"],
        namespace="flower-system",
        grace_period_seconds=0,
    )
    cold_pod = _as_dict(client.create_namespaced_pod.call_args.args[1])
    assert cold_pod["spec"]["containers"][0]["command"] == ["flwr-agentapp"]


@pytest.mark.parametrize(
    ("acknowledge", "send_error"), [(False, False), (True, False), (False, True)]
)
def test_launch_preserves_outcome_when_cleanup_thread_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
    acknowledge: bool,
    send_error: bool,
) -> None:
    """Cleanup startup must neither block accepted tasks nor cause cold fallback."""
    client = Mock()
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        runtime_root_certificates=None,
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
    )
    client.list_namespaced_pod.return_value = {
        "items": [_ready_warm_pod(pool_key, config)]
    }
    response = _WarmExecResponse(acknowledge=acknowledge)
    if send_error:
        monkeypatch.setattr(
            response, "write_stdin", Mock(side_effect=OSError("lost stream"))
        )
    monkeypatch.setattr(
        importlib,
        "import_module",
        Mock(return_value=SimpleNamespace(stream=Mock(return_value=response))),
    )

    def _raise_runtime_error() -> None:
        raise RuntimeError

    monkeypatch.setattr(
        threading,
        "Thread",
        lambda **_kwargs: SimpleNamespace(start=_raise_runtime_error),
    )
    executor = KubernetesExecutor(client=client, config=config)

    result = executor.launch(
        _execution_spec(task_type=TaskType.AGENT_APP, insecure=True)
    )

    expected = (
        LaunchResultStatus.ACCEPTED if acknowledge else LaunchResultStatus.UNKNOWN
    )
    assert result.status == expected
    assert response.written == ([] if send_error else ["task-token\n"])
    assert not response.is_open()
    client.create_namespaced_secret.assert_not_called()
    client.create_namespaced_pod.assert_called_once()
    client.delete_namespaced_pod.assert_not_called()


def test_warm_dispatch_drains_stderr_until_the_child_exits() -> None:
    """Warm child stderr must not accumulate in the Kubernetes exec stream."""
    response = _WarmExecResponse(acknowledge=False, stderr="diagnostic output")
    response._all.write("x" * 1_000_000)  # pylint: disable=protected-access
    dispatch = warm_agentapp_executor.KubernetesWarmAgentAppDispatch(response)

    assert dispatch.wait_for_close()

    assert response.read_stderr_calls == 1
    assert response._all.getvalue() == ""  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "acknowledgement",
    [
        FLWR_TASK_TOKEN_STDIN_ACKNOWLEDGEMENT,
        FLWR_AGENTAPP_TOKEN_STDIN_ACKNOWLEDGEMENT,
    ],
)
def test_warm_dispatch_accepts_fragmented_acknowledgement(
    acknowledgement: str,
) -> None:
    """Current and rollout-compatible acknowledgements can span stdout frames."""
    response = Mock()
    response.read_stdout.side_effect = [acknowledgement[:8], acknowledgement[8:]]
    dispatch = warm_agentapp_executor.KubernetesWarmAgentAppDispatch(response)

    assert dispatch.wait_for_acceptance(1.0)
    assert response.read_stdout.call_count == 2
    response.read_all.assert_not_called()


@pytest.mark.parametrize("wait_error", [False, True])
def test_disconnected_warm_stream_does_not_kill_an_unconfirmed_task(
    wait_error: bool,
) -> None:
    """A missing exit status leaves retirement to process-aware reconciliation."""
    client = Mock()
    client.list_namespaced_pod.return_value = {"items": []}
    pool_key = _warm_executor_pool_key()
    config = _executor_config(
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
    )
    pool = kube._WarmExecutorPoolManager(  # pylint: disable=protected-access
        client, config, lambda: 0
    )
    pool._busy_pods.add("consumed")  # pylint: disable=protected-access
    dispatch = Mock()
    dispatch.wait_for_close.return_value = False
    if wait_error:
        dispatch.wait_for_close.side_effect = RuntimeError("disconnected")

    pool._wait_for_task_and_replace(  # pylint: disable=protected-access
        "consumed", pool_key, dispatch
    )

    client.delete_namespaced_pod.assert_not_called()
    assert "consumed" not in pool._busy_pods  # pylint: disable=protected-access
    dispatch.close.assert_called_once_with()


@pytest.mark.parametrize("state", ["S", "R", "Z", None])
def test_warm_idle_probe_requires_all_task_processes_to_have_exited(
    tmp_path: Path, state: str | None
) -> None:
    """The probe handles live tasks, zombies and disappearing processes."""
    process = tmp_path / "42"
    process.mkdir()
    if state is not None:
        (process / "stat").write_text(
            f"42 (task (child)) {state} 1 0 0", encoding="utf-8"
        )
    probe = warm_agentapp_executor._WARM_EXECUTOR_IDLE_CHECK.replace(  # pylint: disable=protected-access
        "Path('/proc')", f"Path({str(tmp_path)!r})"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert (result.stdout.strip() == "FLWR_WARM_EXECUTOR_IDLE") == (state == "Z")


def test_warm_pool_replaces_consumed_pod_and_cleans_up_idle_pods() -> None:
    """Consumed Pods are replaced without deleting active work during shutdown."""
    client = Mock()
    client.list_namespaced_pod.return_value = {"items": []}
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
    )
    pool = kube._WarmExecutorPoolManager(  # pylint: disable=protected-access
        client, config, lambda: 0
    )
    client.reset_mock()
    pool._busy_pods.add("consumed")  # pylint: disable=protected-access

    response = Mock(returncode=0)
    response.is_open.return_value = False
    response.close.side_effect = RuntimeError
    dispatch = warm_agentapp_executor.KubernetesWarmAgentAppDispatch(response)
    pool._wait_for_task_and_replace(  # pylint: disable=protected-access
        "consumed", pool_key, dispatch
    )

    client.delete_namespaced_pod.assert_called_once_with(
        name="consumed", namespace="flower-system", grace_period_seconds=0
    )
    client.create_namespaced_pod.assert_called_once()

    idle_pod = _ready_warm_pod(pool_key, config, name="idle")
    busy_pod = _ready_warm_pod(pool_key, config, name="busy")
    pool._busy_pods.add("busy")  # pylint: disable=protected-access
    client.reset_mock()
    client.list_namespaced_pod.return_value = {"items": [idle_pod, busy_pod]}

    pool.close()

    client.delete_namespaced_pod.assert_has_calls(
        [
            call(name="idle", namespace="flower-system", grace_period_seconds=0),
        ]
    )
    assert client.delete_namespaced_pod.call_count == 1


def test_warm_pool_preserves_surviving_tasks_until_their_processes_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restart, config changes and shutdown must preserve a consumed live Pod."""
    client = Mock()
    client.list_namespaced_pod.return_value = {"items": []}
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
    )
    pool = kube._WarmExecutorPoolManager(  # pylint: disable=protected-access
        client, config, lambda: 0
    )
    consumed_pod = _ready_warm_pod(pool_key, config, name="consumed")
    consumed_pod["metadata"]["annotations"][
        kube._WARM_EXECUTOR_CONSUMED_ANNOTATION  # pylint: disable=protected-access
    ] = "true"
    # A changed Pod configuration must not override task preservation.
    consumed_pod["metadata"]["annotations"][
        WARM_EXECUTOR_CONFIGURATION_ANNOTATION
    ] = "old"
    response = Mock()
    response.read_all.return_value = ""
    stream = Mock(return_value=response)
    monkeypatch.setattr(
        importlib, "import_module", Mock(return_value=SimpleNamespace(stream=stream))
    )
    client.reset_mock()
    client.list_namespaced_pod.return_value = {"items": [consumed_pod]}

    assert pool._take_ready_pod(pool_key) is None  # pylint: disable=protected-access
    pool.ensure_capacity()

    client.delete_namespaced_pod.assert_not_called()
    response.close.assert_called_once_with()
    # A disconnected stream does not establish that the process has stopped.
    stream.side_effect = RuntimeError("unavailable")
    pool.ensure_capacity()
    client.delete_namespaced_pod.assert_not_called()
    stream.side_effect = None
    pool.close()
    client.delete_namespaced_pod.assert_not_called()

    # Even when pools are disabled, reconciliation retires the Pod after exit.
    response.read_all.return_value = "FLWR_WARM_EXECUTOR_IDLE\n"
    config = _executor_config(warm_executor_owner="superexec-a")
    pool = kube._WarmExecutorPoolManager(  # pylint: disable=protected-access
        client, config, lambda: 0
    )
    pool.ensure_capacity()
    client.delete_namespaced_pod.assert_called_once_with(
        name="consumed", namespace="flower-system", grace_period_seconds=0
    )


def test_disabled_warm_pools_retire_owned_pods() -> None:
    """An owner cleans up its Pods after warm pools are disabled."""
    client = Mock()
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(warm_executor_owner="superexec-a")
    client.list_namespaced_pod.return_value = {
        "items": [_ready_warm_pod(pool_key, config, name="obsolete")]
    }

    executor = KubernetesExecutor(client=client, config=config)
    client.delete_namespaced_pod.assert_not_called()
    executor.reconcile()

    client.delete_namespaced_pod.assert_called_once_with(
        name="obsolete", namespace="flower-system", grace_period_seconds=0
    )


def test_warm_pool_retries_retirement_without_replacing_pending_pod() -> None:
    """A failed retirement is retried without creating more warm Pods."""
    client = Mock()
    client.list_namespaced_pod.return_value = {"items": []}
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
    )
    pool = kube._WarmExecutorPoolManager(  # pylint: disable=protected-access
        client, config, lambda: 0
    )
    client.reset_mock()
    pool._busy_pods.add("consumed")  # pylint: disable=protected-access
    client.delete_namespaced_pod.side_effect = _KubernetesApiError(500, "error")

    pool._wait_for_task_and_replace(  # pylint: disable=protected-access
        "consumed",
        pool_key,
        warm_agentapp_executor.KubernetesWarmAgentAppDispatch(_WarmExecResponse(False)),
    )

    assert "consumed" in pool._busy_pods  # pylint: disable=protected-access
    assert "consumed" in pool._retiring_pods  # pylint: disable=protected-access
    client.create_namespaced_pod.assert_not_called()

    client.list_namespaced_pod.return_value = {
        "items": [_ready_warm_pod(pool_key, config, name="consumed")]
    }
    pool.ensure_capacity()

    assert client.delete_namespaced_pod.call_count == 2
    client.create_namespaced_pod.assert_not_called()

    client.delete_namespaced_pod.side_effect = None
    pool.ensure_capacity()

    assert "consumed" not in pool._busy_pods  # pylint: disable=protected-access
    assert "consumed" not in pool._retiring_pods  # pylint: disable=protected-access


def test_warm_pool_reserves_capacity_for_cold_fallback() -> None:
    """Replacing a missing warm Pod leaves room for the cold task Pod."""
    client = Mock()
    client.list_namespaced_pod.return_value = {"items": []}
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        active_pod_budget=3,
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
    )
    pool = kube._WarmExecutorPoolManager(  # pylint: disable=protected-access
        client, config, lambda: 2
    )
    client.reset_mock()

    pool._ensure_pool_capacity(  # pylint: disable=protected-access
        WarmExecutorPoolConfig(key=pool_key, size=1), reserved_pod_capacity=1
    )

    client.create_namespaced_pod.assert_not_called()


def test_warm_pool_replaces_missing_pod_with_reserved_cold_capacity() -> None:
    """A cold reservation still leaves capacity for a missing warm Pod."""
    client = Mock()
    client.list_namespaced_pod.return_value = {"items": []}
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        active_pod_budget=2,
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
    )
    pool = kube._WarmExecutorPoolManager(  # pylint: disable=protected-access
        client, config, lambda: 0
    )
    client.reset_mock()

    pool._ensure_pool_capacity(  # pylint: disable=protected-access
        WarmExecutorPoolConfig(key=pool_key, size=1), reserved_pod_capacity=1
    )

    client.create_namespaced_pod.assert_called_once()


def test_reconciliation_refills_after_terminating_pods_release_capacity() -> None:
    """Periodic maintenance refills an empty pool without another task launch."""
    client = Mock()
    client.list_namespaced_secret.return_value = {"items": []}
    active_count = 2
    client.list_namespaced_pod.side_effect = lambda _namespace, label_selector: {
        "items": (
            []
            if "warm-executor-owner" in label_selector
            else [_pod("Running")] * active_count
        )
    }
    config = _executor_config(
        active_pod_budget=2,
        warm_executor_owner="superexec-a",
        warm_executor_pools=(
            WarmExecutorPoolConfig(key=_warm_executor_pool_key(), size=1),
        ),
    )
    executor = KubernetesExecutor(client=client, config=config)
    client.create_namespaced_pod.assert_not_called()

    executor.reconcile()
    client.create_namespaced_pod.assert_not_called()

    active_count = 1
    executor.reconcile()
    client.create_namespaced_pod.assert_called_once()


def test_warm_pool_reconciles_obsolete_and_excess_idle_pods() -> None:
    """Reconciliation removes stale Pods and retains Ready excess-pool Pods."""
    client = Mock()
    client.list_namespaced_pod.return_value = {"items": []}
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
    )
    pool = kube._WarmExecutorPoolManager(  # pylint: disable=protected-access
        client, config, lambda: 0
    )
    obsolete_pod = _ready_warm_pod(
        pool_key,
        _executor_config(
            env=[{"name": "OLD_SETTING", "value": "1"}],
            warm_executor_owner="superexec-a",
            warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
        ),
        name="obsolete",
    )
    client.reset_mock()
    pending_pod = _ready_warm_pod(pool_key, config, name="pending")
    pending_pod["status"] = {"phase": "Pending", "conditions": []}
    client.list_namespaced_pod.return_value = {
        "items": [
            pending_pod,
            _ready_warm_pod(pool_key, config, name="keep"),
            obsolete_pod,
        ]
    }

    pool.ensure_capacity()

    client.delete_namespaced_pod.assert_has_calls(
        [
            call(name="pending", namespace="flower-system", grace_period_seconds=0),
            call(name="obsolete", namespace="flower-system", grace_period_seconds=0),
        ],
        any_order=True,
    )
    client.create_namespaced_pod.assert_not_called()

    client.reset_mock()
    client.list_namespaced_pod.return_value = {"items": [obsolete_pod]}
    client.delete_namespaced_pod.side_effect = _KubernetesApiError(500, "error")
    pool.ensure_capacity()

    assert "obsolete" in pool._retiring_pods  # pylint: disable=protected-access
    client.create_namespaced_pod.assert_not_called()


def test_launch_retries_warm_dispatch_after_readiness_recovers_at_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness recovery must unblock a task admitted before readiness was lost."""
    client = Mock()
    sleep = Mock()
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        active_pod_budget=2,
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
        sleep=sleep,
    )
    warm_pod = _ready_warm_pod(pool_key, config)

    def _list_pods(_namespace: str, label_selector: str) -> dict[str, Any]:
        if "warm-executor-owner" in label_selector:
            return {"items": [warm_pod]}
        return {"items": [warm_pod, _pod("Running")]}

    client.list_namespaced_pod.side_effect = _list_pods
    client.list_namespaced_secret.return_value = {"items": []}
    executor = KubernetesExecutor(client=client, config=config)

    executor.wait_for_capacity(TaskType.AGENT_APP, insecure=True)

    sleep.assert_not_called()
    warm_pod["status"]["conditions"][0]["status"] = "False"

    def _restore_readiness(_interval: float) -> None:
        assert sleep.call_count == 1, "Launch kept waiting after readiness recovered"
        warm_pod["status"]["conditions"][0]["status"] = "True"

    sleep.side_effect = _restore_readiness
    response = _WarmExecResponse()
    stream = Mock(return_value=response)
    monkeypatch.setattr(
        importlib, "import_module", Mock(return_value=SimpleNamespace(stream=stream))
    )
    monkeypatch.setattr(threading, "Thread", Mock())

    result = executor.launch(
        _execution_spec(task_type=TaskType.AGENT_APP, insecure=True)
    )

    assert result.status == LaunchResultStatus.ACCEPTED
    sleep.assert_called_once_with(1.0)
    stream.assert_called_once()
    assert response.written == ["task-token\n"]
    client.create_namespaced_pod.assert_not_called()
    client.create_namespaced_secret.assert_not_called()
    client.delete_namespaced_pod.assert_not_called()


def test_wait_for_capacity_reserves_cold_capacity_for_task_specific_ca() -> None:
    """A task-time Runtime CA requires cold capacity."""
    client = Mock()
    sleep = Mock()
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        active_pod_budget=2,
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
        runtime_root_certificates=None,
        sleep=sleep,
    )
    warm_pod = _ready_warm_pod(pool_key, config)
    active_pod_count = 2

    def _list_pods(_namespace: str, label_selector: str) -> dict[str, Any]:
        if "warm-executor-owner" in label_selector:
            return {"items": [warm_pod]}
        return {"items": [warm_pod, _pod("Running")][:active_pod_count]}

    def _release_capacity(_interval: float) -> None:
        nonlocal active_pod_count
        active_pod_count = 1

    sleep.side_effect = _release_capacity
    client.list_namespaced_pod.side_effect = _list_pods
    client.list_namespaced_secret.return_value = {"items": []}
    executor = KubernetesExecutor(client=client, config=config)

    executor.wait_for_capacity(
        TaskType.AGENT_APP,
        insecure=False,
        root_certificates_path="/tmp/task-specific-ca.pem",
    )

    sleep.assert_called_once_with(1.0)


@pytest.mark.parametrize("budget", [None, 1])
def test_cold_fallback_retries_retirement_while_waiting_for_capacity(
    budget: int | None,
) -> None:
    """Transient retirement failures must not stall the cold-capacity wait."""
    client = Mock()
    client.list_namespaced_pod.return_value = {"items": []}
    client.list_namespaced_secret.return_value = {"items": []}
    executor = KubernetesExecutor(
        client=client,
        config=_executor_config(active_pod_budget=budget, sleep=Mock()),
    )
    manager = Mock()
    manager.sweep_completed_pods.side_effect = lambda sweep: sweep()
    executor._warm_executor_pool_manager = manager  # pylint: disable=protected-access
    client.list_namespaced_pod.side_effect = [
        {"items": []},  # Completed-Pod sweep
        {"items": [_pod("Running")]},
        {"items": [_pod("Running")]},
        {"items": []},
        {"items": []},  # Completed-Pod sweep after waiting
    ]

    executor._wait_for_capacity(  # pylint: disable=protected-access
        None,
        allow_warm_dispatch=False,
        reconcile_warm_pools=False,
    )

    assert manager.retry_retiring_pods.call_count == (1 if budget is None else 3)
    manager.ensure_capacity.assert_not_called()


def test_active_budget_counts_surviving_owned_pods_after_labels_change() -> None:
    """A task preserved across restart must still consume the owner's budget."""
    client = Mock()
    client.list_namespaced_pod.side_effect = [
        {"items": [_pod("Running", name="current")]},
        {"items": [_pod("Running", name="current"), _pod("Running", name="old")]},
    ]
    executor = KubernetesExecutor(
        client=client, config=_executor_config(warm_executor_owner="superexec-a")
    )

    assert executor._active_pod_count() == 2  # pylint: disable=protected-access


def test_wait_for_capacity_reserves_space_for_a_cold_task() -> None:
    """Reconciling warm capacity must not consume the cold task's last slot."""
    client = Mock()
    sleep = Mock()
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        active_pod_budget=3,
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
        sleep=sleep,
    )
    active_pod_count = 3

    def _list_pods(_namespace: str, label_selector: str) -> dict[str, Any]:
        if "warm-executor-owner" in label_selector:
            return {"items": []}
        return {"items": [_pod("Running")] * active_pod_count}

    client.list_namespaced_pod.side_effect = _list_pods
    client.list_namespaced_secret.return_value = {"items": []}
    executor = KubernetesExecutor(client=client, config=config)
    client.create_namespaced_pod.reset_mock()
    active_pod_count = 2

    executor.wait_for_capacity(TaskType.MODEL)

    client.create_namespaced_pod.assert_not_called()
    sleep.assert_not_called()


def test_warm_pool_cannot_fill_the_active_pod_budget() -> None:
    """Static warm capacity must leave room for SuperExec's capacity gate."""
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )

    with pytest.raises(ValueError, match="active_pod_budget"):
        _executor_config(
            active_pod_budget=1,
            warm_executor_owner="superexec-a",
            warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
        )


def test_warm_executor_owner_selector_ignores_mutable_pool_labels() -> None:
    """Warm ownership must survive caller-label and resource-pool changes."""
    config = _executor_config(
        labels={"flower.ai/team": "platform"},
        resource_pool="gpu-pool",
        warm_executor_owner="superexec-a",
    )

    selector = (
        kube._warm_executor_owner_label_selector(  # pylint: disable=protected-access
            config
        )
    )

    assert selector == (
        "app.kubernetes.io/component=taskexecutor,app.kubernetes.io/name=flower,"
        "flower.ai/warm-executor=true,flower.ai/warm-executor-owner=superexec-a"
    )


def test_ownerless_sweeper_keeps_warm_secret_before_pod_creation() -> None:
    """A cold-only sweeper must leave another owner's pending warm creation alone."""
    client = Mock()
    secret = kube._build_warm_executor_root_certificates_secret(  # pylint: disable=protected-access
        _warm_executor_pool_key(),
        _executor_config(warm_executor_owner="superexec-a"),
        "creating",
    )
    client.list_namespaced_pod.return_value = {"items": []}
    client.list_namespaced_secret.return_value = {"items": [secret]}

    CompletedPodSweeper(
        client=client, config=_executor_config(warm_executor_owner=None)
    ).sweep()

    client.delete_namespaced_secret.assert_not_called()


def test_sweeper_cleans_orphaned_warm_secret_after_labels_change() -> None:
    """Warm trust cleanup must use the stable owner selector after a restart."""
    client = Mock()
    pool_key = _warm_executor_pool_key()
    previous_config = _executor_config(
        labels={"flower.ai/team": "previous"},
        resource_pool="previous-pool",
        warm_executor_owner="superexec-a",
    )
    config = _executor_config(
        labels={"flower.ai/team": "current"},
        resource_pool="current-pool",
        warm_executor_owner="superexec-a",
    )
    secret = kube._build_warm_executor_root_certificates_secret(  # pylint: disable=protected-access
        pool_key, previous_config, "orphan"
    )
    client.list_namespaced_pod.return_value = {"items": []}
    client.list_namespaced_secret.side_effect = [
        {"items": []},
        {"items": [secret]},
    ]

    CompletedPodSweeper(client=client, config=config).sweep()

    client.delete_namespaced_secret.assert_called_once_with(
        name="flwr-taskexecutor-warm-orphan-runtime-ca",
        namespace="flower-system",
    )


# pylint: disable=too-many-locals
def test_sweeper_waits_for_warm_pod_creation() -> None:
    """Completed-Pod cleanup must not observe a half-created warm Pod."""
    client = Mock()
    objects: dict[str, list[dict[str, Any]]] = {"pods": [], "secrets": []}
    creation_started = threading.Event()
    release_creation = threading.Event()
    sweep_started = threading.Event()
    pool_key = _warm_executor_pool_key(
        runtime_image="ghcr.io/flwrlabs/taskexecutor:dev"
    )
    config = _executor_config(
        warm_executor_owner="superexec-a",
        warm_executor_pools=(WarmExecutorPoolConfig(key=pool_key, size=1),),
    )

    def _list_pods(_namespace: str, **_kwargs: object) -> dict[str, object]:
        return {"items": objects["pods"]}

    def _list_secrets(_namespace: str, **_kwargs: object) -> dict[str, object]:
        sweep_started.set()
        return {"items": objects["secrets"]}

    def _create_secret(_namespace: str, secret: object) -> None:
        objects["secrets"].append(_as_dict(secret))
        creation_started.set()
        assert release_creation.wait(timeout=1.0)

    def _create_pod(_namespace: str, pod: object) -> None:
        objects["pods"].append(_as_dict(pod))

    client.list_namespaced_pod.side_effect = _list_pods
    client.list_namespaced_secret.side_effect = _list_secrets
    client.create_namespaced_secret.side_effect = _create_secret
    client.create_namespaced_pod.side_effect = _create_pod
    executor = KubernetesExecutor(client=client, config=config)
    manager = executor._warm_executor_pool_manager  # pylint: disable=protected-access
    assert manager is not None
    creator = threading.Thread(target=manager.ensure_capacity)
    sweeper = threading.Thread(
        target=executor._sweep_completed_pods  # pylint: disable=protected-access
    )

    creator.start()
    try:
        assert creation_started.wait(timeout=1.0)
        sweeper.start()
        assert not sweep_started.wait(timeout=0.1)
    finally:
        release_creation.set()
        creator.join(timeout=1.0)
        if sweeper.ident is not None:
            sweeper.join(timeout=1.0)

    assert not creator.is_alive()
    assert not sweeper.is_alive()
    assert sweep_started.is_set()
    client.delete_namespaced_secret.assert_not_called()


def test_build_taskexecutor_pod_includes_configured_volumes() -> None:
    """Test configured Pod volumes and container volume mounts are included."""
    spec = _execution_spec()
    config = _executor_config(
        volumes=[
            {
                "name": "shmem",
                "emptyDir": {"medium": "Memory", "sizeLimit": "10Gi"},
            }
        ],
        volume_mounts=[{"name": "shmem", "mountPath": "/dev/shm"}],
    )

    pod = _as_dict(
        _build_taskexecutor_pod(
            spec, config, _appio_root_certificates(spec, config), _LAUNCH_ATTEMPT_ID
        )
    )
    container = pod["spec"]["containers"][0]

    assert {
        "name": "shmem",
        "emptyDir": {"medium": "Memory", "sizeLimit": "10Gi"},
    } in pod["spec"]["volumes"]
    assert {"name": "shmem", "mountPath": "/dev/shm"} in container["volumeMounts"]


@pytest.mark.parametrize(
    ("config_overrides", "expected_message"),
    [
        ({"volumes": {"name": "shmem"}}, "volumes must be a list"),
        ({"volumes": ["shmem"]}, "volume entries must be mappings"),
        (
            {"volumes": [{"name": "appio-credentials", "emptyDir": {}}]},
            "appio-credentials",
        ),
        (
            {"volumes": [{"name": "warm-executor-ready", "emptyDir": {}}]},
            "warm-executor-ready",
        ),
        (
            {"volumes": [{"name": "secret", "secret": {"secretName": "api-key"}}]},
            "secret volumes",
        ),
        (
            {
                "volumes": [
                    {
                        "name": "projected-secret",
                        "projected": {
                            "sources": [
                                {"secret": {"name": "api-key"}},
                            ]
                        },
                    }
                ]
            },
            "projected secret",
        ),
        (
            {
                "volumes": [
                    {
                        "name": "service-account-token",
                        "projected": {
                            "sources": [
                                {"serviceAccountToken": {"path": "token"}},
                            ]
                        },
                    }
                ]
            },
            "serviceAccountToken",
        ),
        ({"volume_mounts": {"name": "shmem"}}, "volume mounts must be a list"),
        ({"volume_mounts": ["shmem"]}, "volume mount entries must be mappings"),
        (
            {
                "volume_mounts": [
                    {"name": "appio-credentials", "mountPath": "/credentials"}
                ]
            },
            "appio-credentials",
        ),
        (
            {"volume_mounts": [{"name": "warm-executor-ready", "mountPath": "/ready"}]},
            "warm-executor-ready",
        ),
        (
            {
                "volume_mounts": [
                    {"name": "credentials", "mountPath": APPIO_CREDENTIALS_MOUNT_PATH}
                ]
            },
            "mount path",
        ),
        (
            {
                "volume_mounts": [
                    {"name": "ready", "mountPath": WARM_EXECUTOR_READY_DIRECTORY}
                ]
            },
            "mount path",
        ),
        (
            {
                "volume_mounts": [
                    {"name": "ready-file", "mountPath": WARM_EXECUTOR_READY_FILE}
                ]
            },
            "mount path",
        ),
    ],
)
def test_kubernetes_executor_config_rejects_invalid_volumes(
    config_overrides: dict[str, object], expected_message: str
) -> None:
    """Test TaskExecutor volumes reject invalid entries."""
    with pytest.raises(ValueError, match=expected_message):
        _executor_config(**config_overrides)


def test_build_taskexecutor_pod_supports_explicit_env() -> None:
    """Test Pod construction includes validated explicit container env."""
    pod = _as_dict(
        _build_taskexecutor_pod(
            _execution_spec(),
            _executor_config(
                env=[
                    {
                        "name": "FLWR_MODEL_API_ENDPOINT",
                        "value": "http://proxy/v1/responses",
                    },
                    {
                        "name": "FLWR_WEB_SEARCH_ENDPOINT",
                        "value": "http://proxy/v1/web-search",
                    },
                    {"name": "UV_INDEX_URL", "value": "https://pypi.org/simple"},
                ]
            ),
            "root-ca",
            _LAUNCH_ATTEMPT_ID,
        )
    )

    assert pod["spec"]["containers"][0]["env"] == [
        {"name": "FLWR_MODEL_API_ENDPOINT", "value": "http://proxy/v1/responses"},
        {"name": "FLWR_WEB_SEARCH_ENDPOINT", "value": "http://proxy/v1/web-search"},
        {"name": "UV_INDEX_URL", "value": "https://pypi.org/simple"},
    ]


@pytest.mark.parametrize(
    "env_name",
    [
        "FLWR_MODEL_API_KEY",
        "BRAVE_API_KEY",
        "TAVILY_API_KEY",
        "EXA_API_KEY",
    ],
)
def test_kubernetes_executor_config_rejects_provider_key_env_names(
    env_name: str,
) -> None:
    """Test TaskExecutor env rejects exact provider key names."""
    with pytest.raises(ValueError, match="TaskExecutor env name"):
        _executor_config(env=[{"name": env_name, "value": "not-forwarded"}])


@pytest.mark.parametrize(
    ("env_entry", "expected_message"),
    [
        (
            {"name": " FLWR_MODEL_API_ENDPOINT ", "value": "not-forwarded"},
            "valid Kubernetes",
        ),
        (
            {"name": "FLWR-MODEL-API-ENDPOINT", "value": "not-forwarded"},
            "valid Kubernetes",
        ),
        ({"name": "1INVALID", "value": "not-forwarded"}, "valid Kubernetes"),
        (
            {
                "name": "FLWR_MODEL_API_ENDPOINT",
                "valueFrom": {"secretKeyRef": {"name": "proxy", "key": "url"}},
            },
            "valueFrom",
        ),
    ],
)
def test_kubernetes_executor_config_rejects_invalid_env_entries(
    env_entry: object, expected_message: str
) -> None:
    """Test TaskExecutor env rejects invalid entries."""
    with pytest.raises(ValueError, match=expected_message):
        _executor_config(env=[env_entry])


def test_build_taskexecutor_pod_supports_clientapp_insecure_args() -> None:
    """Test Pod construction for insecure ClientApp launch args."""
    pod = _as_dict(
        _build_taskexecutor_pod(
            _execution_spec(task_type=TaskType.CLIENT_APP, insecure=True),
            _executor_config(runtime_root_certificates=None),
            None,
            _LAUNCH_ATTEMPT_ID,
        )
    )

    assert pod["spec"]["containers"][0]["command"] == ["flwr-clientapp"]
    assert pod["spec"]["containers"][0]["args"] == [
        "--runtime-api-address",
        "appio.example.com:9092",
        "--token-file",
        APPIO_TOKEN_FILE_PATH,
        "--insecure",
    ]


def test_build_taskexecutor_pod_supports_secure_default_trust_store() -> None:
    """Test secure Pod args can rely on container default trust store."""
    spec = _execution_spec()
    config = _executor_config(runtime_root_certificates=None)
    runtime_root_certificates = _appio_root_certificates(spec, config)

    secret = _as_dict(
        _build_appio_credentials_secret(
            spec, config, runtime_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )
    pod = _as_dict(
        _build_taskexecutor_pod(
            spec, config, runtime_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )

    assert secret["stringData"] == {"token": "task-token"}
    assert pod["spec"]["containers"][0]["args"] == [
        "--runtime-api-address",
        "appio.example.com:9092",
        "--token-file",
        APPIO_TOKEN_FILE_PATH,
    ]


def test_build_taskexecutor_objects_use_execution_spec_root_certificates(
    tmp_path: Path,
) -> None:
    """Test Pod and Secret use root certificates forwarded in ExecutionSpec."""
    root_certificates_path = tmp_path / "appio-ca.pem"
    root_certificates_path.write_text("spec-root-ca", encoding="utf-8")
    spec = _execution_spec(root_certificates_path=str(root_certificates_path))
    config = _executor_config(runtime_root_certificates=None)
    runtime_root_certificates = _appio_root_certificates(spec, config)

    secret = _as_dict(
        _build_appio_credentials_secret(
            spec, config, runtime_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )
    pod = _as_dict(
        _build_taskexecutor_pod(
            spec, config, runtime_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )

    assert secret["stringData"] == {"token": "task-token", "ca.crt": "spec-root-ca"}
    assert pod["spec"]["containers"][0]["args"] == [
        "--runtime-api-address",
        "appio.example.com:9092",
        "--token-file",
        APPIO_TOKEN_FILE_PATH,
        "--root-certificates",
        APPIO_ROOT_CERTIFICATES_FILE_PATH,
    ]


def test_build_taskexecutor_objects_expand_user_root_certificates_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test ExecutionSpec root certificates path supports shell-style home paths."""
    root_certificates_path = tmp_path / "appio-ca.pem"
    root_certificates_path.write_text("home-root-ca", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    spec = _execution_spec(root_certificates_path="~/appio-ca.pem")
    config = _executor_config(runtime_root_certificates=None)
    runtime_root_certificates = _appio_root_certificates(spec, config)

    secret = _as_dict(
        _build_appio_credentials_secret(
            spec, config, runtime_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )
    pod = _as_dict(
        _build_taskexecutor_pod(
            spec, config, runtime_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )

    assert secret["stringData"] == {"token": "task-token", "ca.crt": "home-root-ca"}
    assert pod["spec"]["containers"][0]["args"] == [
        "--runtime-api-address",
        "appio.example.com:9092",
        "--token-file",
        APPIO_TOKEN_FILE_PATH,
        "--root-certificates",
        APPIO_ROOT_CERTIFICATES_FILE_PATH,
    ]


def test_build_taskexecutor_pod_supports_simulation_args() -> None:
    """Test Pod construction for Simulation launch args."""
    pod = _as_dict(
        _build_taskexecutor_pod(
            _execution_spec(task_type=TaskType.SIMULATION),
            _executor_config(),
            "root-ca",
            _LAUNCH_ATTEMPT_ID,
        )
    )

    assert pod["spec"]["containers"][0]["command"] == ["flwr-simulation"]
    assert pod["spec"]["containers"][0]["args"] == [
        "--runtime-api-address",
        "appio.example.com:9092",
        "--token-file",
        APPIO_TOKEN_FILE_PATH,
        "--root-certificates",
        APPIO_ROOT_CERTIFICATES_FILE_PATH,
    ]


def test_build_taskexecutor_pod_supports_optional_container_config() -> None:
    """Test Pod construction includes optional Kubernetes container config."""
    pod = _as_dict(
        _build_taskexecutor_pod(
            _execution_spec(runtime_dependency_install=True),
            _executor_config(
                image_pull_policy="IfNotPresent",
                service_account_name="flower-superexec",
            ),
            "root-ca",
            _LAUNCH_ATTEMPT_ID,
        )
    )
    container = pod["spec"]["containers"][0]

    assert container["imagePullPolicy"] == "IfNotPresent"
    assert "--allow-runtime-dependency-installation" in container["args"]
    assert pod["spec"]["serviceAccountName"] == "flower-superexec"


def test_build_taskexecutor_pod_supports_resources_and_placement() -> None:
    """Test Pod construction includes resource and placement inputs."""
    resources = {
        "requests": {"cpu": "500m", "memory": "1Gi"},
        "limits": {"cpu": "1", "memory": "2Gi"},
    }
    node_selector = {"flower.ai/node-pool": "taskexecutors"}
    tolerations = [
        {
            "key": "flower.ai/taskexecutor",
            "operator": "Equal",
            "value": "true",
            "effect": "NoSchedule",
        }
    ]
    affinity: dict[str, Any] = {
        "podAntiAffinity": {"preferredDuringSchedulingIgnoredDuringExecution": []}
    }

    pod = _as_dict(
        _build_taskexecutor_pod(
            _execution_spec(),
            _executor_config(
                resources=resources,
                node_selector=node_selector,
                tolerations=tolerations,
                affinity=affinity,
                priority_class_name="taskexecutor-priority",
            ),
            "root-ca",
            _LAUNCH_ATTEMPT_ID,
        )
    )

    assert pod["spec"]["containers"][0]["resources"] == resources
    assert pod["spec"]["nodeSelector"] == node_selector
    assert pod["spec"]["tolerations"] == tolerations
    assert pod["spec"]["affinity"] == affinity
    assert pod["spec"]["priorityClassName"] == "taskexecutor-priority"


def test_build_taskexecutor_pod_supports_labels_annotations_and_security() -> None:
    """Test Pod construction includes object metadata and security fields."""
    pod_security_context = {
        "runAsNonRoot": True,
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    container_security_context = {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
    }
    config = _executor_config(
        labels={"flower.ai/team": "platform"},
        annotations={"flower.ai/owner": "superexec"},
        resource_pool="gpu-pool",
        pod_security_context=pod_security_context,
        container_security_context=container_security_context,
    )

    spec = _execution_spec()
    runtime_root_certificates = _appio_root_certificates(spec, config)
    secret = _as_dict(
        _build_appio_credentials_secret(
            spec, config, runtime_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )
    pod = _as_dict(
        _build_taskexecutor_pod(
            spec, config, runtime_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )

    expected_labels = {
        "app.kubernetes.io/name": "flower",
        "app.kubernetes.io/component": "taskexecutor",
        "flower.ai/superexec-task-id": "123",
        "flower.ai/task-type": "flwr-serverapp",
        LAUNCH_ATTEMPT_LABEL: _LAUNCH_ATTEMPT_ID,
        "flower.ai/resource-pool": "gpu-pool",
        "flower.ai/team": "platform",
    }
    assert secret["metadata"]["labels"] == expected_labels
    assert secret["metadata"]["annotations"] == {"flower.ai/owner": "superexec"}
    assert pod["metadata"]["labels"] == expected_labels
    assert pod["metadata"]["annotations"] == {"flower.ai/owner": "superexec"}
    assert pod["spec"]["securityContext"] == pod_security_context
    assert pod["spec"]["containers"][0]["securityContext"] == container_security_context


def test_wait_for_capacity_returns_below_budget_without_sleeping() -> None:
    """Test capacity wait returns immediately when the active Pod count fits."""
    client = Mock()
    client.list_namespaced_pod.side_effect = [
        {"items": []},
        {"items": [_pod("Running")]},
    ]
    client.list_namespaced_secret.return_value = {"items": []}
    sleep = Mock()
    config = _executor_config(
        labels={"flower.ai/team": "platform"},
        resource_pool="gpu-pool",
        active_pod_budget=2,
        capacity_poll_interval=3.0,
        sleep=sleep,
    )

    KubernetesExecutor(client=client, config=config).wait_for_capacity()

    assert client.list_namespaced_pod.call_count == 2
    client.list_namespaced_secret.assert_called_once()
    sleep.assert_not_called()


def test_wait_for_capacity_sleeps_and_polls_again_at_budget() -> None:
    """Test capacity wait sleeps when the active Pod count reaches the budget."""
    client = Mock()
    client.list_namespaced_pod.side_effect = [
        {"items": []},
        {"items": [_pod("Pending")]},
        {"items": []},
        {"items": []},
    ]
    client.list_namespaced_secret.return_value = {"items": []}
    sleep = Mock()
    config = _executor_config(
        resource_pool="gpu-pool",
        active_pod_budget=1,
        capacity_poll_interval=3.0,
        sleep=sleep,
    )

    KubernetesExecutor(client=client, config=config).wait_for_capacity()

    assert client.list_namespaced_pod.call_count == 4
    assert client.list_namespaced_secret.call_count == 2
    sleep.assert_called_once_with(3.0)


def test_wait_for_capacity_sweeps_after_waiting_for_capacity_to_open() -> None:
    """Test completed Pod cleanup runs after a blocking capacity wait opens."""
    client = Mock()
    labels = _task_labels(123)
    client.list_namespaced_pod.side_effect = [
        {"items": []},
        {"items": [_pod("Running", labels=labels)]},
        {"items": [_pod("Succeeded", labels=labels)]},
        {"items": [_pod("Succeeded", labels=labels)]},
    ]
    client.list_namespaced_secret.side_effect = [
        {"items": []},
        {"items": [_secret(_SECRET_NAME, labels)]},
    ]
    sleep = Mock()
    config = _executor_config(
        resource_pool="gpu-pool",
        active_pod_budget=1,
        capacity_poll_interval=3.0,
        sleep=sleep,
    )

    KubernetesExecutor(client=client, config=config).wait_for_capacity()

    assert client.list_namespaced_pod.call_count == 4
    assert client.list_namespaced_secret.call_count == 2
    sleep.assert_called_once_with(3.0)
    client.delete_namespaced_pod.assert_called_once_with(
        name=_POD_NAME,
        namespace="flower-system",
        grace_period_seconds=0,
    )
    client.delete_namespaced_secret.assert_called_once_with(
        name=_SECRET_NAME, namespace="flower-system"
    )


def test_wait_for_capacity_counts_pending_running_and_terminating_active_pods() -> None:
    """Test active Pod counting includes non-terminal terminating Pods."""
    client = Mock()
    client.list_namespaced_pod.side_effect = [
        {"items": []},
        {
            "items": [
                _pod("Pending"),
                _pod("Running"),
                _pod("Pending", deletion_timestamp="2026-05-26T18:30:00Z"),
                _pod("Running", deletion_timestamp="2026-05-26T18:31:00Z"),
            ]
        },
        {"items": []},
    ]
    client.list_namespaced_secret.return_value = {"items": []}
    sleep = Mock()
    config = _executor_config(
        resource_pool="gpu-pool",
        active_pod_budget=4,
        sleep=sleep,
    )

    KubernetesExecutor(client=client, config=config).wait_for_capacity()

    sleep.assert_called_once_with(1.0)


def test_wait_for_capacity_ignores_terminal_pods_even_when_terminating() -> None:
    """Test terminal Pods do not count after cleanup requests deletion."""
    client = Mock()
    client.list_namespaced_pod.side_effect = [
        {"items": []},
        {
            "items": [
                _pod("Succeeded", deletion_timestamp="2026-05-26T18:30:00Z"),
                _pod("Failed", deletion_timestamp="2026-05-26T18:31:00Z"),
            ]
        },
    ]
    client.list_namespaced_secret.return_value = {"items": []}
    sleep = Mock()
    config = _executor_config(
        resource_pool="gpu-pool",
        active_pod_budget=1,
        sleep=sleep,
    )

    KubernetesExecutor(client=client, config=config).wait_for_capacity()

    sleep.assert_not_called()


def test_wait_for_capacity_sweeps_terminal_pods_before_capacity_check() -> None:
    """Test capacity wait runs completed Pod cleanup opportunistically."""
    client = Mock()
    labels = _task_labels(123)
    client.list_namespaced_pod.side_effect = [
        {"items": [_pod("Succeeded", labels=labels)]},
        {"items": []},
    ]
    client.list_namespaced_secret.return_value = {
        "items": [_secret(_SECRET_NAME, labels)]
    }
    config = _executor_config(resource_pool="gpu-pool", active_pod_budget=1)

    KubernetesExecutor(client=client, config=config).wait_for_capacity()

    client.delete_namespaced_pod.assert_called_once_with(
        name=_POD_NAME,
        namespace="flower-system",
        grace_period_seconds=0,
    )
    client.delete_namespaced_secret.assert_called_once_with(
        name=_SECRET_NAME, namespace="flower-system"
    )


def test_wait_for_capacity_throttles_completed_pod_sweeps() -> None:
    """Test capacity wait does not sweep more often than the internal interval."""
    client = Mock()
    client.list_namespaced_pod.return_value = {"items": []}
    client.list_namespaced_secret.return_value = {"items": []}
    config = _executor_config(
        resource_pool="gpu-pool",
        active_pod_budget=1,
        monotonic=Mock(side_effect=[0.0, _COMPLETED_POD_SWEEP_INTERVAL_SECONDS - 1.0]),
    )
    executor = KubernetesExecutor(client=client, config=config)

    executor.wait_for_capacity()
    executor.wait_for_capacity()

    assert client.list_namespaced_pod.call_count == 3
    client.list_namespaced_secret.assert_called_once()


@pytest.mark.parametrize("phase", ["Succeeded", "Failed"])
def test_sweeper_deletes_terminal_pod_and_matching_secret(phase: str) -> None:
    """Test cleanup deletes terminal Pods and their credential Secrets."""
    client = Mock()
    labels = _task_labels(123)
    client.list_namespaced_pod.return_value = {"items": [_pod(phase, labels=labels)]}
    client.list_namespaced_secret.return_value = {
        "items": [_secret(_SECRET_NAME, labels)]
    }
    config = _executor_config(
        labels={
            _TASK_ID_LABEL: "fake-task",
            "flower.ai/task-type": "fake-task-type",
            LAUNCH_ATTEMPT_LABEL: "fake-launch-attempt",
            "flower.ai/resource-pool": "fake-pool",
            "flower.ai/team": "platform",
        },
        resource_pool="gpu-pool",
    )

    CompletedPodSweeper(client=client, config=config).sweep()

    selector = (
        "app.kubernetes.io/component=taskexecutor,"
        "app.kubernetes.io/name=flower,"
        "flower.ai/resource-pool=gpu-pool,"
        "flower.ai/team=platform"
    )
    client.list_namespaced_pod.assert_called_once_with(
        "flower-system", label_selector=selector
    )
    client.list_namespaced_secret.assert_called_once_with(
        "flower-system", label_selector=selector
    )
    client.delete_namespaced_pod.assert_called_once_with(
        name=_POD_NAME,
        namespace="flower-system",
        grace_period_seconds=0,
    )
    client.delete_namespaced_secret.assert_called_once_with(
        name=_SECRET_NAME, namespace="flower-system"
    )


def test_sweeper_keeps_terminal_pod_without_task_id_label() -> None:
    """Test cleanup ignores selector-matching Pods without a task-id label."""
    client = Mock()
    labels = _taskexecutor_labels()
    client.list_namespaced_pod.return_value = {
        "items": [_pod("Succeeded", labels=labels)]
    }
    client.list_namespaced_secret.return_value = {
        "items": [_secret(_SECRET_NAME, labels)]
    }

    CompletedPodSweeper(client=client, config=_executor_config()).sweep()

    client.delete_namespaced_pod.assert_not_called()
    client.delete_namespaced_secret.assert_not_called()


def test_sweeper_keeps_matching_secret_without_task_id_label() -> None:
    """Test cleanup ignores derived Secrets without a task-id label."""
    client = Mock()
    client.list_namespaced_pod.return_value = {
        "items": [_pod("Succeeded", labels=_task_labels(123))]
    }
    client.list_namespaced_secret.return_value = {
        "items": [_secret(_SECRET_NAME, _taskexecutor_labels())]
    }

    CompletedPodSweeper(client=client, config=_executor_config()).sweep()

    client.delete_namespaced_pod.assert_called_once()
    client.delete_namespaced_secret.assert_not_called()


def test_sweeper_keeps_pending_and_running_pods_and_secrets() -> None:
    """Test cleanup ignores non-terminal Pods and their credential Secrets."""
    client = Mock()
    client.list_namespaced_pod.return_value = {
        "items": [
            _pod("Pending", name=_POD_NAME, labels=_task_labels(123)),
            _pod("Running", name=_NEXT_POD_NAME, labels=_task_labels(124)),
        ]
    }
    client.list_namespaced_secret.return_value = {
        "items": [
            _secret(_SECRET_NAME, _task_labels(123)),
            _secret(_NEXT_SECRET_NAME, _task_labels(124)),
        ]
    }

    CompletedPodSweeper(client=client, config=_executor_config()).sweep()

    client.delete_namespaced_pod.assert_not_called()
    client.delete_namespaced_secret.assert_not_called()


def test_sweeper_deletes_orphaned_credential_secret() -> None:
    """Test cleanup deletes a credential Secret when its Pod is gone."""
    client = Mock()
    client.list_namespaced_pod.return_value = {"items": []}
    client.list_namespaced_secret.return_value = {
        "items": [_secret("flwr-taskexecutor-999-appio", _task_labels(999))]
    }

    CompletedPodSweeper(client=client, config=_executor_config()).sweep()

    client.delete_namespaced_pod.assert_not_called()
    client.delete_namespaced_secret.assert_called_once_with(
        name="flwr-taskexecutor-999-appio", namespace="flower-system"
    )


def test_sweeper_keeps_active_retry_secret_for_same_task() -> None:
    """Test cleanup keeps the Secret for a newer active retry of the same task."""
    client = Mock()
    client.list_namespaced_pod.return_value = {
        "items": [
            _pod("Succeeded", name=_POD_NAME, labels=_task_labels(123)),
            _pod("Running", name=_NEXT_POD_NAME, labels=_task_labels(123)),
        ]
    }
    client.list_namespaced_secret.return_value = {
        "items": [
            _secret(_SECRET_NAME, _task_labels(123)),
            _secret(_NEXT_SECRET_NAME, _task_labels(123)),
        ]
    }

    CompletedPodSweeper(client=client, config=_executor_config()).sweep()

    assert client.delete_namespaced_secret.mock_calls == [
        call(name=_SECRET_NAME, namespace="flower-system"),
    ]


def test_sweeper_keeps_secret_without_credential_secret_name() -> None:
    """Test cleanup does not delete Secrets without the credential name suffix."""
    client = Mock()
    client.list_namespaced_pod.return_value = {"items": []}
    client.list_namespaced_secret.return_value = {
        "items": [
            _secret(
                "manual-secret",
                _task_labels(123),
            )
        ]
    }

    CompletedPodSweeper(client=client, config=_executor_config()).sweep()

    client.delete_namespaced_secret.assert_not_called()


def test_sweeper_keeps_orphaned_credential_secret_without_task_id_label() -> None:
    """Test cleanup ignores orphaned credential Secrets without a task-id label."""
    client = Mock()
    client.list_namespaced_pod.return_value = {"items": []}
    client.list_namespaced_secret.return_value = {
        "items": [_secret("flwr-taskexecutor-999-appio", _taskexecutor_labels())]
    }

    CompletedPodSweeper(client=client, config=_executor_config()).sweep()

    client.delete_namespaced_secret.assert_not_called()


def test_sweeper_tolerates_already_deleted_pod_and_secret() -> None:
    """Test cleanup treats 404 delete responses as idempotent success."""
    client = Mock()
    client.list_namespaced_pod.return_value = {
        "items": [_pod("Failed", labels=_task_labels(123))]
    }
    client.list_namespaced_secret.return_value = {
        "items": [_secret(_SECRET_NAME, _task_labels(123))]
    }
    client.delete_namespaced_pod.side_effect = _KubernetesApiError(404, "Not Found")
    client.delete_namespaced_secret.side_effect = _KubernetesApiError(404, "Not Found")

    CompletedPodSweeper(client=client, config=_executor_config()).sweep()

    client.delete_namespaced_pod.assert_called_once()
    client.delete_namespaced_secret.assert_called_once()


def test_launch_submits_secret_before_pod_and_returns_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test launch creates the Secret before the Pod and returns accepted."""
    client = Mock()
    config = _executor_config()
    spec = _execution_spec()
    monkeypatch.setattr(
        kube, "_new_launch_attempt_id", Mock(return_value=_LAUNCH_ATTEMPT_ID)
    )

    result = KubernetesExecutor(client=client, config=config).launch(spec)

    runtime_root_certificates = _appio_root_certificates(spec, config)
    secret = _as_dict(
        _build_appio_credentials_secret(
            spec, config, runtime_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )
    pod = _as_dict(
        _build_taskexecutor_pod(
            spec, config, runtime_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )
    assert result.status == LaunchResultStatus.ACCEPTED
    assert client.mock_calls == [
        call.create_namespaced_secret("flower-system", secret),
        call.create_namespaced_pod("flower-system", pod),
    ]


def test_launch_generates_distinct_object_names_for_same_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test repeated launch calls for one task do not reuse Pod/Secret names."""
    client = Mock()
    monkeypatch.setattr(
        kube,
        "_new_launch_attempt_id",
        Mock(side_effect=[_LAUNCH_ATTEMPT_ID, _NEXT_LAUNCH_ATTEMPT_ID]),
    )
    executor = KubernetesExecutor(client=client, config=_executor_config())
    spec = _execution_spec()

    first_result = executor.launch(spec)
    second_result = executor.launch(spec)

    assert first_result.status == LaunchResultStatus.ACCEPTED
    assert second_result.status == LaunchResultStatus.ACCEPTED
    secret_bodies = [
        _as_dict(call_args.args[1])
        for call_args in client.create_namespaced_secret.call_args_list
    ]
    pod_bodies = [
        _as_dict(call_args.args[1])
        for call_args in client.create_namespaced_pod.call_args_list
    ]

    assert [secret["metadata"]["name"] for secret in secret_bodies] == [
        _SECRET_NAME,
        _NEXT_SECRET_NAME,
    ]
    assert [pod["metadata"]["name"] for pod in pod_bodies] == [
        _POD_NAME,
        _NEXT_POD_NAME,
    ]
    assert [
        pod["spec"]["volumes"][0]["secret"]["secretName"] for pod in pod_bodies
    ] == [secret["metadata"]["name"] for secret in secret_bodies]
    assert [
        secret["metadata"]["labels"][LAUNCH_ATTEMPT_LABEL] for secret in secret_bodies
    ] == [_LAUNCH_ATTEMPT_ID, _NEXT_LAUNCH_ATTEMPT_ID]


def test_launch_returns_capacity_rejected_if_secret_create_hits_quota() -> None:
    """Test launch maps Secret quota rejection without creating the Pod."""
    client = Mock()
    client.create_namespaced_secret.side_effect = _KubernetesApiError(
        403, "exceeded quota: object-counts"
    )

    result = KubernetesExecutor(client=client, config=_executor_config()).launch(
        _execution_spec()
    )

    assert result.status == LaunchResultStatus.CAPACITY_REJECTED
    assert result.message == "_KubernetesApiError: exceeded quota: object-counts"
    client.create_namespaced_pod.assert_not_called()
    client.delete_namespaced_secret.assert_not_called()


def test_launch_deletes_new_secret_if_pod_create_is_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Pod capacity rejection cleans up the just-created Secret."""
    client = Mock()
    client.create_namespaced_pod.side_effect = _KubernetesApiError(
        429, "too many requests"
    )
    monkeypatch.setattr(
        kube, "_new_launch_attempt_id", Mock(return_value=_LAUNCH_ATTEMPT_ID)
    )

    result = KubernetesExecutor(client=client, config=_executor_config()).launch(
        _execution_spec()
    )

    assert result.status == LaunchResultStatus.CAPACITY_REJECTED
    assert result.message == "_KubernetesApiError: too many requests"
    client.create_namespaced_secret.assert_called_once()
    client.create_namespaced_pod.assert_called_once()
    client.delete_namespaced_secret.assert_called_once_with(
        _SECRET_NAME, "flower-system"
    )


def test_launch_delete_failure_does_not_mask_pod_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test best-effort Secret cleanup failure preserves the launch result."""
    client = Mock()
    client.create_namespaced_pod.side_effect = _KubernetesApiError(
        429, "too many requests"
    )
    client.delete_namespaced_secret.side_effect = _KubernetesApiError(
        500, "delete failed"
    )
    monkeypatch.setattr(
        kube, "_new_launch_attempt_id", Mock(return_value=_LAUNCH_ATTEMPT_ID)
    )

    result = KubernetesExecutor(client=client, config=_executor_config()).launch(
        _execution_spec()
    )

    assert result.status == LaunchResultStatus.CAPACITY_REJECTED
    assert result.message == "_KubernetesApiError: too many requests"
    client.delete_namespaced_secret.assert_called_once_with(
        _SECRET_NAME, "flower-system"
    )


def test_launch_returns_failed_for_clear_non_capacity_failure() -> None:
    """Test launch maps clear non-capacity API failures to failed."""
    client = Mock()
    client.create_namespaced_secret.side_effect = _KubernetesApiError(
        401, "unauthorized"
    )

    result = KubernetesExecutor(client=client, config=_executor_config()).launch(
        _execution_spec()
    )

    assert result.status == LaunchResultStatus.FAILED
    assert result.message == "_KubernetesApiError: unauthorized"
    client.create_namespaced_pod.assert_not_called()


def test_launch_returns_unknown_for_ambiguous_server_failure() -> None:
    """Test launch maps ambiguous server failures to unknown."""
    client = Mock()
    client.create_namespaced_pod.side_effect = _KubernetesApiError(
        503, "service unavailable"
    )

    result = KubernetesExecutor(client=client, config=_executor_config()).launch(
        _execution_spec()
    )

    assert result.status == LaunchResultStatus.UNKNOWN
    assert result.message == "_KubernetesApiError: service unavailable"
    client.create_namespaced_secret.assert_called_once()
    client.create_namespaced_pod.assert_called_once()
    client.delete_namespaced_secret.assert_not_called()


def test_launch_returns_failed_if_root_certificates_file_cannot_be_read() -> None:
    """Test launch fails before submission if spec root certificates cannot be read."""
    client = Mock()

    result = KubernetesExecutor(
        client=client, config=_executor_config(runtime_root_certificates=None)
    ).launch(_execution_spec(root_certificates_path="/missing/appio-ca.pem"))

    assert result.status == LaunchResultStatus.FAILED
    assert result.message is not None
    assert result.message.startswith("FileNotFoundError:")
    client.create_namespaced_secret.assert_not_called()
    client.create_namespaced_pod.assert_not_called()


def test_execution_spec_rejects_invalid_task_id() -> None:
    """Test ExecutionSpec rejects invalid task IDs."""
    with pytest.raises(ValueError, match="positive integer task_id"):
        _execution_spec(task_id=0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runtime_api_address", "", "Runtime API address"),
        ("token", "", "task token"),
    ],
)
def test_execution_spec_rejects_empty_required_strings(
    field: str, value: str, message: str
) -> None:
    """Test ExecutionSpec rejects empty string fields required by all executors."""
    with pytest.raises(ValueError, match=message):
        _execution_spec(**{field: value})
