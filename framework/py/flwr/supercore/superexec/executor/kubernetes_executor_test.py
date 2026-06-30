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

from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, call

import pytest

from flwr.supercore.constant import TaskType

from . import kubernetes_executor as kube
from .kubernetes_executor import (
    _COMPLETED_POD_SWEEP_INTERVAL_SECONDS,
    _TASK_ID_LABEL,
    APPIO_CREDENTIALS_MOUNT_PATH,
    APPIO_ROOT_CERTIFICATES_FILE_PATH,
    APPIO_TOKEN_FILE_PATH,
    LAUNCH_ATTEMPT_LABEL,
    CompletedPodSweeper,
    KubernetesExecutor,
    KubernetesExecutorConfig,
    _build_appio_credentials_secret,
    _build_taskexecutor_pod,
    _get_appio_root_certificates,
)
from .types import ExecutionSpec, LaunchResultStatus


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
        "appio_api_address": "appio.example.com:9092",
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
        "appio_root_certificates": "root-ca",
    }
    base.update(overrides)
    return KubernetesExecutorConfig(**base)


def _as_dict(value: object) -> dict[str, Any]:
    """Return a typed dict for nested JSON assertions."""
    return cast(dict[str, Any], value)


def _appio_root_certificates(
    spec: ExecutionSpec, config: KubernetesExecutorConfig
) -> str | None:
    """Return AppIo root certificates for object-building tests."""
    return _get_appio_root_certificates(spec, config)


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
        "--serverappio-api-address",
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
            _executor_config(appio_root_certificates=None),
            None,
            _LAUNCH_ATTEMPT_ID,
        )
    )

    assert pod["spec"]["containers"][0]["command"] == ["flwr-clientapp"]
    assert pod["spec"]["containers"][0]["args"] == [
        "--clientappio-api-address",
        "appio.example.com:9092",
        "--token-file",
        APPIO_TOKEN_FILE_PATH,
        "--insecure",
    ]


def test_build_taskexecutor_pod_supports_secure_default_trust_store() -> None:
    """Test secure Pod args can rely on container default trust store."""
    spec = _execution_spec()
    config = _executor_config(appio_root_certificates=None)
    appio_root_certificates = _appio_root_certificates(spec, config)

    secret = _as_dict(
        _build_appio_credentials_secret(
            spec, config, appio_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )
    pod = _as_dict(
        _build_taskexecutor_pod(
            spec, config, appio_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )

    assert secret["stringData"] == {"token": "task-token"}
    assert pod["spec"]["containers"][0]["args"] == [
        "--serverappio-api-address",
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
    config = _executor_config(appio_root_certificates=None)
    appio_root_certificates = _appio_root_certificates(spec, config)

    secret = _as_dict(
        _build_appio_credentials_secret(
            spec, config, appio_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )
    pod = _as_dict(
        _build_taskexecutor_pod(
            spec, config, appio_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )

    assert secret["stringData"] == {"token": "task-token", "ca.crt": "spec-root-ca"}
    assert pod["spec"]["containers"][0]["args"] == [
        "--serverappio-api-address",
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
    config = _executor_config(appio_root_certificates=None)
    appio_root_certificates = _appio_root_certificates(spec, config)

    secret = _as_dict(
        _build_appio_credentials_secret(
            spec, config, appio_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )
    pod = _as_dict(
        _build_taskexecutor_pod(
            spec, config, appio_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )

    assert secret["stringData"] == {"token": "task-token", "ca.crt": "home-root-ca"}
    assert pod["spec"]["containers"][0]["args"] == [
        "--serverappio-api-address",
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
        "--serverappio-api-address",
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
    appio_root_certificates = _appio_root_certificates(spec, config)
    secret = _as_dict(
        _build_appio_credentials_secret(
            spec, config, appio_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )
    pod = _as_dict(
        _build_taskexecutor_pod(
            spec, config, appio_root_certificates, _LAUNCH_ATTEMPT_ID
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

    appio_root_certificates = _appio_root_certificates(spec, config)
    secret = _as_dict(
        _build_appio_credentials_secret(
            spec, config, appio_root_certificates, _LAUNCH_ATTEMPT_ID
        )
    )
    pod = _as_dict(
        _build_taskexecutor_pod(
            spec, config, appio_root_certificates, _LAUNCH_ATTEMPT_ID
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
        client=client, config=_executor_config(appio_root_certificates=None)
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
        ("appio_api_address", "", "AppIo API address"),
        ("token", "", "task token"),
    ],
)
def test_execution_spec_rejects_empty_required_strings(
    field: str, value: str, message: str
) -> None:
    """Test ExecutionSpec rejects empty string fields required by all executors."""
    with pytest.raises(ValueError, match=message):
        _execution_spec(**{field: value})
