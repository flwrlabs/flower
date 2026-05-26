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


from typing import Any
from unittest.mock import Mock, call

import pytest

from flwr.supercore.constant import TaskType

from .kubernetes_executor import (
    APPIO_CREDENTIALS_MOUNT_PATH,
    APPIO_ROOT_CERTIFICATES_FILE_PATH,
    APPIO_TOKEN_FILE_PATH,
    KubernetesExecutor,
    KubernetesExecutorConfig,
    build_appio_credentials_secret,
    build_taskexecutor_pod,
)
from .types import ExecutionSpec, LaunchResultStatus


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


def test_build_appio_credentials_secret_contains_token_and_ca() -> None:
    """Test building the AppIo credential Secret."""
    secret = build_appio_credentials_secret(_execution_spec(), _executor_config())

    assert secret == {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": "flwr-taskexecutor-123-appio",
            "namespace": "flower-system",
            "labels": {
                "app.kubernetes.io/name": "flower",
                "app.kubernetes.io/component": "taskexecutor",
                "flwr.ai/superexec-task-id": "123",
                "flwr.ai/task-type": "flwr-serverapp",
            },
        },
        "type": "Opaque",
        "stringData": {"token": "task-token", "ca.crt": "root-ca"},
    }


def test_build_taskexecutor_pod_uses_secret_files_for_credentials() -> None:
    """Test Pod construction uses mounted files instead of credential args."""
    pod = build_taskexecutor_pod(_execution_spec(), _executor_config())
    container = pod["spec"]["containers"][0]

    assert APPIO_CREDENTIALS_MOUNT_PATH == "/run/flwr/appio"
    assert pod["metadata"]["name"] == "flwr-taskexecutor-123"
    assert pod["metadata"]["namespace"] == "flower-system"
    assert container["image"] == "ghcr.io/flwrlabs/taskexecutor:dev"
    assert container["command"] == ["flwr-serverapp"]
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
            "secret": {"secretName": "flwr-taskexecutor-123-appio"},
        }
    ]
    assert pod["spec"]["automountServiceAccountToken"] is False
    assert pod["spec"]["restartPolicy"] == "Never"


def test_build_taskexecutor_pod_supports_clientapp_insecure_args() -> None:
    """Test Pod construction for insecure ClientApp launch args."""
    pod = build_taskexecutor_pod(
        _execution_spec(task_type=TaskType.CLIENT_APP, insecure=True),
        _executor_config(appio_root_certificates=None),
    )

    assert pod["spec"]["containers"][0]["command"] == ["flwr-clientapp"]
    assert pod["spec"]["containers"][0]["args"] == [
        "--clientappio-api-address",
        "appio.example.com:9092",
        "--token-file",
        APPIO_TOKEN_FILE_PATH,
        "--insecure",
    ]


def test_build_taskexecutor_pod_supports_simulation_args() -> None:
    """Test Pod construction for Simulation launch args."""
    pod = build_taskexecutor_pod(
        _execution_spec(task_type=TaskType.SIMULATION),
        _executor_config(),
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
    pod = build_taskexecutor_pod(
        _execution_spec(runtime_dependency_install=True),
        _executor_config(
            image_pull_policy="IfNotPresent",
            service_account_name="flower-superexec",
        ),
    )
    container = pod["spec"]["containers"][0]

    assert container["imagePullPolicy"] == "IfNotPresent"
    assert "--allow-runtime-dependency-installation" in container["args"]
    assert pod["spec"]["serviceAccountName"] == "flower-superexec"


def test_launch_submits_secret_before_pod_and_returns_accepted() -> None:
    """Test launch creates the Secret before the Pod and returns accepted."""
    client = Mock()
    config = _executor_config()
    spec = _execution_spec()

    result = KubernetesExecutor(client=client, config=config).launch(spec)

    secret = build_appio_credentials_secret(spec, config)
    pod = build_taskexecutor_pod(spec, config)
    assert result.status == LaunchResultStatus.ACCEPTED
    assert client.mock_calls == [
        call.create_namespaced_secret("flower-system", secret),
        call.create_namespaced_pod("flower-system", pod),
    ]


def test_launch_returns_failed_if_secret_create_fails() -> None:
    """Test launch fails without creating the Pod if Secret creation fails."""
    client = Mock()
    client.create_namespaced_secret.side_effect = RuntimeError("secret denied")

    result = KubernetesExecutor(client=client, config=_executor_config()).launch(
        _execution_spec()
    )

    assert result.status == LaunchResultStatus.FAILED
    assert result.message == "RuntimeError: secret denied"
    client.create_namespaced_pod.assert_not_called()


def test_launch_returns_failed_if_pod_create_fails() -> None:
    """Test launch fails after Secret creation if Pod creation fails."""
    client = Mock()
    client.create_namespaced_pod.side_effect = RuntimeError("pod denied")

    result = KubernetesExecutor(client=client, config=_executor_config()).launch(
        _execution_spec()
    )

    assert result.status == LaunchResultStatus.FAILED
    assert result.message == "RuntimeError: pod denied"
    client.create_namespaced_secret.assert_called_once()
    client.create_namespaced_pod.assert_called_once()


def test_build_rejects_invalid_task_id() -> None:
    """Test Kubernetes object construction rejects invalid task IDs."""
    with pytest.raises(ValueError, match="positive integer task_id"):
        build_taskexecutor_pod(
            _execution_spec(task_id=0),
            _executor_config(),
        )


def test_build_rejects_missing_ca_for_secure_connection() -> None:
    """Test secure Kubernetes object construction requires root certificate material."""
    with pytest.raises(ValueError, match="root certificates"):
        build_taskexecutor_pod(
            _execution_spec(insecure=False),
            _executor_config(appio_root_certificates=None),
        )
