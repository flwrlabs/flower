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

from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, call

import pytest

from flwr.supercore.constant import TaskType

from .kubernetes_executor import (
    APPIO_CREDENTIALS_MOUNT_PATH,
    APPIO_ROOT_CERTIFICATES_FILE_PATH,
    APPIO_TOKEN_FILE_PATH,
    KubernetesExecutor,
    KubernetesExecutorConfig,
    _build_appio_credentials_secret,
    _build_taskexecutor_pod,
    _get_appio_root_certificates,
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


def _as_dict(value: object) -> dict[str, Any]:
    """Return a typed dict for nested JSON assertions."""
    return cast(dict[str, Any], value)


def _appio_root_certificates(
    spec: ExecutionSpec, config: KubernetesExecutorConfig
) -> str | None:
    """Return AppIo root certificates for object-building tests."""
    return _get_appio_root_certificates(spec, config)


def test_build_appio_credentials_secret_contains_token_and_ca() -> None:
    """Test building the AppIo credential Secret."""
    spec = _execution_spec()
    config = _executor_config()

    secret = _as_dict(
        _build_appio_credentials_secret(
            spec, config, _appio_root_certificates(spec, config)
        )
    )

    assert secret == {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": "flwr-taskexecutor-123-appio",
            "namespace": "flower-system",
            "labels": {
                "app.kubernetes.io/name": "flower",
                "app.kubernetes.io/component": "taskexecutor",
                "flower.ai/superexec-task-id": "123",
                "flower.ai/task-type": "flwr-serverapp",
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
        _build_taskexecutor_pod(spec, config, _appio_root_certificates(spec, config))
    )
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
            "secret": {
                "secretName": "flwr-taskexecutor-123-appio",
                "defaultMode": 0o444,
            },
        }
    ]
    assert pod["spec"]["automountServiceAccountToken"] is False
    assert pod["spec"]["restartPolicy"] == "Never"


def test_build_taskexecutor_pod_supports_clientapp_insecure_args() -> None:
    """Test Pod construction for insecure ClientApp launch args."""
    pod = _as_dict(
        _build_taskexecutor_pod(
            _execution_spec(task_type=TaskType.CLIENT_APP, insecure=True),
            _executor_config(appio_root_certificates=None),
            None,
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
        _build_appio_credentials_secret(spec, config, appio_root_certificates)
    )
    pod = _as_dict(_build_taskexecutor_pod(spec, config, appio_root_certificates))

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
        _build_appio_credentials_secret(spec, config, appio_root_certificates)
    )
    pod = _as_dict(_build_taskexecutor_pod(spec, config, appio_root_certificates))

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
        _build_appio_credentials_secret(spec, config, appio_root_certificates)
    )
    pod = _as_dict(_build_taskexecutor_pod(spec, config, appio_root_certificates))

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
        )
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

    appio_root_certificates = _appio_root_certificates(spec, config)
    secret = _as_dict(
        _build_appio_credentials_secret(spec, config, appio_root_certificates)
    )
    pod = _as_dict(_build_taskexecutor_pod(spec, config, appio_root_certificates))
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
