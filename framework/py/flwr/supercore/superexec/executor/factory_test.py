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
"""Tests for SuperExec executor factory."""


from pathlib import Path
from unittest.mock import Mock

import pytest

from flwr.supercore.constant import ExecutorType

from . import factory as factory_module
from .factory import get_executor
from .kubernetes_executor import KubernetesExecutor


def test_get_executor_requires_kubernetes_config() -> None:
    """Test Kubernetes selection requires executor config."""
    with pytest.raises(ValueError, match="requires --executor-config"):
        get_executor(ExecutorType.KUBERNETES)


def test_get_executor_builds_kubernetes_executor_from_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test Kubernetes config fields are mapped into executor construction."""
    root_certificates_path = tmp_path / "ca.pem"
    root_certificates_path.write_text("root-ca", encoding="utf-8")
    client = Mock()
    exec_client = Mock()
    create_clients = Mock(return_value=(client, exec_client))
    monkeypatch.setattr(
        factory_module, "create_incluster_kubernetes_clients", create_clients
    )

    executor = get_executor(
        ExecutorType.KUBERNETES,
        executor_config={
            "namespace": "flower-system",
            "image": "ghcr.io/flwrlabs/taskexecutor:dev",
            "image-pull-policy": "IfNotPresent",
            "active-pod-budget": 5,
            "appio-root-certificates-path": str(root_certificates_path),
            "env": [
                {
                    "name": "FLWR_MODEL_API_ENDPOINT",
                    "value": "http://proxy/v1/responses",
                }
            ],
            "volumes": [
                {
                    "name": "shmem",
                    "emptyDir": {"medium": "Memory", "sizeLimit": "10Gi"},
                }
            ],
            "volume-mounts": [{"name": "shmem", "mountPath": "/dev/shm"}],
            "resources": {"requests": {"cpu": "1"}},
            "node-selector": {"kubernetes.io/os": "linux"},
            "unknown-field": "ignored",
        },
    )

    assert isinstance(executor, KubernetesExecutor)
    assert executor._client is client  # pylint: disable=protected-access
    config = executor._config  # pylint: disable=protected-access
    assert config.namespace == "flower-system"
    assert config.image == "ghcr.io/flwrlabs/taskexecutor:dev"
    assert config.image_pull_policy == "IfNotPresent"
    assert config.active_pod_budget == 5
    assert config.runtime_root_certificates == "root-ca"
    assert config.env == [
        {"name": "FLWR_MODEL_API_ENDPOINT", "value": "http://proxy/v1/responses"}
    ]
    assert config.volumes == [
        {"name": "shmem", "emptyDir": {"medium": "Memory", "sizeLimit": "10Gi"}}
    ]
    assert config.volume_mounts == [{"name": "shmem", "mountPath": "/dev/shm"}]
    assert config.resources == {"requests": {"cpu": "1"}}
    assert config.node_selector == {"kubernetes.io/os": "linux"}
    assert not hasattr(config, "unknown_field")
    create_clients.assert_called_once_with()


@pytest.mark.parametrize("insecure", [False, True])
@pytest.mark.parametrize("ca_source", [None, "executor", "task"])
def test_get_executor_configures_usable_agentapp_warm_executor_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    insecure: bool,
    ca_source: str | None,
) -> None:
    """Only provision warm pools usable with the configured Runtime transport."""
    client = Mock()
    exec_client = Mock()
    client.list_namespaced_pod.return_value = {"items": []}
    client.list_namespaced_secret.return_value = {"items": []}
    monkeypatch.setattr(
        factory_module,
        "create_incluster_kubernetes_clients",
        Mock(return_value=(client, exec_client)),
    )

    executor_config: dict[str, object] = {
        "namespace": "flower-system",
        "image": "ghcr.io/flwrlabs/taskexecutor:dev",
        "warm-executor-owner": "superexec-a",
        "warm-executor-pools": [{"task-type": "flwr-agentapp", "size": 2}],
    }
    ca_path = tmp_path / "ca.pem"
    ca_path.write_text("root-ca", encoding="utf-8")
    if ca_source == "executor":
        executor_config["appio-root-certificates-path"] = str(ca_path)
    root_certificates_path = str(ca_path) if ca_source == "task" else None
    executor = get_executor(
        ExecutorType.KUBERNETES,
        executor_config=executor_config,
        insecure=insecure,
        root_certificates_path=root_certificates_path,
    )

    assert isinstance(executor, KubernetesExecutor)
    config = executor._config  # pylint: disable=protected-access
    enabled = insecure or ca_source is None
    assert bool(config.warm_executor_pools) == enabled
    if enabled:
        pool = config.warm_executor_pools[0]
        assert pool.key.task_type.value == "flwr-agentapp"
        assert pool.key.runtime_image == "ghcr.io/flwrlabs/taskexecutor:dev"
        assert pool.size == 2
    assert config.warm_executor_owner == "superexec-a"
    assert config.runtime_root_certificates == (
        "root-ca" if ca_source == "executor" else None
    )
    manager = executor._warm_executor_pool_manager  # pylint: disable=protected-access
    assert manager is not None
    assert manager._exec_client is exec_client  # pylint: disable=protected-access
    executor.reconcile()
    assert client.create_namespaced_pod.call_count == (2 if enabled else 0)


def test_get_executor_rejects_non_string_warm_executor_owner() -> None:
    """Warm executor owners must reach the normal invalid-config error path."""
    with pytest.raises(ValueError, match="warm_executor_owner must be a string"):
        factory_module._kubernetes_executor_config_from_mapping(  # pylint: disable=protected-access
            {
                "namespace": "flower-system",
                "image": "ghcr.io/flwrlabs/taskexecutor:dev",
                "warm-executor-owner": 1,
                "warm-executor-pools": [
                    {
                        "task-type": "flwr-agentapp",
                        "size": 1,
                    }
                ],
            }
        )


def test_get_executor_rejects_non_agentapp_warm_pool() -> None:
    """The first warm dispatch feature must not enable Model or Connector pools."""
    with pytest.raises(ValueError, match="flwr-agentapp"):
        factory_module._kubernetes_executor_config_from_mapping(  # pylint: disable=protected-access
            {
                "namespace": "flower-system",
                "image": "ghcr.io/flwrlabs/taskexecutor:dev",
                "warm-executor-owner": "superexec-a",
                "warm-executor-pools": [
                    {
                        "task-type": "flwr-model",
                        "size": 1,
                    }
                ],
            }
        )


@pytest.mark.parametrize("field_name", ["namespace", "image"])
def test_get_executor_rejects_missing_required_kubernetes_field(
    field_name: str,
) -> None:
    """Test required Kubernetes construction fields fail clearly."""
    executor_config: dict[str, object] = {
        "namespace": "flower-system",
        "image": "ghcr.io/flwrlabs/taskexecutor:dev",
    }
    del executor_config[field_name]

    with pytest.raises(ValueError, match=f"'{field_name}'"):
        get_executor(ExecutorType.KUBERNETES, executor_config=executor_config)


def test_get_executor_rejects_unreadable_appio_root_certificates_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test Runtime API root certificate load failures do not reach client
    creation."""
    create_clients = Mock()
    monkeypatch.setattr(
        factory_module, "create_incluster_kubernetes_clients", create_clients
    )

    with pytest.raises(ValueError) as exc_info:
        get_executor(
            ExecutorType.KUBERNETES,
            executor_config={
                "namespace": "flower-system",
                "image": "ghcr.io/flwrlabs/taskexecutor:dev",
                "appio-root-certificates-path": str(tmp_path / "missing-ca.pem"),
            },
        )

    assert "appio-root-certificates-path" in str(exc_info.value)
    create_clients.assert_not_called()


def test_get_executor_wraps_kubernetes_client_construction_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Kubernetes dependency/auth failures surface as config failures."""
    monkeypatch.setattr(
        factory_module,
        "create_incluster_kubernetes_clients",
        Mock(side_effect=RuntimeError("in-cluster auth unavailable")),
    )

    with pytest.raises(ValueError, match="in-cluster auth unavailable"):
        get_executor(
            ExecutorType.KUBERNETES,
            executor_config={
                "namespace": "flower-system",
                "image": "ghcr.io/flwrlabs/taskexecutor:dev",
            },
        )
