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

# pylint: disable=protected-access

from pathlib import Path
from unittest.mock import Mock

import pytest

from flwr.supercore.constant import ExecutorType

from . import factory as factory_module
from .factory import get_executor
from .kubernetes_executor import KubernetesExecutor
from .subprocess_executor import SubprocessExecutor


def test_get_executor_returns_subprocess_executor_by_default() -> None:
    """Test subprocess selection preserves the default executor."""
    executor = get_executor(ExecutorType.SUBPROCESS)

    assert isinstance(executor, SubprocessExecutor)


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
    create_client = Mock(return_value=client)
    monkeypatch.setattr(
        factory_module, "create_incluster_kubernetes_client", create_client
    )

    executor = get_executor(
        ExecutorType.KUBERNETES,
        executor_config={
            "namespace": "flower-system",
            "image": "ghcr.io/flwrlabs/taskexecutor:dev",
            "image-pull-policy": "IfNotPresent",
            "resource-pool": "gpu-pool",
            "active-pod-budget": 5,
            "capacity-poll-interval": 0.25,
            "capacity-log-interval": 30.0,
            "appio-root-certificates-path": str(root_certificates_path),
            "labels": {"flower.ai/deployment": "dev"},
            "annotations": {"flower.ai/owner": "superexec"},
            "resources": {"requests": {"cpu": "1"}},
            "node-selector": {"kubernetes.io/os": "linux"},
            "tolerations": [{"key": "dedicated", "operator": "Exists"}],
            "affinity": {"podAntiAffinity": {}},
            "priority-class-name": "high-priority",
            "pod-security-context": {"runAsNonRoot": True},
            "container-security-context": {"allowPrivilegeEscalation": False},
            "service-account-name": "taskexecutor",
            "unknown-field": "ignored",
        },
    )

    assert isinstance(executor, KubernetesExecutor)
    assert executor._client is client
    config = executor._config
    assert config.namespace == "flower-system"
    assert config.image == "ghcr.io/flwrlabs/taskexecutor:dev"
    assert config.image_pull_policy == "IfNotPresent"
    assert config.resource_pool == "gpu-pool"
    assert config.active_pod_budget == 5
    assert config.capacity_poll_interval == 0.25
    assert config.capacity_log_interval == 30.0
    assert config.appio_root_certificates == "root-ca"
    assert config.labels == {"flower.ai/deployment": "dev"}
    assert config.annotations == {"flower.ai/owner": "superexec"}
    assert config.resources == {"requests": {"cpu": "1"}}
    assert config.node_selector == {"kubernetes.io/os": "linux"}
    assert config.tolerations == [{"key": "dedicated", "operator": "Exists"}]
    assert config.affinity == {"podAntiAffinity": {}}
    assert config.priority_class_name == "high-priority"
    assert config.pod_security_context == {"runAsNonRoot": True}
    assert config.container_security_context == {"allowPrivilegeEscalation": False}
    assert config.service_account_name == "taskexecutor"
    assert not hasattr(config, "unknown_field")
    create_client.assert_called_once_with()


@pytest.mark.parametrize("field_name", ["namespace", "image"])
def test_get_executor_rejects_missing_required_kubernetes_field(
    field_name: str,
) -> None:
    """Test required Kubernetes construction fields fail clearly."""
    executor_config: dict[object, object] = {
        "namespace": "flower-system",
        "image": "ghcr.io/flwrlabs/taskexecutor:dev",
    }
    del executor_config[field_name]

    with pytest.raises(ValueError, match=f"'{field_name}'"):
        get_executor(ExecutorType.KUBERNETES, executor_config=executor_config)


def test_get_executor_rejects_unreadable_appio_root_certificates_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test AppIo root certificate load failures do not reach client
    creation."""
    create_client = Mock()
    monkeypatch.setattr(
        factory_module, "create_incluster_kubernetes_client", create_client
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
    create_client.assert_not_called()


def test_get_executor_wraps_kubernetes_client_construction_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Kubernetes dependency/auth failures surface as config failures."""
    monkeypatch.setattr(
        factory_module,
        "create_incluster_kubernetes_client",
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
