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
            "active-pod-budget": 5,
            "appio-root-certificates-path": str(root_certificates_path),
            "env": [
                {
                    "name": "FLWR_MODEL_API_ENDPOINT",
                    "value": "http://proxy/v1/responses",
                }
            ],
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
    assert config.appio_root_certificates == "root-ca"
    assert config.env == [
        {"name": "FLWR_MODEL_API_ENDPOINT", "value": "http://proxy/v1/responses"}
    ]
    assert config.resources == {"requests": {"cpu": "1"}}
    assert config.node_selector == {"kubernetes.io/os": "linux"}
    assert not hasattr(config, "unknown_field")
    create_client.assert_called_once_with()


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
