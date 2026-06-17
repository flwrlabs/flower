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
"""Tests for the Kubernetes executor dev smoke script."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from textwrap import dedent
from types import ModuleType
from unittest.mock import Mock

import pytest

from . import factory as factory_module


def _load_smoke_module() -> ModuleType:
    """Load the dev smoke script from its file path."""
    smoke_path = (
        Path(__file__).resolve().parents[5] / "dev" / ("kubernetes_executor_smoke.py")
    )
    spec = importlib.util.spec_from_file_location(
        "kubernetes_executor_smoke", smoke_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load Kubernetes executor smoke module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke_module = _load_smoke_module()


class _KubernetesList:
    """Small Kubernetes list response stand-in."""

    def __init__(self, items: list[object]) -> None:
        self.items = items


class _FakeKubernetesClient:
    """Fake CoreV1Api for the smoke script."""

    def __init__(self, *, pod_create_error: Exception | None = None) -> None:
        self.pod_create_error = pod_create_error
        self.pods: dict[str, object] = {}
        self.secrets: dict[str, object] = {}
        self.created_pods: list[object] = []
        self.created_secrets: list[object] = []

    def create_namespaced_secret(self, namespace: str, body: object) -> object:
        """Create a fake Secret."""
        del namespace
        name = _metadata_name(body)
        self.created_secrets.append(body)
        self.secrets[name] = body
        return body

    def create_namespaced_pod(self, namespace: str, body: object) -> object:
        """Create a fake Pod."""
        del namespace
        if self.pod_create_error is not None:
            raise self.pod_create_error
        name = _metadata_name(body)
        self.created_pods.append(body)
        self.pods[name] = body
        return body

    def list_namespaced_pod(
        self, namespace: str, label_selector: str
    ) -> _KubernetesList:
        """List fake Pods."""
        del namespace
        return _KubernetesList(
            [
                pod
                for pod in self.pods.values()
                if _matches_selector(pod, label_selector)
            ]
        )

    def list_namespaced_secret(
        self, namespace: str, label_selector: str
    ) -> _KubernetesList:
        """List fake Secrets."""
        del namespace
        return _KubernetesList(
            [
                secret
                for secret in self.secrets.values()
                if _matches_selector(secret, label_selector)
            ]
        )

    def delete_namespaced_pod(
        self, name: str, namespace: str, grace_period_seconds: int = 0
    ) -> object:
        """Delete a fake Pod."""
        del namespace, grace_period_seconds
        return self.pods.pop(name)

    def delete_namespaced_secret(self, name: str, namespace: str) -> object:
        """Delete a fake Secret."""
        del namespace
        return self.secrets.pop(name)


def test_run_smoke_creates_visible_objects_and_cleans_them_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test the smoke path creates, lists, and deletes scoped objects."""
    client = _FakeKubernetesClient()
    monkeypatch.setattr(
        factory_module, "create_incluster_kubernetes_client", Mock(return_value=client)
    )
    config_path = _write_executor_config(tmp_path)
    logs: list[str] = []

    summary = smoke_module.run_smoke(
        str(config_path), smoke_run="smoke-test", log=logs.append
    )

    assert summary.status == "passed"
    assert len(client.created_secrets) == 1
    assert len(client.created_pods) == 1
    assert client.pods == {}
    assert client.secrets == {}
    assert "flower-kubernetes-executor-smoke-token" not in "\n".join(logs)

    pod = client.created_pods[0]
    labels = _metadata_labels(pod)
    assert labels["flower.ai/smoke-run"] == "smoke-test"
    assert labels["flower.ai/resource-pool"] == "gpu-pool"
    assert _pod_spec(pod)["automountServiceAccountToken"] is False


def test_run_smoke_cleans_up_secret_if_pod_creation_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test a rejected Pod still cleans up the created credential Secret."""
    client = _FakeKubernetesClient(
        pod_create_error=RuntimeError(
            "Pod rejected for token flower-kubernetes-executor-smoke-token"
        )
    )
    monkeypatch.setattr(
        factory_module, "create_incluster_kubernetes_client", Mock(return_value=client)
    )
    config_path = _write_executor_config(tmp_path)

    summary = smoke_module.run_smoke(
        str(config_path), smoke_run="smoke-test", log=lambda _: None
    )

    assert summary.status == "failed"
    assert len(client.created_secrets) == 1
    assert client.created_pods == []
    assert client.secrets == {}
    assert summary.error is not None
    assert "flower-kubernetes-executor-smoke-token" not in summary.error
    assert "<redacted>" in summary.error


@pytest.mark.parametrize(
    ("status", "expected_exit_code"), [("passed", 0), ("failed", 1)]
)
def test_main_emits_json_summary_and_uses_status_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: str,
    expected_exit_code: int,
) -> None:
    """Test command output shape and exit status without Kubernetes access."""
    config_path = _write_executor_config(tmp_path)

    def _run_smoke(
        executor_config_path: str,
        *,
        smoke_run: str | None = None,
        log: Callable[[str], None] = print,
    ) -> object:
        del smoke_run
        assert executor_config_path == str(config_path)
        log("human-readable smoke log")
        return smoke_module.SmokeSummary(
            status=status,
            config_path=executor_config_path,
            smoke_run="smoke-test",
            namespace="flower-system",
        )

    monkeypatch.setattr(smoke_module, "run_smoke", _run_smoke)

    exit_code = smoke_module.main(["--executor-config", str(config_path), "--json"])

    captured = capsys.readouterr()
    assert exit_code == expected_exit_code
    assert captured.err == "human-readable smoke log\n"
    assert json.loads(captured.out) == {
        "cleanup_errors": [],
        "config_path": str(config_path),
        "error": None,
        "namespace": "flower-system",
        "pod_created": False,
        "pod_deleted": False,
        "pod_name": None,
        "pod_visible": False,
        "pool_selector": None,
        "resource_pool": None,
        "secret_created": False,
        "secret_deleted": False,
        "secret_name": None,
        "secret_visible": False,
        "smoke_run": "smoke-test",
        "smoke_selector": None,
        "status": status,
    }


def test_help_documents_incluster_scope_without_running_smoke(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test help text makes the non-gate in-cluster scope discoverable."""
    with pytest.raises(SystemExit) as exc_info:
        smoke_module.main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "not a stable flwr-* command" in captured.out
    assert "production in-cluster auth" in captured.out
    assert "runner ServiceAccount permissions" in captured.out
    assert "Flower PR merge requirement" in captured.out


def _write_executor_config(tmp_path: Path) -> Path:
    """Write a minimal Kubernetes executor config."""
    config_path = tmp_path / "executor.yaml"
    config_path.write_text(
        dedent(
            """
            namespace: flower-system
            image: ghcr.io/flwrlabs/taskexecutor:dev
            resource-pool: gpu-pool
            labels:
              flower.ai/deployment: dev
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _metadata_name(value: object) -> str:
    metadata = _field(value, "metadata")
    name = _field(metadata, "name")
    if not isinstance(name, str):
        raise AssertionError("Object does not have metadata.name")
    return name


def _metadata_labels(value: object) -> dict[str, str]:
    metadata = _field(value, "metadata")
    labels = _field(metadata, "labels")
    if not isinstance(labels, dict):
        raise AssertionError("Object does not have metadata.labels")
    return labels


def _pod_spec(value: object) -> dict[str, object]:
    spec = _field(value, "spec")
    if not isinstance(spec, dict):
        raise AssertionError("Object does not have spec")
    return spec


def _matches_selector(value: object, selector: str) -> bool:
    labels = _metadata_labels(value)
    for requirement in selector.split(","):
        key, expected = requirement.split("=", maxsplit=1)
        if labels.get(key) != expected:
            return False
    return True


def _field(value: object, field_name: str) -> object | None:
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)
