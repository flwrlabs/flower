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
"""Tests for the optional Kubernetes executor k3d smoke harness."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import Mock, call

import kubernetes_executor_k3d_probe.probe_taskexecutor as probe
import kubernetes_executor_k3d_smoke as smoke
import pytest


def _list_response(*names: str) -> object:
    """Return a Kubernetes-list-like object."""
    return {"items": [{"metadata": {"name": name}} for name in names]}


def test_parse_args_uses_environment_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test smoke config defaults can come from environment variables."""
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_SMOKE_CLUSTER_NAME", "env-cluster")
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_SMOKE_NAMESPACE", "env-namespace")
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_SMOKE_IMAGE", "example/image:dev")
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_SMOKE_IMAGE_PULL_POLICY", "IfNotPresent")
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_SMOKE_APPIO_API_ADDRESS", "appio:9092")
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_SMOKE_ACTIVE_POD_BUDGET", "2")
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_SMOKE_CAPACITY_TIMEOUT", "3.5")
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_SMOKE_CAPACITY_POLL_INTERVAL", "0.1")
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_SMOKE_KEEP_RESOURCES", "true")
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_SMOKE_DELETE_CLUSTER", "yes")

    config = smoke.parse_args([])

    assert config.cluster_name == "env-cluster"
    assert config.namespace == "env-namespace"
    assert config.image == "example/image:dev"
    assert config.image_pull_policy == "IfNotPresent"
    assert config.appio_api_address == "appio:9092"
    assert config.active_pod_budget == 2
    assert config.capacity_timeout == 3.5
    assert config.capacity_poll_interval == 0.1
    assert config.keep_resources is True
    assert config.delete_cluster is True


def test_parse_args_probe_image_forces_local_image_and_completion() -> None:
    """Test probe image mode selects the controlled image and strict Pod wait."""
    config = smoke.parse_args(["--probe-image"])

    assert config.probe_image is True
    assert config.image == smoke.DEFAULT_PROBE_IMAGE
    assert config.image_pull_policy == "Never"
    assert config.require_pod_succeeded is True


def test_parse_args_can_require_pod_succeeded_without_building_probe() -> None:
    """Test strict Pod completion can be requested for a caller-provided image."""
    config = smoke.parse_args(
        ["--require-pod-succeeded", "--image", "example/taskexecutor:local"]
    )

    assert config.probe_image is False
    assert config.image == "example/taskexecutor:local"
    assert config.require_pod_succeeded is True


def test_probe_taskexecutor_validates_mounted_token_and_insecure_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Test the probe command accepts executor-rendered insecure args."""
    token_path = tmp_path / "token"
    token_path.write_text(
        "smoke-token-0123456789abcdef0123456789abcdef-0", encoding="utf-8"
    )
    monkeypatch.setattr(probe, "APPIO_TOKEN_FILE_PATH", str(token_path))

    probe._run_probe(
        [
            "flwr-serverapp",
            "--serverappio-api-address",
            "appio:9092",
            "--token-file",
            str(token_path),
            "--insecure",
        ]
    )


def test_core_v1_api_adapter_forwards_executor_calls() -> None:
    """Test CoreV1Api calls are adapted to the executor client protocol."""
    api = Mock()
    adapter = smoke.CoreV1ApiAdapter(api)
    secret: dict[str, Any] = {"kind": "Secret"}
    pod: dict[str, Any] = {"kind": "Pod"}

    adapter.create_namespaced_secret("namespace", secret)
    adapter.create_namespaced_pod("namespace", pod)
    adapter.list_namespaced_pod("namespace", "label=value")

    api.create_namespaced_secret.assert_called_once_with(
        namespace="namespace", body=secret
    )
    api.create_namespaced_pod.assert_called_once_with(namespace="namespace", body=pod)
    api.list_namespaced_pod.assert_called_once_with(
        namespace="namespace", label_selector="label=value"
    )


def test_build_smoke_executor_scopes_capacity_selector_to_run_label() -> None:
    """Test executor config includes local-only pool and unique run label."""
    config = smoke.SmokeConfig(
        cluster_name="cluster",
        namespace="namespace",
        image="image:dev",
        image_pull_policy="Never",
        appio_api_address="appio:9092",
        active_pod_budget=1,
        capacity_timeout=5.0,
        capacity_poll_interval=0.1,
        keep_resources=False,
        delete_cluster=False,
    )

    executor_config = smoke.build_executor_config(config, "run-123")
    selector, _executor = smoke.build_smoke_executor(Mock(), config, "run-123")

    assert "app.kubernetes.io/component=taskexecutor" in selector
    assert f"{smoke.RUN_LABEL_KEY}=run-123" in selector
    assert f"flower.ai/resource-pool={smoke.LOCAL_RESOURCE_POOL}" in selector
    assert executor_config.image_pull_policy == "Never"


def test_build_execution_spec_uses_unique_task_ids_and_tokens() -> None:
    """Test smoke specs remain unique when active Pod budget is overridden."""
    config = smoke.SmokeConfig(
        cluster_name="cluster",
        namespace="namespace",
        image="image:dev",
        image_pull_policy="Never",
        appio_api_address="appio:9092",
        active_pod_budget=2,
        capacity_timeout=5.0,
        capacity_poll_interval=0.1,
        keep_resources=False,
        delete_cluster=False,
    )
    run_id = "0123456789abcdef0123456789abcdef"

    first = smoke.build_execution_spec(config, run_id, task_offset=0)
    second = smoke.build_execution_spec(config, run_id, task_offset=1)

    assert first.task_id > 0
    assert second.task_id == first.task_id + 1
    assert first.token != second.token


def test_pod_lifecycle_diagnostics_reports_waiting_pod_without_logs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test lifecycle diagnostics print waiting state without failing."""
    api = Mock()
    api.list_namespaced_event.return_value = {
        "items": [
            {
                "type": "Warning",
                "reason": "ErrImageNeverPull",
                "message": "Container image is not present with pull policy Never",
                "count": 2,
                "lastTimestamp": "2026-06-02T12:00:00Z",
            }
        ]
    }
    pod = {
        "metadata": {
            "name": "pod-a",
            "deletionTimestamp": "2026-06-02T12:00:01Z",
        },
        "status": {
            "phase": "Pending",
            "conditions": [
                {
                    "type": "PodScheduled",
                    "status": "True",
                    "lastTransitionTime": "2026-06-02T12:00:01Z",
                }
            ],
            "containerStatuses": [
                {
                    "name": "taskexecutor",
                    "ready": False,
                    "restartCount": 0,
                    "image": "image:dev",
                    "state": {
                        "waiting": {
                            "reason": "ErrImageNeverPull",
                            "message": "image is absent locally",
                        }
                    },
                }
            ],
        },
    }

    smoke.report_pod_lifecycle_diagnostics(api, "namespace", [pod])

    output = capsys.readouterr().out
    assert "phase: Pending" in output
    assert "deletion timestamp: 2026-06-02T12:00:01Z" in output
    assert "kubectl describe pod -n namespace pod-a" in output
    assert "kubectl logs -n namespace pod-a -c taskexecutor --tail=80" in output
    assert "state: waiting reason=ErrImageNeverPull" in output
    assert "type=Warning reason=ErrImageNeverPull count=2" in output
    assert "Container logs: no container has started yet" in output

    api.list_namespaced_event.assert_called_once_with(
        namespace="namespace",
        field_selector="involvedObject.kind=Pod,involvedObject.name=pod-a",
    )
    api.read_namespaced_pod_log.assert_not_called()


def test_pod_lifecycle_diagnostics_reads_started_container_logs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test lifecycle diagnostics read logs for a started container."""
    api = Mock()
    api.list_namespaced_event.return_value = {"items": []}
    api.read_namespaced_pod_log.return_value = b"hello from taskexecutor\n"
    pod = {
        "metadata": {"name": "pod-a"},
        "status": {
            "phase": "Succeeded",
            "containerStatuses": [
                {
                    "name": "taskexecutor",
                    "ready": False,
                    "restartCount": 0,
                    "state": {
                        "terminated": {
                            "reason": "Completed",
                            "exitCode": 0,
                            "finishedAt": "2026-06-02T12:00:01Z",
                        }
                    },
                }
            ],
        },
    }

    smoke.report_pod_lifecycle_diagnostics(api, "namespace", [pod])

    output = capsys.readouterr().out
    assert "phase: Succeeded" in output
    assert "state: terminated reason=Completed exitCode=0" in output
    assert "hello from taskexecutor" in output
    api.read_namespaced_pod_log.assert_called_once_with(
        name="pod-a",
        namespace="namespace",
        container="taskexecutor",
        tail_lines=80,
    )


def test_command_output_text_decodes_bytes_literal_string() -> None:
    """Test Kubernetes log byte-literal strings print as normal text."""
    assert smoke._command_output_text("b'hello from taskexecutor\\n'") == (
        "hello from taskexecutor\n"
    )


def test_validate_smoke_config_rejects_invalid_budget() -> None:
    """Test harness config rejects a non-positive active Pod budget."""
    config = smoke.SmokeConfig(
        cluster_name="cluster",
        namespace="namespace",
        image="image:dev",
        image_pull_policy="Never",
        appio_api_address="appio:9092",
        active_pod_budget=0,
        capacity_timeout=5.0,
        capacity_poll_interval=0.1,
        keep_resources=False,
        delete_cluster=False,
    )

    with pytest.raises(smoke.SmokeFailure, match="Active Pod budget"):
        smoke.validate_smoke_config(config)


def test_validate_smoke_config_rejects_keep_resources_with_delete_cluster() -> None:
    """Test mutually exclusive cleanup flags are rejected."""
    config = smoke.SmokeConfig(
        cluster_name="cluster",
        namespace="namespace",
        image="image:dev",
        image_pull_policy="Never",
        appio_api_address="appio:9092",
        active_pod_budget=1,
        capacity_timeout=5.0,
        capacity_poll_interval=0.1,
        keep_resources=True,
        delete_cluster=True,
    )

    with pytest.raises(smoke.SmokeFailure, match="cannot be combined"):
        smoke.validate_smoke_config(config)


def test_validate_smoke_config_rejects_probe_mode_without_never_pull() -> None:
    """Test controlled probe mode keeps imagePullPolicy=Never."""
    config = smoke.SmokeConfig(
        cluster_name="cluster",
        namespace="namespace",
        image=smoke.DEFAULT_PROBE_IMAGE,
        image_pull_policy="IfNotPresent",
        appio_api_address="appio:9092",
        active_pod_budget=1,
        capacity_timeout=5.0,
        capacity_poll_interval=0.1,
        keep_resources=False,
        delete_cluster=False,
        probe_image=True,
    )

    with pytest.raises(smoke.SmokeFailure, match="imagePullPolicy=Never"):
        smoke.validate_smoke_config(config)


def test_prepare_probe_image_builds_and_imports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Test probe mode builds the local image and imports it into k3d."""
    commands = []
    monkeypatch.setattr(smoke, "_run_command", commands.append)
    monkeypatch.setattr(smoke, "_probe_image_dir", lambda: tmp_path)
    config = smoke.SmokeConfig(
        cluster_name="cluster",
        namespace="namespace",
        image=smoke.DEFAULT_PROBE_IMAGE,
        image_pull_policy="Never",
        appio_api_address="appio:9092",
        active_pod_budget=1,
        capacity_timeout=5.0,
        capacity_poll_interval=0.1,
        keep_resources=False,
        delete_cluster=False,
        probe_image=True,
    )

    smoke.prepare_probe_image(config)

    assert commands == [
        [
            "docker",
            "build",
            "--tag",
            smoke.DEFAULT_PROBE_IMAGE,
            str(tmp_path),
        ],
        [
            "k3d",
            "image",
            "import",
            smoke.DEFAULT_PROBE_IMAGE,
            "--cluster",
            "cluster",
        ],
    ]


def test_cleanup_deletes_only_objects_matching_selector() -> None:
    """Test cleanup deletes Pods and Secrets returned by labeled list calls."""
    api = Mock()
    api.list_namespaced_pod.return_value = _list_response("pod-a", "pod-b")
    api.list_namespaced_secret.return_value = _list_response("secret-a")

    smoke.cleanup_labeled_objects(api, "namespace", "label=value")

    api.list_namespaced_pod.assert_called_once_with(
        namespace="namespace", label_selector="label=value"
    )
    api.list_namespaced_secret.assert_called_once_with(
        namespace="namespace", label_selector="label=value"
    )
    assert api.delete_namespaced_pod.mock_calls == [
        call(name="pod-a", namespace="namespace", grace_period_seconds=0),
        call(name="pod-b", namespace="namespace", grace_period_seconds=0),
    ]
    api.delete_namespaced_secret.assert_called_once_with(
        name="secret-a", namespace="namespace"
    )


def test_cleanup_ignores_objects_already_removed() -> None:
    """Test cleanup tolerates Kubernetes not-found races."""

    class _NotFound(Exception):
        status = 404

    api = Mock()
    api.list_namespaced_pod.return_value = _list_response("pod-a")
    api.list_namespaced_secret.return_value = _list_response("secret-a")
    api.delete_namespaced_pod.side_effect = _NotFound()
    api.delete_namespaced_secret.side_effect = _NotFound()

    smoke.cleanup_labeled_objects(api, "namespace", "label=value")


def test_wait_for_capacity_proof_deletes_and_unblocks() -> None:
    """Test bounded wait proof runs cleanup and observes unblock."""
    can_finish = threading.Event()
    cleanup = Mock(side_effect=can_finish.set)

    class _Executor:
        def wait_for_capacity(self) -> None:
            can_finish.wait()

    smoke.prove_wait_for_capacity_blocks_and_unblocks(
        executor=_Executor(), cleanup=cleanup, timeout=1.0, block_check_timeout=0.01
    )

    cleanup.assert_called_once_with()


def test_wait_for_capacity_proof_rejects_immediate_return() -> None:
    """Test bounded wait proof fails if wait_for_capacity does not block."""

    class _Executor:
        def wait_for_capacity(self) -> None:
            return

    with pytest.raises(smoke.SmokeFailure, match="returned before smoke Pod cleanup"):
        smoke.prove_wait_for_capacity_blocks_and_unblocks(
            executor=_Executor(),
            cleanup=Mock(),
            timeout=1.0,
            block_check_timeout=0.01,
        )


def test_wait_for_pods_succeeded_returns_when_all_selected_pods_succeed() -> None:
    """Test strict probe wait returns selected Pods once they reach Succeeded."""
    api = Mock()
    pending = {"metadata": {"name": "pod-a"}, "status": {"phase": "Pending"}}
    succeeded = {"metadata": {"name": "pod-a"}, "status": {"phase": "Succeeded"}}
    api.list_namespaced_pod.side_effect = [
        {"items": [pending]},
        {"items": [succeeded]},
    ]

    pods = smoke.wait_for_pods_succeeded(
        api,
        "namespace",
        "label=value",
        expected_count=1,
        timeout=1.0,
        poll_interval=0.01,
    )

    assert pods == [succeeded]


def test_wait_for_pods_succeeded_fails_fast_for_failed_pod() -> None:
    """Test strict probe wait fails when a selected Pod enters Failed."""
    api = Mock()
    api.list_namespaced_pod.return_value = {
        "items": [{"metadata": {"name": "pod-a"}, "status": {"phase": "Failed"}}]
    }
    api.list_namespaced_event.return_value = {"items": []}
    api.read_namespaced_pod.return_value = {
        "metadata": {"name": "pod-a"},
        "status": {"phase": "Failed"},
    }

    with pytest.raises(smoke.SmokeFailure, match="failed before Succeeded"):
        smoke.wait_for_pods_succeeded(
            api,
            "namespace",
            "label=value",
            expected_count=1,
            timeout=1.0,
            poll_interval=0.01,
        )
