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
"""Tests for the local k8s launch-path harness."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml


def _load_harness_module() -> ModuleType:
    """Load the dev harness scaffold from its file path."""
    harness_path = Path(__file__).resolve().parents[5] / "dev" / "k8s" / "harness.py"
    spec = importlib.util.spec_from_file_location(
        "kubernetes_executor_harness", harness_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load Kubernetes executor harness module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_verifier_module() -> ModuleType:
    """Load the dev harness verifier from its file path."""
    verifier_path = (
        Path(__file__).resolve().parents[5] / "dev" / "k8s" / "verify_evidence.py"
    )
    spec = importlib.util.spec_from_file_location(
        "verify_kubernetes_executor_harness", verifier_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load Kubernetes executor verifier module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness_module = _load_harness_module()
verifier_module = _load_verifier_module()


def test_run_contract_scaffold_writes_bundle_to_selected_output_dir(
    tmp_path: Path,
) -> None:
    """Test the scaffold writes the portable evidence bundle layout."""
    output_dir = tmp_path / "evidence"

    summary = harness_module.run_contract_scaffold(output_dir)

    assert summary.status == "passed"
    assert summary.result == "scaffold-written"
    assert summary.output_dir == str(output_dir)
    assert summary.event_count == 4
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "events.jsonl").is_file()
    assert (output_dir / "harness.log").is_file()
    assert (output_dir / "sanitized-config.yaml").is_file()
    assert (output_dir / "objects" / "pods.json").is_file()
    assert (output_dir / "objects" / "services.json").is_file()
    assert (output_dir / "objects" / "rbac.json").is_file()
    assert (output_dir / "diagnostics" / "commands.txt").is_file()
    assert (output_dir / "diagnostics" / "failures.txt").is_file()

    events = _read_jsonl(output_dir / "events.jsonl")
    assert [event["event"] for event in events] == [
        "harness.start",
        "profile.loaded",
        "policy.not_validated_locally",
        "harness.result",
    ]
    assert events[2]["status"] == "not_validated"

    summary_record = json.loads((output_dir / "summary.json").read_text())
    assert summary_record["event_count"] == 4
    assert "TaskExecutor Pod launch" in summary_record["not_validated"]

    config = yaml.safe_load((output_dir / "sanitized-config.yaml").read_text())
    assert config["name"] == "generic-k3d"
    assert config["executor-config"]["namespace"] == "flower-local-k8s"
    assert config["executor-config"]["image"] == "flwr/superexec:dev"


def test_writer_redacts_events_summary_yaml_and_text(tmp_path: Path) -> None:
    """Test the writer redacts sensitive fields in each output format."""
    writer = harness_module.EvidenceBundleWriter(tmp_path)
    secret = {
        "kind": "Secret",
        "metadata": {"name": "task-1-appio"},
        "stringData": {
            "token": "task-token",
            "ca.crt": "-----BEGIN CERTIFICATE-----\nroot-ca\n-----END CERTIFICATE-----",
        },
    }

    writer.write_event(
        harness_module.HarnessEvent(
            event="harness.start",
            status="passed",
            message="token=task-token",
            details={"secret_name": "task-1-appio", "secret": secret},
        )
    )
    writer.write_summary(
        harness_module.HarnessSummary(
            status="failed",
            result="failed",
            profile_name="generic-k3d",
            output_dir=str(tmp_path),
            started_at="2026-06-15T00:00:00Z",
            namespace="flower-local-k8s",
            resource_pool="generic-k3d",
            details={"task_token": "task-token"},
        )
    )
    writer.write_yaml("sanitized-config.yaml", secret)
    writer.write_text(
        "harness.log",
        "authorization: bearer-token\n"
        "-----BEGIN CERTIFICATE-----\nroot-ca\n-----END CERTIFICATE-----\n",
    )

    event = _read_jsonl(tmp_path / "events.jsonl")[0]
    assert event["message"] == f"token={harness_module.REDACTED}"
    assert event["details"]["secret_name"] == "task-1-appio"
    assert event["details"]["secret"]["metadata"]["name"] == "task-1-appio"
    assert event["details"]["secret"]["stringData"] == {
        "ca.crt": harness_module.REDACTED,
        "token": harness_module.REDACTED,
    }

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["details"]["task_token"] == harness_module.REDACTED

    config = yaml.safe_load((tmp_path / "sanitized-config.yaml").read_text())
    assert config["metadata"]["name"] == "task-1-appio"
    assert config["stringData"]["token"] == harness_module.REDACTED

    log_text = (tmp_path / "harness.log").read_text()
    assert "bearer-token" not in log_text
    assert "BEGIN CERTIFICATE" not in log_text
    assert harness_module.REDACTED in log_text


def test_redact_command_args_removes_credentials_but_preserves_safe_paths() -> None:
    """Test command redaction keeps argument shape and safe file paths."""
    pem = "-----BEGIN CERTIFICATE-----\nroot-ca\n-----END CERTIFICATE-----"

    redacted = harness_module.redact_command_args(
        [
            "flwr-serverapp",
            "--token",
            "task-token",
            "--token-file",
            "/run/flwr/appio/token",
            "--root-certificates",
            pem,
            "--executor-config=config.yaml",
        ]
    )

    assert redacted == [
        "flwr-serverapp",
        "--token",
        harness_module.REDACTED,
        "--token-file",
        "/run/flwr/appio/token",
        "--root-certificates",
        harness_module.REDACTED,
        "--executor-config=config.yaml",
    ]


def test_writer_rejects_paths_outside_bundle(tmp_path: Path) -> None:
    """Test evidence writes cannot escape the selected output directory."""
    writer = harness_module.EvidenceBundleWriter(tmp_path)

    with pytest.raises(ValueError, match="inside bundle"):
        writer.write_text("../outside.txt", "content")

    with pytest.raises(ValueError, match="inside bundle"):
        writer.write_json(str(tmp_path / "outside.json"), {})


def test_main_writes_json_summary(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Test the minimal command surface writes the scaffold and prints JSON."""
    output_dir = tmp_path / "from-main"

    exit_code = harness_module.main(
        [
            "--output-dir",
            str(output_dir),
            "--namespace",
            "flower-dev",
            "--resource-pool",
            "pool-a",
            "--image",
            "example/taskexecutor:test",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary["status"] == "passed"
    assert summary["namespace"] == "flower-dev"
    assert summary["resource_pool"] == "pool-a"
    assert (output_dir / "summary.json").is_file()


def test_render_superexec_rbac_manifests_scope_pods_and_secrets() -> None:
    """Test local k8s infra proof RBAC manifests grant only Pod and Secret access."""
    profile = harness_module.generic_k3d_profile()

    manifests = harness_module.render_superexec_rbac_manifests(profile)

    assert [manifest["kind"] for manifest in manifests] == [
        "ServiceAccount",
        "Role",
        "RoleBinding",
    ]
    service_account, role, role_binding = manifests
    assert service_account["metadata"]["name"] == "flower-superexec"
    assert service_account["automountServiceAccountToken"] is True

    rules = role["rules"]
    assert rules == [
        {
            "apiGroups": [""],
            "resources": ["pods", "secrets"],
            "verbs": ["get", "list", "create", "delete"],
        }
    ]
    assert "*" not in rules[0]["resources"]
    assert "deployments" not in rules[0]["resources"]
    assert role_binding["subjects"] == [
        {
            "kind": "ServiceAccount",
            "name": "flower-superexec",
            "namespace": "flower-local-k8s",
        }
    ]


def test_tls_material_contract_records_fingerprint_without_pem(tmp_path: Path) -> None:
    """Test TLS evidence records fingerprint and never writes PEM content."""
    pem = "-----BEGIN CERTIFICATE-----\nroot-ca\n-----END CERTIFICATE-----\n"
    ca_path = tmp_path / "appio-ca.pem"
    ca_path.write_text(pem, encoding="utf-8")
    profile = harness_module.generic_k3d_profile()
    profile.appio_root_certificates_path = str(ca_path)

    summary = harness_module.run_infra_proof(tmp_path / "evidence", profile=profile)

    expected_sha256 = hashlib.sha256(pem.encode()).hexdigest()
    tls_text = (tmp_path / "evidence" / "objects" / "tls.json").read_text()
    tls = json.loads(tls_text)
    assert summary.status == "passed"
    assert tls["ready"] is True
    assert tls["root_certificates"]["sha256"] == expected_sha256
    assert "BEGIN CERTIFICATE" not in tls_text
    assert "root-ca" not in tls_text


def test_run_infra_proof_dry_run_writes_local_k8s_infra_evidence(
    tmp_path: Path,
) -> None:
    """Test infra dry-run writes TLS, RBAC, and command evidence."""
    output_dir = tmp_path / "k8s-infra"

    summary = harness_module.run_infra_proof(
        output_dir,
        create_cluster=True,
        apply_manifests=True,
    )

    assert summary.status == "passed"
    assert summary.result == "infra-proof-dry-run"
    assert summary.event_count == 9
    assert (output_dir / "objects" / "namespace.yaml").is_file()
    assert (output_dir / "objects" / "rbac.yaml").is_file()
    assert (output_dir / "objects" / "tls.json").is_file()
    rbac_apply_document = yaml.safe_load(
        (output_dir / "objects" / "rbac.yaml").read_text()
    )
    assert rbac_apply_document["kind"] == "List"
    assert [item["kind"] for item in rbac_apply_document["items"]] == [
        "ServiceAccount",
        "Role",
        "RoleBinding",
    ]

    events = _read_jsonl(output_dir / "events.jsonl")
    assert [event["event"] for event in events] == [
        "harness.start",
        "profile.loaded",
        "cluster.detected",
        "namespace.ready",
        "tls.material.ready",
        "rbac.applied",
        "rbac.negative_check",
        "policy.not_validated_locally",
        "harness.result",
    ]
    assert events[2]["status"] == "planned"
    assert events[5]["status"] == "planned"
    assert events[6]["status"] == "planned"

    summary_record = json.loads((output_dir / "summary.json").read_text())
    assert "host command execution" in summary_record["not_validated"]

    commands_text = (output_dir / "diagnostics" / "commands.txt").read_text()
    assert "DRY-RUN $ k3d cluster list flower-local-k8s" in commands_text
    assert "DRY-RUN $ k3d cluster create flower-local-k8s --wait" in commands_text
    assert "kubectl --context k3d-flower-local-k8s apply -f" in commands_text
    assert "auth can-i delete pods" in commands_text
    assert "auth can-i get secrets" in commands_text
    assert "auth can-i create pods --subresource=exec" in commands_text


def test_run_infra_proof_fails_when_negative_rbac_check_allows_too_much(
    tmp_path: Path,
) -> None:
    """Test local k8s infra proof rejects broader RBAC access."""
    runner = _AllowEverythingRunner()

    summary = harness_module.run_infra_proof(
        tmp_path / "k8s-infra",
        runner=runner,
        execute=True,
        apply_manifests=True,
    )

    assert summary.status == "failed"
    assert any("cannot-create-deployments" in item for item in summary.failures)
    events = _read_jsonl(tmp_path / "k8s-infra" / "events.jsonl")
    assert events[6]["event"] == "rbac.negative_check"
    assert events[6]["status"] == "failed"


def test_main_writes_infra_proof_json_summary(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Test the CLI can write local k8s infra proof dry-run evidence explicitly."""
    output_dir = tmp_path / "from-main"

    exit_code = harness_module.main(
        [
            "--mode",
            "infra-proof",
            "--output-dir",
            str(output_dir),
            "--namespace",
            "flower-dev",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary["status"] == "passed"
    assert summary["result"] == "infra-proof-dry-run"
    assert summary["namespace"] == "flower-dev"
    assert (output_dir / "objects" / "rbac.yaml").is_file()


def test_render_real_launch_manifests_use_superexec_kubernetes_executor() -> None:
    """Test launch manifests run real SuperExec with the Kubernetes executor."""
    profile = harness_module.generic_k3d_profile()

    manifests = harness_module.render_real_launch_manifests(profile, "k8s-launch-test")

    assert [manifest["kind"] for manifest in manifests] == [
        "Service",
        "Pod",
        "ConfigMap",
        "Pod",
    ]
    superlink_service, superlink_pod, executor_config, superexec_pod = manifests
    assert superlink_service["metadata"]["name"] == "flower-superlink"
    assert [port["name"] for port in superlink_service["spec"]["ports"]] == [
        "serverappio",
        "control",
    ]
    assert (
        superlink_pod["metadata"]["labels"]["flower.ai/harness-run"]
        == "k8s-launch-test"
    )
    assert superlink_pod["spec"]["automountServiceAccountToken"] is False
    assert superlink_pod["spec"]["containers"][0]["args"] == [
        "--insecure",
        "--isolation",
        "process",
        "--serverappio-api-address",
        "0.0.0.0:9091",
        "--control-api-address",
        "0.0.0.0:9093",
    ]

    config_yaml = executor_config["data"]["executor-config.yaml"]
    config = yaml.safe_load(config_yaml)
    assert config["namespace"] == "flower-local-k8s"
    assert config["image"] == "flwr/superexec:dev"
    assert config["labels"]["flower.ai/harness-run"] == "k8s-launch-test"

    assert superexec_pod["spec"]["serviceAccountName"] == "flower-superexec"
    superexec_container = superexec_pod["spec"]["containers"][0]
    assert superexec_container["args"] == [
        "--insecure",
        "--appio-api-address",
        "flower-superlink:9091",
        "--plugin-type",
        "serverapp",
        "--executor",
        "kubernetes",
        "--executor-config",
        "/etc/flower/executor-config.yaml",
    ]


def test_render_appio_seed_manifests_create_control_api_job() -> None:
    """Test seed manifests create one Control API StartRun Job."""
    profile = harness_module.generic_k3d_profile()

    manifests = harness_module.render_appio_seed_manifests(profile, "k8s-launch-test")

    assert [manifest["kind"] for manifest in manifests] == ["ConfigMap", "Job"]
    seed_config, seed_job = manifests
    seed_script = seed_config["data"]["seed_run.py"]
    assert sorted(seed_config["data"]) == [
        "launch_probe_client_app.py",
        "launch_probe_init.py",
        "launch_probe_server_app.py",
        "probe_pyproject.toml",
        "seed_run.py",
    ]
    assert "StartRunRequest" in seed_script
    assert "build_fab_from_disk(_PROBE_APP_DIR)" in seed_script
    assert "K8s launch seed created run_id=" in seed_script
    assert "launch_probe.server_app:app" in seed_config["data"]["probe_pyproject.toml"]
    assert (
        "K8s launch probe ServerApp ran"
        in seed_config["data"]["launch_probe_server_app.py"]
    )
    container = seed_job["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["python"]
    assert container["args"] == [
        "/opt/flower-local-k8s/seed_run.py",
        "--control-api-address",
        "flower-superlink:9093",
    ]
    assert container["volumeMounts"] == [
        {
            "name": "seed-assets",
            "mountPath": "/opt/flower-local-k8s",
            "readOnly": True,
        }
    ]
    items = seed_job["spec"]["template"]["spec"]["volumes"][0]["configMap"]["items"]
    assert [item["path"] for item in items] == [
        "seed_run.py",
        "probe_app/pyproject.toml",
        "probe_app/launch_probe/__init__.py",
        "probe_app/launch_probe/server_app.py",
        "probe_app/launch_probe/client_app.py",
    ]
    assert seed_job["spec"]["template"]["spec"]["automountServiceAccountToken"] is False


def test_rendered_local_k8s_outputs_do_not_use_sprint_identifiers() -> None:
    """Test rendered launch objects do not expose sprint-local identifiers."""
    profile = harness_module.generic_k3d_profile()

    rendered = yaml.safe_dump(
        {
            "profile": profile.to_mapping(),
            "runtime": harness_module.render_real_launch_manifests(
                profile, "k8s-launch-test"
            ),
            "seed": harness_module.render_appio_seed_manifests(
                profile, "k8s-launch-test"
            ),
        }
    )

    forbidden_identifiers = (
        "F" + "7",
        "f" + "7c",
        "flower-" + "f" + "7",
        "f" + "7_probe",
        "F" + "7 probe",
    )
    for forbidden in forbidden_identifiers:
        assert forbidden not in rendered


def test_run_local_k8s_launch_path_dry_run_writes_evidence(tmp_path: Path) -> None:
    """Test dry-run writes launch manifests, events, and commands."""
    output_dir = tmp_path / "k8s-launch"

    summary = harness_module.run_local_k8s_launch_path(
        output_dir,
        create_cluster=True,
        apply_manifests=True,
    )

    assert summary.status == "passed"
    assert summary.result == "local-k8s-launch-path-dry-run"
    assert summary.event_count == 15
    assert (output_dir / "objects" / "real-launch.yaml").is_file()
    assert (output_dir / "objects" / "seed-job.yaml").is_file()
    assert (output_dir / "objects" / "executor-config.yaml").is_file()
    assert (output_dir / "invocation.json").is_file()
    assert (output_dir / "task-lineage.json").is_file()
    assert (output_dir / "taskexecutor-pods.json").is_file()
    assert (output_dir / "taskexecutor-secrets.redacted.json").is_file()
    assert (output_dir / "final-state.json").is_file()
    assert (output_dir / "proof-checklist.json").is_file()

    events = _read_jsonl(output_dir / "events.jsonl")
    assert [event["event"] for event in events] == [
        "harness.start",
        "profile.loaded",
        "cluster.detected",
        "namespace.ready",
        "tls.material.ready",
        "rbac.applied",
        "rbac.negative_check",
        "superlink.pod.ready",
        "superexec.pod.ready",
        "appio.seeded",
        "superexec.claim_observed",
        "kubernetes_executor.pod_created",
        "taskexecutor.pod_phase",
        "taskexecutor.appio_connectivity",
        "harness.result",
    ]
    assert events[7]["status"] == "planned"
    assert events[9]["status"] == "planned"
    assert events[11]["status"] == "planned"
    assert events[13]["status"] == "not_validated"

    commands_text = (output_dir / "diagnostics" / "commands.txt").read_text()
    assert "kubectl --context k3d-flower-local-k8s apply -f" in commands_text
    assert "docker image inspect flwr/superlink:dev flwr/superexec:dev" in commands_text
    assert "delete pod flower-superlink flower-superexec" in commands_text
    assert "wait --for=condition=Ready pod/flower-superlink" in commands_text
    assert "wait --for=condition=Ready pod/flower-superexec" in commands_text
    assert "delete job flower-local-k8s-seed-run" in commands_text
    assert (
        "wait --for=condition=Complete job/flower-local-k8s-seed-run" in commands_text
    )
    assert "app.kubernetes.io/component=taskexecutor" in commands_text
    assert (output_dir / "diagnostics" / "image-preflight.txt").is_file()
    assert (output_dir / "diagnostics" / "cleanup.txt").is_file()
    assert (output_dir / "diagnostics" / "taskexecutor-logs.txt").is_file()
    cleanup_text = (output_dir / "diagnostics" / "cleanup.txt").read_text()
    assert "Cleanup requested for this run: no" in cleanup_text
    assert "One-command wrapper default: pass --cleanup" in cleanup_text


def test_run_local_k8s_launch_path_records_terminal_pod_logs_and_cleanup(
    tmp_path: Path,
) -> None:
    """Test execute-mode evidence captures terminal Pod state and logs."""
    runner = _RealLaunchRunner()
    output_dir = tmp_path / "k8s-launch-real"

    summary = harness_module.run_local_k8s_launch_path(
        output_dir,
        runner=runner,
        execute=True,
        apply_manifests=True,
        import_images=True,
        cleanup=True,
    )

    assert summary.status == "passed"
    assert summary.result == "local-k8s-launch-path"
    assert summary.details["seed_run_id"] == 123
    assert summary.details["pods"][0]["phase"] == "Succeeded"
    assert summary.details["credential_secrets"][0]["name"] == (
        "flwr-taskexecutor-123-abc-appio"
    )
    assert summary.details["final_state_counts"]["taskexecutor_pods"] == 1
    assert summary.details["final_state_counts"]["taskexecutor_secrets"] == 1
    assert summary.details["cleanup"]["requested"] is True
    assert summary.details["cleanup"]["result"]["returncode"] == 0
    cleanup_text = (output_dir / "diagnostics" / "cleanup.txt").read_text()
    assert "Cleanup requested for this run: yes" in cleanup_text

    pods = json.loads((output_dir / "objects" / "pods.json").read_text())
    assert pods["phases"] == ["Succeeded"]
    taskexecutor_pods = json.loads((output_dir / "taskexecutor-pods.json").read_text())
    assert taskexecutor_pods["items"][0]["metadata"]["uid"] == "pod-uid-123"

    secret_text = (output_dir / "taskexecutor-secrets.redacted.json").read_text()
    assert "task-token" not in secret_text
    assert "dGFzay10b2tlbg==" not in secret_text
    taskexecutor_secrets = json.loads(secret_text)
    assert taskexecutor_secrets["redacted"] is True
    assert taskexecutor_secrets["command"]["stdout"] == (
        f"{harness_module.REDACTED} Secret list JSON; see summarized items"
    )
    assert taskexecutor_secrets["items"][0]["data_keys"] == ["token"]
    assert taskexecutor_secrets["items"][0]["data_byte_lengths"] == [
        {"bytes": 10, "key": "token"}
    ]

    lineage = json.loads((output_dir / "task-lineage.json").read_text())
    assert lineage["seeded_run_id"] == 123
    assert lineage["tasks"] == [
        {
            "credential_secret_name": "flwr-taskexecutor-123-abc-appio",
            "credential_secret_uid": "secret-uid-123",
            "launch_attempt": "abc",
            "pod_name": "flwr-taskexecutor-123-abc",
            "pod_phase": "Succeeded",
            "pod_uid": "pod-uid-123",
            "resource_pool": "generic-k3d",
            "seeded_run_id": 123,
            "task_id": "123",
            "task_type": "flwr-serverapp",
            "terminal_phase": "Succeeded",
        }
    ]

    final_state_text = (output_dir / "final-state.json").read_text()
    assert "task-token" not in final_state_text
    assert "dGFzay10b2tlbg==" not in final_state_text
    final_state = json.loads((output_dir / "final-state.json").read_text())
    assert final_state["captured_before_namespace_cleanup"] is True
    assert final_state["cleanup_requested"] is True
    assert final_state["commands"]["taskexecutor_secrets"]["stdout"] == (
        f"{harness_module.REDACTED} Secret list JSON; see summarized items"
    )
    assert final_state["counts"] == {
        "jobs": 1,
        "namespace": 1,
        "services": 1,
        "taskexecutor_pods": 1,
        "taskexecutor_secrets": 1,
    }

    checklist = json.loads((output_dir / "proof-checklist.json").read_text())
    assert any("capacity" in item for item in checklist["out_of_scope"])
    assert checklist["claims"][4]["artifact"] == "taskexecutor-secrets.redacted.json"

    taskexecutor_logs = (
        output_dir / "diagnostics" / "taskexecutor-logs.txt"
    ).read_text()
    assert "K8s launch probe ServerApp ran" in taskexecutor_logs

    command_text = (output_dir / "diagnostics" / "commands.txt").read_text()
    assert "task-token" not in command_text
    assert "dGFzay10b2tlbg==" not in command_text
    assert f"stdout: {harness_module.REDACTED} Secret list JSON" in command_text

    commands = [" ".join(command) for command in runner.commands]
    assert any(command.startswith("docker image inspect ") for command in commands)
    assert any(command.startswith("k3d image import ") for command in commands)
    assert any(
        "delete pod flower-superlink flower-superexec" in command
        for command in commands
    )
    assert any(
        "wait --for=jsonpath={.status.phase}=Succeeded pod/flwr-taskexecutor-123-abc"
        in command
        for command in commands
    )
    assert any(
        "delete job flower-local-k8s-seed-run" in command for command in commands
    )
    assert any("logs pod/flwr-taskexecutor-123-abc" in command for command in commands)
    assert any("delete namespace flower-local-k8s" in command for command in commands)


def test_run_local_k8s_launch_path_polls_until_taskexecutor_pod_appears(
    tmp_path: Path,
) -> None:
    """Test delayed TaskExecutor Pod creation is polled before phase wait."""
    runner = _RealLaunchRunner(empty_pod_gets=2)
    output_dir = tmp_path / "k8s-launch-real"
    original_interval = (
        harness_module.real_launch._TASKEXECUTOR_POD_POLL_INTERVAL_SECONDS
    )
    harness_module.real_launch._TASKEXECUTOR_POD_POLL_INTERVAL_SECONDS = 0.0
    try:
        summary = harness_module.run_local_k8s_launch_path(
            output_dir,
            runner=runner,
            execute=True,
            apply_manifests=True,
        )
    finally:
        harness_module.real_launch._TASKEXECUTOR_POD_POLL_INTERVAL_SECONDS = (
            original_interval
        )

    assert summary.status == "passed"
    assert summary.details["pods"][0]["phase"] == "Succeeded"
    assert runner.pod_get_count == 4

    events = _read_jsonl(output_dir / "events.jsonl")
    pod_created = next(
        event for event in events if event["event"] == "kubernetes_executor.pod_created"
    )
    assert pod_created["status"] == "passed"
    attempts = pod_created["details"]["data"]["creation_attempts"]
    assert len(attempts) == 3


def test_run_local_k8s_launch_path_fails_when_taskexecutor_is_not_succeeded(
    tmp_path: Path,
) -> None:
    """Test execute-mode fails if the terminal Pod phase is not Succeeded."""
    runner = _RealLaunchRunner(terminal_phase="Running")
    output_dir = tmp_path / "k8s-launch-real"

    summary = harness_module.run_local_k8s_launch_path(
        output_dir,
        runner=runner,
        execute=True,
        apply_manifests=True,
    )

    assert summary.status == "failed"
    assert any("did not reach Succeeded phase" in item for item in summary.failures)

    events = _read_jsonl(output_dir / "events.jsonl")
    phase_event = next(
        event for event in events if event["event"] == "taskexecutor.pod_phase"
    )
    assert phase_event["status"] == "failed"


def test_main_writes_local_k8s_launch_path_json_summary(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Test the CLI can write launch dry-run evidence explicitly."""
    output_dir = tmp_path / "from-main"

    exit_code = harness_module.main(
        [
            "--mode",
            "local-k8s-launch-path",
            "--output-dir",
            str(output_dir),
            "--namespace",
            "flower-dev",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary["status"] == "passed"
    assert summary["result"] == "local-k8s-launch-path-dry-run"
    assert summary["namespace"] == "flower-dev"
    assert (output_dir / "objects" / "real-launch.yaml").is_file()


def test_verify_local_k8s_launch_evidence_accepts_passing_bundle(
    tmp_path: Path,
) -> None:
    """Test the verifier accepts a passing real-run evidence bundle."""
    output_dir = tmp_path / "evidence"
    _write_verifier_evidence(output_dir)

    failures, report = verifier_module.verify_evidence(output_dir)

    assert failures == []
    assert "Verification: PASSED" in report
    assert "TaskExecutor Pods: 1" in report


def test_verify_local_k8s_launch_evidence_rejects_missing_serverapp_marker(
    tmp_path: Path,
) -> None:
    """Test the verifier requires the probe ServerApp log marker."""
    output_dir = tmp_path / "evidence"
    _write_verifier_evidence(output_dir, taskexecutor_log_text="no marker\n")

    failures, report = verifier_module.verify_evidence(output_dir)

    assert any("K8s launch probe ServerApp ran" in failure for failure in failures)
    assert "Verification: FAILED" in report


class _AllowEverythingRunner:
    """Fake command runner that reports yes for every RBAC can-i check."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, args: list[str]) -> Any:
        """Return success for all commands and broad allow for RBAC checks."""
        self.commands.append(list(args))
        stdout = "yes\n" if "can-i" in args else ""
        return harness_module.CommandResult(
            args=list(args),
            returncode=0,
            stdout=stdout,
        )


class _RealLaunchRunner:
    """Fake command runner for an execute-mode local k8s launch path."""

    def __init__(
        self, *, terminal_phase: str = "Succeeded", empty_pod_gets: int = 0
    ) -> None:
        self.commands: list[list[str]] = []
        self.pod_get_count = 0
        self.terminal_phase = terminal_phase
        self.empty_pod_gets = empty_pod_gets

    def run(self, args: list[str]) -> Any:
        """Return realistic command output for the real-launch harness."""
        self.commands.append(list(args))
        if args[:3] == ["docker", "image", "inspect"]:
            return self._result(args)
        if args[:3] == ["k3d", "cluster", "list"]:
            return self._result(args, stdout="NAME\nflower-local-k8s\n")
        if args[:3] == ["k3d", "image", "import"]:
            return self._result(args, stdout="imported\n")
        if "auth" in args and "can-i" in args:
            allowed = self._rbac_allowed(args)
            return self._result(
                args,
                returncode=0 if allowed else 1,
                stdout="yes\n" if allowed else "no\n",
            )
        if "wait" in args and "--for=jsonpath={.status.phase}=Succeeded" in args:
            return self._result(
                args,
                returncode=0 if self.terminal_phase == "Succeeded" else 1,
                stderr=(
                    ""
                    if self.terminal_phase == "Succeeded"
                    else "timed out waiting for the condition\n"
                ),
            )
        if "get" in args and "pods" in args and "-o" in args and "json" in args:
            self.pod_get_count += 1
            if self.pod_get_count <= self.empty_pod_gets:
                return self._result(args, stdout=json.dumps({"items": []}))
            phase = (
                "Running"
                if self.pod_get_count == self.empty_pod_gets + 1
                else self.terminal_phase
            )
            return self._result(args, stdout=json.dumps(_pod_list(phase)))
        if "get" in args and "secrets" in args and "-o" in args and "json" in args:
            return self._result(args, stdout=json.dumps(_secret_list()))
        if "get" in args and "jobs" in args and "-o" in args and "json" in args:
            return self._result(args, stdout=json.dumps(_object_list("Job")))
        if "get" in args and "services" in args and "-o" in args and "json" in args:
            return self._result(args, stdout=json.dumps(_object_list("Service")))
        if "get" in args and "namespace" in args and "-o" in args and "json" in args:
            return self._result(args, stdout=json.dumps(_namespace()))
        if "logs" in args and "job/flower-local-k8s-seed-run" in args:
            return self._result(args, stdout="K8s launch seed created run_id=123\n")
        if "logs" in args and "pod/flower-superexec" in args:
            return self._result(args, stdout="claim launch task_id taskexecutor\n")
        if "logs" in args and "pod/flwr-taskexecutor-123-abc" in args:
            return self._result(args, stdout="K8s launch probe ServerApp ran\n")
        return self._result(args)

    @staticmethod
    def _result(
        args: list[str], *, returncode: int = 0, stdout: str = "", stderr: str = ""
    ) -> Any:
        return harness_module.CommandResult(
            args=list(args),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    def _rbac_allowed(args: list[str]) -> bool:
        if "--subresource=exec" in args:
            return False
        allowed_specs = {
            ("get", "pods"),
            ("list", "pods"),
            ("create", "pods"),
            ("delete", "pods"),
            ("get", "secrets"),
            ("list", "secrets"),
            ("create", "secrets"),
            ("delete", "secrets"),
        }
        try:
            index = args.index("can-i")
        except ValueError:
            return False
        spec = tuple(args[index + 1 : index + 3])
        return spec in allowed_specs and "-n" in args and "flower-local-k8s" in args


def _pod_list(phase: str) -> dict[str, Any]:
    return {
        "items": [
            {
                "metadata": {
                    "name": "flwr-taskexecutor-123-abc",
                    "namespace": "flower-local-k8s",
                    "uid": "pod-uid-123",
                    "labels": {
                        "app.kubernetes.io/name": "flower",
                        "app.kubernetes.io/component": "taskexecutor",
                        "flower.ai/harness-run": "k8s-launch-test",
                        "flower.ai/launch-attempt": "abc",
                        "flower.ai/resource-pool": "generic-k3d",
                        "flower.ai/superexec-task-id": "123",
                        "flower.ai/task-type": "flwr-serverapp",
                    },
                },
                "spec": {
                    "containers": [
                        {
                            "name": "taskexecutor",
                            "image": "flwr/superexec:dev",
                            "args": [
                                "--serverappio-api-address",
                                "flower-superlink:9091",
                                "--token-file",
                                "/run/flwr/appio/token",
                                "--insecure",
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "appio-credentials",
                            "secret": {"secretName": "flwr-taskexecutor-123-abc-appio"},
                        }
                    ],
                },
                "status": {"phase": phase},
            }
        ]
    }


def _secret_list() -> dict[str, Any]:
    return {
        "items": [
            {
                "kind": "Secret",
                "metadata": {
                    "name": "flwr-taskexecutor-123-abc-appio",
                    "namespace": "flower-local-k8s",
                    "uid": "secret-uid-123",
                    "labels": {
                        "app.kubernetes.io/name": "flower",
                        "app.kubernetes.io/component": "taskexecutor",
                        "flower.ai/harness-run": "k8s-launch-test",
                        "flower.ai/launch-attempt": "abc",
                        "flower.ai/resource-pool": "generic-k3d",
                        "flower.ai/superexec-task-id": "123",
                        "flower.ai/task-type": "flwr-serverapp",
                    },
                },
                "type": "Opaque",
                "data": {"token": "dGFzay10b2tlbg=="},
            }
        ]
    }


def _object_list(kind: str) -> dict[str, Any]:
    return {
        "items": [
            {
                "kind": kind,
                "metadata": {
                    "name": f"flower-local-k8s-{kind.lower()}",
                    "namespace": "flower-local-k8s",
                    "uid": f"{kind.lower()}-uid",
                    "labels": {"flower.ai/harness-run": "k8s-launch-test"},
                },
                "status": {},
            }
        ]
    }


def _namespace() -> dict[str, Any]:
    return {
        "kind": "Namespace",
        "metadata": {"name": "flower-local-k8s", "uid": "namespace-uid"},
        "status": {"phase": "Active"},
    }


def _write_verifier_evidence(
    output_dir: Path, *, taskexecutor_log_text: str = "K8s launch probe ServerApp ran\n"
) -> None:
    (output_dir / "diagnostics").mkdir(parents=True)
    summary = {
        "status": "passed",
        "result": "local-k8s-launch-path",
        "failures": [],
        "details": {
            "run_id": "k8s-launch-test",
            "seed_run_id": 123,
            "dry_run": False,
            "image_preflight": {
                "docker_inspect": {"returncode": 0},
                "k3d_import": {"returncode": 0},
            },
            "rbac": {"status": "passed"},
            "pods": [{"name": "flwr-taskexecutor-test", "phase": "Succeeded"}],
            "taskexecutor_logs": [{"returncode": 0}],
            "cleanup": {"requested": True, "result": {"returncode": 0}},
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (output_dir / "invocation.json").write_text(
        json.dumps({"mode": "local-k8s-launch-path", "dry_run": False}),
        encoding="utf-8",
    )
    (output_dir / "task-lineage.json").write_text(
        json.dumps(
            {
                "seeded_run_id": 123,
                "tasks": [
                    {
                        "pod_name": "flwr-taskexecutor-test",
                        "credential_secret_name": "flwr-taskexecutor-test-appio",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "taskexecutor-pods.json").write_text(
        json.dumps({"items": [{"metadata": {"name": "flwr-taskexecutor-test"}}]}),
        encoding="utf-8",
    )
    (output_dir / "taskexecutor-secrets.redacted.json").write_text(
        json.dumps(
            {
                "redacted": True,
                "items": [
                    {
                        "name": "flwr-taskexecutor-test-appio",
                        "data_keys": ["token"],
                        "redacted": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "final-state.json").write_text(
        json.dumps(
            {
                "captured_before_namespace_cleanup": True,
                "counts": {"taskexecutor_pods": 1, "taskexecutor_secrets": 1},
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "proof-checklist.json").write_text(
        json.dumps(
            {
                "claims": [{"claim": "TaskExecutor Pod observed"}],
                "out_of_scope": ["capacity wait proof"],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "diagnostics" / "taskexecutor-logs.txt").write_text(
        taskexecutor_log_text,
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into records."""
    return [json.loads(line) for line in path.read_text().splitlines()]
