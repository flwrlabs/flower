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
"""Tests for the F7a Kubernetes executor harness contract scaffold."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml


def _load_harness_module() -> ModuleType:
    """Load the dev harness scaffold from its file path."""
    harness_path = (
        Path(__file__).resolve().parents[5] / "dev" / ("kubernetes_executor_harness.py")
    )
    spec = importlib.util.spec_from_file_location(
        "kubernetes_executor_harness", harness_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load Kubernetes executor harness module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness_module = _load_harness_module()


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
    assert config["executor-config"]["namespace"] == "flower-f7"
    assert config["executor-config"]["image"] == "ghcr.io/flwrlabs/taskexecutor:dev"


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
            namespace="flower-f7",
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
    """Test F7b RBAC manifests grant only Pod and Secret access."""
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
            "verbs": ["get", "list", "watch", "create", "delete"],
        }
    ]
    assert "*" not in rules[0]["resources"]
    assert "deployments" not in rules[0]["resources"]
    assert role_binding["subjects"] == [
        {
            "kind": "ServiceAccount",
            "name": "flower-superexec",
            "namespace": "flower-f7",
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


def test_run_infra_proof_dry_run_writes_f7b_evidence(tmp_path: Path) -> None:
    """Test F7b dry-run writes infra, TLS, RBAC, and command evidence."""
    output_dir = tmp_path / "f7b"

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
    assert "DRY-RUN $ k3d cluster list flower-f7" in commands_text
    assert "DRY-RUN $ k3d cluster create flower-f7 --wait" in commands_text
    assert "kubectl --context k3d-flower-f7 apply -f" in commands_text
    assert "auth can-i create pods" in commands_text


def test_run_infra_proof_fails_when_negative_rbac_check_allows_too_much(
    tmp_path: Path,
) -> None:
    """Test F7b records a failure when broader RBAC access is allowed."""
    runner = _AllowEverythingRunner()

    summary = harness_module.run_infra_proof(
        tmp_path / "f7b",
        runner=runner,
        execute=True,
        apply_manifests=True,
    )

    assert summary.status == "failed"
    assert any("cannot-create-deployments" in item for item in summary.failures)
    events = _read_jsonl(tmp_path / "f7b" / "events.jsonl")
    assert events[6]["event"] == "rbac.negative_check"
    assert events[6]["status"] == "failed"


def test_main_writes_infra_proof_json_summary(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Test the CLI can write F7b dry-run evidence explicitly."""
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into records."""
    return [json.loads(line) for line in path.read_text().splitlines()]
