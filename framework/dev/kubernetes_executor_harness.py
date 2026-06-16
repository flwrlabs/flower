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
"""Evidence tooling for the integrated k3d launch-path harness.

This dev-maintained module defines the portable event, summary, redaction,
evidence-bundle, and infrastructure-proof shape for the future integrated
harness. The default command only writes the F7a contract scaffold. F7b
infrastructure checks are explicit and dry-run host commands unless ``--execute``
is passed.

Usage:
    python dev/kubernetes_executor_harness.py --output-dir ./f7a-evidence
    python dev/kubernetes_executor_harness.py --output-dir ./f7a-evidence --json
    python dev/kubernetes_executor_harness.py --mode infra-proof \
        --output-dir ./f7b-evidence
    python dev/kubernetes_executor_harness.py --mode real-launch-path \
        --output-dir ./f7c-evidence --execute --apply-manifests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

import yaml

SCHEMA_VERSION = "f7a-harness-v1"
REDACTED = "<redacted>"
HARNESS_EVENT_CLASSES = (
    "harness.start",
    "profile.loaded",
    "cluster.detected",
    "namespace.ready",
    "tls.material.ready",
    "rbac.applied",
    "rbac.negative_check",
    "superlink.pod.ready",
    "appio.seeded",
    "superexec.pod.ready",
    "superexec.claim_observed",
    "kubernetes_executor.pod_created",
    "taskexecutor.pod_phase",
    "taskexecutor.appio_connectivity",
    "capacity.wait_observed",
    "cleanup.observed",
    "policy.not_validated_locally",
    "harness.result",
)

_PEM_BLOCK_RE = re.compile(r"-----BEGIN [^-]+-----.*?-----END [^-]+-----", re.DOTALL)
_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(token|password|api[-_]?key|authorization)\s*([:=])\s*([^\s,;]+)"
)
_SECRET_DATA_KEYS = {"data", "stringData", "binaryData"}
_SENSITIVE_KEYS = {
    "api_key",
    "appio_token",
    "authorization",
    "client_secret",
    "credential",
    "credentials",
    "password",
    "private_key",
    "root_ca_pem",
    "secret_data",
    "task_token",
    "token",
}
_CREDENTIAL_ARG_NAMES = {
    "--api-key",
    "--appio-token",
    "--authorization",
    "--credential",
    "--password",
    "--secret",
    "--token",
}
_ROOT_CERT_ARG_NAMES = {
    "--appio-root-certificates",
    "--root-certificates",
    "--root-certificates-bytes",
}
_TASKEXECUTOR_POD_POLL_INTERVAL_SECONDS = 1.0


@dataclass
class HarnessEvent:
    """One JSONL event emitted by the future integrated harness."""

    event: str
    status: str
    message: str
    details: dict[str, object] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: _utc_now())
    schema_version: str = SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        """Return a JSON-ready event record."""
        if self.event not in HARNESS_EVENT_CLASSES:
            raise ValueError(f"Unsupported harness event class: {self.event}")
        return asdict(self)


@dataclass
class HarnessSummary:
    """Machine-readable summary for one harness evidence bundle."""

    status: str
    result: str
    profile_name: str
    output_dir: str
    started_at: str
    namespace: str
    resource_pool: str
    event_count: int = 0
    completed_at: str | None = None
    failures: list[str] = field(default_factory=list)
    not_validated: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        """Return a JSON-ready summary record."""
        return asdict(self)


@dataclass
class HarnessProfile:  # pylint: disable=too-many-instance-attributes
    """Portable profile sketch for a future integrated k3d harness run."""

    name: str
    cluster_name: str
    namespace: str
    resource_pool: str
    image: str
    image_pull_policy: str
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 600
    cleanup_mode: str = "always"
    appio_tls_mode: str = "test-ca-secret"
    expected_cni: str = "not-validated-locally"
    executor_config_path: str | None = None
    kubectl_context: str | None = None
    appio_root_certificates_path: str | None = None
    superlink_image: str = "flwr/superlink:dev"
    superexec_image: str = "flwr/superexec:dev"
    runtime_image_pull_policy: str = "IfNotPresent"
    tls_secret_name: str = "flower-appio-root-certificates"
    superexec_service_account: str = "flower-superexec"
    rbac_name: str = "flower-superexec-kubernetes-executor"
    superlink_name: str = "flower-superlink"
    superexec_name: str = "flower-superexec"
    seed_job_name: str = "flower-f7-seed-run"
    seed_config_name: str = "flower-f7-seed-run"
    executor_config_name: str = "flower-f7-executor-config"
    appio_api_port: int = 9091
    control_api_port: int = 9093

    def to_mapping(self) -> dict[str, object]:
        """Return the profile as a JSON/YAML-ready mapping."""
        executor_config: dict[str, object] = {
            "namespace": self.namespace,
            "image": self.image,
            "image-pull-policy": self.image_pull_policy,
            "resource-pool": self.resource_pool,
            "labels": dict(self.labels),
            "annotations": dict(self.annotations),
        }
        if self.executor_config_path is not None:
            executor_config["path"] = self.executor_config_path
        return {
            "schema-version": SCHEMA_VERSION,
            "name": self.name,
            "cluster": {
                "type": "k3d",
                "name": self.cluster_name,
                "kubectl-context": _kubectl_context(self),
            },
            "executor-config": executor_config,
            "harness": {
                "timeout-seconds": self.timeout_seconds,
                "cleanup-mode": self.cleanup_mode,
                "appio-tls-mode": self.appio_tls_mode,
                "expected-cni": self.expected_cni,
            },
            "tls": {
                "appio-root-certificates-path": self.appio_root_certificates_path,
                "secret-name": self.tls_secret_name,
            },
            "runtime": {
                "superlink-image": self.superlink_image,
                "superexec-image": self.superexec_image,
                "image-pull-policy": self.runtime_image_pull_policy,
                "superlink-name": self.superlink_name,
                "superexec-name": self.superexec_name,
                "seed-job-name": self.seed_job_name,
                "appio-api-port": self.appio_api_port,
                "control-api-port": self.control_api_port,
            },
            "rbac": {
                "name": self.rbac_name,
                "superexec-service-account": self.superexec_service_account,
                "resources": ["pods", "secrets"],
            },
        }


@dataclass
class CommandResult:
    """Result of one host command used by the optional F7b proof."""

    args: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    dry_run: bool = False


class HostCommandRunner:
    """Run host commands, or report them without execution in dry-run mode."""

    def __init__(self, *, dry_run: bool = True) -> None:
        self.dry_run = dry_run

    def run(self, args: Sequence[str]) -> CommandResult:
        """Run one command and return a captured result."""
        command = [str(arg) for arg in args]
        if self.dry_run:
            return CommandResult(args=command, returncode=0, dry_run=True)
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
        )
        return CommandResult(
            args=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


class EvidenceBundleWriter:
    """Write a portable harness evidence bundle under a selected directory."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.event_count = 0

    def initialize(self) -> None:
        """Create the evidence bundle directory layout."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for directory in ("objects", "diagnostics"):
            (self.output_dir / directory).mkdir(exist_ok=True)

    def write_event(self, event: HarnessEvent) -> None:
        """Append one redacted JSONL event."""
        self.initialize()
        record = redact_sensitive_data(event.to_record())
        with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            file.write("\n")
        self.event_count += 1

    def write_summary(self, summary: HarnessSummary) -> None:
        """Write the redacted ``summary.json`` file."""
        self.write_json("summary.json", summary.to_record())

    def write_json(self, relative_path: str, value: object) -> None:
        """Write redacted JSON inside the evidence bundle."""
        path = self._bundle_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(redact_sensitive_data(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_yaml(self, relative_path: str, value: object) -> None:
        """Write redacted YAML inside the evidence bundle."""
        path = self._bundle_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(redact_sensitive_data(value), sort_keys=True),
            encoding="utf-8",
        )

    def write_text(self, relative_path: str, content: str) -> None:
        """Write redacted text inside the evidence bundle."""
        path = self._bundle_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(redact_text(content), encoding="utf-8")

    def _bundle_path(self, relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Evidence path must stay inside bundle: {relative_path}")
        return self.output_dir / path


def generic_k3d_profile() -> HarnessProfile:
    """Return the default OSS-friendly profile sketch for future k3d runs."""
    return HarnessProfile(
        name="generic-k3d",
        cluster_name="flower-f7",
        namespace="flower-f7",
        resource_pool="generic-k3d",
        image="flwr/superexec:dev",
        image_pull_policy="IfNotPresent",
        labels={
            "flower.ai/harness": "f7",
            "flower.ai/profile": "generic-k3d",
        },
    )


def run_contract_scaffold(
    output_dir: str | Path, *, profile: HarnessProfile | None = None
) -> HarnessSummary:
    """Write a scaffold evidence bundle without running Kubernetes work."""
    profile = profile or generic_k3d_profile()
    writer = EvidenceBundleWriter(output_dir)
    writer.initialize()
    run_id = f"f7a-{uuid4().hex[:12]}"
    started_at = _utc_now()

    writer.write_event(
        HarnessEvent(
            event="harness.start",
            status="passed",
            message="F7a contract scaffold started.",
            details={
                "run_id": run_id,
                "profile": profile.name,
                "output_dir": str(writer.output_dir),
            },
            timestamp=started_at,
        )
    )
    writer.write_event(
        HarnessEvent(
            event="profile.loaded",
            status="passed",
            message="Generic harness profile loaded.",
            details=profile.to_mapping(),
        )
    )
    writer.write_event(
        HarnessEvent(
            event="policy.not_validated_locally",
            status="not_validated",
            message="F7a does not validate CNI, RBAC, or Kubernetes runtime policy.",
            details={
                "expected_cni": profile.expected_cni,
                "reason": "contract scaffold only",
            },
        )
    )

    writer.write_yaml("sanitized-config.yaml", profile.to_mapping())
    writer.write_json("objects/pods.json", {"items": []})
    writer.write_json("objects/services.json", {"items": []})
    writer.write_json("objects/rbac.json", {"items": []})
    writer.write_text(
        "harness.log",
        "F7a scaffold only; no Kubernetes resources were created.\n",
    )
    writer.write_text(
        "diagnostics/commands.txt",
        "No external commands were executed by the F7a scaffold.\n",
    )
    writer.write_text("diagnostics/failures.txt", "")

    writer.write_event(
        HarnessEvent(
            event="harness.result",
            status="passed",
            message="F7a contract scaffold written.",
            details={"run_id": run_id, "result": "scaffold-written"},
        )
    )

    summary = HarnessSummary(
        status="passed",
        result="scaffold-written",
        profile_name=profile.name,
        output_dir=str(writer.output_dir),
        started_at=started_at,
        completed_at=_utc_now(),
        namespace=profile.namespace,
        resource_pool=profile.resource_pool,
        event_count=writer.event_count,
        not_validated=[
            "k3d cluster detection",
            "Kubernetes namespace setup",
            "SuperLink AppIo",
            "SuperExec task claim",
            "TaskExecutor Pod launch",
            "RBAC negative checks",
            "NetworkPolicy/CNI enforcement",
        ],
    )
    writer.write_summary(summary)
    return summary


def run_infra_proof(
    output_dir: str | Path,
    *,
    profile: HarnessProfile | None = None,
    runner: HostCommandRunner | None = None,
    execute: bool = False,
    create_cluster: bool = False,
    apply_manifests: bool = False,
) -> HarnessSummary:
    """Write F7b infra/TLS/RBAC evidence with optional host command execution."""
    profile = profile or generic_k3d_profile()
    runner = runner or HostCommandRunner(dry_run=not execute)
    writer = EvidenceBundleWriter(output_dir)
    writer.initialize()
    run_id = f"f7b-{uuid4().hex[:12]}"
    started_at = _utc_now()
    command_results: list[CommandResult] = []
    failures: list[str] = []

    namespace_manifest = render_namespace_manifest(profile)
    rbac_manifests = render_superexec_rbac_manifests(profile)
    rbac_manifest_list = {
        "apiVersion": "v1",
        "kind": "List",
        "items": rbac_manifests,
    }
    tls_contract = build_tls_material_contract(profile)
    writer.write_yaml("sanitized-config.yaml", profile.to_mapping())
    writer.write_json("objects/namespace.json", namespace_manifest)
    writer.write_yaml("objects/namespace.yaml", namespace_manifest)
    writer.write_json("objects/rbac.json", rbac_manifest_list)
    writer.write_yaml("objects/rbac.yaml", rbac_manifest_list)
    writer.write_json("objects/tls.json", tls_contract)
    writer.write_json("objects/pods.json", {"items": []})
    writer.write_json("objects/services.json", {"items": []})

    def write_event(event: str, status: str, message: str, details: object) -> None:
        writer.write_event(
            HarnessEvent(
                event=event,
                status=status,
                message=message,
                details={
                    "run_id": run_id,
                    "mode": "f7b-infra-proof",
                    "dry_run": not execute,
                    "data": details,
                },
            )
        )

    def run_command(
        args: Sequence[str], failure_context: str, *, record_failure: bool = True
    ) -> CommandResult:
        result = runner.run(args)
        command_results.append(result)
        if result.returncode != 0 and not result.dry_run and record_failure:
            failures.append(f"{failure_context}: {_command_error(result)}")
        return result

    write_event(
        "harness.start",
        "passed",
        "F7b infrastructure proof started.",
        {
            "output_dir": str(writer.output_dir),
            "execute": execute,
            "create_cluster": create_cluster,
            "apply_manifests": apply_manifests,
        },
    )
    write_event(
        "profile.loaded",
        "passed",
        "Generic harness profile loaded.",
        profile.to_mapping(),
    )

    cluster_result = run_command(
        ["k3d", "cluster", "list", profile.cluster_name],
        "k3d cluster detection",
        record_failure=False,
    )
    if cluster_result.returncode != 0 and not cluster_result.dry_run and create_cluster:
        cluster_result = run_command(
            ["k3d", "cluster", "create", profile.cluster_name, "--wait"],
            "k3d cluster creation",
        )
    elif cluster_result.dry_run and create_cluster:
        cluster_result = run_command(
            ["k3d", "cluster", "create", profile.cluster_name, "--wait"],
            "k3d cluster creation",
        )
    elif cluster_result.returncode != 0 and not cluster_result.dry_run:
        failures.append(f"k3d cluster detection: {_command_error(cluster_result)}")
    write_event(
        "cluster.detected",
        _status_from_command(cluster_result, planned_status="planned"),
        "k3d cluster detection command recorded.",
        {
            "cluster_name": profile.cluster_name,
            "kubectl_context": _kubectl_context(profile),
            "command": _command_record(cluster_result),
        },
    )

    namespace_file = writer.output_dir / "objects" / "namespace.yaml"
    namespace_command = (
        _kubectl_args(profile, ["apply", "-f", str(namespace_file)])
        if apply_manifests
        else _kubectl_args(profile, ["get", "namespace", profile.namespace])
    )
    namespace_result = run_command(namespace_command, "namespace readiness")
    write_event(
        "namespace.ready",
        _status_from_command(namespace_result, planned_status="planned"),
        "Namespace manifest rendered and readiness command recorded.",
        {
            "namespace": profile.namespace,
            "manifest": "objects/namespace.yaml",
            "command": _command_record(namespace_result),
        },
    )

    tls_status = "passed" if tls_contract["ready"] else "not_validated"
    if profile.appio_root_certificates_path is not None and not tls_contract["ready"]:
        tls_status = "failed"
        failures.append(
            "TLS material path was configured but could not be read: "
            f"{profile.appio_root_certificates_path}"
        )
    write_event(
        "tls.material.ready",
        tls_status,
        "TLS material contract recorded without PEM content.",
        tls_contract,
    )

    rbac_file = writer.output_dir / "objects" / "rbac.yaml"
    rbac_result = CommandResult(args=[], returncode=0, dry_run=not execute)
    if apply_manifests:
        rbac_result = run_command(
            _kubectl_args(profile, ["apply", "-f", str(rbac_file)]),
            "RBAC manifest apply",
        )
    rbac_status = (
        _status_from_command(rbac_result, planned_status="planned")
        if apply_manifests
        else "planned"
    )
    write_event(
        "rbac.applied",
        rbac_status,
        "SuperExec RBAC manifests rendered and apply command recorded.",
        {
            "manifest": "objects/rbac.yaml",
            "service_account": profile.superexec_service_account,
            "command": _command_record(rbac_result) if rbac_result.args else None,
        },
    )

    rbac_check = _run_rbac_checks(profile, runner, command_results)
    if rbac_check["status"] == "failed":
        failures.extend(str(failure) for failure in rbac_check["failures"])
    write_event(
        "rbac.negative_check",
        str(rbac_check["status"]),
        "RBAC positive and negative authorization checks recorded.",
        rbac_check,
    )

    write_event(
        "policy.not_validated_locally",
        "not_validated",
        "NetworkPolicy/CNI enforcement is not validated by F7b.",
        {
            "expected_cni": profile.expected_cni,
            "reason": "F7b only proves infrastructure/TLS/RBAC readiness",
        },
    )

    result = "infra-proof" if execute else "infra-proof-dry-run"
    status = "failed" if failures else "passed"
    write_event(
        "harness.result",
        status,
        "F7b infrastructure proof evidence written.",
        {"result": result, "failures": failures},
    )
    _write_command_log(writer, command_results)
    writer.write_text("diagnostics/failures.txt", "\n".join(failures))
    writer.write_text(
        "harness.log",
        (
            f"F7b {result} wrote infra/TLS/RBAC evidence for "
            f"namespace {profile.namespace}.\n"
        ),
    )

    not_validated = [
        "SuperLink AppIo",
        "SuperExec task claim",
        "TaskExecutor Pod launch",
        "TaskExecutor AppIo connectivity",
        "capacity wait proof",
        "completed Pod and Secret cleanup proof",
        "NetworkPolicy/CNI enforcement",
    ]
    if not execute:
        not_validated.append("host command execution")
    summary = HarnessSummary(
        status=status,
        result=result,
        profile_name=profile.name,
        output_dir=str(writer.output_dir),
        started_at=started_at,
        completed_at=_utc_now(),
        namespace=profile.namespace,
        resource_pool=profile.resource_pool,
        event_count=writer.event_count,
        failures=failures,
        not_validated=not_validated,
        details={
            "run_id": run_id,
            "dry_run": not execute,
            "cluster_name": profile.cluster_name,
            "kubectl_context": _kubectl_context(profile),
            "tls": tls_contract,
            "rbac": rbac_check,
        },
    )
    writer.write_summary(summary)
    return summary


def run_real_launch_path(
    output_dir: str | Path,
    *,
    profile: HarnessProfile | None = None,
    runner: HostCommandRunner | None = None,
    execute: bool = False,
    create_cluster: bool = False,
    apply_manifests: bool = False,
    import_images: bool = False,
    cleanup: bool = False,
) -> HarnessSummary:
    """Write F7c real AppIo/SuperExec/TaskExecutor launch-path evidence."""
    profile = profile or generic_k3d_profile()
    runner = runner or HostCommandRunner(dry_run=not execute)
    writer = EvidenceBundleWriter(output_dir)
    writer.initialize()
    run_id = f"f7c-{uuid4().hex[:12]}"
    started_at = _utc_now()
    command_results: list[CommandResult] = []
    failures: list[str] = []

    namespace_manifest = render_namespace_manifest(profile)
    rbac_manifest_list = {
        "apiVersion": "v1",
        "kind": "List",
        "items": render_superexec_rbac_manifests(profile),
    }
    executor_config = render_kubernetes_executor_config(profile, run_id)
    real_launch_manifests = render_real_launch_manifests(profile, run_id)
    seed_manifests = render_appio_seed_manifests(profile, run_id)
    tls_contract = build_tls_material_contract(profile)
    image_preflight = build_image_preflight(profile)
    cleanup_plan = build_cleanup_plan(profile)

    writer.write_yaml("sanitized-config.yaml", profile.to_mapping())
    writer.write_json("diagnostics/image-preflight.json", image_preflight)
    writer.write_text(
        "diagnostics/image-preflight.txt", _format_image_preflight(image_preflight)
    )
    writer.write_json("diagnostics/cleanup.json", cleanup_plan)
    writer.write_text(
        "diagnostics/cleanup.txt",
        _format_cleanup_plan(cleanup_plan, cleanup_requested=cleanup),
    )
    writer.write_yaml("objects/namespace.yaml", namespace_manifest)
    writer.write_json("objects/namespace.json", namespace_manifest)
    writer.write_yaml("objects/rbac.yaml", rbac_manifest_list)
    writer.write_json("objects/rbac.json", rbac_manifest_list)
    writer.write_yaml("objects/executor-config.yaml", executor_config)
    writer.write_json("objects/executor-config.json", executor_config)
    writer.write_yaml("objects/real-launch.yaml", _manifest_list(real_launch_manifests))
    writer.write_json("objects/real-launch.json", _manifest_list(real_launch_manifests))
    writer.write_yaml("objects/seed-job.yaml", _manifest_list(seed_manifests))
    writer.write_json("objects/seed-job.json", _manifest_list(seed_manifests))
    writer.write_json("objects/tls.json", tls_contract)
    writer.write_json("objects/pods.json", {"items": []})
    writer.write_json("objects/services.json", {"items": []})

    def write_event(event: str, status: str, message: str, details: object) -> None:
        writer.write_event(
            HarnessEvent(
                event=event,
                status=status,
                message=message,
                details={
                    "run_id": run_id,
                    "mode": "f7c-real-launch-path",
                    "dry_run": not execute,
                    "data": details,
                },
            )
        )

    def run_command(
        args: Sequence[str], failure_context: str, *, record_failure: bool = True
    ) -> CommandResult:
        result = runner.run(args)
        command_results.append(result)
        if result.returncode != 0 and not result.dry_run and record_failure:
            failures.append(f"{failure_context}: {_command_error(result)}")
        return result

    write_event(
        "harness.start",
        "passed",
        "F7c real launch-path proof started.",
        {
            "output_dir": str(writer.output_dir),
            "execute": execute,
            "create_cluster": create_cluster,
            "apply_manifests": apply_manifests,
            "import_images": import_images,
            "cleanup": cleanup,
        },
    )
    write_event(
        "profile.loaded",
        "passed",
        "Generic harness profile loaded.",
        profile.to_mapping(),
    )

    cluster_result = run_command(
        ["k3d", "cluster", "list", profile.cluster_name],
        "k3d cluster detection",
        record_failure=False,
    )
    if cluster_result.returncode != 0 and not cluster_result.dry_run and create_cluster:
        cluster_result = run_command(
            ["k3d", "cluster", "create", profile.cluster_name, "--wait"],
            "k3d cluster creation",
        )
    elif cluster_result.dry_run and create_cluster:
        cluster_result = run_command(
            ["k3d", "cluster", "create", profile.cluster_name, "--wait"],
            "k3d cluster creation",
        )
    elif cluster_result.returncode != 0 and not cluster_result.dry_run:
        failures.append(f"k3d cluster detection: {_command_error(cluster_result)}")
    write_event(
        "cluster.detected",
        _status_from_command(cluster_result, planned_status="planned"),
        "k3d cluster detection command recorded.",
        {
            "cluster_name": profile.cluster_name,
            "kubectl_context": _kubectl_context(profile),
            "command": _command_record(cluster_result),
        },
    )

    image_inspect_result = run_command(
        cast(list[str], image_preflight["docker_inspect_command"]),
        "local runtime image inspection",
    )
    image_import_result = CommandResult(args=[], returncode=0, dry_run=not execute)
    if import_images:
        image_import_result = run_command(
            cast(list[str], image_preflight["k3d_import_command"]),
            "k3d runtime image import",
        )

    namespace_file = writer.output_dir / "objects" / "namespace.yaml"
    namespace_command = (
        _kubectl_args(profile, ["apply", "-f", str(namespace_file)])
        if apply_manifests
        else _kubectl_args(profile, ["get", "namespace", profile.namespace])
    )
    namespace_result = run_command(namespace_command, "namespace readiness")
    write_event(
        "namespace.ready",
        _status_from_command(namespace_result, planned_status="planned"),
        "Namespace manifest rendered and readiness command recorded.",
        {
            "namespace": profile.namespace,
            "manifest": "objects/namespace.yaml",
            "command": _command_record(namespace_result),
        },
    )

    tls_status = "passed" if tls_contract["ready"] else "not_validated"
    write_event(
        "tls.material.ready",
        tls_status,
        "TLS material contract recorded; F7c uses insecure AppIo unless supplied.",
        tls_contract,
    )

    rbac_file = writer.output_dir / "objects" / "rbac.yaml"
    rbac_result = CommandResult(args=[], returncode=0, dry_run=not execute)
    if apply_manifests:
        rbac_result = run_command(
            _kubectl_args(profile, ["apply", "-f", str(rbac_file)]),
            "RBAC manifest apply",
        )
    write_event(
        "rbac.applied",
        (
            _status_from_command(rbac_result, planned_status="planned")
            if apply_manifests
            else "planned"
        ),
        "SuperExec RBAC manifests rendered and apply command recorded.",
        {
            "manifest": "objects/rbac.yaml",
            "service_account": profile.superexec_service_account,
            "command": _command_record(rbac_result) if rbac_result.args else None,
        },
    )

    rbac_check = _run_rbac_checks(profile, runner, command_results)
    if rbac_check["status"] == "failed":
        failures.extend(str(failure) for failure in rbac_check["failures"])
    write_event(
        "rbac.negative_check",
        str(rbac_check["status"]),
        "RBAC positive and negative authorization checks recorded.",
        rbac_check,
    )

    real_launch_file = writer.output_dir / "objects" / "real-launch.yaml"
    runtime_prune_result = CommandResult(args=[], returncode=0, dry_run=not execute)
    if apply_manifests:
        runtime_prune_result = run_command(
            _kubectl_args(
                profile,
                [
                    "delete",
                    "pod",
                    profile.superlink_name,
                    profile.superexec_name,
                    "-n",
                    profile.namespace,
                    "--ignore-not-found=true",
                    "--wait=true",
                ],
            ),
            "previous SuperLink and SuperExec Pod cleanup",
        )
        runtime_apply_result = run_command(
            _kubectl_args(profile, ["apply", "-f", str(real_launch_file)]),
            "SuperLink and SuperExec manifest apply",
        )
    else:
        runtime_apply_result = CommandResult(args=[], returncode=0, dry_run=not execute)

    superlink_ready_result = run_command(
        _kubectl_args(
            profile,
            [
                "wait",
                "--for=condition=Ready",
                f"pod/{profile.superlink_name}",
                "-n",
                profile.namespace,
                f"--timeout={profile.timeout_seconds}s",
            ],
        ),
        "SuperLink Pod readiness",
    )
    write_event(
        "superlink.pod.ready",
        _combined_status(
            [runtime_apply_result, superlink_ready_result],
            planned_status="planned",
        ),
        "SuperLink Pod manifest and readiness command recorded.",
        {
            "pod": profile.superlink_name,
            "service": profile.superlink_name,
            "manifest": "objects/real-launch.yaml",
            "delete_previous": (
                _command_record(runtime_prune_result)
                if runtime_prune_result.args
                else None
            ),
            "apply": (
                _command_record(runtime_apply_result)
                if runtime_apply_result.args
                else None
            ),
            "wait": _command_record(superlink_ready_result),
        },
    )

    superexec_ready_result = run_command(
        _kubectl_args(
            profile,
            [
                "wait",
                "--for=condition=Ready",
                f"pod/{profile.superexec_name}",
                "-n",
                profile.namespace,
                f"--timeout={profile.timeout_seconds}s",
            ],
        ),
        "SuperExec Pod readiness",
    )
    write_event(
        "superexec.pod.ready",
        _combined_status(
            [runtime_apply_result, superexec_ready_result],
            planned_status="planned",
        ),
        "SuperExec Pod manifest and readiness command recorded.",
        {
            "pod": profile.superexec_name,
            "executor_config": "objects/executor-config.yaml",
            "command": _command_record(superexec_ready_result),
        },
    )

    seed_file = writer.output_dir / "objects" / "seed-job.yaml"
    seed_prune_result = CommandResult(args=[], returncode=0, dry_run=not execute)
    seed_apply_result = CommandResult(args=[], returncode=0, dry_run=not execute)
    if apply_manifests:
        seed_prune_result = run_command(
            _kubectl_args(
                profile,
                [
                    "delete",
                    "job",
                    profile.seed_job_name,
                    "-n",
                    profile.namespace,
                    "--ignore-not-found=true",
                    "--wait=true",
                ],
            ),
            "previous AppIo seed Job cleanup",
        )
        seed_apply_result = run_command(
            _kubectl_args(profile, ["apply", "-f", str(seed_file)]),
            "AppIo seed Job manifest apply",
        )
    seed_wait_result = run_command(
        _kubectl_args(
            profile,
            [
                "wait",
                "--for=condition=Complete",
                f"job/{profile.seed_job_name}",
                "-n",
                profile.namespace,
                f"--timeout={profile.timeout_seconds}s",
            ],
        ),
        "AppIo seed Job completion",
    )
    seed_logs_result = run_command(
        _kubectl_args(
            profile,
            ["logs", f"job/{profile.seed_job_name}", "-n", profile.namespace],
        ),
        "AppIo seed Job logs",
        record_failure=False,
    )
    seed_observation = _seed_observation(seed_logs_result)
    if execute and seed_observation["run_id"] is None:
        failures.append("AppIo seed Job did not report a run_id.")
    write_event(
        "appio.seeded",
        _appio_seed_status(seed_apply_result, seed_wait_result, seed_observation),
        "Control API seed Job recorded one deterministic ServerApp run.",
        {
            "job": profile.seed_job_name,
            "run_id": seed_observation["run_id"],
            "manifest": "objects/seed-job.yaml",
            "delete_previous": (
                _command_record(seed_prune_result) if seed_prune_result.args else None
            ),
            "apply": _command_record(seed_apply_result) if seed_apply_result.args else None,
            "wait": _command_record(seed_wait_result),
            "logs": _command_record(seed_logs_result),
        },
    )

    superexec_logs_result = run_command(
        _kubectl_args(
            profile,
            [
                "logs",
                f"pod/{profile.superexec_name}",
                "-n",
                profile.namespace,
                "--tail=200",
            ],
        ),
        "SuperExec logs",
        record_failure=False,
    )
    claim_observation = _superexec_claim_observation(superexec_logs_result)
    write_event(
        "superexec.claim_observed",
        _observation_status(superexec_logs_result, claim_observation["observed"]),
        "SuperExec logs inspected for a task claim or launch marker.",
        {
            "observed": claim_observation["observed"],
            "markers": claim_observation["markers"],
            "command": _command_record(superexec_logs_result),
        },
    )

    taskexecutor_selector = _taskexecutor_selector(profile, run_id)
    taskexecutor_pod_attempts: list[CommandResult] = []
    taskexecutor_pods_result = CommandResult(args=[], returncode=0, dry_run=not execute)
    taskexecutor_observation: dict[str, object] = {"items": [], "phases": []}
    taskexecutor_deadline = time.monotonic() + profile.timeout_seconds
    while True:
        taskexecutor_pods_result = run_command(
            _taskexecutor_pods_args(profile, taskexecutor_selector),
            "TaskExecutor Pod observation",
            record_failure=False,
        )
        taskexecutor_pod_attempts.append(taskexecutor_pods_result)
        taskexecutor_observation = _pod_observation(taskexecutor_pods_result)
        if (
            taskexecutor_observation["items"]
            or not execute
            or taskexecutor_pods_result.dry_run
        ):
            break
        if time.monotonic() >= taskexecutor_deadline:
            if taskexecutor_pods_result.returncode != 0:
                failures.append(
                    f"TaskExecutor Pod observation: "
                    f"{_command_error(taskexecutor_pods_result)}"
                )
            failures.append(
                "No TaskExecutor Pod was observed through the F7c selector "
                "before timeout."
            )
            break
        time.sleep(
            min(
                _TASKEXECUTOR_POD_POLL_INTERVAL_SECONDS,
                max(0.0, taskexecutor_deadline - time.monotonic()),
            )
        )
    taskexecutor_wait_results: list[CommandResult] = []
    taskexecutor_pod_names = _pod_names(taskexecutor_observation)
    if execute and taskexecutor_pod_names:
        for pod_name in taskexecutor_pod_names:
            taskexecutor_wait_results.append(
                run_command(
                    _kubectl_args(
                        profile,
                        [
                            "wait",
                            "--for=jsonpath={.status.phase}=Succeeded",
                            f"pod/{pod_name}",
                            "-n",
                            profile.namespace,
                            f"--timeout={profile.timeout_seconds}s",
                        ],
                    ),
                    f"TaskExecutor Pod {pod_name} terminal phase",
                    record_failure=False,
                )
            )
        taskexecutor_pods_result = run_command(
            _taskexecutor_pods_args(profile, taskexecutor_selector),
            "TaskExecutor Pod terminal observation",
        )
        taskexecutor_observation = _pod_observation(taskexecutor_pods_result)
        taskexecutor_pod_names = _pod_names(taskexecutor_observation)
    taskexecutor_phases = _pod_phases(taskexecutor_observation)
    if execute and taskexecutor_observation["items"]:
        unexpected_phases = [
            phase for phase in taskexecutor_phases if phase != "Succeeded"
        ]
        if not taskexecutor_phases:
            failures.append("TaskExecutor Pod phase was not reported.")
        elif unexpected_phases:
            failures.append(
                "TaskExecutor Pod did not reach Succeeded phase: "
                f"{', '.join(taskexecutor_phases)}."
            )
    taskexecutor_log_results = [
        run_command(
            _kubectl_args(
                profile,
                ["logs", f"pod/{pod_name}", "-n", profile.namespace, "--tail=200"],
            ),
            f"TaskExecutor Pod {pod_name} logs",
            record_failure=False,
        )
        for pod_name in taskexecutor_pod_names
    ]
    writer.write_json("objects/pods.json", taskexecutor_observation)
    writer.write_text(
        "diagnostics/taskexecutor-logs.txt",
        _format_taskexecutor_logs(taskexecutor_log_results),
    )
    write_event(
        "kubernetes_executor.pod_created",
        _taskexecutor_status(taskexecutor_pods_result, taskexecutor_observation),
        "Kubernetes API inspected for executor-created TaskExecutor Pods.",
        {
            "selector": taskexecutor_selector,
            "pods": taskexecutor_observation["items"],
            "command": _command_record(taskexecutor_pods_result),
            "creation_attempts": [
                _command_record(result) for result in taskexecutor_pod_attempts
            ],
            "terminal_waits": [
                _command_record(result) for result in taskexecutor_wait_results
            ],
        },
    )
    write_event(
        "taskexecutor.pod_phase",
        _taskexecutor_phase_status(taskexecutor_pods_result, taskexecutor_observation),
        "TaskExecutor Pod phases recorded from Kubernetes.",
        {
            "selector": taskexecutor_selector,
            "phases": taskexecutor_phases,
            "logs": [_command_record(result) for result in taskexecutor_log_results],
        },
    )
    write_event(
        "taskexecutor.appio_connectivity",
        "not_validated",
        "F7c does not prove TaskExecutor AppIo RPC completion in this slice.",
        {
            "reason": "first slice observes Pod launch only",
            "future_stage": "F7d or follow-up F7c hardening",
        },
    )

    cleanup_result = CommandResult(args=[], returncode=0, dry_run=not execute)
    if cleanup:
        cleanup_result = run_command(
            cast(list[str], cleanup_plan["command"]),
            "harness namespace cleanup",
        )

    result = "real-launch-path" if execute else "real-launch-path-dry-run"
    status = "failed" if failures else "passed"
    write_event(
        "harness.result",
        status,
        "F7c real launch-path evidence written.",
        {"result": result, "failures": failures},
    )
    _write_command_log(writer, command_results)
    writer.write_text("diagnostics/failures.txt", "\n".join(failures))
    writer.write_text(
        "harness.log",
        (
            f"F7c {result} wrote integrated launch-path evidence for "
            f"namespace {profile.namespace}.\n"
        ),
    )

    not_validated = [
        "TaskExecutor AppIo RPC completion",
        "capacity wait proof",
        "completed Pod and Secret cleanup proof",
        "NetworkPolicy/CNI enforcement",
        "production RBAC posture",
    ]
    if not execute:
        not_validated.append("host command execution")
    if tls_status != "passed":
        not_validated.append("AppIo TLS handshake")
    summary = HarnessSummary(
        status=status,
        result=result,
        profile_name=profile.name,
        output_dir=str(writer.output_dir),
        started_at=started_at,
        completed_at=_utc_now(),
        namespace=profile.namespace,
        resource_pool=profile.resource_pool,
        event_count=writer.event_count,
        failures=failures,
        not_validated=not_validated,
        details={
            "run_id": run_id,
            "dry_run": not execute,
            "cluster_name": profile.cluster_name,
            "kubectl_context": _kubectl_context(profile),
            "component_locations": {
                "driver": "host",
                "superlink": f"pod/{profile.superlink_name}",
                "superexec": f"pod/{profile.superexec_name}",
                "taskexecutor": "executor-created pod",
            },
            "selector": taskexecutor_selector,
            "seed_run_id": seed_observation["run_id"],
            "pods": taskexecutor_observation["items"],
            "rbac": rbac_check,
            "image_preflight": {
                "required_images": image_preflight["required_images"],
                "unique_images": image_preflight["unique_images"],
                "docker_inspect": _command_record(image_inspect_result),
                "k3d_import": (
                    _command_record(image_import_result)
                    if image_import_result.args
                    else None
                ),
            },
            "taskexecutor_logs": [
                _command_record(result) for result in taskexecutor_log_results
            ],
            "cleanup": {
                "requested": cleanup,
                "command": cleanup_plan["command"],
                "result": (
                    _command_record(cleanup_result) if cleanup_result.args else None
                ),
            },
        },
    )
    writer.write_summary(summary)
    return summary


def render_namespace_manifest(profile: HarnessProfile) -> dict[str, object]:
    """Render the namespace manifest for the optional local k3d proof."""
    return {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {
            "name": profile.namespace,
            "labels": _harness_object_labels(profile),
        },
    }


def render_superexec_rbac_manifests(
    profile: HarnessProfile,
) -> list[dict[str, object]]:
    """Render minimal SuperExec ServiceAccount, Role, and RoleBinding objects."""
    labels = _harness_object_labels(profile)
    subject = {
        "kind": "ServiceAccount",
        "name": profile.superexec_service_account,
        "namespace": profile.namespace,
    }
    return [
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": profile.superexec_service_account,
                "namespace": profile.namespace,
                "labels": labels,
            },
            "automountServiceAccountToken": True,
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {
                "name": profile.rbac_name,
                "namespace": profile.namespace,
                "labels": labels,
            },
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["pods", "secrets"],
                    "verbs": ["get", "list", "create", "delete"],
                }
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {
                "name": profile.rbac_name,
                "namespace": profile.namespace,
                "labels": labels,
            },
            "subjects": [subject],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": profile.rbac_name,
            },
        },
    ]


def render_kubernetes_executor_config(
    profile: HarnessProfile, run_id: str
) -> dict[str, object]:
    """Render the trusted root-mapping config consumed by SuperExec F1-F4."""
    labels = dict(profile.labels)
    labels["flower.ai/harness-run"] = run_id
    return {
        "namespace": profile.namespace,
        "image": profile.image,
        "image-pull-policy": profile.image_pull_policy,
        "resource-pool": profile.resource_pool,
        "labels": labels,
    }


def render_real_launch_manifests(
    profile: HarnessProfile, run_id: str
) -> list[dict[str, object]]:
    """Render SuperLink, executor config, and SuperExec objects for F7c."""
    labels = _run_object_labels(profile, run_id)
    executor_config_yaml = yaml.safe_dump(
        render_kubernetes_executor_config(profile, run_id),
        sort_keys=True,
    )
    appio_address = f"{profile.superlink_name}:{profile.appio_api_port}"
    executor_config_path = "/etc/flower/executor-config.yaml"
    return [
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": profile.superlink_name,
                "namespace": profile.namespace,
                "labels": labels,
            },
            "spec": {
                "selector": {
                    "app.kubernetes.io/name": "flower",
                    "app.kubernetes.io/component": "superlink",
                    "flower.ai/harness-run": run_id,
                },
                "ports": [
                    {
                        "name": "serverappio",
                        "port": profile.appio_api_port,
                        "targetPort": "serverappio",
                    },
                    {
                        "name": "control",
                        "port": profile.control_api_port,
                        "targetPort": "control",
                    },
                ],
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": profile.superlink_name,
                "namespace": profile.namespace,
                "labels": labels
                | {"app.kubernetes.io/component": "superlink"},
            },
            "spec": {
                "restartPolicy": "Never",
                "automountServiceAccountToken": False,
                "containers": [
                    {
                        "name": "superlink",
                        "image": profile.superlink_image,
                        "imagePullPolicy": profile.runtime_image_pull_policy,
                        "args": [
                            "--insecure",
                            "--isolation",
                            "process",
                            "--serverappio-api-address",
                            f"0.0.0.0:{profile.appio_api_port}",
                            "--control-api-address",
                            f"0.0.0.0:{profile.control_api_port}",
                        ],
                        "ports": [
                            {
                                "name": "serverappio",
                                "containerPort": profile.appio_api_port,
                            },
                            {
                                "name": "control",
                                "containerPort": profile.control_api_port,
                            },
                        ],
                    }
                ],
            },
        },
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": profile.executor_config_name,
                "namespace": profile.namespace,
                "labels": labels,
            },
            "data": {"executor-config.yaml": executor_config_yaml},
        },
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": profile.superexec_name,
                "namespace": profile.namespace,
                "labels": labels
                | {"app.kubernetes.io/component": "superexec"},
            },
            "spec": {
                "serviceAccountName": profile.superexec_service_account,
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": "superexec",
                        "image": profile.superexec_image,
                        "imagePullPolicy": profile.runtime_image_pull_policy,
                        "args": [
                            "--insecure",
                            "--appio-api-address",
                            appio_address,
                            "--plugin-type",
                            "serverapp",
                            "--executor",
                            "kubernetes",
                            "--executor-config",
                            executor_config_path,
                        ],
                        "volumeMounts": [
                            {
                                "name": "executor-config",
                                "mountPath": executor_config_path,
                                "subPath": "executor-config.yaml",
                                "readOnly": True,
                            }
                        ],
                    }
                ],
                "volumes": [
                    {
                        "name": "executor-config",
                        "configMap": {"name": profile.executor_config_name},
                    }
                ],
            },
        },
    ]


def render_appio_seed_manifests(
    profile: HarnessProfile, run_id: str
) -> list[dict[str, object]]:
    """Render the Control API seed Job that creates one ServerApp task."""
    labels = _run_object_labels(profile, run_id)
    control_address = f"{profile.superlink_name}:{profile.control_api_port}"
    return [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": profile.seed_config_name,
                "namespace": profile.namespace,
                "labels": labels,
            },
            "data": {"seed_run.py": _seed_run_script()},
        },
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": profile.seed_job_name,
                "namespace": profile.namespace,
                "labels": labels
                | {"app.kubernetes.io/component": "appio-seed"},
            },
            "spec": {
                "backoffLimit": 0,
                "template": {
                    "metadata": {
                        "labels": labels
                        | {"app.kubernetes.io/component": "appio-seed"},
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "automountServiceAccountToken": False,
                        "containers": [
                            {
                                "name": "seed-run",
                                "image": profile.superlink_image,
                                "imagePullPolicy": profile.runtime_image_pull_policy,
                                "command": ["python"],
                                "args": [
                                    "/opt/flower-f7/seed_run.py",
                                    "--control-api-address",
                                    control_address,
                                ],
                                "volumeMounts": [
                                    {
                                        "name": "seed-script",
                                        "mountPath": "/opt/flower-f7",
                                        "readOnly": True,
                                    }
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "seed-script",
                                "configMap": {"name": profile.seed_config_name},
                            }
                        ],
                    },
                },
            },
        },
    ]


def build_tls_material_contract(profile: HarnessProfile) -> dict[str, object]:
    """Return sanitized TLS material evidence without storing PEM contents."""
    path = (
        Path(profile.appio_root_certificates_path).expanduser()
        if profile.appio_root_certificates_path is not None
        else None
    )
    ready = path is not None and path.is_file()
    root_certificates: dict[str, object] = {
        "source": "configured-path" if path is not None else "planned-test-ca",
        "path": str(path) if path is not None else None,
        "ready": ready,
        "sha256": _sha256_file(path) if ready and path is not None else None,
        "pem_redacted": True,
    }
    return {
        "ready": ready,
        "appio_tls_mode": profile.appio_tls_mode,
        "secret_name": profile.tls_secret_name,
        "root_certificates": root_certificates,
        "taskexecutor_mount_path": "/run/flwr/appio/ca.crt",
    }


def redact_sensitive_data(value: object) -> object:
    """Return ``value`` with credentials, Secret data, and PEM blocks redacted."""
    if isinstance(value, Mapping):
        return _redact_mapping(value)
    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_command_args(args: Sequence[str]) -> list[str]:
    """Redact credential-bearing command arguments while preserving structure."""
    redacted: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg.startswith("--") and "=" in arg:
            flag, value = arg.split("=", maxsplit=1)
            if _is_credential_arg(flag):
                redacted.append(f"{flag}={REDACTED}")
            elif _is_root_cert_arg(flag) and _contains_pem(value):
                redacted.append(f"{flag}={REDACTED}")
            else:
                redacted.append(redact_text(arg))
            index += 1
            continue

        if _is_credential_arg(arg):
            redacted.append(arg)
            if index + 1 < len(args):
                redacted.append(REDACTED)
                index += 2
            else:
                index += 1
            continue

        if _is_root_cert_arg(arg) and index + 1 < len(args):
            redacted.append(arg)
            next_arg = args[index + 1]
            redacted.append(REDACTED if _contains_pem(next_arg) else next_arg)
            index += 2
            continue

        redacted.append(redact_text(arg))
        index += 1

    return redacted


def redact_text(content: str) -> str:
    """Redact inline PEM blocks and credential assignments in text."""
    content = _PEM_BLOCK_RE.sub(REDACTED, content)
    return _CREDENTIAL_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", content
    )


def _redact_mapping(mapping: Mapping[object, object]) -> dict[str, object]:
    is_secret = str(mapping.get("kind")) == "Secret"
    redacted: dict[str, object] = {}
    for key, value in mapping.items():
        key_str = str(key)
        if is_secret and key_str in _SECRET_DATA_KEYS:
            redacted[key_str] = _redact_secret_payload(value)
        elif _is_sensitive_key(key_str):
            redacted[key_str] = REDACTED
        else:
            redacted[key_str] = redact_sensitive_data(value)
    return redacted


def _redact_secret_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): REDACTED for key in value}
    return REDACTED


def _is_sensitive_key(key: str) -> bool:
    return _normalize_key(key) in _SENSITIVE_KEYS


def _is_credential_arg(arg: str) -> bool:
    return arg in _CREDENTIAL_ARG_NAMES


def _is_root_cert_arg(arg: str) -> bool:
    return arg in _ROOT_CERT_ARG_NAMES


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("-", "_").replace(".", "_")


def _contains_pem(value: str) -> bool:
    return "-----BEGIN " in value and "-----END " in value


def _kubectl_context(profile: HarnessProfile) -> str:
    if profile.kubectl_context is not None:
        return profile.kubectl_context
    return f"k3d-{profile.cluster_name}"


def _kubectl_args(profile: HarnessProfile, args: Sequence[str]) -> list[str]:
    return ["kubectl", "--context", _kubectl_context(profile), *args]


def _harness_object_labels(profile: HarnessProfile) -> dict[str, str]:
    labels = dict(profile.labels)
    labels["flower.ai/harness"] = "f7"
    labels["flower.ai/profile"] = profile.name
    labels["flower.ai/resource-pool"] = profile.resource_pool
    return labels


def _run_object_labels(profile: HarnessProfile, run_id: str) -> dict[str, str]:
    labels = _harness_object_labels(profile)
    labels["app.kubernetes.io/name"] = "flower"
    labels["flower.ai/harness-run"] = run_id
    return labels


def _manifest_list(manifests: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [dict(manifest) for manifest in manifests],
    }


def build_image_preflight(profile: HarnessProfile) -> dict[str, object]:
    """Return local image checks and the k3d import command for this profile."""
    required_images = {
        "superlink": profile.superlink_image,
        "superexec": profile.superexec_image,
        "taskexecutor": profile.image,
    }
    unique_images = _unique_values(list(required_images.values()))
    return {
        "required_images": required_images,
        "unique_images": unique_images,
        "docker_inspect_command": ["docker", "image", "inspect", *unique_images],
        "k3d_import_command": [
            "k3d",
            "image",
            "import",
            *unique_images,
            "-c",
            profile.cluster_name,
        ],
    }


def build_cleanup_plan(profile: HarnessProfile) -> dict[str, object]:
    """Return the namespace cleanup command for the harness resources."""
    return {
        "default": "inspectable",
        "command": _kubectl_args(
            profile,
            [
                "delete",
                "namespace",
                profile.namespace,
                "--ignore-not-found=true",
                "--wait=true",
            ],
        ),
        "scope": "namespace",
        "namespace": profile.namespace,
    }


def _format_image_preflight(preflight: Mapping[str, object]) -> str:
    required = cast(Mapping[str, str], preflight["required_images"])
    unique_images = cast(Sequence[str], preflight["unique_images"])
    docker_command = cast(Sequence[str], preflight["docker_inspect_command"])
    import_command = cast(Sequence[str], preflight["k3d_import_command"])
    lines = [
        "F7c runtime image preflight",
        "",
        "Required images:",
        *(f"- {name}: {image}" for name, image in sorted(required.items())),
        "",
        "Unique images:",
        *(f"- {image}" for image in unique_images),
        "",
        "Verify locally:",
        f"$ {shlex.join(docker_command)}",
        "",
        "Import into k3d before a real run, or run with --import-images:",
        f"$ {shlex.join(import_command)}",
        "",
    ]
    return "\n".join(lines)


def _format_cleanup_plan(
    cleanup_plan: Mapping[str, object], *, cleanup_requested: bool
) -> str:
    command = cast(Sequence[str], cleanup_plan["command"])
    requested = "yes" if cleanup_requested else "no"
    return "\n".join(
        [
            "F7c cleanup plan",
            "",
            f"Cleanup requested for this run: {requested}",
            "",
            "Harness module default: leave resources in place for inspection.",
            "One-command wrapper default: pass --cleanup and delete the namespace.",
            "Wrapper --skip-cleanup: leave resources in place for inspection.",
            "",
            "Cleanup command:",
            f"$ {shlex.join(command)}",
            "",
        ]
    )


def _format_taskexecutor_logs(results: Sequence[CommandResult]) -> str:
    if not results:
        return "No TaskExecutor Pod logs were captured.\n"
    sections: list[str] = []
    for result in results:
        command = shlex.join(result.args)
        sections.append(
            "\n".join(
                [
                    f"$ {command}",
                    f"returncode={result.returncode}",
                    "stdout:",
                    result.stdout.rstrip() or "<empty>",
                    "stderr:",
                    result.stderr.rstrip() or "<empty>",
                ]
            )
        )
    return "\n\n".join(sections) + "\n"


def _unique_values(values: Sequence[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        if value and value not in unique:
            unique.append(value)
    return unique


def _seed_run_script() -> str:
    return "\n".join(
        [
            "import argparse",
            "import hashlib",
            "import tempfile",
            "from pathlib import Path",
            "",
            "import grpc",
            "",
            "from flwr.cli.build import build_fab_from_disk",
            "from flwr.common.serde import fab_to_proto",
            "from flwr.proto.control_pb2 import StartRunRequest",
            "from flwr.proto.control_pb2_grpc import ControlStub",
            "from flwr.supercore.constant import NOOP_FEDERATION",
            "from flwr.supercore.fab import Fab",
            "",
            "",
            "def _write_probe_app(app_dir: Path) -> None:",
            "    package_dir = app_dir / 'f7_probe'",
            "    package_dir.mkdir()",
            "    (package_dir / '__init__.py').write_text('', encoding='utf-8')",
            "    (package_dir / 'server_app.py').write_text(",
            "        'import flwr as fl\\n'",
            "        '\\n'",
            "        'app = fl.serverapp.ServerApp()\\n'",
            "        '\\n'",
            "        '\\n'",
            "        '@app.main()\\n'",
            "        'def main(grid, context):\\n'",
            "        '    print(\"F7 probe ServerApp ran\")\\n',",
            "        encoding='utf-8',",
            "    )",
            "    (package_dir / 'client_app.py').write_text(",
            "        'import flwr as fl\\n'",
            "        '\\n'",
            "        'app = fl.clientapp.ClientApp()\\n',",
            "        encoding='utf-8',",
            "    )",
            "    (app_dir / 'pyproject.toml').write_text(",
            "        '[build-system]\\n'",
            "        'requires = [\"hatchling\"]\\n'",
            "        'build-backend = \"hatchling.build\"\\n'",
            "        '\\n'",
            "        '[project]\\n'",
            "        'name = \"f7-probe\"\\n'",
            "        'version = \"0.1.0\"\\n'",
            "        '\\n'",
            "        '[tool.hatch.build.targets.wheel]\\n'",
            "        'packages = [\"f7_probe\"]\\n'",
            "        '\\n'",
            "        '[tool.flwr.app]\\n'",
            "        'publisher = \"flwrlabs\"\\n'",
            "        '\\n'",
            "        '[tool.flwr.app.components]\\n'",
            "        'serverapp = \"f7_probe.server_app:app\"\\n'",
            "        'clientapp = \"f7_probe.client_app:app\"\\n'",
            "        '\\n'",
            "        '[tool.flwr.app.config]\\n',",
            "        encoding='utf-8',",
            "    )",
            "",
            "",
            "def main() -> None:",
            "    parser = argparse.ArgumentParser()",
            "    parser.add_argument('--control-api-address', required=True)",
            "    args = parser.parse_args()",
            "    with tempfile.TemporaryDirectory() as tmp_dir:",
            "        app_dir = Path(tmp_dir)",
            "        _write_probe_app(app_dir)",
            "        fab_bytes = build_fab_from_disk(app_dir)",
            "        fab_hash = hashlib.sha256(fab_bytes).hexdigest()",
            "        channel = grpc.insecure_channel(args.control_api_address)",
            "        grpc.channel_ready_future(channel).result(timeout=60)",
            "        stub = ControlStub(channel)",
            "        response = stub.StartRun(",
            "            StartRunRequest(",
            "                fab=fab_to_proto(Fab(fab_hash, fab_bytes, {})),",
            "                federation=NOOP_FEDERATION,",
            "            )",
            "        )",
            "        if not response.HasField('run_id'):",
            "            raise RuntimeError('Control API did not return a run_id')",
            "        print(f'F7 seed created run_id={response.run_id}')",
            "",
            "",
            "if __name__ == '__main__':",
            "    main()",
            "",
        ]
    )


def _taskexecutor_selector(profile: HarnessProfile, run_id: str) -> str:
    labels = {
        "app.kubernetes.io/name": "flower",
        "app.kubernetes.io/component": "taskexecutor",
        "flower.ai/harness-run": run_id,
    }
    if profile.resource_pool:
        labels["flower.ai/resource-pool"] = profile.resource_pool
    return _label_selector(labels)


def _taskexecutor_pods_args(profile: HarnessProfile, selector: str) -> list[str]:
    return _kubectl_args(
        profile,
        [
            "get",
            "pods",
            "-n",
            profile.namespace,
            "-l",
            selector,
            "-o",
            "json",
        ],
    )


def _label_selector(labels: Mapping[str, str]) -> str:
    return ",".join(f"{key}={value}" for key, value in sorted(labels.items()))


def _run_rbac_checks(
    profile: HarnessProfile,
    runner: HostCommandRunner,
    command_results: list[CommandResult],
) -> dict[str, object]:
    subject = (
        f"system:serviceaccount:{profile.namespace}:"
        f"{profile.superexec_service_account}"
    )
    checks: list[dict[str, object]] = []
    failures: list[str] = []
    planned = False

    for name, expect_allowed, args in _rbac_check_specs(profile, subject):
        result = runner.run(args)
        command_results.append(result)
        if result.dry_run:
            planned = True
            check_status = "planned"
            allowed: bool | None = None
        else:
            allowed = _is_yes_result(result)
            check_status = "passed" if allowed is expect_allowed else "failed"
            if check_status == "failed":
                expectation = "allowed" if expect_allowed else "denied"
                failures.append(f"{name} expected {expectation}.")
        checks.append(
            {
                "name": name,
                "expected_allowed": expect_allowed,
                "allowed": allowed,
                "status": check_status,
                "command": _command_record(result),
            }
        )

    if failures:
        status = "failed"
    elif planned:
        status = "planned"
    else:
        status = "passed"
    return {
        "status": status,
        "subject": subject,
        "checks": checks,
        "failures": failures,
    }


def _rbac_check_specs(
    profile: HarnessProfile, subject: str
) -> list[tuple[str, bool, list[str]]]:
    namespace_args = ["--as", subject, "-n", profile.namespace]
    other_namespace = "default" if profile.namespace != "default" else "kube-system"
    return [
        (
            "can-get-pods",
            True,
            _kubectl_args(profile, ["auth", "can-i", "get", "pods", *namespace_args]),
        ),
        (
            "can-list-pods",
            True,
            _kubectl_args(profile, ["auth", "can-i", "list", "pods", *namespace_args]),
        ),
        (
            "can-create-pods",
            True,
            _kubectl_args(
                profile, ["auth", "can-i", "create", "pods", *namespace_args]
            ),
        ),
        (
            "can-delete-pods",
            True,
            _kubectl_args(
                profile, ["auth", "can-i", "delete", "pods", *namespace_args]
            ),
        ),
        (
            "can-get-secrets",
            True,
            _kubectl_args(
                profile, ["auth", "can-i", "get", "secrets", *namespace_args]
            ),
        ),
        (
            "can-list-secrets",
            True,
            _kubectl_args(
                profile, ["auth", "can-i", "list", "secrets", *namespace_args]
            ),
        ),
        (
            "can-create-secrets",
            True,
            _kubectl_args(
                profile, ["auth", "can-i", "create", "secrets", *namespace_args]
            ),
        ),
        (
            "can-delete-secrets",
            True,
            _kubectl_args(
                profile, ["auth", "can-i", "delete", "secrets", *namespace_args]
            ),
        ),
        (
            "cannot-create-deployments",
            False,
            _kubectl_args(
                profile,
                ["auth", "can-i", "create", "deployments.apps", *namespace_args],
            ),
        ),
        (
            "cannot-create-rolebindings",
            False,
            _kubectl_args(
                profile,
                [
                    "auth",
                    "can-i",
                    "create",
                    "rolebindings.rbac.authorization.k8s.io",
                    *namespace_args,
                ],
            ),
        ),
        (
            "cannot-create-roles",
            False,
            _kubectl_args(
                profile,
                [
                    "auth",
                    "can-i",
                    "create",
                    "roles.rbac.authorization.k8s.io",
                    *namespace_args,
                ],
            ),
        ),
        (
            "cannot-create-serviceaccounts",
            False,
            _kubectl_args(
                profile,
                ["auth", "can-i", "create", "serviceaccounts", *namespace_args],
            ),
        ),
        (
            "cannot-create-pods-exec",
            False,
            _kubectl_args(
                profile,
                [
                    "auth",
                    "can-i",
                    "create",
                    "pods",
                    "--subresource=exec",
                    *namespace_args,
                ],
            ),
        ),
        (
            "cannot-get-secrets-outside-namespace",
            False,
            _kubectl_args(
                profile,
                [
                    "auth",
                    "can-i",
                    "get",
                    "secrets",
                    "--as",
                    subject,
                    "-n",
                    other_namespace,
                ],
            ),
        ),
        (
            "cannot-read-nodes",
            False,
            _kubectl_args(profile, ["auth", "can-i", "get", "nodes", "--as", subject]),
        ),
    ]


def _is_yes_result(result: CommandResult) -> bool:
    if result.returncode != 0:
        return False
    return result.stdout.strip().lower().splitlines()[:1] == ["yes"]


def _status_from_command(result: CommandResult, *, planned_status: str) -> str:
    if result.dry_run:
        return planned_status
    if result.returncode == 0:
        return "passed"
    return "failed"


def _combined_status(
    results: Sequence[CommandResult], *, planned_status: str
) -> str:
    if any(result.returncode != 0 and not result.dry_run for result in results):
        return "failed"
    if any(result.dry_run for result in results):
        return planned_status
    return "passed"


def _appio_seed_status(
    seed_apply_result: CommandResult,
    seed_wait_result: CommandResult,
    seed_observation: Mapping[str, object],
) -> str:
    status = _combined_status(
        [seed_apply_result, seed_wait_result], planned_status="planned"
    )
    if status != "passed":
        return status
    return "passed" if seed_observation.get("run_id") is not None else "failed"


def _observation_status(result: CommandResult, observed: object) -> str:
    if result.dry_run:
        return "planned"
    if result.returncode != 0:
        return "failed"
    if bool(observed):
        return "passed"
    return "not_validated"


def _taskexecutor_status(
    result: CommandResult, observation: Mapping[str, object]
) -> str:
    if result.dry_run:
        return "planned"
    if result.returncode != 0:
        return "failed"
    return "passed" if observation.get("items") else "failed"


def _taskexecutor_phase_status(
    result: CommandResult, observation: Mapping[str, object]
) -> str:
    if result.dry_run:
        return "planned"
    if result.returncode != 0:
        return "failed"
    if not observation.get("items"):
        return "failed"
    phases = _pod_phases(observation)
    if not phases:
        return "failed"
    return "passed" if all(phase == "Succeeded" for phase in phases) else "failed"


def _seed_observation(result: CommandResult) -> dict[str, object]:
    match = re.search(r"\brun_id=(\d+)\b", result.stdout)
    return {
        "run_id": int(match.group(1)) if match is not None else None,
        "dry_run": result.dry_run,
    }


def _superexec_claim_observation(result: CommandResult) -> dict[str, object]:
    combined = f"{result.stdout}\n{result.stderr}".lower()
    markers = [
        marker
        for marker in ("claim", "launch", "task_id", "taskexecutor")
        if marker in combined
    ]
    return {"observed": bool(markers), "markers": markers}


def _pod_observation(result: CommandResult) -> dict[str, object]:
    if result.dry_run or not result.stdout.strip():
        return {"items": [], "phases": []}
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as err:
        return {"items": [], "phases": [], "error": f"invalid pod JSON: {err}"}
    items: list[dict[str, object]] = []
    phases: list[str] = []
    for pod in raw.get("items", []):
        if not isinstance(pod, Mapping):
            continue
        metadata = pod.get("metadata", {})
        status = pod.get("status", {})
        if not isinstance(metadata, Mapping) or not isinstance(status, Mapping):
            continue
        phase = status.get("phase")
        if isinstance(phase, str):
            phases.append(phase)
        items.append(
            {
                "name": metadata.get("name"),
                "namespace": metadata.get("namespace"),
                "labels": metadata.get("labels", {}),
                "phase": phase,
                "reason": status.get("reason"),
                "message": status.get("message"),
            }
        )
    return {"items": items, "phases": phases}


def _pod_names(observation: Mapping[str, object]) -> list[str]:
    names: list[str] = []
    raw_items = observation.get("items", [])
    if not isinstance(raw_items, Sequence):
        return names
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _pod_phases(observation: Mapping[str, object]) -> list[str]:
    raw_phases = observation.get("phases", [])
    if not isinstance(raw_phases, Sequence):
        return []
    return [phase for phase in raw_phases if isinstance(phase, str)]


def _command_record(result: CommandResult) -> dict[str, object]:
    return {
        "args": redact_command_args(result.args),
        "command": _format_command(result.args),
        "returncode": result.returncode,
        "dry_run": result.dry_run,
        "stdout": redact_text(result.stdout.strip()),
        "stderr": redact_text(result.stderr.strip()),
    }


def _command_error(result: CommandResult) -> str:
    stderr = redact_text(result.stderr.strip())
    stdout = redact_text(result.stdout.strip())
    detail = stderr or stdout or "no output"
    return f"{_format_command(result.args)} exited {result.returncode}: {detail}"


def _write_command_log(
    writer: EvidenceBundleWriter, command_results: Sequence[CommandResult]
) -> None:
    lines: list[str] = []
    if not command_results:
        lines.append("No external commands were executed or planned.")
    for result in command_results:
        prefix = "DRY-RUN " if result.dry_run else ""
        lines.append(f"{prefix}$ {_format_command(result.args)}")
        if not result.dry_run:
            lines.append(f"returncode: {result.returncode}")
            if result.stdout.strip():
                lines.append(f"stdout: {redact_text(result.stdout.strip())}")
            if result.stderr.strip():
                lines.append(f"stderr: {redact_text(result.stderr.strip())}")
    writer.write_text("diagnostics/commands.txt", "\n".join(lines) + "\n")


def _format_command(args: Sequence[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in redact_command_args(args))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write F7 integrated k3d harness evidence. The default mode writes "
            "the F7a contract scaffold; infra-proof mode writes F7b "
            "infra/TLS/RBAC evidence; real-launch-path mode writes F7c "
            "SuperLink/SuperExec/TaskExecutor evidence. Host commands only run "
            "with --execute."
        )
    )
    default_profile = generic_k3d_profile()
    parser.add_argument(
        "--mode",
        choices=("contract-scaffold", "infra-proof", "real-launch-path"),
        default="contract-scaffold",
        help=(
            "Write the F7a contract scaffold, F7b infra/TLS/RBAC proof bundle, "
            "or F7c real launch-path bundle."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the portable evidence bundle should be written.",
    )
    parser.add_argument(
        "--profile",
        choices=("generic-k3d",),
        default=default_profile.name,
        help="Harness profile sketch to write.",
    )
    parser.add_argument(
        "--cluster-name",
        default=default_profile.cluster_name,
        help="k3d cluster name planned or used by infra-proof mode.",
    )
    parser.add_argument(
        "--kubectl-context",
        default=None,
        help="kubectl context for infra-proof mode; defaults to k3d-<cluster>.",
    )
    parser.add_argument(
        "--namespace",
        default=default_profile.namespace,
        help="Namespace planned for future k3d harness objects.",
    )
    parser.add_argument(
        "--resource-pool",
        default=default_profile.resource_pool,
        help="Resource-pool label value planned for TaskExecutor Pods.",
    )
    parser.add_argument(
        "--image",
        default=default_profile.image,
        help="TaskExecutor image reference planned for the future harness.",
    )
    parser.add_argument(
        "--superlink-image",
        default=default_profile.superlink_image,
        help="SuperLink image reference for real-launch-path mode.",
    )
    parser.add_argument(
        "--superexec-image",
        default=default_profile.superexec_image,
        help="SuperExec image reference for real-launch-path mode.",
    )
    parser.add_argument(
        "--image-pull-policy",
        default=default_profile.image_pull_policy,
        help="TaskExecutor imagePullPolicy planned for the future harness.",
    )
    parser.add_argument(
        "--runtime-image-pull-policy",
        default=default_profile.runtime_image_pull_policy,
        help="SuperLink/SuperExec/seed imagePullPolicy for real-launch-path mode.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=default_profile.timeout_seconds,
        help="Future harness timeout recorded in the profile sketch.",
    )
    parser.add_argument(
        "--cleanup-mode",
        default=default_profile.cleanup_mode,
        help="Future harness cleanup mode recorded in the profile sketch.",
    )
    parser.add_argument(
        "--appio-root-certificates-path",
        default=None,
        help=(
            "Optional local AppIo root certificate PEM path. The evidence "
            "records only path and SHA-256, never PEM content."
        ),
    )
    parser.add_argument(
        "--tls-secret-name",
        default=default_profile.tls_secret_name,
        help="Planned local Kubernetes Secret name for AppIo root certificates.",
    )
    parser.add_argument(
        "--superexec-service-account",
        default=default_profile.superexec_service_account,
        help="ServiceAccount name rendered for the SuperExec infra proof.",
    )
    parser.add_argument(
        "--superlink-name",
        default=default_profile.superlink_name,
        help="SuperLink Pod and Service name rendered for real-launch-path mode.",
    )
    parser.add_argument(
        "--superexec-name",
        default=default_profile.superexec_name,
        help="SuperExec Pod name rendered for real-launch-path mode.",
    )
    parser.add_argument(
        "--seed-job-name",
        default=default_profile.seed_job_name,
        help="Control API seed Job name rendered for real-launch-path mode.",
    )
    parser.add_argument(
        "--rbac-name",
        default=default_profile.rbac_name,
        help="Role and RoleBinding name rendered for the SuperExec infra proof.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Run host k3d/kubectl commands in infra-proof or real-launch-path "
            "mode. Without this, commands are only recorded as dry-run evidence."
        ),
    )
    parser.add_argument(
        "--create-cluster",
        action="store_true",
        help="Create the k3d cluster if detection fails in infra-proof mode.",
    )
    parser.add_argument(
        "--apply-manifests",
        action="store_true",
        help=(
            "Apply rendered namespace/RBAC/runtime manifests in infra-proof or "
            "real-launch-path mode."
        ),
    )
    parser.add_argument(
        "--import-images",
        action="store_true",
        help=(
            "Import the rendered runtime images into the k3d cluster before "
            "applying real-launch-path manifests."
        ),
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help=(
            "Delete the rendered harness namespace before writing the final "
            "real-launch-path summary. By default resources are left for "
            "inspection."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit compact JSON summary on stdout.",
    )
    return parser.parse_args(argv)


def _profile_from_args(args: argparse.Namespace) -> HarnessProfile:
    return HarnessProfile(
        name=args.profile,
        cluster_name=args.cluster_name,
        namespace=args.namespace,
        resource_pool=args.resource_pool,
        image=args.image,
        image_pull_policy=args.image_pull_policy,
        superlink_image=args.superlink_image,
        superexec_image=args.superexec_image,
        runtime_image_pull_policy=args.runtime_image_pull_policy,
        timeout_seconds=args.timeout_seconds,
        cleanup_mode=args.cleanup_mode,
        kubectl_context=args.kubectl_context,
        appio_root_certificates_path=args.appio_root_certificates_path,
        tls_secret_name=args.tls_secret_name,
        superexec_service_account=args.superexec_service_account,
        superlink_name=args.superlink_name,
        superexec_name=args.superexec_name,
        seed_job_name=args.seed_job_name,
        rbac_name=args.rbac_name,
        labels={
            "flower.ai/harness": "f7",
            "flower.ai/profile": args.profile,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write the requested F7 evidence bundle."""
    args = _parse_args(argv)
    if args.mode == "real-launch-path":
        summary = run_real_launch_path(
            args.output_dir,
            profile=_profile_from_args(args),
            execute=args.execute,
            create_cluster=args.create_cluster,
            apply_manifests=args.apply_manifests,
            import_images=args.import_images,
            cleanup=args.cleanup,
        )
    elif args.mode == "infra-proof":
        summary = run_infra_proof(
            args.output_dir,
            profile=_profile_from_args(args),
            execute=args.execute,
            create_cluster=args.create_cluster,
            apply_manifests=args.apply_manifests,
        )
    else:
        summary = run_contract_scaffold(
            args.output_dir,
            profile=_profile_from_args(args),
        )
    if args.json:
        print(
            json.dumps(
                redact_sensitive_data(summary.to_record()),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        print(f"Wrote {summary.result} harness bundle to {summary.output_dir}")
    return 0 if summary.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
