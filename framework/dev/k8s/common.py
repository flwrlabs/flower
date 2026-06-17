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
"""Shared helpers for the local k8s launch-path harness."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import yaml

SCHEMA_VERSION = "local-k8s-harness-v1"
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
@dataclass
class HarnessEvent:
    """One JSONL event emitted by the local k8s harness."""

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
    """Portable profile sketch for a local k8s launch-path harness run."""

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
    seed_job_name: str = "flower-local-k8s-seed-run"
    seed_config_name: str = "flower-local-k8s-seed-run"
    executor_config_name: str = "flower-local-k8s-executor-config"
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
    """Result of one host command used by the optional local k8s infra proof."""

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
    """Return the default OSS-friendly profile sketch for local k8s runs."""
    return HarnessProfile(
        name="generic-k3d",
        cluster_name="flower-local-k8s",
        namespace="flower-local-k8s",
        resource_pool="generic-k3d",
        image="flwr/superexec:dev",
        image_pull_policy="IfNotPresent",
        labels={
            "flower.ai/harness": "local-k8s-launch-path",
            "flower.ai/profile": "generic-k3d",
        },
    )


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
    labels["flower.ai/harness"] = "local-k8s-launch-path"
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
        "Local k8s launch-path runtime image preflight",
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
            "Local k8s launch-path cleanup plan",
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
