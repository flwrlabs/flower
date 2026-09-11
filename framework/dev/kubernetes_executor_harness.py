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
"""F7a evidence contract scaffold for the integrated k3d launch-path harness.

This dev-maintained module defines the portable event, summary, redaction, and
evidence-bundle shape for a future integrated harness. It does not create a k3d
cluster, talk to Kubernetes, start SuperLink, start SuperExec, or launch a
TaskExecutor Pod.

Usage:
    python dev/kubernetes_executor_harness.py --output-dir ./f7a-evidence
    python dev/kubernetes_executor_harness.py --output-dir ./f7a-evidence --json
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
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
            "executor-config": executor_config,
            "harness": {
                "timeout-seconds": self.timeout_seconds,
                "cleanup-mode": self.cleanup_mode,
                "appio-tls-mode": self.appio_tls_mode,
                "expected-cni": self.expected_cni,
            },
        }


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
        namespace="flower-f7",
        resource_pool="generic-k3d",
        image="ghcr.io/flwrlabs/taskexecutor:dev",
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


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write the F7a integrated k3d harness contract scaffold. This does "
            "not start k3d, Kubernetes, SuperLink, SuperExec, or TaskExecutor."
        )
    )
    default_profile = generic_k3d_profile()
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
        "--image-pull-policy",
        default=default_profile.image_pull_policy,
        help="TaskExecutor imagePullPolicy planned for the future harness.",
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
        "--json",
        action="store_true",
        help="Emit compact JSON summary on stdout.",
    )
    return parser.parse_args(argv)


def _profile_from_args(args: argparse.Namespace) -> HarnessProfile:
    return HarnessProfile(
        name=args.profile,
        namespace=args.namespace,
        resource_pool=args.resource_pool,
        image=args.image,
        image_pull_policy=args.image_pull_policy,
        timeout_seconds=args.timeout_seconds,
        cleanup_mode=args.cleanup_mode,
        labels={
            "flower.ai/harness": "f7",
            "flower.ai/profile": args.profile,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write the F7a contract scaffold evidence bundle."""
    args = _parse_args(argv)
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
        print(f"Wrote F7a harness contract bundle to {summary.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
