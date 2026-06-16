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
"""CLI entrypoint for the local k8s launch-path harness."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import real_launch
from common import (
    CommandResult,
    EvidenceBundleWriter,
    HarnessEvent,
    HarnessProfile,
    HarnessSummary,
    HostCommandRunner,
    REDACTED,
    _command_error,
    _command_record,
    _kubectl_args,
    _kubectl_context,
    _run_rbac_checks,
    _status_from_command,
    _utc_now,
    _write_command_log,
    build_tls_material_contract,
    generic_k3d_profile,
    redact_command_args,
    redact_sensitive_data,
)
from manifests import (
    render_appio_seed_manifests,
    render_kubernetes_executor_config,
    render_namespace_manifest,
    render_real_launch_manifests,
    render_superexec_rbac_manifests,
)
from real_launch import run_local_k8s_launch_path


def run_contract_scaffold(
    output_dir: str | Path, *, profile: HarnessProfile | None = None
) -> HarnessSummary:
    """Write a scaffold evidence bundle without running Kubernetes work."""
    profile = profile or generic_k3d_profile()
    writer = EvidenceBundleWriter(output_dir)
    writer.initialize()
    run_id = f"k8s-scaffold-{uuid4().hex[:12]}"
    started_at = _utc_now()

    writer.write_event(
        HarnessEvent(
            event="harness.start",
            status="passed",
            message="Local k8s contract scaffold started.",
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
            message="The local k8s contract scaffold does not validate CNI, RBAC, or Kubernetes runtime policy.",
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
        "Local k8s scaffold only; no Kubernetes resources were created.\n",
    )
    writer.write_text(
        "diagnostics/commands.txt",
        "No external commands were executed by the local k8s scaffold.\n",
    )
    writer.write_text("diagnostics/failures.txt", "")

    writer.write_event(
        HarnessEvent(
            event="harness.result",
            status="passed",
            message="Local k8s contract scaffold written.",
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
    """Write infra/TLS/RBAC evidence with optional host command execution."""
    profile = profile or generic_k3d_profile()
    runner = runner or HostCommandRunner(dry_run=not execute)
    writer = EvidenceBundleWriter(output_dir)
    writer.initialize()
    run_id = f"k8s-infra-{uuid4().hex[:12]}"
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
                    "mode": "infra-proof",
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
        "Local k8s infrastructure proof started.",
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
        "NetworkPolicy/CNI enforcement is not validated by infra-proof mode.",
        {
            "expected_cni": profile.expected_cni,
            "reason": "infra-proof mode only proves infrastructure/TLS/RBAC readiness",
        },
    )

    result = "infra-proof" if execute else "infra-proof-dry-run"
    status = "failed" if failures else "passed"
    write_event(
        "harness.result",
        status,
        "Local k8s infrastructure proof evidence written.",
        {"result": result, "failures": failures},
    )
    _write_command_log(writer, command_results)
    writer.write_text("diagnostics/failures.txt", "\n".join(failures))
    writer.write_text(
        "harness.log",
        (
            f"Local k8s {result} wrote infra/TLS/RBAC evidence for "
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write local k8s harness evidence. The default mode writes "
            "the contract scaffold; infra-proof mode writes "
            "infra/TLS/RBAC evidence; local-k8s-launch-path mode writes "
            "SuperLink/SuperExec/TaskExecutor evidence. Host commands only run "
            "with --execute."
        )
    )
    default_profile = generic_k3d_profile()
    parser.add_argument(
        "--mode",
        choices=("contract-scaffold", "infra-proof", "local-k8s-launch-path"),
        default="contract-scaffold",
        help=(
            "Write the contract scaffold, infra/TLS/RBAC proof bundle, "
            "or local k8s launch-path bundle."
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
        help="SuperLink image reference for local-k8s-launch-path mode.",
    )
    parser.add_argument(
        "--superexec-image",
        default=default_profile.superexec_image,
        help="SuperExec image reference for local-k8s-launch-path mode.",
    )
    parser.add_argument(
        "--image-pull-policy",
        default=default_profile.image_pull_policy,
        help="TaskExecutor imagePullPolicy planned for the future harness.",
    )
    parser.add_argument(
        "--runtime-image-pull-policy",
        default=default_profile.runtime_image_pull_policy,
        help="SuperLink/SuperExec/seed imagePullPolicy for local-k8s-launch-path mode.",
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
        help="SuperLink Pod and Service name rendered for local-k8s-launch-path mode.",
    )
    parser.add_argument(
        "--superexec-name",
        default=default_profile.superexec_name,
        help="SuperExec Pod name rendered for local-k8s-launch-path mode.",
    )
    parser.add_argument(
        "--seed-job-name",
        default=default_profile.seed_job_name,
        help="Control API seed Job name rendered for local-k8s-launch-path mode.",
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
            "Run host k3d/kubectl commands in infra-proof or local-k8s-launch-path "
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
            "local-k8s-launch-path mode."
        ),
    )
    parser.add_argument(
        "--import-images",
        action="store_true",
        help=(
            "Import the rendered runtime images into the k3d cluster before "
            "applying local k8s launch-path manifests."
        ),
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help=(
            "Delete the rendered harness namespace before writing the final "
            "local k8s launch-path summary. By default resources are left for "
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
            "flower.ai/harness": "local-k8s-launch-path",
            "flower.ai/profile": args.profile,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write the requested local k8s evidence bundle."""
    args = _parse_args(argv)
    if args.mode == "local-k8s-launch-path":
        summary = run_local_k8s_launch_path(
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
