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
"""Local k8s launch-path orchestration."""

from __future__ import annotations

import base64
import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

from common import (
    CommandResult,
    EvidenceBundleWriter,
    HarnessEvent,
    HarnessProfile,
    HarnessSummary,
    HostCommandRunner,
    REDACTED,
    SCHEMA_VERSION,
    _command_error,
    _command_record,
    _combined_status,
    _format_cleanup_plan,
    _format_image_preflight,
    _format_taskexecutor_logs,
    _kubectl_args,
    _kubectl_context,
    _manifest_list,
    _run_rbac_checks,
    _status_from_command,
    _utc_now,
    _write_command_log,
    build_cleanup_plan,
    build_image_preflight,
    build_tls_material_contract,
    capacity_cleanup_profile,
    generic_k3d_profile,
    redact_command_args,
)
from manifests import (
    render_appio_seed_manifests,
    render_kubernetes_executor_config,
    render_namespace_manifest,
    render_real_launch_manifests,
    render_superexec_rbac_manifests,
)
from observations import (
    _appio_seed_status,
    _observation_status,
    _pod_names,
    _pod_observation,
    _pod_phases,
    _seed_observation,
    _secret_names,
    _secret_observation,
    _superexec_capacity_wait_observation,
    _superexec_claim_observation,
    _taskexecutor_phase_status,
    _taskexecutor_pods_args,
    _taskexecutor_selector,
    _taskexecutor_status,
)

_TASKEXECUTOR_POD_POLL_INTERVAL_SECONDS = 1.0
_TASK_ID_LABEL = "flower.ai/superexec-task-id"
_LAUNCH_ATTEMPT_LABEL = "flower.ai/launch-attempt"


def run_local_k8s_launch_path(
    output_dir: str | Path,
    *,
    profile: HarnessProfile | None = None,
    runner: HostCommandRunner | None = None,
    execute: bool = False,
    create_cluster: bool = False,
    apply_manifests: bool = False,
    import_images: bool = False,
    cleanup: bool = False,
    capacity_cleanup_proof: bool = False,
) -> HarnessSummary:
    """Write local k8s AppIo/SuperExec/TaskExecutor launch-path evidence."""
    profile = profile or generic_k3d_profile()
    if capacity_cleanup_proof:
        profile = capacity_cleanup_profile(profile)
    runner = runner or HostCommandRunner(dry_run=not execute)
    writer = EvidenceBundleWriter(output_dir)
    writer.initialize()
    run_id = f"k8s-launch-{uuid4().hex[:12]}"
    started_at = _utc_now()
    command_results: list[CommandResult] = []
    failures: list[str] = []
    mode_name = (
        "local-k8s-capacity-cleanup-proof"
        if capacity_cleanup_proof
        else "local-k8s-launch-path"
    )

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

    writer.write_json(
        "invocation.json",
        _invocation_record(
            profile=profile,
            writer=writer,
            run_id=run_id,
            execute=execute,
            create_cluster=create_cluster,
            apply_manifests=apply_manifests,
            import_images=import_images,
            cleanup=cleanup,
            capacity_cleanup_proof=capacity_cleanup_proof,
        ),
    )
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
                    "mode": mode_name,
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
        "Local k8s launch-path proof started.",
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
        (
            "TLS material contract recorded; local k8s launch-path uses insecure "
            "AppIo unless supplied."
        ),
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
    seed_run_ids = cast(list[int], seed_observation["run_ids"])
    if execute and len(seed_run_ids) < profile.seed_run_count:
        failures.append(
            "AppIo seed Job reported "
            f"{len(seed_run_ids)} run IDs, expected {profile.seed_run_count}."
        )
    write_event(
        "appio.seeded",
        _appio_seed_status(seed_apply_result, seed_wait_result, seed_observation),
        "Control API seed Job recorded deterministic ServerApp runs.",
        {
            "job": profile.seed_job_name,
            "run_id": seed_observation["run_id"],
            "run_ids": seed_run_ids,
            "expected_run_count": profile.seed_run_count,
            "manifest": "objects/seed-job.yaml",
            "delete_previous": (
                _command_record(seed_prune_result) if seed_prune_result.args else None
            ),
            "apply": (
                _command_record(seed_apply_result) if seed_apply_result.args else None
            ),
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
    taskexecutor_wait_results: list[CommandResult] = []
    capacity_wait_results: list[CommandResult] = []
    before_cleanup_secret_evidence: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "redacted": True,
        "selector": taskexecutor_selector,
        "items": [],
    }
    after_cleanup_secret_evidence: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "redacted": True,
        "selector": taskexecutor_selector,
        "items": [],
    }
    cleanup_observation: dict[str, object] = {
        "observed": False,
        "removed_pods": [],
        "removed_secrets": [],
        "remaining_pods": [],
        "remaining_secrets": [],
    }
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
                "No TaskExecutor Pod was observed through the local k8s selector "
                "before timeout."
            )
            break
        time.sleep(
            min(
                _TASKEXECUTOR_POD_POLL_INTERVAL_SECONDS,
                max(0.0, taskexecutor_deadline - time.monotonic()),
            )
        )
    first_taskexecutor_observation = taskexecutor_observation
    first_taskexecutor_pod_names = _pod_names(first_taskexecutor_observation)
    blocked_pod_snapshot = _json_list_snapshot(taskexecutor_pods_result)
    if capacity_cleanup_proof:
        before_cleanup_secrets_result = run_command(
            _taskexecutor_secrets_args(profile, taskexecutor_selector),
            "TaskExecutor credential Secret before cleanup observation",
            record_failure=False,
        )
        before_cleanup_secret_snapshot = _json_list_snapshot(
            before_cleanup_secrets_result
        )
        before_cleanup_secret_observation = _secret_observation(
            before_cleanup_secrets_result
        )
        _redact_secret_observation_stdout(before_cleanup_secrets_result)
        before_cleanup_secret_evidence = _taskexecutor_secret_evidence(
            before_cleanup_secret_snapshot,
            selector=taskexecutor_selector,
            command=_command_record(before_cleanup_secrets_result),
        )

        capacity_wait_observation: dict[str, object] = {
            "observed": False,
            "markers": [],
        }
        capacity_wait_result = CommandResult(args=[], returncode=0, dry_run=not execute)
        capacity_deadline = time.monotonic() + profile.timeout_seconds
        while True:
            capacity_wait_result = run_command(
                _kubectl_args(
                    profile,
                    [
                        "logs",
                        f"pod/{profile.superexec_name}",
                        "-n",
                        profile.namespace,
                        "--tail=400",
                    ],
                ),
                "SuperExec capacity wait observation",
                record_failure=False,
            )
            capacity_wait_results.append(capacity_wait_result)
            capacity_wait_observation = _superexec_capacity_wait_observation(
                capacity_wait_result
            )
            if (
                capacity_wait_observation["observed"]
                or not execute
                or capacity_wait_result.dry_run
            ):
                break
            if time.monotonic() >= capacity_deadline:
                failures.append("SuperExec capacity wait was not observed.")
                break
            time.sleep(
                min(
                    _TASKEXECUTOR_POD_POLL_INTERVAL_SECONDS,
                    max(0.0, capacity_deadline - time.monotonic()),
                )
            )
        write_event(
            "capacity.wait_observed",
            _observation_status(
                capacity_wait_result, capacity_wait_observation["observed"]
            ),
            "SuperExec logs inspected for Kubernetes capacity waiting.",
            {
                "active_pod_budget": profile.active_pod_budget,
                "seed_run_ids": seed_run_ids,
                "first_pods": first_taskexecutor_observation["items"],
                "markers": capacity_wait_observation["markers"],
                "commands": [
                    _command_record(result) for result in capacity_wait_results
                ],
            },
        )

        observed_pod_names = set(first_taskexecutor_pod_names)
        second_pod_attempts: list[CommandResult] = []
        second_deadline = time.monotonic() + profile.timeout_seconds
        while True:
            taskexecutor_pods_result = run_command(
                _taskexecutor_pods_args(profile, taskexecutor_selector),
                "Second TaskExecutor Pod observation",
                record_failure=False,
            )
            taskexecutor_pod_attempts.append(taskexecutor_pods_result)
            second_pod_attempts.append(taskexecutor_pods_result)
            taskexecutor_observation = _pod_observation(taskexecutor_pods_result)
            current_pod_names = _pod_names(taskexecutor_observation)
            observed_pod_names.update(current_pod_names)
            new_pod_names = [
                name
                for name in current_pod_names
                if name not in first_taskexecutor_pod_names
            ]
            if (
                new_pod_names
                or len(observed_pod_names) >= profile.seed_run_count
                or not execute
                or taskexecutor_pods_result.dry_run
            ):
                break
            if time.monotonic() >= second_deadline:
                failures.append(
                    "Second TaskExecutor Pod was not observed after capacity opened."
                )
                break
            time.sleep(
                min(
                    _TASKEXECUTOR_POD_POLL_INTERVAL_SECONDS,
                    max(0.0, second_deadline - time.monotonic()),
                )
            )

        after_cleanup_secrets_result = run_command(
            _taskexecutor_secrets_args(profile, taskexecutor_selector),
            "TaskExecutor credential Secret after cleanup observation",
            record_failure=False,
        )
        after_cleanup_secret_snapshot = _json_list_snapshot(
            after_cleanup_secrets_result
        )
        after_cleanup_secret_observation = _secret_observation(
            after_cleanup_secrets_result
        )
        _redact_secret_observation_stdout(after_cleanup_secrets_result)
        after_cleanup_secret_evidence = _taskexecutor_secret_evidence(
            after_cleanup_secret_snapshot,
            selector=taskexecutor_selector,
            command=_command_record(after_cleanup_secrets_result),
        )
        cleanup_observation = _cleanup_observation(
            first_pod_names=first_taskexecutor_pod_names,
            before_cleanup_secrets=before_cleanup_secret_observation,
            after_cleanup_pods=taskexecutor_observation,
            after_cleanup_secrets=after_cleanup_secret_observation,
        )
        if execute and not cleanup_observation["observed"]:
            failures.append(
                "Completed TaskExecutor Pod and credential Secret cleanup was "
                "not observed before namespace cleanup."
            )
        writer.write_json("objects/capacity-blocked-pods.json", blocked_pod_snapshot)
        writer.write_json(
            "objects/secrets-before-cleanup.redacted.json",
            before_cleanup_secret_evidence,
        )
        writer.write_json("objects/cleanup-pods.json", taskexecutor_observation)
        writer.write_json(
            "objects/secrets-after-cleanup.redacted.json",
            after_cleanup_secret_evidence,
        )
        write_event(
            "cleanup.observed",
            (
                "planned"
                if not execute or taskexecutor_pods_result.dry_run
                else ("passed" if cleanup_observation["observed"] else "failed")
            ),
            "Completed TaskExecutor Pod and Secret cleanup inspected.",
            {
                "selector": taskexecutor_selector,
                "observation": cleanup_observation,
                "first_pods": first_taskexecutor_observation["items"],
                "pods_after_cleanup": taskexecutor_observation["items"],
                "secrets_before_cleanup": before_cleanup_secret_observation["items"],
                "secrets_after_cleanup": after_cleanup_secret_observation["items"],
                "second_pod_attempts": [
                    _command_record(result) for result in second_pod_attempts
                ],
            },
        )

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
    taskexecutor_pod_snapshot = _json_list_snapshot(taskexecutor_pods_result)
    lineage_pod_snapshot = (
        _merge_object_list_snapshots(blocked_pod_snapshot, taskexecutor_pod_snapshot)
        if capacity_cleanup_proof
        else taskexecutor_pod_snapshot
    )
    writer.write_json("objects/pods.json", taskexecutor_observation)
    writer.write_json("taskexecutor-pods.json", lineage_pod_snapshot)
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
        "This local k8s slice does not prove TaskExecutor AppIo RPC completion.",
        {
            "reason": "first slice observes Pod launch only",
            "future_stage": "future local k8s hardening",
        },
    )

    taskexecutor_secrets_result = run_command(
        _taskexecutor_secrets_args(profile, taskexecutor_selector),
        "TaskExecutor credential Secret observation",
    )
    taskexecutor_secret_snapshot = _json_list_snapshot(taskexecutor_secrets_result)
    _redact_secret_observation_stdout(taskexecutor_secrets_result)
    taskexecutor_secret_evidence = _taskexecutor_secret_evidence(
        taskexecutor_secret_snapshot,
        selector=taskexecutor_selector,
        command=_command_record(taskexecutor_secrets_result),
    )
    lineage_secret_evidence = (
        _merge_secret_evidence(
            before_cleanup_secret_evidence, taskexecutor_secret_evidence
        )
        if capacity_cleanup_proof
        else taskexecutor_secret_evidence
    )
    writer.write_json(
        "taskexecutor-secrets.redacted.json", taskexecutor_secret_evidence
    )
    if (
        execute
        and taskexecutor_observation["items"]
        and not taskexecutor_secret_evidence["items"]
    ):
        failures.append(
            "No TaskExecutor credential Secret was observed through the local k8s "
            "selector before namespace cleanup."
        )

    task_lineage = _task_lineage(
        profile=profile,
        run_id=run_id,
        mode=mode_name,
        seed_run_id=seed_observation["run_id"],
        seed_run_ids=seed_run_ids,
        selector=taskexecutor_selector,
        pod_snapshot=lineage_pod_snapshot,
        secret_evidence=lineage_secret_evidence,
    )
    writer.write_json("task-lineage.json", task_lineage)

    final_state = _final_state_record(
        profile=profile,
        run_id=run_id,
        mode=mode_name,
        cleanup_requested=cleanup,
        taskexecutor_selector=taskexecutor_selector,
        pod_snapshot=taskexecutor_pod_snapshot,
        pod_result=taskexecutor_pods_result,
        secret_evidence=taskexecutor_secret_evidence,
        secret_result=taskexecutor_secrets_result,
        run_command=run_command,
    )
    writer.write_json("final-state.json", final_state)

    cleanup_result = CommandResult(args=[], returncode=0, dry_run=not execute)
    if cleanup:
        cleanup_result = run_command(
            cast(list[str], cleanup_plan["command"]),
            "harness namespace cleanup",
        )

    if capacity_cleanup_proof:
        result = (
            "local-k8s-capacity-cleanup-proof"
            if execute
            else "local-k8s-capacity-cleanup-proof-dry-run"
        )
    else:
        result = "local-k8s-launch-path" if execute else "local-k8s-launch-path-dry-run"
    status = "failed" if failures else "passed"
    writer.write_json(
        "proof-checklist.json",
        _proof_checklist(
            status=status,
            dry_run=not execute,
            run_id=run_id,
            cleanup_requested=cleanup,
            capacity_cleanup_proof=capacity_cleanup_proof,
            active_pod_budget=profile.active_pod_budget,
            seed_run_count=profile.seed_run_count,
        ),
    )
    write_event(
        "harness.result",
        status,
        "Local k8s launch-path evidence written.",
        {"result": result, "failures": failures},
    )
    _write_command_log(writer, command_results)
    writer.write_text("diagnostics/failures.txt", "\n".join(failures))
    writer.write_text(
        "harness.log",
        (
            f"Local k8s {result} wrote integrated launch-path evidence for "
            f"namespace {profile.namespace}.\n"
        ),
    )

    not_validated = [
        "TaskExecutor AppIo RPC completion",
        "NetworkPolicy/CNI enforcement",
        "production RBAC posture",
    ]
    if not capacity_cleanup_proof or not execute:
        not_validated.extend(
            [
                "capacity wait proof",
                "completed Pod and Secret cleanup proof",
            ]
        )
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
            "seed_run_ids": seed_run_ids,
            "expected_seed_run_count": profile.seed_run_count,
            "active_pod_budget": profile.active_pod_budget,
            "pods": taskexecutor_observation["items"],
            "credential_secrets": taskexecutor_secret_evidence["items"],
            "final_state_counts": final_state["counts"],
            "artifacts": {
                "invocation": "invocation.json",
                "task_lineage": "task-lineage.json",
                "taskexecutor_pods": "taskexecutor-pods.json",
                "taskexecutor_secrets": "taskexecutor-secrets.redacted.json",
                "final_state": "final-state.json",
                "proof_checklist": "proof-checklist.json",
                "capacity_blocked_pods": (
                    "objects/capacity-blocked-pods.json"
                    if capacity_cleanup_proof
                    else None
                ),
                "secrets_before_cleanup": (
                    "objects/secrets-before-cleanup.redacted.json"
                    if capacity_cleanup_proof
                    else None
                ),
                "cleanup_pods": (
                    "objects/cleanup-pods.json" if capacity_cleanup_proof else None
                ),
                "secrets_after_cleanup": (
                    "objects/secrets-after-cleanup.redacted.json"
                    if capacity_cleanup_proof
                    else None
                ),
            },
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
            "capacity_wait": {
                "observed": bool(capacity_wait_results)
                and any(
                    _superexec_capacity_wait_observation(result)["observed"]
                    for result in capacity_wait_results
                ),
                "commands": [
                    _command_record(result) for result in capacity_wait_results
                ],
            },
            "cleanup_observed": cleanup_observation,
            "secrets": {
                "before_cleanup": before_cleanup_secret_evidence["items"],
                "after_cleanup": after_cleanup_secret_evidence["items"],
            },
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


def _invocation_record(
    *,
    profile: HarnessProfile,
    writer: EvidenceBundleWriter,
    run_id: str,
    execute: bool,
    create_cluster: bool,
    apply_manifests: bool,
    import_images: bool,
    cleanup: bool,
    capacity_cleanup_proof: bool,
) -> dict[str, object]:
    """Return reviewer-facing inputs for one local k8s launch-path run."""
    cwd = Path.cwd()
    mode_name = (
        "local-k8s-capacity-cleanup-proof"
        if capacity_cleanup_proof
        else "local-k8s-launch-path"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode_name,
        "run_id": run_id,
        "dry_run": not execute,
        "cwd": str(cwd),
        "repo": _git_context(cwd),
        "output_dir": str(writer.output_dir),
        "profile_name": profile.name,
        "profile": profile.to_mapping(),
        "settings": {
            "execute": execute,
            "create_cluster": create_cluster,
            "apply_manifests": apply_manifests,
            "import_images": import_images,
            "cleanup_requested": cleanup,
            "capacity_cleanup_proof": capacity_cleanup_proof,
            "active_pod_budget": profile.active_pod_budget,
            "seed_run_count": profile.seed_run_count,
            "probe_hold_seconds": profile.probe_hold_seconds,
        },
        "equivalent_argv": _equivalent_argv(
            profile=profile,
            output_dir=writer.output_dir,
            execute=execute,
            create_cluster=create_cluster,
            apply_manifests=apply_manifests,
            import_images=import_images,
            cleanup=cleanup,
            capacity_cleanup_proof=capacity_cleanup_proof,
        ),
    }


def _equivalent_argv(
    *,
    profile: HarnessProfile,
    output_dir: Path,
    execute: bool,
    create_cluster: bool,
    apply_manifests: bool,
    import_images: bool,
    cleanup: bool,
    capacity_cleanup_proof: bool,
) -> list[str]:
    args = [
        "python",
        "framework/dev/k8s/harness.py",
        "--mode",
        (
            "capacity-cleanup-proof"
            if capacity_cleanup_proof
            else "local-k8s-launch-path"
        ),
        "--output-dir",
        str(output_dir),
        "--cluster-name",
        profile.cluster_name,
        "--namespace",
        profile.namespace,
        "--resource-pool",
        profile.resource_pool,
        "--image",
        profile.image,
        "--superlink-image",
        profile.superlink_image,
        "--superexec-image",
        profile.superexec_image,
        "--timeout-seconds",
        str(profile.timeout_seconds),
        "--seed-run-count",
        str(profile.seed_run_count),
        "--probe-hold-seconds",
        str(profile.probe_hold_seconds),
    ]
    if profile.active_pod_budget is not None:
        args.extend(["--active-pod-budget", str(profile.active_pod_budget)])
    if profile.capacity_poll_interval is not None:
        args.extend(["--capacity-poll-interval", str(profile.capacity_poll_interval)])
    if profile.capacity_log_interval is not None:
        args.extend(["--capacity-log-interval", str(profile.capacity_log_interval)])
    if execute:
        args.append("--execute")
    if create_cluster:
        args.append("--create-cluster")
    if apply_manifests:
        args.append("--apply-manifests")
    if import_images:
        args.append("--import-images")
    if cleanup:
        args.append("--cleanup")
    return redact_command_args(args)


def _git_context(cwd: Path) -> dict[str, object]:
    return {
        "root": _git_output(cwd, "rev-parse", "--show-toplevel"),
        "branch": _git_output(cwd, "rev-parse", "--abbrev-ref", "HEAD"),
        "sha": _git_output(cwd, "rev-parse", "HEAD"),
    }


def _git_output(cwd: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _json_list_snapshot(result: CommandResult) -> dict[str, object]:
    snapshot = _json_snapshot(result)
    items = snapshot.get("items")
    if not isinstance(items, list):
        snapshot["items"] = []
    return snapshot


def _json_snapshot(result: CommandResult) -> dict[str, object]:
    if result.dry_run or not result.stdout.strip():
        return {"items": []}
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as err:
        return {"items": [], "parse_error": f"invalid JSON: {err}"}
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return {"items": [], "parse_error": "JSON output was not an object"}


def _merge_object_list_snapshots(
    *snapshots: Mapping[str, object],
) -> dict[str, object]:
    """Merge Kubernetes List snapshots by object name, preserving first-seen order."""
    merged: dict[str, object] = {"items": []}
    item_by_name: dict[str, Mapping[str, object]] = {}
    order: list[str] = []
    for snapshot in snapshots:
        for item in _object_items(snapshot):
            metadata = _mapping(item.get("metadata"))
            name = _string_or_none(metadata.get("name"))
            if name is None:
                continue
            if name not in item_by_name:
                order.append(name)
            item_by_name[name] = item
    merged["items"] = [dict(item_by_name[name]) for name in order]
    return merged


def _taskexecutor_secret_evidence(
    secret_snapshot: Mapping[str, object],
    *,
    selector: str,
    command: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "redacted": True,
        "selector": selector,
        "command": dict(command),
        "items": [_secret_summary(secret) for secret in _object_items(secret_snapshot)],
    }


def _merge_secret_evidence(
    *evidence_records: Mapping[str, object],
) -> dict[str, object]:
    """Merge Secret evidence summaries by name."""
    merged: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "redacted": True,
        "items": [],
    }
    item_by_name: dict[str, Mapping[str, object]] = {}
    order: list[str] = []
    for evidence in evidence_records:
        if "selector" not in merged and evidence.get("selector") is not None:
            merged["selector"] = evidence["selector"]
        for item in _object_items(evidence):
            name = _string_or_none(item.get("name"))
            if name is None:
                continue
            if name not in item_by_name:
                order.append(name)
            item_by_name[name] = item
    merged["items"] = [dict(item_by_name[name]) for name in order]
    return merged


def _redact_secret_observation_stdout(result: CommandResult) -> None:
    if result.stdout.strip():
        result.stdout = f"{REDACTED} Secret list JSON; see summarized items"


def _secret_summary(secret: Mapping[str, object]) -> dict[str, object]:
    metadata = _mapping(secret.get("metadata"))
    data = _mapping(secret.get("data"))
    string_data = _mapping(secret.get("stringData"))
    binary_data = _mapping(secret.get("binaryData"))
    return {
        "name": _string_or_none(metadata.get("name")),
        "namespace": _string_or_none(metadata.get("namespace")),
        "uid": _string_or_none(metadata.get("uid")),
        "labels": _string_mapping(metadata.get("labels")),
        "annotations": _string_mapping(metadata.get("annotations")),
        "type": _string_or_none(secret.get("type")),
        "data_keys": sorted(str(key) for key in data),
        "data_byte_lengths": _base64_byte_lengths(data),
        "stringData_keys": sorted(str(key) for key in string_data),
        "binaryData_keys": sorted(str(key) for key in binary_data),
        "redacted": True,
    }


def _task_lineage(
    *,
    profile: HarnessProfile,
    run_id: str,
    mode: str,
    seed_run_id: object,
    seed_run_ids: Sequence[int],
    selector: str,
    pod_snapshot: Mapping[str, object],
    secret_evidence: Mapping[str, object],
) -> dict[str, object]:
    secrets_by_name = {
        str(secret["name"]): secret
        for secret in _object_items(secret_evidence)
        if isinstance(secret.get("name"), str)
    }
    tasks: list[dict[str, object]] = []
    for pod in _object_items(pod_snapshot):
        metadata = _mapping(pod.get("metadata"))
        labels = _string_mapping(metadata.get("labels"))
        status = _mapping(pod.get("status"))
        pod_name = _string_or_none(metadata.get("name"))
        secret_name = _pod_secret_name(pod)
        secret = secrets_by_name.get(secret_name or "")
        task_id = labels.get(_TASK_ID_LABEL)
        if secret is None:
            secret = _secret_matching_task(
                secret_evidence,
                task_id=task_id,
                launch_attempt=labels.get(_LAUNCH_ATTEMPT_LABEL),
            )
            if isinstance(secret.get("name"), str):
                secret_name = str(secret["name"])
        tasks.append(
            {
                "task_id": task_id,
                "task_type": labels.get("flower.ai/task-type"),
                "pod_name": pod_name,
                "pod_uid": _string_or_none(metadata.get("uid")),
                "pod_phase": _string_or_none(status.get("phase")),
                "terminal_phase": _string_or_none(status.get("phase")),
                "launch_attempt": labels.get(_LAUNCH_ATTEMPT_LABEL),
                "resource_pool": labels.get("flower.ai/resource-pool"),
                "credential_secret_name": secret_name,
                "credential_secret_uid": _string_or_none(secret.get("uid")),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "run_id": run_id,
        "seeded_run_id": seed_run_id,
        "seeded_run_ids": list(seed_run_ids),
        "seeded_task_count": len(seed_run_ids),
        "observed_task_count": len(tasks),
        "lineage_note": (
            "TaskExecutor Pod labels expose executor task IDs, not ServerApp run IDs; "
            "this record captures seeded run IDs and observed TaskExecutor objects "
            "without claiming a per-Pod run-ID mapping."
        ),
        "resource_pool": profile.resource_pool,
        "selector": selector,
        "tasks": tasks,
    }


def _final_state_record(
    *,
    profile: HarnessProfile,
    run_id: str,
    mode: str,
    cleanup_requested: bool,
    taskexecutor_selector: str,
    pod_snapshot: Mapping[str, object],
    pod_result: CommandResult,
    secret_evidence: Mapping[str, object],
    secret_result: CommandResult,
    run_command: object,
) -> dict[str, object]:
    command_runner = cast(
        CallableCommand,
        run_command,
    )
    run_selector = f"flower.ai/harness-run={run_id}"
    jobs_result = command_runner(
        _kubectl_args(
            profile,
            ["get", "jobs", "-n", profile.namespace, "-l", run_selector, "-o", "json"],
        ),
        "final-state Job observation",
    )
    services_result = command_runner(
        _kubectl_args(
            profile,
            [
                "get",
                "services",
                "-n",
                profile.namespace,
                "-l",
                run_selector,
                "-o",
                "json",
            ],
        ),
        "final-state Service observation",
    )
    namespace_result = command_runner(
        _kubectl_args(profile, ["get", "namespace", profile.namespace, "-o", "json"]),
        "final-state namespace observation",
    )

    jobs_snapshot = _json_list_snapshot(jobs_result)
    services_snapshot = _json_list_snapshot(services_result)
    namespace_snapshot = _json_snapshot(namespace_result)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "run_id": run_id,
        "captured_before_namespace_cleanup": True,
        "cleanup_requested": cleanup_requested,
        "selectors": {
            "taskexecutor": taskexecutor_selector,
            "run": run_selector,
        },
        "commands": {
            "taskexecutor_pods": _command_record(pod_result),
            "taskexecutor_secrets": _command_record(secret_result),
            "jobs": _command_record(jobs_result),
            "services": _command_record(services_result),
            "namespace": _command_record(namespace_result),
        },
        "counts": {
            "taskexecutor_pods": len(_object_items(pod_snapshot)),
            "taskexecutor_secrets": len(_object_items(secret_evidence)),
            "jobs": len(_object_items(jobs_snapshot)),
            "services": len(_object_items(services_snapshot)),
            "namespace": 1 if namespace_snapshot.get("metadata") else 0,
        },
        "resources": {
            "pods": _pod_summaries(pod_snapshot),
            "secrets": list(_object_items(secret_evidence)),
            "jobs": _object_summaries(jobs_snapshot),
            "services": _object_summaries(services_snapshot),
            "namespace": _namespace_summary(namespace_snapshot),
        },
    }


def _proof_checklist(
    *,
    status: str,
    dry_run: bool,
    run_id: str,
    cleanup_requested: bool,
    capacity_cleanup_proof: bool,
    active_pod_budget: int | None,
    seed_run_count: int,
) -> dict[str, object]:
    proof_status = "planned" if dry_run else status
    claims: list[dict[str, object]] = [
        {
            "claim": "Harness invocation and selected inputs are inspectable.",
            "status": "proved",
            "artifact": "invocation.json",
            "fields": ["equivalent_argv", "repo", "profile", "settings"],
        },
        {
            "claim": "SuperExec is configured to use the Kubernetes executor.",
            "status": proof_status,
            "artifact": "objects/real-launch.yaml",
            "fields": [
                "SuperExec container args include --executor kubernetes",
                "executor config is mounted at /etc/flower/executor-config.yaml",
            ],
        },
        {
            "claim": (
                "Seeded ServerApp run count and observed TaskExecutor objects are "
                "captured together."
            ),
            "status": proof_status,
            "artifact": "task-lineage.json",
            "fields": [
                "seeded_run_id",
                "seeded_run_ids",
                "seeded_task_count",
                "observed_task_count",
                "tasks[].pod_name",
                "tasks[].credential_secret_name",
            ],
        },
        {
            "claim": (
                "The executor-created TaskExecutor Pod is captured as a full "
                "redacted object snapshot."
            ),
            "status": proof_status,
            "artifact": "taskexecutor-pods.json",
            "fields": ["items[].metadata", "items[].spec", "items[].status"],
        },
        {
            "claim": (
                "The per-task credential Secret existed without exposing "
                "credential values."
            ),
            "status": proof_status,
            "artifact": "taskexecutor-secrets.redacted.json",
            "fields": [
                "items[].name",
                "items[].data_keys",
                "items[].data_byte_lengths",
                "items[].redacted",
            ],
        },
        {
            "claim": (
                "Pre-cleanup resource state is captured before namespace deletion."
            ),
            "status": proof_status,
            "artifact": "final-state.json",
            "fields": ["counts", "resources", "captured_before_namespace_cleanup"],
        },
    ]
    out_of_scope = [
        "budget-2/three-task cardinality behavior",
        "AppIo TLS proof",
        "production deployment readiness",
    ]
    if capacity_cleanup_proof:
        claims.extend(
            [
                {
                    "claim": "The Kubernetes executor active Pod budget is one.",
                    "status": proof_status,
                    "artifact": "objects/executor-config.yaml",
                    "fields": ["active-pod-budget"],
                    "expected": {"active-pod-budget": active_pod_budget},
                },
                {
                    "claim": "Two deterministic ServerApp tasks were seeded.",
                    "status": proof_status,
                    "artifact": "summary.json",
                    "fields": ["details.seed_run_ids"],
                    "expected": {"seed_run_count": seed_run_count},
                },
                {
                    "claim": "SuperExec waited because TaskExecutor capacity was full.",
                    "status": proof_status,
                    "artifact": "events.jsonl",
                    "fields": ["capacity.wait_observed"],
                },
                {
                    "claim": (
                        "A second TaskExecutor Pod launched after capacity opened."
                    ),
                    "status": proof_status,
                    "artifact": "objects/cleanup-pods.json",
                    "fields": ["items[].name", "items[].phase"],
                },
                {
                    "claim": (
                        "The completed first TaskExecutor Pod and credential Secret "
                        "were removed before namespace cleanup."
                    ),
                    "status": proof_status,
                    "artifact": "summary.json",
                    "fields": [
                        "details.cleanup_observed.removed_pods",
                        "details.cleanup_observed.removed_secrets",
                    ],
                },
            ]
        )
    else:
        out_of_scope.extend(
            [
                "active Pod budget behavior",
                "two-task capacity waiting",
                "executor-owned completed Pod cleanup proof",
                "executor-owned per-task Secret cleanup proof",
            ]
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": (
            "local-k8s-capacity-cleanup-proof"
            if capacity_cleanup_proof
            else "local-k8s-launch-path"
        ),
        "run_id": run_id,
        "claims": claims,
        "out_of_scope": out_of_scope,
        "cleanup_requested": cleanup_requested,
    }


class CallableCommand(Protocol):
    """Callable shape for the local run_command closure."""

    def __call__(
        self, args: Sequence[str], failure_context: str, *, record_failure: bool = True
    ) -> CommandResult:
        """Run a command and return its result."""


def _taskexecutor_secrets_args(profile: HarnessProfile, selector: str) -> list[str]:
    return _kubectl_args(
        profile,
        [
            "get",
            "secrets",
            "-n",
            profile.namespace,
            "-l",
            selector,
            "-o",
            "json",
        ],
    )


def _cleanup_observation(
    *,
    first_pod_names: Sequence[str],
    before_cleanup_secrets: Mapping[str, object],
    after_cleanup_pods: Mapping[str, object],
    after_cleanup_secrets: Mapping[str, object],
) -> dict[str, object]:
    after_pod_names = set(_pod_names(after_cleanup_pods))
    before_secret_names = set(_secret_names(before_cleanup_secrets))
    after_secret_names = set(_secret_names(after_cleanup_secrets))
    removed_pods = [
        pod_name for pod_name in first_pod_names if pod_name not in after_pod_names
    ]
    removed_secrets = sorted(before_secret_names - after_secret_names)
    return {
        "observed": bool(removed_pods) and bool(removed_secrets),
        "removed_pods": removed_pods,
        "removed_secrets": removed_secrets,
        "remaining_pods": sorted(after_pod_names),
        "remaining_secrets": sorted(after_secret_names),
    }


def _pod_summaries(snapshot: Mapping[str, object]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for pod in _object_items(snapshot):
        metadata = _mapping(pod.get("metadata"))
        status = _mapping(pod.get("status"))
        summaries.append(
            {
                "name": _string_or_none(metadata.get("name")),
                "namespace": _string_or_none(metadata.get("namespace")),
                "uid": _string_or_none(metadata.get("uid")),
                "labels": _string_mapping(metadata.get("labels")),
                "phase": _string_or_none(status.get("phase")),
                "deletionTimestamp": _string_or_none(metadata.get("deletionTimestamp")),
            }
        )
    return summaries


def _object_summaries(snapshot: Mapping[str, object]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for item in _object_items(snapshot):
        metadata = _mapping(item.get("metadata"))
        status = _mapping(item.get("status"))
        summaries.append(
            {
                "kind": _string_or_none(item.get("kind")),
                "name": _string_or_none(metadata.get("name")),
                "namespace": _string_or_none(metadata.get("namespace")),
                "uid": _string_or_none(metadata.get("uid")),
                "labels": _string_mapping(metadata.get("labels")),
                "phase": _string_or_none(status.get("phase")),
            }
        )
    return summaries


def _namespace_summary(snapshot: Mapping[str, object]) -> dict[str, object]:
    metadata = _mapping(snapshot.get("metadata"))
    status = _mapping(snapshot.get("status"))
    return {
        "name": _string_or_none(metadata.get("name")),
        "uid": _string_or_none(metadata.get("uid")),
        "phase": _string_or_none(status.get("phase")),
    }


def _secret_matching_task(
    secret_evidence: Mapping[str, object],
    *,
    task_id: str | None,
    launch_attempt: str | None,
) -> Mapping[str, object]:
    for secret in _object_items(secret_evidence):
        labels = _string_mapping(secret.get("labels"))
        if labels.get(_TASK_ID_LABEL) != task_id:
            continue
        if labels.get(_LAUNCH_ATTEMPT_LABEL) == launch_attempt:
            return secret
    return {}


def _pod_secret_name(pod: Mapping[str, object]) -> str | None:
    spec = _mapping(pod.get("spec"))
    volumes = spec.get("volumes", [])
    if not isinstance(volumes, Sequence) or isinstance(volumes, str):
        return None
    for volume in volumes:
        if not isinstance(volume, Mapping):
            continue
        secret = _mapping(volume.get("secret"))
        secret_name = _string_or_none(secret.get("secretName"))
        if secret_name is not None:
            return secret_name
    return None


def _object_items(snapshot: Mapping[str, object]) -> list[Mapping[str, object]]:
    items = snapshot.get("items", [])
    if not isinstance(items, Sequence) or isinstance(items, str):
        return []
    return [item for item in items if isinstance(item, Mapping)]


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items() if isinstance(item, str)}


def _base64_byte_lengths(data: Mapping[str, object]) -> list[dict[str, object]]:
    lengths: list[dict[str, object]] = []
    for key, value in data.items():
        if not isinstance(value, str):
            lengths.append({"key": str(key), "bytes": None})
            continue
        try:
            byte_length = len(base64.b64decode(value, validate=True))
        except ValueError:
            byte_length = len(value.encode("utf-8"))
        lengths.append({"key": str(key), "bytes": byte_length})
    return lengths


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None
