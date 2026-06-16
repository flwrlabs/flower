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

import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast
from uuid import uuid4

from common import (
    CommandResult,
    EvidenceBundleWriter,
    HarnessEvent,
    HarnessProfile,
    HarnessSummary,
    HostCommandRunner,
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
    generic_k3d_profile,
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
    _superexec_claim_observation,
    _taskexecutor_phase_status,
    _taskexecutor_pods_args,
    _taskexecutor_selector,
    _taskexecutor_status,
)

_TASKEXECUTOR_POD_POLL_INTERVAL_SECONDS = 1.0


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
) -> HarnessSummary:
    """Write local k8s AppIo/SuperExec/TaskExecutor launch-path evidence."""
    profile = profile or generic_k3d_profile()
    runner = runner or HostCommandRunner(dry_run=not execute)
    writer = EvidenceBundleWriter(output_dir)
    writer.initialize()
    run_id = f"k8s-launch-{uuid4().hex[:12]}"
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
                    "mode": "local-k8s-launch-path",
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
        "TLS material contract recorded; local k8s launch-path uses insecure AppIo unless supplied.",
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
        "This local k8s slice does not prove TaskExecutor AppIo RPC completion.",
        {
            "reason": "first slice observes Pod launch only",
            "future_stage": "future local k8s hardening",
        },
    )

    cleanup_result = CommandResult(args=[], returncode=0, dry_run=not execute)
    if cleanup:
        cleanup_result = run_command(
            cast(list[str], cleanup_plan["command"]),
            "harness namespace cleanup",
        )

    result = "local-k8s-launch-path" if execute else "local-k8s-launch-path-dry-run"
    status = "failed" if failures else "passed"
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
