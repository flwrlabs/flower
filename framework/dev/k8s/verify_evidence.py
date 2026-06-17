#!/usr/bin/env python

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
"""Verify and present local k8s launch-path harness evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

_SERVERAPP_MARKER = "K8s launch probe ServerApp ran"


def verify_evidence(
    evidence_dir: str | Path,
    *,
    require_cleanup: bool = True,
    expected_result: str = "local-k8s-launch-path",
) -> tuple[list[str], str]:
    """Verify a local k8s launch-path evidence bundle."""
    evidence_path = Path(evidence_dir)
    summary_path = evidence_path / "summary.json"
    taskexecutor_logs_path = evidence_path / "diagnostics" / "taskexecutor-logs.txt"

    failures: list[str] = []
    if not summary_path.is_file():
        return [f"summary.json not found under {evidence_path}"], (
            f"local k8s launch-path verification failed\nEvidence: {evidence_path}\n"
        )

    summary = _read_json(summary_path, failures)
    details = _mapping(summary.get("details"))
    invocation = _read_required_json(evidence_path, "invocation.json", failures)
    task_lineage = _read_required_json(evidence_path, "task-lineage.json", failures)
    taskexecutor_pods = _read_required_json(
        evidence_path, "taskexecutor-pods.json", failures
    )
    taskexecutor_secrets = _read_required_json(
        evidence_path, "taskexecutor-secrets.redacted.json", failures
    )
    final_state = _read_required_json(evidence_path, "final-state.json", failures)
    proof_checklist = _read_required_json(
        evidence_path, "proof-checklist.json", failures
    )

    _expect(summary.get("status") == "passed", "summary status is not passed", failures)
    _expect(
        summary.get("result") == expected_result,
        f"summary result is not {expected_result}",
        failures,
    )
    _expect(not summary.get("failures"), "summary contains failures", failures)
    _expect(
        details.get("dry_run") is False,
        "harness did not execute host commands",
        failures,
    )
    expected_mode = (
        "local-k8s-capacity-cleanup-proof"
        if expected_result == "local-k8s-capacity-cleanup-proof"
        else "local-k8s-launch-path"
    )
    _expect(
        invocation.get("mode") == expected_mode,
        f"invocation.json mode is not {expected_mode}",
        failures,
    )
    _expect(
        invocation.get("dry_run") is False,
        "invocation.json does not record a real execution",
        failures,
    )

    image_preflight = _mapping(details.get("image_preflight"))
    docker_inspect = _mapping(image_preflight.get("docker_inspect"))
    k3d_import = _mapping(image_preflight.get("k3d_import"))
    _expect(
        docker_inspect.get("returncode") == 0,
        "Docker image inspection did not pass",
        failures,
    )
    _expect(
        k3d_import.get("returncode") == 0, "k3d image import did not pass", failures
    )

    rbac = _mapping(details.get("rbac"))
    _expect(rbac.get("status") == "passed", "RBAC verification did not pass", failures)

    pods = _sequence(details.get("pods"))
    _expect(bool(pods), "no TaskExecutor Pods were recorded", failures)
    pod_phases = [
        str(_mapping(pod).get("phase"))
        for pod in pods
        if _mapping(pod).get("phase") is not None
    ]
    _expect(bool(pod_phases), "no TaskExecutor Pod phases were recorded", failures)
    _expect(
        all(phase == "Succeeded" for phase in pod_phases),
        f"TaskExecutor Pod phases were not all Succeeded: {', '.join(pod_phases)}",
        failures,
    )
    task_lineage_tasks = _sequence(task_lineage.get("tasks"))
    _expect(bool(task_lineage_tasks), "task-lineage.json contains no tasks", failures)
    _expect(
        task_lineage.get("seeded_run_id") == details.get("seed_run_id"),
        "task-lineage.json seeded_run_id does not match summary",
        failures,
    )
    lineage_seed_run_ids = _sequence(task_lineage.get("seeded_run_ids"))
    _expect(
        list(lineage_seed_run_ids) == list(_sequence(details.get("seed_run_ids"))),
        "task-lineage.json seeded_run_ids does not match summary",
        failures,
    )
    _expect(
        task_lineage.get("seeded_task_count") == len(lineage_seed_run_ids),
        "task-lineage.json seeded_task_count does not match seeded_run_ids",
        failures,
    )
    _expect(
        task_lineage.get("observed_task_count") == len(task_lineage_tasks),
        "task-lineage.json observed_task_count does not match tasks",
        failures,
    )
    _expect(
        bool(_sequence(taskexecutor_pods.get("items"))),
        "taskexecutor-pods.json contains no Pod items",
        failures,
    )

    secret_items = _sequence(taskexecutor_secrets.get("items"))
    _expect(
        taskexecutor_secrets.get("redacted") is True,
        "taskexecutor-secrets.redacted.json does not declare redaction",
        failures,
    )
    _expect(
        bool(secret_items),
        "taskexecutor-secrets.redacted.json contains no Secret items",
        failures,
    )
    _expect(
        all(_mapping(secret).get("redacted") is True for secret in secret_items),
        "one or more Secret evidence records are not marked redacted",
        failures,
    )
    _expect(
        all(
            "token" in _sequence(_mapping(secret).get("data_keys"))
            for secret in secret_items
        ),
        "Secret evidence does not include the token key name",
        failures,
    )

    final_counts = _mapping(final_state.get("counts"))
    _expect(
        final_state.get("captured_before_namespace_cleanup") is True,
        "final-state.json was not marked as pre-cleanup state",
        failures,
    )
    _expect(
        _int_value(final_counts.get("taskexecutor_pods")) >= 1,
        "final-state.json does not count TaskExecutor Pods",
        failures,
    )
    _expect(
        _int_value(final_counts.get("taskexecutor_secrets")) >= 1,
        "final-state.json does not count TaskExecutor Secrets",
        failures,
    )

    checklist_claims = _sequence(proof_checklist.get("claims"))
    out_of_scope = [
        str(item) for item in _sequence(proof_checklist.get("out_of_scope"))
    ]
    _expect(bool(checklist_claims), "proof-checklist.json contains no claims", failures)
    if expected_result == "local-k8s-capacity-cleanup-proof":
        _expect(
            not any("capacity wait proof" == item for item in out_of_scope),
            "proof-checklist.json incorrectly keeps capacity wait proof out of scope",
            failures,
        )
    else:
        _expect(
            any("capacity" in item for item in out_of_scope),
            "proof-checklist.json does not keep capacity claims out of scope",
            failures,
        )

    taskexecutor_logs = _sequence(details.get("taskexecutor_logs"))
    _expect(
        bool(taskexecutor_logs),
        "no TaskExecutor log command records were captured",
        failures,
    )
    _expect(
        all(_mapping(record).get("returncode") == 0 for record in taskexecutor_logs),
        "one or more TaskExecutor log commands failed",
        failures,
    )

    taskexecutor_log_text = (
        taskexecutor_logs_path.read_text(encoding="utf-8")
        if taskexecutor_logs_path.is_file()
        else ""
    )
    _expect(
        _SERVERAPP_MARKER in taskexecutor_log_text,
        f"TaskExecutor logs do not contain marker: {_SERVERAPP_MARKER}",
        failures,
    )

    cleanup = _mapping(details.get("cleanup"))
    cleanup_result = _mapping(cleanup.get("result"))
    if require_cleanup:
        _expect(cleanup.get("requested") is True, "cleanup was not requested", failures)
        _expect(
            cleanup_result.get("returncode") == 0,
            "cleanup command did not pass",
            failures,
        )

    capacity_wait = _mapping(details.get("capacity_wait"))
    cleanup_observed = _mapping(details.get("cleanup_observed"))
    if expected_result == "local-k8s-capacity-cleanup-proof":
        removed_pods = _sequence(cleanup_observed.get("removed_pods"))
        remaining_pods = _sequence(cleanup_observed.get("remaining_pods"))
        removed_pod_names = {str(name) for name in removed_pods}
        remaining_pod_names = {str(name) for name in remaining_pods}
        final_pod_names = {
            str(_mapping(pod).get("name"))
            for pod in pods
            if _mapping(pod).get("name") is not None
        }
        _expect(
            details.get("active_pod_budget") == 1,
            "active Pod budget is not 1",
            failures,
        )
        _expect(
            len(lineage_seed_run_ids) >= 2,
            "capacity proof did not record at least two seeded run IDs",
            failures,
        )
        _expect(
            len(task_lineage_tasks) >= 2,
            "capacity proof did not record at least two observed TaskExecutor tasks",
            failures,
        )
        _expect(
            capacity_wait.get("observed") is True,
            "capacity wait was not observed",
            failures,
        )
        _expect(
            cleanup_observed.get("observed") is True,
            "completed Pod/Secret cleanup was not observed",
            failures,
        )
        _expect(
            bool(removed_pods),
            "capacity proof did not record removed Pods",
            failures,
        )
        _expect(
            bool(_sequence(cleanup_observed.get("removed_secrets"))),
            "capacity proof did not record removed Secrets",
            failures,
        )
        _expect(
            bool(remaining_pods),
            "capacity proof did not record a remaining TaskExecutor Pod after cleanup",
            failures,
        )
        _expect(
            remaining_pod_names.issubset(final_pod_names),
            "remaining cleanup Pods are not present in final TaskExecutor pod records",
            failures,
        )
        _expect(
            remaining_pod_names.isdisjoint(removed_pod_names),
            "remaining cleanup Pods overlap removed Pods",
            failures,
        )

    return failures, _format_report(
        evidence_path=evidence_path,
        summary=summary,
        details=details,
        task_lineage_tasks=task_lineage_tasks,
        secret_items=secret_items,
        final_counts=final_counts,
        pod_phases=pod_phases,
        cleanup_result=cleanup_result,
        require_cleanup=require_cleanup,
        capacity_wait=capacity_wait,
        cleanup_observed=cleanup_observed,
        failures=failures,
    )


def _read_json(path: Path, failures: list[str]) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        failures.append(f"{path} is not valid JSON: {err}")
        return {}
    if not isinstance(value, dict):
        failures.append(f"{path} does not contain a JSON object")
        return {}
    return value


def _read_required_json(
    evidence_path: Path, relative_path: str, failures: list[str]
) -> dict[str, object]:
    path = evidence_path / relative_path
    if not path.is_file():
        failures.append(f"{relative_path} not found under {evidence_path}")
        return {}
    return _read_json(path, failures)


def _format_report(
    *,
    evidence_path: Path,
    summary: Mapping[str, object],
    details: Mapping[str, object],
    task_lineage_tasks: Sequence[object],
    secret_items: Sequence[object],
    final_counts: Mapping[str, object],
    pod_phases: Sequence[str],
    cleanup_result: Mapping[str, object],
    require_cleanup: bool,
    capacity_wait: Mapping[str, object],
    cleanup_observed: Mapping[str, object],
    failures: Sequence[str],
) -> str:
    pods = [_mapping(pod) for pod in _sequence(details.get("pods"))]
    taskexecutor_logs = _sequence(details.get("taskexecutor_logs"))
    removed_pods = ", ".join(
        str(item) for item in _sequence(cleanup_observed.get("removed_pods"))
    )
    removed_secrets = ", ".join(
        str(item) for item in _sequence(cleanup_observed.get("removed_secrets"))
    )
    cleanup_status = (
        f"returncode={cleanup_result.get('returncode')}"
        if cleanup_result
        else "not requested"
    )
    lines = [
        "=== local k8s launch-path verification ===",
        f"Evidence: {evidence_path}",
        f"Status: {summary.get('status')}",
        f"Result: {summary.get('result')}",
        f"Run ID: {details.get('run_id')}",
        f"Seed run ID: {details.get('seed_run_id')}",
        f"TaskExecutor Pods: {len(pods)}",
    ]
    for pod in pods:
        lines.append(f"  - {pod.get('name')} phase={pod.get('phase')}")
    lines.extend(
        [
            f"Task lineage records: {len(task_lineage_tasks)}",
            f"Credential Secret records: {len(secret_items)}",
            (
                "Final state counts: "
                f"pods={final_counts.get('taskexecutor_pods')} "
                f"secrets={final_counts.get('taskexecutor_secrets')}"
            ),
            f"TaskExecutor log captures: {len(taskexecutor_logs)}",
            f"TaskExecutor phases: {', '.join(pod_phases) or '<none>'}",
            f"Capacity wait observed: {capacity_wait.get('observed')}",
            f"Removed Pods: {removed_pods or '<none>'}",
            f"Removed Secrets: {removed_secrets or '<none>'}",
            f"Cleanup required: {str(require_cleanup).lower()}",
            f"Cleanup: {cleanup_status}",
        ]
    )
    if failures:
        lines.append("Verification: FAILED")
        lines.extend(f"  - {failure}" for failure in failures)
    else:
        lines.append("Verification: PASSED")
    return "\n".join(lines) + "\n"


def _expect(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, list) else []


def _int_value(value: object) -> int:
    return value if isinstance(value, int) else 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and summarize local k8s launch-path harness evidence."
    )
    parser.add_argument(
        "evidence_dir",
        help="Directory containing the harness evidence bundle.",
    )
    parser.add_argument(
        "--no-require-cleanup",
        action="store_true",
        help="Do not require the final namespace cleanup command to have run.",
    )
    parser.add_argument(
        "--expected-result",
        choices=("local-k8s-launch-path", "local-k8s-capacity-cleanup-proof"),
        default="local-k8s-launch-path",
        help="Expected harness result to verify.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Verify an evidence bundle from the command line."""
    args = _parse_args(argv)
    failures, report = verify_evidence(
        args.evidence_dir,
        require_cleanup=not args.no_require_cleanup,
        expected_result=args.expected_result,
    )
    print(report, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
