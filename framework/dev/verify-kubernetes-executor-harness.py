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
"""Verify and present F7c Kubernetes executor harness evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

_SERVERAPP_MARKER = "F7 probe ServerApp ran"


def verify_evidence(
    evidence_dir: str | Path, *, require_cleanup: bool = True
) -> tuple[list[str], str]:
    """Verify an F7c evidence bundle and return failures plus a report."""
    evidence_path = Path(evidence_dir)
    summary_path = evidence_path / "summary.json"
    taskexecutor_logs_path = evidence_path / "diagnostics" / "taskexecutor-logs.txt"

    failures: list[str] = []
    if not summary_path.is_file():
        return [f"summary.json not found under {evidence_path}"], (
            f"F7c k3d verification failed\nEvidence: {evidence_path}\n"
        )

    summary = _read_json(summary_path, failures)
    details = _mapping(summary.get("details"))

    _expect(summary.get("status") == "passed", "summary status is not passed", failures)
    _expect(
        summary.get("result") == "real-launch-path",
        "summary result is not real-launch-path",
        failures,
    )
    _expect(not summary.get("failures"), "summary contains failures", failures)
    _expect(details.get("dry_run") is False, "harness did not execute host commands", failures)

    image_preflight = _mapping(details.get("image_preflight"))
    docker_inspect = _mapping(image_preflight.get("docker_inspect"))
    k3d_import = _mapping(image_preflight.get("k3d_import"))
    _expect(
        docker_inspect.get("returncode") == 0,
        "Docker image inspection did not pass",
        failures,
    )
    _expect(k3d_import.get("returncode") == 0, "k3d image import did not pass", failures)

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

    return failures, _format_report(
        evidence_path=evidence_path,
        summary=summary,
        details=details,
        pod_phases=pod_phases,
        cleanup_result=cleanup_result,
        require_cleanup=require_cleanup,
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


def _format_report(
    *,
    evidence_path: Path,
    summary: Mapping[str, object],
    details: Mapping[str, object],
    pod_phases: Sequence[str],
    cleanup_result: Mapping[str, object],
    require_cleanup: bool,
    failures: Sequence[str],
) -> str:
    pods = [_mapping(pod) for pod in _sequence(details.get("pods"))]
    taskexecutor_logs = _sequence(details.get("taskexecutor_logs"))
    cleanup_status = (
        f"returncode={cleanup_result.get('returncode')}"
        if cleanup_result
        else "not requested"
    )
    lines = [
        "=== F7c k3d verification ===",
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
            f"TaskExecutor log captures: {len(taskexecutor_logs)}",
            f"TaskExecutor phases: {', '.join(pod_phases) or '<none>'}",
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and summarize F7c Kubernetes executor harness evidence."
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Verify an evidence bundle from the command line."""
    args = _parse_args(argv)
    failures, report = verify_evidence(
        args.evidence_dir,
        require_cleanup=not args.no_require_cleanup,
    )
    print(report, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
