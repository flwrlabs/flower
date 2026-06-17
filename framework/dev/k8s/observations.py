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
"""Observation parsing and status helpers for local k8s evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from common import CommandResult, HarnessProfile, _combined_status, _kubectl_args


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
