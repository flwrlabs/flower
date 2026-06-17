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
"""Render local k8s launch-path Kubernetes manifests from YAML templates."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from string import Template

import yaml

from common import HarnessProfile, _harness_object_labels, _run_object_labels

_THIS_DIR = Path(__file__).resolve().parent
_MANIFEST_DIR = _THIS_DIR / "manifests"
_ASSET_DIR = _THIS_DIR / "assets"
_LOCAL_K8S_ROOT = "/opt/flower-local-k8s"
_SEED_CONFIGMAP_ASSETS = {
    "seed_run.py": _ASSET_DIR / "seed_run.py",
    "probe_pyproject.toml": _ASSET_DIR / "probe_app" / "pyproject.toml",
    "launch_probe_init.py": _ASSET_DIR
    / "probe_app"
    / "launch_probe"
    / "__init__.py",
    "launch_probe_server_app.py": _ASSET_DIR
    / "probe_app"
    / "launch_probe"
    / "server_app.py",
    "launch_probe_client_app.py": _ASSET_DIR
    / "probe_app"
    / "launch_probe"
    / "client_app.py",
}


def render_namespace_manifest(profile: HarnessProfile) -> dict[str, object]:
    """Render the namespace manifest for the optional local k8s proof."""
    namespace = _load_template("namespace.yaml", {"namespace": profile.namespace})[0]
    _merge_metadata_labels(namespace, _harness_object_labels(profile))
    return namespace


def render_superexec_rbac_manifests(
    profile: HarnessProfile,
) -> list[dict[str, object]]:
    """Render minimal SuperExec ServiceAccount, Role, and RoleBinding objects."""
    manifests = _load_template(
        "rbac.yaml",
        {
            "namespace": profile.namespace,
            "rbac_name": profile.rbac_name,
            "service_account": profile.superexec_service_account,
        },
    )
    labels = _harness_object_labels(profile)
    for manifest in manifests:
        _merge_metadata_labels(manifest, labels)
    return manifests


def render_kubernetes_executor_config(
    profile: HarnessProfile, run_id: str
) -> dict[str, object]:
    """Render the trusted root-mapping config consumed by SuperExec."""
    labels = dict(profile.labels)
    labels["flower.ai/harness-run"] = run_id
    config: dict[str, object] = {
        "namespace": profile.namespace,
        "image": profile.image,
        "image-pull-policy": profile.image_pull_policy,
        "resource-pool": profile.resource_pool,
        "labels": labels,
    }
    if profile.active_pod_budget is not None:
        config["active-pod-budget"] = profile.active_pod_budget
    if profile.capacity_poll_interval is not None:
        config["capacity-poll-interval"] = profile.capacity_poll_interval
    if profile.capacity_log_interval is not None:
        config["capacity-log-interval"] = profile.capacity_log_interval
    return config


def render_real_launch_manifests(
    profile: HarnessProfile, run_id: str
) -> list[dict[str, object]]:
    """Render SuperLink, executor config, and SuperExec objects."""
    labels = _run_object_labels(profile, run_id)
    manifests = _load_template(
        "runtime.yaml",
        {
            "namespace": profile.namespace,
            "superlink_name": profile.superlink_name,
            "superexec_name": profile.superexec_name,
            "executor_config_name": profile.executor_config_name,
            "superlink_image": profile.superlink_image,
            "superexec_image": profile.superexec_image,
            "runtime_image_pull_policy": profile.runtime_image_pull_policy,
            "appio_api_port": str(profile.appio_api_port),
            "control_api_port": str(profile.control_api_port),
            "executor_config_path": "/etc/flower/executor-config.yaml",
            "appio_address": f"{profile.superlink_name}:{profile.appio_api_port}",
            "service_account": profile.superexec_service_account,
        },
    )

    for manifest in manifests:
        _merge_metadata_labels(manifest, labels)

    superlink_service, superlink_pod, executor_config, superexec_pod = manifests
    _set_service_selector(superlink_service, run_id, "superlink")
    _merge_metadata_labels(superlink_pod, {"app.kubernetes.io/component": "superlink"})
    _merge_metadata_labels(superexec_pod, {"app.kubernetes.io/component": "superexec"})
    executor_config["data"] = {
        "executor-config.yaml": yaml.safe_dump(
            render_kubernetes_executor_config(profile, run_id),
            sort_keys=True,
        )
    }
    return manifests


def render_appio_seed_manifests(
    profile: HarnessProfile, run_id: str
) -> list[dict[str, object]]:
    """Render the Control API seed Job that creates one ServerApp task."""
    labels = _run_object_labels(profile, run_id)
    manifests = _load_template(
        "seed-job.yaml",
        {
            "namespace": profile.namespace,
            "seed_config_name": profile.seed_config_name,
            "seed_job_name": profile.seed_job_name,
            "superlink_image": profile.superlink_image,
            "runtime_image_pull_policy": profile.runtime_image_pull_policy,
            "control_address": f"{profile.superlink_name}:{profile.control_api_port}",
            "local_k8s_root": _LOCAL_K8S_ROOT,
            "seed_run_count": str(profile.seed_run_count),
            "probe_hold_seconds": str(profile.probe_hold_seconds),
        },
    )
    seed_config, seed_job = manifests
    seed_labels = labels | {"app.kubernetes.io/component": "appio-seed"}

    _merge_metadata_labels(seed_config, labels)
    seed_config["data"] = _read_seed_assets()
    _merge_metadata_labels(seed_job, seed_labels)
    template = _mapping(seed_job["spec"])["template"]
    _merge_template_labels(template, seed_labels)
    return manifests


def _load_template(name: str, substitutions: Mapping[str, str]) -> list[dict[str, object]]:
    content = (_MANIFEST_DIR / name).read_text(encoding="utf-8")
    rendered = Template(content).substitute(substitutions)
    documents = [document for document in yaml.safe_load_all(rendered) if document]
    if not all(isinstance(document, dict) for document in documents):
        raise ValueError(f"{name} must contain only YAML mapping documents")
    return [dict(document) for document in documents]


def _read_seed_assets() -> dict[str, str]:
    return {
        key: path.read_text(encoding="utf-8")
        for key, path in sorted(_SEED_CONFIGMAP_ASSETS.items())
    }


def _merge_metadata_labels(manifest: dict[str, object], labels: Mapping[str, str]) -> None:
    metadata = _mapping(manifest.setdefault("metadata", {}))
    existing = _mapping(metadata.setdefault("labels", {}))
    existing.update(labels)


def _merge_template_labels(template: object, labels: Mapping[str, str]) -> None:
    metadata = _mapping(_mapping(template).setdefault("metadata", {}))
    existing = _mapping(metadata.setdefault("labels", {}))
    existing.update(labels)


def _set_service_selector(
    service: dict[str, object], run_id: str, component: str
) -> None:
    spec = _mapping(service["spec"])
    selector = _mapping(spec.setdefault("selector", {}))
    selector.update(
        {
            "app.kubernetes.io/name": "flower",
            "app.kubernetes.io/component": component,
            "flower.ai/harness-run": run_id,
        }
    )


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Expected YAML mapping")
    return value
