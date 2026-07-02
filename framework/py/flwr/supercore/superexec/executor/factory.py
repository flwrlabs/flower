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
"""Executor factory for SuperExec TaskExecutor processes."""

from pathlib import Path
from typing import Any

from flwr.supercore.constant import ExecutorType

from .config import ExecutorConfig
from .kubernetes_executor import (
    KubernetesExecutor,
    KubernetesExecutorConfig,
    create_incluster_kubernetes_client,
)
from .subprocess_executor import SubprocessExecutor
from .types import Executor

_KUBERNETES_CONFIG_FIELD_MAP = {
    "image-pull-policy": "image_pull_policy",
    "resource-pool": "resource_pool",
    "active-pod-budget": "active_pod_budget",
    "capacity-poll-interval": "capacity_poll_interval",
    "capacity-log-interval": "capacity_log_interval",
    "labels": "labels",
    "annotations": "annotations",
    "env": "env",
    "resources": "resources",
    "node-selector": "node_selector",
    "tolerations": "tolerations",
    "affinity": "affinity",
    "priority-class-name": "priority_class_name",
    "pod-security-context": "pod_security_context",
    "container-security-context": "container_security_context",
    "service-account-name": "service_account_name",
}


def get_executor(
    executor_type: ExecutorType, executor_config: ExecutorConfig | None = None
) -> Executor:
    """Return the executor for the configured executor type."""
    if executor_type == ExecutorType.SUBPROCESS:
        return SubprocessExecutor()

    if executor_type == ExecutorType.KUBERNETES:
        if executor_config is None:
            raise ValueError("Kubernetes executor requires --executor-config.")
        config = _kubernetes_executor_config_from_mapping(executor_config)
        try:
            client = create_incluster_kubernetes_client()
        except RuntimeError as err:
            raise ValueError(str(err)) from err
        return KubernetesExecutor(
            client=client,
            config=config,
        )

    raise ValueError(f"Unsupported executor selection: {executor_type}")


def _kubernetes_executor_config_from_mapping(
    config: ExecutorConfig,
) -> KubernetesExecutorConfig:
    """Build Kubernetes executor config from the trusted root mapping."""
    namespace = _required_nonempty_string(config, "namespace")
    image = _required_nonempty_string(config, "image")
    kwargs: dict[str, Any] = {
        "namespace": namespace,
        "image": image,
    }

    for config_key, field_name in _KUBERNETES_CONFIG_FIELD_MAP.items():
        if config_key in config:
            kwargs[field_name] = config[config_key]

    if "appio-root-certificates-path" in config:
        path_value = config["appio-root-certificates-path"]
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(
                "Kubernetes executor config field 'appio-root-certificates-path' "
                "must be a non-empty path."
            )
        kwargs["appio_root_certificates"] = _read_appio_root_certificates(path_value)

    return KubernetesExecutorConfig(**kwargs)


def _required_nonempty_string(config: ExecutorConfig, field_name: str) -> str:
    """Return a required string field or raise a clear construction error."""
    value = config.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Kubernetes executor config requires non-empty '{field_name}'."
        )
    return value


def _read_appio_root_certificates(path_value: str) -> str:
    """Read AppIo root certificate PEM data from the configured path."""
    try:
        return Path(path_value).expanduser().read_text(encoding="utf-8")
    except OSError as err:
        message = err.strerror or str(err)
        raise ValueError(
            "Failed to read Kubernetes executor config field "
            f"'appio-root-certificates-path' from '{path_value}': {message}."
        ) from err
