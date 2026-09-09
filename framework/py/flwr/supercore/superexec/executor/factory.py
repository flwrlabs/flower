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

from logging import WARNING
from pathlib import Path
from typing import Any

from flwr.supercore import log
from flwr.supercore.constant import ExecutorType, TaskType

from .config import ExecutorConfig
from .kubernetes_executor import (
    KubernetesExecutor,
    KubernetesExecutorConfig,
    create_incluster_kubernetes_clients,
)
from .subprocess_executor import SubprocessExecutor
from .types import Executor
from .warm_executor_pool import WarmExecutorPoolConfig, WarmExecutorPoolKey

_KUBERNETES_CONFIG_FIELD_MAP = {
    "image-pull-policy": "image_pull_policy",
    "resource-pool": "resource_pool",
    "active-pod-budget": "active_pod_budget",
    "capacity-poll-interval": "capacity_poll_interval",
    "capacity-log-interval": "capacity_log_interval",
    "labels": "labels",
    "annotations": "annotations",
    "env": "env",
    "volumes": "volumes",
    "volume-mounts": "volume_mounts",
    "resources": "resources",
    "node-selector": "node_selector",
    "tolerations": "tolerations",
    "affinity": "affinity",
    "priority-class-name": "priority_class_name",
    "pod-security-context": "pod_security_context",
    "container-security-context": "container_security_context",
    "service-account-name": "service_account_name",
    "warm-executor-owner": "warm_executor_owner",
}


def get_executor(
    executor_type: ExecutorType,
    executor_config: ExecutorConfig | None = None,
    *,
    insecure: bool = False,
    root_certificates_path: str | None = None,
) -> Executor:
    """Return the executor for the configured executor type."""
    if executor_type == ExecutorType.SUBPROCESS:
        return SubprocessExecutor()

    if executor_type == ExecutorType.KUBERNETES:
        if executor_config is None:
            raise ValueError("Kubernetes executor requires --executor-config.")
        config = _kubernetes_executor_config_from_mapping(executor_config)
        if (
            config.warm_executor_pools
            and not insecure
            and config.runtime_root_certificates is None
            and root_certificates_path is not None
        ):
            log(
                WARNING,
                "Warm executor pools are disabled because task-specific Runtime CA "
                "certificates require cold dispatch.",
            )
            # Retain the owner so reconciliation can clean up surviving idle Pods.
            config.warm_executor_pools = ()
        try:
            client, exec_client = create_incluster_kubernetes_clients()
        except RuntimeError as err:
            raise ValueError(str(err)) from err
        return KubernetesExecutor(
            client=client,
            config=config,
            exec_client=exec_client,
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
        kwargs["runtime_root_certificates"] = _read_runtime_root_certificates(
            path_value
        )

    if "warm-executor-pools" in config:
        kwargs["warm_executor_pools"] = _warm_executor_pools_from_config(
            config["warm-executor-pools"], image
        )

    return KubernetesExecutorConfig(**kwargs)


def _required_nonempty_string(config: ExecutorConfig, field_name: str) -> str:
    """Return a required string field or raise a clear construction error."""
    value = config.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Kubernetes executor config requires non-empty '{field_name}'."
        )
    return value


def _read_runtime_root_certificates(path_value: str) -> str:
    """Read Runtime API root certificate PEM data from the configured path."""
    try:
        return Path(path_value).expanduser().read_text(encoding="utf-8")
    except OSError as err:
        message = err.strerror or str(err)
        raise ValueError(
            "Failed to read Kubernetes executor config field "
            f"'appio-root-certificates-path' from '{path_value}': {message}."
        ) from err


def _warm_executor_pools_from_config(
    value: object, runtime_image: str
) -> tuple[WarmExecutorPoolConfig, ...]:
    """Parse AgentApp-only warm-executor pools from trusted executor YAML."""
    if not isinstance(value, list):
        raise ValueError(
            "Kubernetes executor config field 'warm-executor-pools' must be a list."
        )

    pools: list[WarmExecutorPoolConfig] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("Warm executor pool entries must be mappings.")
        allowed_fields = {
            "task-type",
            "size",
        }
        if set(entry) - allowed_fields:
            raise ValueError("Warm executor pool entries contain an unknown field.")

        if entry.get("task-type") != TaskType.AGENT_APP.value:
            raise ValueError(
                "Warm executor pools support only task-type 'flwr-agentapp'."
            )
        size = entry.get("size")
        if isinstance(size, bool) or not isinstance(size, int):
            raise ValueError("Warm executor pool requires integer 'size'.")

        pools.append(
            WarmExecutorPoolConfig(
                key=WarmExecutorPoolKey(
                    task_type=TaskType.AGENT_APP,
                    runtime_image=runtime_image,
                ),
                size=size,
            )
        )

    keys = [pool.key for pool in pools]
    if len(keys) != len(set(keys)):
        raise ValueError("Warm executor pools must not repeat a compatibility key.")
    return tuple(pools)
