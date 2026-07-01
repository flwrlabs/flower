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
"""Kubernetes executor for SuperExec TaskExecutor processes."""


import importlib
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from logging import INFO, WARNING
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

from flwr.common.logger import log
from flwr.supercore.constant import (
    TASK_TYPE_TO_APPIO_API_ADDRESS_ARG,
    TASK_TYPE_TO_COMMAND,
)
from flwr.supercore.typing import JSONObject

from .types import ExecutionSpec, LaunchResult

APPIO_CREDENTIALS_MOUNT_PATH = "/run/flwr/appio"
APPIO_TOKEN_FILE_PATH = f"{APPIO_CREDENTIALS_MOUNT_PATH}/token"
APPIO_ROOT_CERTIFICATES_FILE_PATH = f"{APPIO_CREDENTIALS_MOUNT_PATH}/ca.crt"
LAUNCH_ATTEMPT_LABEL = "flower.ai/launch-attempt"
_TASK_ID_LABEL = "flower.ai/superexec-task-id"
_NAME_LABEL = "app.kubernetes.io/name"
_COMPONENT_LABEL = "app.kubernetes.io/component"
_TASK_TYPE_LABEL = "flower.ai/task-type"
_RESOURCE_POOL_LABEL = "flower.ai/resource-pool"
_EXECUTOR_OWNED_LABELS = frozenset(
    {
        _NAME_LABEL,
        _COMPONENT_LABEL,
        _TASK_ID_LABEL,
        _TASK_TYPE_LABEL,
        LAUNCH_ATTEMPT_LABEL,
        _RESOURCE_POOL_LABEL,
    }
)
_APPIO_CREDENTIAL_SECRET_SUFFIX = "-appio"
_COMPLETED_POD_SWEEP_INTERVAL_SECONDS = 60.0
_FORBIDDEN_TASKEXECUTOR_ENV_NAMES = frozenset(
    {
        "FLWR_MODEL_API_KEY",
        "BRAVE_API_KEY",
        "TAVILY_API_KEY",
        "EXA_API_KEY",
    }
)
_KUBERNETES_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class KubernetesList(Protocol):
    """Subset of Kubernetes list responses used by executor list helpers."""

    items: Sequence[object]


class KubernetesClient(Protocol):
    """Subset of Kubernetes CoreV1Api used by the executor."""

    def create_namespaced_secret(self, namespace: str, body: JSONObject) -> object:
        """Create a Kubernetes Secret in the selected namespace."""

    def create_namespaced_pod(self, namespace: str, body: JSONObject) -> object:
        """Create a Kubernetes Pod in the selected namespace."""

    def delete_namespaced_secret(self, name: str, namespace: str) -> object:
        """Delete a Kubernetes Secret from the selected namespace."""

    def list_namespaced_secret(
        self, namespace: str, label_selector: str
    ) -> KubernetesList:
        """List Kubernetes Secrets in the selected namespace."""

    def delete_namespaced_pod(
        self, name: str, namespace: str, grace_period_seconds: int = 0
    ) -> object:
        """Delete a Kubernetes Pod in the selected namespace."""

    def list_namespaced_pod(
        self, namespace: str, label_selector: str
    ) -> KubernetesList:
        """List Kubernetes Pods in the selected namespace."""


def create_incluster_kubernetes_client() -> KubernetesClient:
    """Create a KubernetesClient backed by in-cluster ServiceAccount auth."""
    try:
        kubernetes_client = importlib.import_module("kubernetes.client")
        kubernetes_config = importlib.import_module("kubernetes.config")
    except ModuleNotFoundError as exc:
        missing_module = exc.name
        if missing_module in {"kubernetes", "kubernetes.client", "kubernetes.config"}:
            raise RuntimeError(
                "Kubernetes Python client package is required for the Kubernetes "
                "executor. Install the official 'kubernetes' package in the "
                "SuperExec environment."
            ) from exc
        raise

    try:
        kubernetes_config.load_incluster_config()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise RuntimeError(
            "Failed to load in-cluster Kubernetes configuration for the Kubernetes "
            "executor. Run SuperExec in a Kubernetes Pod with ServiceAccount "
            "credentials."
        ) from exc

    client: KubernetesClient = kubernetes_client.CoreV1Api()
    return client


@dataclass
class KubernetesExecutorConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration needed to build one TaskExecutor Pod and Secret.

    Parameters
    ----------
    namespace : str
        Kubernetes namespace for TaskExecutor Pods and credential Secrets.
    image : str
        Container image used for TaskExecutor Pods.
    appio_root_certificates : str | None
        Optional PEM data mounted as ca.crt. If unset, launch uses
        ExecutionSpec.root_certificates_path when provided.
    image_pull_policy : str | None
        Optional Kubernetes imagePullPolicy for the TaskExecutor container.
    labels : dict[str, str] | None
        Extra labels added to generated Pods and Secrets.
    annotations : dict[str, str] | None
        Extra annotations added to generated Pods and Secrets.
    resource_pool : str | None
        Optional Flower resource-pool label value.
    resources : JSONObject | None
        Optional Kubernetes container resource requests and limits.
    env : list[JSONObject] | None
        Optional explicit TaskExecutor container environment. Only literal
        name/value entries are supported.
    node_selector : dict[str, str] | None
        Optional Kubernetes nodeSelector.
    tolerations : list[JSONObject] | None
        Optional Kubernetes tolerations.
    affinity : JSONObject | None
        Optional Kubernetes affinity.
    priority_class_name : str | None
        Optional Kubernetes priorityClassName.
    pod_security_context : JSONObject | None
        Optional Kubernetes Pod securityContext.
    container_security_context : JSONObject | None
        Optional TaskExecutor container securityContext.
    service_account_name : str | None
        Optional Kubernetes serviceAccountName. Service account policy/RBAC is
        decided outside this executor.
    """

    namespace: str
    image: str
    appio_root_certificates: str | None = None
    image_pull_policy: str | None = None
    labels: dict[str, str] | None = None
    annotations: dict[str, str] | None = None
    resource_pool: str | None = None
    resources: JSONObject | None = None
    env: list[JSONObject] | None = None
    node_selector: dict[str, str] | None = None
    tolerations: list[JSONObject] | None = None
    affinity: JSONObject | None = None
    priority_class_name: str | None = None
    pod_security_context: JSONObject | None = None
    container_security_context: JSONObject | None = None
    # Optional Pod field only; service account policy/RBAC is decided elsewhere.
    service_account_name: str | None = None
    active_pod_budget: int | None = None
    capacity_poll_interval: float = 1.0
    capacity_log_interval: float | None = None
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        """Validate config values used to build TaskExecutor Pods."""
        if self.env is not None:
            self.env = _taskexecutor_env(self.env)


class KubernetesExecutor:
    """Submit TaskExecutor Pods to Kubernetes."""

    def __init__(
        self,
        *,
        client: KubernetesClient,
        config: KubernetesExecutorConfig,
    ) -> None:
        self._client = client
        self._config = config
        self._completed_pod_sweeper = CompletedPodSweeper(client=client, config=config)
        self._last_completed_pod_sweep_at: float | None = None

    def wait_for_capacity(self) -> None:
        """Wait until the configured resource pool is below its active Pod budget."""
        self._sweep_completed_pods_if_due()
        if self._config.active_pod_budget is None:
            return

        last_log_at: float | None = None
        waited_for_capacity = False
        while True:
            try:
                active_pod_count = self._active_pod_count()
            except Exception:  # pylint: disable=broad-exception-caught
                log(
                    WARNING,
                    "Kubernetes capacity check failed; proceeding without waiting. "
                    "selector=%s",
                    _capacity_label_selector(self._config),
                    exc_info=True,
                )
                return
            if active_pod_count < self._config.active_pod_budget:
                if waited_for_capacity:
                    self._last_completed_pod_sweep_at = self._config.monotonic()
                    self._sweep_completed_pods()
                return

            if self._config.capacity_log_interval is not None:
                now = self._config.monotonic()
                if (
                    last_log_at is None
                    or now - last_log_at >= self._config.capacity_log_interval
                ):
                    log(
                        INFO,
                        "Waiting for Kubernetes TaskExecutor capacity: "
                        "%s active Pods, budget %s, selector %s",
                        active_pod_count,
                        self._config.active_pod_budget,
                        _capacity_label_selector(self._config),
                    )
                    last_log_at = now

            waited_for_capacity = True
            self._config.sleep(self._config.capacity_poll_interval)

    def _sweep_completed_pods_if_due(self) -> None:
        """Run best-effort completed Pod cleanup if the internal throttle allows it."""
        now = self._config.monotonic()
        if (
            self._last_completed_pod_sweep_at is not None
            and now - self._last_completed_pod_sweep_at
            < _COMPLETED_POD_SWEEP_INTERVAL_SECONDS
        ):
            return

        self._last_completed_pod_sweep_at = now
        self._sweep_completed_pods()

    def _sweep_completed_pods(self) -> None:
        """Run best-effort completed Pod cleanup."""
        try:
            self._completed_pod_sweeper.sweep()
        except Exception:  # pylint: disable=broad-exception-caught
            log(
                WARNING,
                "Kubernetes completed Pod sweep failed; proceeding. selector=%s",
                _taskexecutor_pool_label_selector(self._config),
                exc_info=True,
            )

    def launch(self, spec: ExecutionSpec) -> LaunchResult:
        """Submit the TaskExecutor Pod and credential Secret."""
        try:
            appio_root_certificates = _get_appio_root_certificates(spec, self._config)
            launch_attempt_id = _new_launch_attempt_id()
            secret_name = _credential_secret_name(spec, launch_attempt_id)
            secret = _build_appio_credentials_secret(
                spec, self._config, appio_root_certificates, launch_attempt_id
            )
            pod = _build_taskexecutor_pod(
                spec, self._config, appio_root_certificates, launch_attempt_id
            )
            self._client.create_namespaced_secret(self._config.namespace, secret)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return _launch_result_from_exception(exc)

        try:
            self._client.create_namespaced_pod(self._config.namespace, pod)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result = _launch_result_from_exception(exc)
            if _is_definite_pod_rejection(exc):
                _delete_secret_best_effort(
                    self._client, self._config.namespace, secret_name
                )
            return result

        return LaunchResult.accepted()

    def _active_pod_count(self) -> int:
        """Return the active TaskExecutor Pod count for the configured pool."""
        pod_list = self._client.list_namespaced_pod(
            self._config.namespace,
            label_selector=_capacity_label_selector(self._config),
        )
        return sum(1 for pod in _pod_items(pod_list) if _is_active_pod(pod))


class CompletedPodSweeper:
    """Delete terminal TaskExecutor Pods and orphaned credential Secrets."""

    def __init__(
        self,
        *,
        client: KubernetesClient,
        config: KubernetesExecutorConfig,
    ) -> None:
        self._client = client
        self._config = config

    def sweep(self) -> None:
        """Delete terminal Pods and orphaned credential Secrets."""
        selector = _taskexecutor_pool_label_selector(self._config)
        pods = _pod_items(
            self._client.list_namespaced_pod(
                self._config.namespace, label_selector=selector
            )
        )
        secrets = _secret_items(
            self._client.list_namespaced_secret(
                self._config.namespace, label_selector=selector
            )
        )
        pod_names = {name for pod in pods if (name := _object_name(pod)) is not None}
        task_secret_names = {
            name
            for secret in secrets
            if (name := _object_name(secret)) is not None and _has_task_id_label(secret)
        }

        for pod in pods:
            pod_name = _object_name(pod)
            if (
                pod_name is None
                or not _has_task_id_label(pod)
                or not _is_terminal_pod(pod)
            ):
                continue
            self._delete_pod(pod_name)
            credential_secret_name = _credential_secret_name_from_pod_name(pod_name)
            if credential_secret_name in task_secret_names:
                self._delete_secret(credential_secret_name)

        # Delete credential Secrets whose owner Pod is no longer listed.
        for secret in secrets:
            secret_name = _object_name(secret)
            if secret_name is None or not _has_task_id_label(secret):
                continue
            pod_name = _pod_name_from_credential_secret_name(secret_name)
            if pod_name is None or pod_name in pod_names:
                continue
            self._delete_secret(secret_name)

    def _delete_pod(self, name: str) -> None:
        """Delete a Pod, tolerating already-deleted objects."""
        try:
            self._client.delete_namespaced_pod(
                name=name,
                namespace=self._config.namespace,
                grace_period_seconds=0,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _raise_unless_not_found(exc)

    def _delete_secret(self, name: str) -> None:
        """Delete a Secret, tolerating already-deleted objects."""
        try:
            self._client.delete_namespaced_secret(
                name=name, namespace=self._config.namespace
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _raise_unless_not_found(exc)


def _build_appio_credentials_secret(
    spec: ExecutionSpec,
    config: KubernetesExecutorConfig,
    appio_root_certificates: str | None,
    launch_attempt_id: str,
) -> JSONObject:
    """Build the AppIo credential Secret for a TaskExecutor Pod."""
    data: JSONObject = {"token": spec.token}
    if appio_root_certificates is not None:
        data["ca.crt"] = appio_root_certificates

    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": _metadata(
            _credential_secret_name(spec, launch_attempt_id),
            spec,
            config,
            launch_attempt_id,
        ),
        "type": "Opaque",
        "stringData": data,
    }


def _build_taskexecutor_pod(
    spec: ExecutionSpec,
    config: KubernetesExecutorConfig,
    appio_root_certificates: str | None,
    launch_attempt_id: str,
) -> JSONObject:
    """Build the TaskExecutor Pod for a claimed SuperExec task."""
    container: JSONObject = {
        "name": "taskexecutor",
        "image": config.image,
        "command": [TASK_TYPE_TO_COMMAND[spec.task_type]],
        "args": _taskexecutor_args(spec, appio_root_certificates),
        "volumeMounts": [
            {
                "name": "appio-credentials",
                "mountPath": APPIO_CREDENTIALS_MOUNT_PATH,
                "readOnly": True,
            }
        ],
    }
    if config.image_pull_policy is not None:
        container["imagePullPolicy"] = config.image_pull_policy
    if config.resources is not None:
        container["resources"] = config.resources
    if config.env is not None:
        container["env"] = config.env
    if config.container_security_context is not None:
        container["securityContext"] = config.container_security_context

    pod_spec: JSONObject = {
        "automountServiceAccountToken": False,
        "restartPolicy": "Never",
        "containers": [container],
        "volumes": [
            {
                "name": "appio-credentials",
                "secret": {
                    "secretName": _credential_secret_name(spec, launch_attempt_id),
                    "defaultMode": 0o444,
                },
            }
        ],
    }
    if config.service_account_name is not None:
        pod_spec["serviceAccountName"] = config.service_account_name
    if config.node_selector is not None:
        pod_spec["nodeSelector"] = cast(JSONObject, config.node_selector)
    if config.tolerations is not None:
        pod_spec["tolerations"] = config.tolerations
    if config.affinity is not None:
        pod_spec["affinity"] = config.affinity
    if config.priority_class_name is not None:
        pod_spec["priorityClassName"] = config.priority_class_name
    if config.pod_security_context is not None:
        pod_spec["securityContext"] = config.pod_security_context

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": _metadata(
            _pod_name(spec, launch_attempt_id), spec, config, launch_attempt_id
        ),
        "spec": pod_spec,
    }


def _taskexecutor_args(
    spec: ExecutionSpec, appio_root_certificates: str | None
) -> list[str]:
    """Build TaskExecutor arguments with file-based credential delivery."""
    args = [
        TASK_TYPE_TO_APPIO_API_ADDRESS_ARG[spec.task_type],
        spec.appio_api_address,
        "--token-file",
        APPIO_TOKEN_FILE_PATH,
    ]

    if spec.insecure:
        args.append("--insecure")
    elif appio_root_certificates is not None:
        args.extend(["--root-certificates", APPIO_ROOT_CERTIFICATES_FILE_PATH])

    if spec.runtime_dependency_install:
        args.append("--allow-runtime-dependency-installation")

    return args


def _taskexecutor_env(env: list[JSONObject]) -> list[JSONObject]:
    """Build validated TaskExecutor container environment entries."""
    if not isinstance(env, list):
        raise ValueError("TaskExecutor env must be a list of mappings.")
    entries: list[JSONObject] = []
    for entry in env:
        if not isinstance(entry, dict):
            raise ValueError("TaskExecutor env entries must be mappings.")
        # Keep this path limited to non-secret literal config. Design secret
        # references separately before allowing them into TaskExecutor Pods.
        if "valueFrom" in entry:
            raise ValueError(
                "TaskExecutor env entries support literal 'value' only; "
                "'valueFrom' is not supported."
            )
        if set(entry) != {"name", "value"}:
            raise ValueError(
                "TaskExecutor env entries must contain exactly 'name' and 'value'."
            )
        name = entry["name"]
        value = entry["value"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("TaskExecutor env names must be non-empty strings.")
        # Validate env name locally so invalid executor config fails before Pod creation
        if not _KUBERNETES_ENV_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"TaskExecutor env name {name!r} must be a valid Kubernetes "
                "environment variable name."
            )
        if not isinstance(value, str):
            raise ValueError(f"TaskExecutor env value for {name!r} must be a string.")
        # Reject task-visible provider API key env names before Pod construction
        if name in _FORBIDDEN_TASKEXECUTOR_ENV_NAMES:
            raise ValueError(
                f"TaskExecutor env name {name!r} is not allowed because it is a "
                "provider API key."
            )
        # Copy only the validated Kubernetes env shape into the generated Pod spec.
        entries.append({"name": name, "value": value})
    return entries


def _get_appio_root_certificates(
    spec: ExecutionSpec, config: KubernetesExecutorConfig
) -> str | None:
    """Return PEM data for AppIo root certificates, if configured."""
    if config.appio_root_certificates is not None:
        return config.appio_root_certificates
    if spec.root_certificates_path is not None:
        return (
            Path(spec.root_certificates_path).expanduser().read_text(encoding="utf-8")
        )
    return None


def _new_launch_attempt_id() -> str:
    """Return a DNS-label-safe opaque identifier for one local launch call."""
    return uuid4().hex[:12]


def _pod_name(spec: ExecutionSpec, launch_attempt_id: str) -> str:
    """Return the TaskExecutor Pod name."""
    return f"flwr-taskexecutor-{spec.task_id}-{launch_attempt_id}"


def _credential_secret_name(spec: ExecutionSpec, launch_attempt_id: str) -> str:
    """Return the AppIo credential Secret name."""
    return _credential_secret_name_from_pod_name(_pod_name(spec, launch_attempt_id))


def _credential_secret_name_from_pod_name(pod_name: str) -> str:
    """Return the AppIo credential Secret name for a TaskExecutor Pod name."""
    return f"{pod_name}{_APPIO_CREDENTIAL_SECRET_SUFFIX}"


def _pod_name_from_credential_secret_name(secret_name: str) -> str | None:
    """Return the owner Pod name encoded in a credential Secret name."""
    if not secret_name.endswith(_APPIO_CREDENTIAL_SECRET_SUFFIX):
        return None
    pod_name = secret_name[: -len(_APPIO_CREDENTIAL_SECRET_SUFFIX)]
    if not pod_name:
        return None
    return pod_name


def _metadata(
    name: str,
    spec: ExecutionSpec,
    config: KubernetesExecutorConfig,
    launch_attempt_id: str,
) -> JSONObject:
    """Return Kubernetes object metadata."""
    metadata: JSONObject = {
        "name": name,
        "namespace": config.namespace,
        "labels": _labels(spec, config, launch_attempt_id),
    }
    if config.annotations is not None:
        metadata["annotations"] = cast(JSONObject, config.annotations)
    return metadata


def _labels(
    spec: ExecutionSpec, config: KubernetesExecutorConfig, launch_attempt_id: str
) -> JSONObject:
    """Return stable labels for Kubernetes objects."""
    labels: JSONObject = {}
    labels.update(_caller_labels(config))
    # Apply executor-owned labels last; selectors and cleanup rely on them.
    labels.update(
        {
            _NAME_LABEL: "flower",
            _COMPONENT_LABEL: "taskexecutor",
            _TASK_ID_LABEL: str(spec.task_id),
            _TASK_TYPE_LABEL: spec.task_type.value,
            LAUNCH_ATTEMPT_LABEL: launch_attempt_id,
        }
    )
    if config.resource_pool is not None:
        labels[_RESOURCE_POOL_LABEL] = config.resource_pool
    return labels


def _capacity_label_selector(config: KubernetesExecutorConfig) -> str:
    """Return the label selector used for resource-pool capacity checks."""
    return _taskexecutor_pool_label_selector(config)


def _taskexecutor_pool_label_selector(config: KubernetesExecutorConfig) -> str:
    """Return the label selector for TaskExecutor pool-scoped operations."""
    return _label_selector(_taskexecutor_pool_labels(config))


def _taskexecutor_pool_labels(config: KubernetesExecutorConfig) -> dict[str, str]:
    """Return labels identifying a scoped TaskExecutor pool."""
    labels = _caller_labels(config)
    labels.update(
        {
            _NAME_LABEL: "flower",
            _COMPONENT_LABEL: "taskexecutor",
        }
    )
    if config.resource_pool is not None:
        labels[_RESOURCE_POOL_LABEL] = config.resource_pool
    return labels


def _caller_labels(config: KubernetesExecutorConfig) -> dict[str, str]:
    """Return caller-provided labels that are not owned by the executor."""
    return {
        key: value
        for key, value in (config.labels or {}).items()
        if key not in _EXECUTOR_OWNED_LABELS
    }


def _label_selector(labels: dict[str, str]) -> str:
    """Return a Kubernetes equality label selector."""
    return ",".join(f"{key}={value}" for key, value in sorted(labels.items()))


def _pod_items(pod_list: KubernetesList | Mapping[str, object]) -> list[object]:
    """Return Pod items from a Kubernetes list response."""
    items = _object_field(pod_list, "items")
    if isinstance(items, Sequence) and not isinstance(items, str):
        return list(items)
    return []


def _secret_items(secret_list: KubernetesList | Mapping[str, object]) -> list[object]:
    """Return Secret items from a Kubernetes list response."""
    items = _object_field(secret_list, "items")
    if isinstance(items, Sequence) and not isinstance(items, str):
        return list(items)
    return []


def _is_active_pod(pod: object) -> bool:
    """Return true if a Pod counts against best-effort launch capacity."""
    status = _object_field(pod, "status")
    if _object_field(status, "phase") in {"Succeeded", "Failed"}:
        return False

    metadata = _object_field(pod, "metadata")
    deletion_timestamp = _object_field(metadata, "deletion_timestamp")
    if deletion_timestamp is None:
        deletion_timestamp = _object_field(metadata, "deletionTimestamp")
    if deletion_timestamp is not None:
        return True

    return _object_field(status, "phase") not in {"Succeeded", "Failed"}


def _is_terminal_pod(pod: object) -> bool:
    """Return true if a Pod reached a terminal phase."""
    status = _object_field(pod, "status")
    return _object_field(status, "phase") in {"Succeeded", "Failed"}


def _object_name(value: object) -> str | None:
    """Return an object's metadata name."""
    metadata = _object_field(value, "metadata")
    name = _object_field(metadata, "name")
    if isinstance(name, str) and name.strip():
        return name
    return None


def _has_task_id_label(value: object) -> bool:
    """Return true if an object carries a TaskExecutor task-id label."""
    metadata = _object_field(value, "metadata")
    labels = _object_field(metadata, "labels")
    task_id = _object_field(labels, _TASK_ID_LABEL)
    return isinstance(task_id, str) and bool(task_id.strip())


def _object_field(value: object, field_name: str) -> object | None:
    """Return a field from a Kubernetes dict or model object."""
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _launch_result_from_exception(exc: Exception) -> LaunchResult:
    """Map immediate Kubernetes API exceptions to launch results."""
    message = f"{type(exc).__name__}: {exc}"
    status = _exception_status(exc)
    lower_message = message.lower()

    if isinstance(exc, (ConnectionError, TimeoutError)):
        return LaunchResult.unknown(message)

    if status == 429 or _is_capacity_message(lower_message):
        return LaunchResult.capacity_rejected(message)

    if status is not None and (status == 408 or status >= 500):
        return LaunchResult.unknown(message)

    return LaunchResult.failed(message)


def _is_definite_pod_rejection(exc: Exception) -> bool:
    """Return true when Pod submission definitely failed before acceptance."""
    status = _exception_status(exc)
    if status is None:
        return False

    # Cleanup only relies on the Kubernetes API status contract. Message matching
    # remains useful for LaunchResult mapping, but it is too heuristic to decide
    # whether deleting the just-created Secret is safe.
    return 400 <= status < 500 and status != 408


def _delete_secret_best_effort(
    client: KubernetesClient, namespace: str, secret_name: str
) -> None:
    """Best-effort cleanup for a Secret whose Pod was definitely rejected."""
    try:
        client.delete_namespaced_secret(secret_name, namespace)
    except Exception:  # pylint: disable=broad-exception-caught
        log(
            WARNING,
            "Failed to delete Kubernetes credential Secret %r in namespace %r",
            secret_name,
            namespace,
            exc_info=True,
        )


def _exception_status(exc: Exception) -> int | None:
    """Return an HTTP-like status from Kubernetes client exceptions."""
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    if isinstance(status, str) and status.isdigit():
        return int(status)
    return None


def _raise_unless_not_found(exc: Exception) -> None:
    """Raise Kubernetes client exceptions except already-deleted objects."""
    if _exception_status(exc) == 404:
        return
    raise exc


def _is_capacity_message(message: str) -> bool:
    """Return true for quota/admission capacity rejection messages."""
    # Status codes are preferred above; this intentionally brittle fallback covers
    # common Kubernetes quota/scheduler wording when clients only expose messages.
    capacity_markers = (
        "exceeded quota",
        "resourcequota",
        "quota exceeded",
        "too many requests",
        "rate limit",
        "insufficient cpu",
        "insufficient memory",
        "insufficient pods",
    )
    return any(marker in message for marker in capacity_markers)
