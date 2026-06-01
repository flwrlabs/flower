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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from flwr.supercore.constant import (
    TASK_TYPE_TO_APPIO_API_ADDRESS_ARG,
    TASK_TYPE_TO_COMMAND,
)
from flwr.supercore.typing import JSONObject

from .types import ExecutionSpec, LaunchResult

APPIO_CREDENTIALS_MOUNT_PATH = "/run/flwr/appio"
APPIO_TOKEN_FILE_PATH = f"{APPIO_CREDENTIALS_MOUNT_PATH}/token"
APPIO_ROOT_CERTIFICATES_FILE_PATH = f"{APPIO_CREDENTIALS_MOUNT_PATH}/ca.crt"


class KubernetesClient(Protocol):
    """Subset of Kubernetes CoreV1Api used by the executor."""

    def create_namespaced_secret(self, namespace: str, body: JSONObject) -> object:
        """Create a Kubernetes Secret in the selected namespace."""

    def create_namespaced_pod(self, namespace: str, body: JSONObject) -> object:
        """Create a Kubernetes Pod in the selected namespace."""


@dataclass(frozen=True)
class KubernetesExecutorConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration needed to build one TaskExecutor Pod and Secret.

    appio_root_certificates contains optional PEM data mounted as ca.crt. If unset,
    launch uses ExecutionSpec.root_certificates_path when provided.
    """

    namespace: str
    image: str
    appio_root_certificates: str | None = None
    image_pull_policy: str | None = None
    labels: dict[str, str] | None = None
    annotations: dict[str, str] | None = None
    resource_pool: str | None = None
    resources: JSONObject | None = None
    node_selector: dict[str, str] | None = None
    tolerations: list[JSONObject] | None = None
    affinity: JSONObject | None = None
    priority_class_name: str | None = None
    pod_security_context: JSONObject | None = None
    container_security_context: JSONObject | None = None
    # Optional Pod field only; service account policy/RBAC is decided elsewhere.
    service_account_name: str | None = None

    def __post_init__(self) -> None:
        """Validate required object-building inputs."""
        if not self.namespace.strip():
            raise ValueError("Kubernetes namespace must not be empty.")
        if not self.image.strip():
            raise ValueError("TaskExecutor image must not be empty.")
        if self.appio_root_certificates is not None and not (
            self.appio_root_certificates.strip()
        ):
            raise ValueError("AppIo root certificates must not be empty.")
        if self.image_pull_policy is not None and not self.image_pull_policy.strip():
            raise ValueError("Image pull policy must not be empty.")
        if self.service_account_name is not None and not (
            self.service_account_name.strip()
        ):
            raise ValueError("Service account name must not be empty.")
        if self.resource_pool is not None:
            if not self.resource_pool.strip():
                raise ValueError("Resource pool must not be empty.")
            if self.resource_pool != self.resource_pool.strip():
                raise ValueError(
                    "Resource pool must not include leading/trailing whitespace."
                )
        if self.priority_class_name is not None:
            if not self.priority_class_name.strip():
                raise ValueError("Priority class name must not be empty.")
            if self.priority_class_name != self.priority_class_name.strip():
                raise ValueError(
                    "Priority class name must not include leading/trailing whitespace."
                )
        if self.labels is not None:
            _validate_labels(self.labels)
        if self.annotations is not None:
            _validate_string_map("Kubernetes annotations", self.annotations)
        if self.node_selector is not None:
            _validate_string_map("Node selector", self.node_selector)


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

    def wait_for_capacity(self) -> None:
        """Return immediately.

        Kubernetes capacity checks are not implemented yet.
        """

    def launch(self, spec: ExecutionSpec) -> LaunchResult:
        """Submit the TaskExecutor Pod and credential Secret."""
        try:
            appio_root_certificates = _get_appio_root_certificates(spec, self._config)
            secret = _build_appio_credentials_secret(
                spec, self._config, appio_root_certificates
            )
            pod = _build_taskexecutor_pod(spec, self._config, appio_root_certificates)
            self._client.create_namespaced_secret(self._config.namespace, secret)
            self._client.create_namespaced_pod(self._config.namespace, pod)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return _launch_result_from_exception(exc)

        return LaunchResult.accepted()


def _build_appio_credentials_secret(
    spec: ExecutionSpec,
    config: KubernetesExecutorConfig,
    appio_root_certificates: str | None,
) -> JSONObject:
    """Build the AppIo credential Secret for a TaskExecutor Pod."""
    data: JSONObject = {"token": spec.token}
    if appio_root_certificates is not None:
        data["ca.crt"] = appio_root_certificates

    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": _metadata(_credential_secret_name(spec), spec, config),
        "type": "Opaque",
        "stringData": data,
    }


def _build_taskexecutor_pod(
    spec: ExecutionSpec,
    config: KubernetesExecutorConfig,
    appio_root_certificates: str | None,
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
        container["resources"] = _copy_json_object(config.resources)
    if config.container_security_context is not None:
        container["securityContext"] = _copy_json_object(
            config.container_security_context
        )

    pod_spec: JSONObject = {
        "automountServiceAccountToken": False,
        "restartPolicy": "Never",
        "containers": [container],
        "volumes": [
            {
                "name": "appio-credentials",
                "secret": {
                    "secretName": _credential_secret_name(spec),
                    "defaultMode": 0o444,
                },
            }
        ],
    }
    if config.service_account_name is not None:
        pod_spec["serviceAccountName"] = config.service_account_name
    if config.node_selector is not None:
        pod_spec["nodeSelector"] = cast(JSONObject, dict(config.node_selector))
    if config.tolerations is not None:
        pod_spec["tolerations"] = [
            _copy_json_object(toleration) for toleration in config.tolerations
        ]
    if config.affinity is not None:
        pod_spec["affinity"] = _copy_json_object(config.affinity)
    if config.priority_class_name is not None:
        pod_spec["priorityClassName"] = config.priority_class_name
    if config.pod_security_context is not None:
        pod_spec["securityContext"] = _copy_json_object(config.pod_security_context)

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": _metadata(_pod_name(spec), spec, config),
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


def _pod_name(spec: ExecutionSpec) -> str:
    """Return the TaskExecutor Pod name."""
    return f"flwr-taskexecutor-{spec.task_id}"


def _credential_secret_name(spec: ExecutionSpec) -> str:
    """Return the AppIo credential Secret name."""
    return f"{_pod_name(spec)}-appio"


def _metadata(
    name: str, spec: ExecutionSpec, config: KubernetesExecutorConfig
) -> JSONObject:
    """Return Kubernetes object metadata."""
    metadata: JSONObject = {
        "name": name,
        "namespace": config.namespace,
        "labels": _labels(spec, config),
    }
    if config.annotations is not None:
        metadata["annotations"] = cast(JSONObject, dict(config.annotations))
    return metadata


def _labels(spec: ExecutionSpec, config: KubernetesExecutorConfig) -> JSONObject:
    """Return stable labels for Kubernetes objects."""
    labels: JSONObject = {
        "app.kubernetes.io/name": "flower",
        "app.kubernetes.io/component": "taskexecutor",
        "flower.ai/superexec-task-id": str(spec.task_id),
        "flower.ai/task-type": spec.task_type.value,
    }
    if config.resource_pool is not None:
        labels["flower.ai/resource-pool"] = config.resource_pool
    if config.labels is not None:
        labels.update(config.labels)
    return labels


def _validate_labels(labels: dict[str, str]) -> None:
    """Validate that caller-provided labels do not replace stable labels."""
    stable_label_names = {
        "app.kubernetes.io/name",
        "app.kubernetes.io/component",
        "flower.ai/superexec-task-id",
        "flower.ai/task-type",
        "flower.ai/resource-pool",
    }
    conflicts = sorted(stable_label_names.intersection(labels))
    if conflicts:
        raise ValueError(
            f"Kubernetes labels must not override stable labels: {conflicts}"
        )
    _validate_string_map("Kubernetes labels", labels)


def _validate_string_map(name: str, values: dict[str, str]) -> None:
    """Validate that mapping entries are non-empty strings."""
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{name} entries must be strings.")
        if not key.strip() or not value.strip():
            raise ValueError(f"{name} entries must not be empty.")


def _copy_json_object(value: object) -> JSONObject:
    """Return a plain dict/list copy of a JSON object."""
    return cast(JSONObject, _copy_json_value(value))


def _copy_json_value(value: object) -> object:
    """Return a mutable copy of a JSON-compatible value."""
    if isinstance(value, Mapping):
        return {key: _copy_json_value(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_copy_json_value(nested) for nested in value]
    return value


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


def _exception_status(exc: Exception) -> int | None:
    """Return an HTTP-like status from Kubernetes client exceptions."""
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    if isinstance(status, str) and status.isdigit():
        return int(status)
    return None


def _is_capacity_message(message: str) -> bool:
    """Return true for quota/admission capacity rejection messages."""
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
