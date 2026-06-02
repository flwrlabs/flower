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

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

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

LOGGER = logging.getLogger(__name__)


class KubernetesClient(Protocol):
    """Subset of Kubernetes CoreV1Api used by the executor."""

    def create_namespaced_secret(self, namespace: str, body: JSONObject) -> object:
        """Create a Kubernetes Secret in the selected namespace."""

    def create_namespaced_pod(self, namespace: str, body: JSONObject) -> object:
        """Create a Kubernetes Pod in the selected namespace."""

    def delete_namespaced_secret(self, name: str, namespace: str) -> object:
        """Delete a Kubernetes Secret from the selected namespace."""


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
        _validate_required_string("Kubernetes namespace", self.namespace)
        _validate_required_string("TaskExecutor image", self.image)
        _validate_optional_string(
            "AppIo root certificates", self.appio_root_certificates
        )
        _validate_optional_string("Image pull policy", self.image_pull_policy)
        _validate_optional_string("Service account name", self.service_account_name)
        _validate_optional_string(
            "Resource pool", self.resource_pool, reject_outer_whitespace=True
        )
        _validate_optional_string(
            "Priority class name",
            self.priority_class_name,
            reject_outer_whitespace=True,
        )
        _validate_optional_labels(self.labels)
        _validate_optional_string_map("Kubernetes annotations", self.annotations)
        _validate_optional_string_map("Node selector", self.node_selector)


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
            launch_attempt = _new_launch_attempt_suffix()
            secret_name = _credential_secret_name(spec, launch_attempt)
            secret = _build_appio_credentials_secret(
                spec, self._config, appio_root_certificates, launch_attempt
            )
            pod = _build_taskexecutor_pod(
                spec, self._config, appio_root_certificates, launch_attempt
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


def _build_appio_credentials_secret(
    spec: ExecutionSpec,
    config: KubernetesExecutorConfig,
    appio_root_certificates: str | None,
    launch_attempt: str,
) -> JSONObject:
    """Build the AppIo credential Secret for a TaskExecutor Pod."""
    data: JSONObject = {"token": spec.token}
    if appio_root_certificates is not None:
        data["ca.crt"] = appio_root_certificates

    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": _metadata(
            _credential_secret_name(spec, launch_attempt), spec, config, launch_attempt
        ),
        "type": "Opaque",
        "stringData": data,
    }


def _build_taskexecutor_pod(
    spec: ExecutionSpec,
    config: KubernetesExecutorConfig,
    appio_root_certificates: str | None,
    launch_attempt: str,
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
                    "secretName": _credential_secret_name(spec, launch_attempt),
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
        "metadata": _metadata(
            _pod_name(spec, launch_attempt), spec, config, launch_attempt
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


def _new_launch_attempt_suffix() -> str:
    """Return a DNS-label-safe identifier for one local launch attempt."""
    return uuid4().hex[:12]


def _pod_name(spec: ExecutionSpec, launch_attempt: str) -> str:
    """Return the TaskExecutor Pod name."""
    return f"flwr-taskexecutor-{spec.task_id}-{launch_attempt}"


def _credential_secret_name(spec: ExecutionSpec, launch_attempt: str) -> str:
    """Return the AppIo credential Secret name."""
    return f"{_pod_name(spec, launch_attempt)}-appio"


def _metadata(
    name: str,
    spec: ExecutionSpec,
    config: KubernetesExecutorConfig,
    launch_attempt: str,
) -> JSONObject:
    """Return Kubernetes object metadata."""
    metadata: JSONObject = {
        "name": name,
        "namespace": config.namespace,
        "labels": _labels(spec, config, launch_attempt),
    }
    if config.annotations is not None:
        metadata["annotations"] = cast(JSONObject, dict(config.annotations))
    return metadata


def _labels(
    spec: ExecutionSpec, config: KubernetesExecutorConfig, launch_attempt: str
) -> JSONObject:
    """Return stable labels for Kubernetes objects."""
    labels: JSONObject = {
        "app.kubernetes.io/name": "flower",
        "app.kubernetes.io/component": "taskexecutor",
        "flower.ai/superexec-task-id": str(spec.task_id),
        "flower.ai/task-type": spec.task_type.value,
        LAUNCH_ATTEMPT_LABEL: launch_attempt,
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
        LAUNCH_ATTEMPT_LABEL,
        "flower.ai/resource-pool",
    }
    conflicts = sorted(stable_label_names.intersection(labels))
    if conflicts:
        raise ValueError(
            f"Kubernetes labels must not override stable labels: {conflicts}"
        )
    _validate_string_map("Kubernetes labels", labels)


def _validate_optional_labels(labels: dict[str, str] | None) -> None:
    """Validate optional caller-provided labels."""
    if labels is None:
        return
    _validate_labels(labels)


def _validate_required_string(name: str, value: str) -> None:
    """Validate that a required string field is not empty."""
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _validate_optional_string(
    name: str, value: str | None, *, reject_outer_whitespace: bool = False
) -> None:
    """Validate that an optional string field is not empty when provided."""
    if value is None:
        return
    _validate_required_string(name, value)
    if reject_outer_whitespace and value != value.strip():
        raise ValueError(f"{name} must not include leading/trailing whitespace.")


def _validate_optional_string_map(name: str, values: dict[str, str] | None) -> None:
    """Validate optional string mapping entries when provided."""
    if values is None:
        return
    _validate_string_map(name, values)


def _validate_string_map(name: str, values: dict[str, str]) -> None:
    """Validate that mapping entries are non-empty strings."""
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{name} entries must be strings.")
        if not key.strip() or not value.strip():
            raise ValueError(f"{name} entries must not be empty.")


def _copy_json_object(value: object) -> JSONObject:
    """Return a plain dict/list copy of a JSON object."""
    # Keep config objects normal and copy only when rendering Kubernetes payloads.
    # This avoids recursive freeze/thaw while preventing rendered bodies from sharing
    # mutable nested structures with caller-owned config.
    return cast(JSONObject, _copy_json_value(value))


def _copy_json_value(value: object) -> object:
    """Return a mutable copy of a JSON-compatible value."""
    if isinstance(value, Mapping):
        return {key: _copy_json_value(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        # Normalize tuples and other sequences to JSON lists at the API boundary.
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
        LOGGER.warning(
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
