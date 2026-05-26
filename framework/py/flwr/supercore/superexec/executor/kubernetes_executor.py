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

from dataclasses import dataclass
from typing import Any, Protocol

from flwr.supercore.constant import (
    TASK_TYPE_TO_APPIO_API_ADDRESS_ARG,
    TASK_TYPE_TO_COMMAND,
)

from .types import ExecutionSpec, LaunchResult

APPIO_CREDENTIALS_MOUNT_PATH = "/run/flwr/appio"
APPIO_TOKEN_FILE_PATH = f"{APPIO_CREDENTIALS_MOUNT_PATH}/token"
APPIO_ROOT_CERTIFICATES_FILE_PATH = f"{APPIO_CREDENTIALS_MOUNT_PATH}/ca.crt"


class KubernetesClient(Protocol):
    """Subset of Kubernetes CoreV1Api used by the executor."""

    def create_namespaced_secret(self, namespace: str, body: dict[str, Any]) -> object:
        """Create a Kubernetes Secret in the selected namespace."""

    def create_namespaced_pod(self, namespace: str, body: dict[str, Any]) -> object:
        """Create a Kubernetes Pod in the selected namespace."""


@dataclass(frozen=True)
class KubernetesExecutorConfig:
    """Configuration needed to build one TaskExecutor Pod and Secret.

    appio_root_certificates contains optional PEM data mounted as ca.crt.
    """

    namespace: str
    image: str
    appio_root_certificates: str | None = None
    image_pull_policy: str | None = None
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
            secret = build_appio_credentials_secret(spec, self._config)
            pod = build_taskexecutor_pod(spec, self._config)
            self._client.create_namespaced_secret(self._config.namespace, secret)
            self._client.create_namespaced_pod(self._config.namespace, pod)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return LaunchResult.failed(f"{type(exc).__name__}: {exc}")

        return LaunchResult.accepted()


def build_appio_credentials_secret(
    spec: ExecutionSpec, config: KubernetesExecutorConfig
) -> dict[str, Any]:
    """Build the AppIo credential Secret for a TaskExecutor Pod."""
    _validate_kubernetes_spec(spec, config)
    data = {"token": spec.token}
    if config.appio_root_certificates is not None:
        data["ca.crt"] = config.appio_root_certificates

    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": _credential_secret_name(spec),
            "namespace": config.namespace,
            "labels": _labels(spec),
        },
        "type": "Opaque",
        "stringData": data,
    }


def build_taskexecutor_pod(
    spec: ExecutionSpec, config: KubernetesExecutorConfig
) -> dict[str, Any]:
    """Build the TaskExecutor Pod for a claimed SuperExec task."""
    _validate_kubernetes_spec(spec, config)

    container = {
        "name": "taskexecutor",
        "image": config.image,
        "command": [TASK_TYPE_TO_COMMAND[spec.task_type]],
        "args": _taskexecutor_args(spec, config),
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

    pod_spec: dict[str, Any] = {
        "automountServiceAccountToken": False,
        "restartPolicy": "Never",
        "containers": [container],
        "volumes": [
            {
                "name": "appio-credentials",
                "secret": {
                    "secretName": _credential_secret_name(spec),
                    "defaultMode": 0o400,
                },
            }
        ],
    }
    if config.service_account_name is not None:
        pod_spec["serviceAccountName"] = config.service_account_name

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": _pod_name(spec),
            "namespace": config.namespace,
            "labels": _labels(spec),
        },
        "spec": pod_spec,
    }


def _taskexecutor_args(
    spec: ExecutionSpec, config: KubernetesExecutorConfig
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
    elif config.appio_root_certificates is not None:
        args.extend(["--root-certificates", APPIO_ROOT_CERTIFICATES_FILE_PATH])

    if spec.runtime_dependency_install:
        args.append("--allow-runtime-dependency-installation")

    return args


def _validate_kubernetes_spec(
    spec: ExecutionSpec, config: KubernetesExecutorConfig
) -> None:
    """Validate spec inputs required for Kubernetes object construction."""
    if not isinstance(spec.task_id, int) or spec.task_id <= 0:
        raise ValueError("Kubernetes executor requires a positive integer task_id.")
    if not spec.appio_api_address.strip():
        raise ValueError("Kubernetes executor requires an AppIo API address.")
    if not spec.token.strip():
        raise ValueError("Kubernetes executor requires a task token.")


def _pod_name(spec: ExecutionSpec) -> str:
    """Return the TaskExecutor Pod name."""
    return f"flwr-taskexecutor-{spec.task_id}"


def _credential_secret_name(spec: ExecutionSpec) -> str:
    """Return the AppIo credential Secret name."""
    return f"{_pod_name(spec)}-appio"


def _labels(spec: ExecutionSpec) -> dict[str, str]:
    """Return stable labels for Kubernetes objects."""
    return {
        "app.kubernetes.io/name": "flower",
        "app.kubernetes.io/component": "taskexecutor",
        "flower.ai/superexec-task-id": str(spec.task_id),
        "flower.ai/task-type": spec.task_type.value,
    }
