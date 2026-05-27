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
"""Optional local k3d smoke harness for the SuperExec Kubernetes executor."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from flwr.supercore.constant import TaskType

# This dev harness intentionally exercises the exact capacity selector path
# without exporting a runtime API from kubernetes_executor.py.
from flwr.supercore.superexec.executor.kubernetes_executor import (
    KubernetesExecutor,
    KubernetesExecutorConfig,
    _capacity_label_selector,
)
from flwr.supercore.superexec.executor.types import ExecutionSpec, LaunchResultStatus

DEFAULT_CLUSTER_NAME = "flwr-k8s-executor-smoke"
DEFAULT_NAMESPACE = "flwr-k8s-executor-smoke"
DEFAULT_IMAGE = "ghcr.io/flwrlabs/taskexecutor:dev"
DEFAULT_IMAGE_PULL_POLICY = "Never"
DEFAULT_APPIO_API_ADDRESS = "127.0.0.1:9092"
DEFAULT_ACTIVE_POD_BUDGET = 1
DEFAULT_CAPACITY_TIMEOUT = 30.0
DEFAULT_CAPACITY_POLL_INTERVAL = 0.25
LOCAL_RESOURCE_POOL = "local-k3d-smoke"
RUN_LABEL_KEY = "flower.ai/k8s-executor-smoke-run"
ENV_PREFIX = "FLWR_K8S_EXECUTOR_SMOKE_"


class SkipSmoke(RuntimeError):
    """Raised when optional local smoke prerequisites are unavailable."""


class SmokeFailure(RuntimeError):
    """Raised when the smoke harness runs but the proof fails."""


@dataclass(frozen=True)
class SmokeConfig:
    """Configuration for one optional local k3d smoke run."""

    cluster_name: str
    namespace: str
    image: str
    image_pull_policy: str
    appio_api_address: str
    active_pod_budget: int
    capacity_timeout: float
    capacity_poll_interval: float
    keep_resources: bool
    delete_cluster: bool


class CoreV1ApiAdapter:
    """Harness-local adapter from Kubernetes CoreV1Api to KubernetesClient."""

    def __init__(self, api: Any) -> None:
        self._api = api

    def create_namespaced_secret(self, namespace: str, body: dict[str, Any]) -> object:
        """Create a Kubernetes Secret in the selected namespace."""
        return self._api.create_namespaced_secret(namespace=namespace, body=body)

    def create_namespaced_pod(self, namespace: str, body: dict[str, Any]) -> object:
        """Create a Kubernetes Pod in the selected namespace."""
        return self._api.create_namespaced_pod(namespace=namespace, body=body)

    def list_namespaced_pod(self, namespace: str, label_selector: str) -> object:
        """List Kubernetes Pods in the selected namespace."""
        return self._api.list_namespaced_pod(
            namespace=namespace, label_selector=label_selector
        )


def parse_args(argv: Sequence[str] | None = None) -> SmokeConfig:
    """Parse CLI args and environment defaults for the smoke harness."""
    parser = argparse.ArgumentParser(
        description="Run the optional local k3d smoke harness for KubernetesExecutor."
    )
    parser.add_argument(
        "--cluster-name",
        default=_env("CLUSTER_NAME", DEFAULT_CLUSTER_NAME),
        help="Local k3d cluster name.",
    )
    parser.add_argument(
        "--namespace",
        default=_env("NAMESPACE", DEFAULT_NAMESPACE),
        help="Namespace used for smoke-run objects.",
    )
    parser.add_argument(
        "--image",
        default=_env("IMAGE", DEFAULT_IMAGE),
        help="TaskExecutor image reference accepted by Kubernetes.",
    )
    parser.add_argument(
        "--image-pull-policy",
        default=_env("IMAGE_PULL_POLICY", DEFAULT_IMAGE_PULL_POLICY),
        help=(
            "Pod imagePullPolicy. The local default keeps the proof in "
            "API-acceptance mode without pulling an image."
        ),
    )
    parser.add_argument(
        "--appio-api-address",
        default=_env("APPIO_API_ADDRESS", DEFAULT_APPIO_API_ADDRESS),
        help="AppIo API address passed into the rendered TaskExecutor Pod.",
    )
    parser.add_argument(
        "--active-pod-budget",
        type=int,
        default=_env_int("ACTIVE_POD_BUDGET", DEFAULT_ACTIVE_POD_BUDGET),
        help="Active Pod budget for the local capacity wait proof.",
    )
    parser.add_argument(
        "--capacity-timeout",
        type=float,
        default=_env_float("CAPACITY_TIMEOUT", DEFAULT_CAPACITY_TIMEOUT),
        help="Harness-level timeout for bounded capacity wait proof.",
    )
    parser.add_argument(
        "--capacity-poll-interval",
        type=float,
        default=_env_float("CAPACITY_POLL_INTERVAL", DEFAULT_CAPACITY_POLL_INTERVAL),
        help="Capacity polling interval used by the executor.",
    )
    parser.add_argument(
        "--keep-resources",
        action="store_true",
        default=_env_bool("KEEP_RESOURCES", False),
        help="Keep smoke Pods and Secrets for debugging.",
    )
    parser.add_argument(
        "--delete-cluster",
        action="store_true",
        default=_env_bool("DELETE_CLUSTER", False),
        help="Delete the k3d cluster at the end only if this harness created it.",
    )
    args = parser.parse_args(argv)

    return SmokeConfig(
        cluster_name=args.cluster_name,
        namespace=args.namespace,
        image=args.image,
        image_pull_policy=args.image_pull_policy,
        appio_api_address=args.appio_api_address,
        active_pod_budget=args.active_pod_budget,
        capacity_timeout=args.capacity_timeout,
        capacity_poll_interval=args.capacity_poll_interval,
        keep_resources=args.keep_resources,
        delete_cluster=args.delete_cluster,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the optional local k3d smoke harness."""
    config = parse_args(argv)
    try:
        run_smoke(config)
    except SkipSmoke as exc:
        print(f"SKIP: {exc}")
        return 0
    except SmokeFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


def run_smoke(config: SmokeConfig) -> None:
    """Run one local Kubernetes API smoke proof."""
    validate_smoke_config(config)
    _check_local_tools()
    client_module, kube_config = _load_kubernetes()

    api: Any | None = None
    selector: str | None = None
    cluster_created = ensure_k3d_cluster(config.cluster_name)
    try:
        kube_config.load_kube_config()
        api = client_module.CoreV1Api()
        ensure_namespace(api, config.namespace)

        run_id = uuid.uuid4().hex
        selector, executor = build_smoke_executor(api, config, run_id)

        print(f"Using cluster: {config.cluster_name}")
        print(f"Using namespace: {config.namespace}")
        print(f"Using smoke selector: {selector}")

        launch_smoke_pods(executor, config, run_id)

        pods = _list_items(
            api.list_namespaced_pod(namespace=config.namespace, label_selector=selector)
        )
        if len(pods) < config.active_pod_budget:
            raise SmokeFailure(
                "KubernetesExecutor.launch returned accepted, but fewer smoke Pods "
                "than the configured active Pod budget were found by the capacity "
                f"selector: {len(pods)} of {config.active_pod_budget}."
            )
        print(f"Created smoke Pod count visible to capacity selector: {len(pods)}")

        if config.keep_resources:
            print(
                "Skipping bounded wait proof because --keep-resources preserves "
                "the active smoke Pod."
            )
        else:
            prove_wait_for_capacity_blocks_and_unblocks(
                executor=executor,
                cleanup=lambda: delete_labeled_pods(api, config.namespace, selector),
                timeout=config.capacity_timeout,
            )
            cleanup_labeled_objects(api, config.namespace, selector)
            wait_for_no_labeled_objects(
                api, config.namespace, selector, timeout=config.capacity_timeout
            )

        print("Kubernetes executor k3d smoke harness passed.")
        print(
            "Cleanup check: "
            f"kubectl get pods,secrets -n {config.namespace} -l '{selector}'"
        )
    finally:
        if not config.keep_resources:
            try:
                if api is not None and selector is not None:
                    cleanup_labeled_objects(api, config.namespace, selector)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"WARN: cleanup failed: {exc}", file=sys.stderr)
        if config.delete_cluster and cluster_created:
            _run_command(["k3d", "cluster", "delete", config.cluster_name])
        elif config.delete_cluster and not cluster_created:
            print(
                "Preserving reused k3d cluster despite --delete-cluster: "
                f"{config.cluster_name}"
            )


def validate_smoke_config(config: SmokeConfig) -> None:
    """Validate harness-local smoke configuration."""
    for name, value in (
        ("Cluster name", config.cluster_name),
        ("Namespace", config.namespace),
        ("TaskExecutor image", config.image),
        ("Image pull policy", config.image_pull_policy),
        ("AppIo API address", config.appio_api_address),
    ):
        if not value.strip():
            raise SmokeFailure(f"{name} must not be empty.")
    if config.active_pod_budget <= 0:
        raise SmokeFailure("Active Pod budget must be positive.")
    if config.capacity_timeout <= 0:
        raise SmokeFailure("Capacity timeout must be positive.")
    if config.capacity_poll_interval <= 0:
        raise SmokeFailure("Capacity poll interval must be positive.")
    if config.keep_resources and config.delete_cluster:
        raise SmokeFailure("--keep-resources cannot be combined with --delete-cluster.")


def ensure_k3d_cluster(cluster_name: str) -> bool:
    """Create or reuse the local k3d cluster and select its kube context."""
    cluster_names = _k3d_cluster_names()
    cluster_created = cluster_name not in cluster_names
    if cluster_created:
        print(f"Creating local k3d cluster: {cluster_name}")
        _run_command(["k3d", "cluster", "create", cluster_name])
    else:
        print(f"Reusing local k3d cluster: {cluster_name}")

    _run_command(
        ["k3d", "kubeconfig", "merge", cluster_name, "--kubeconfig-switch-context"]
    )
    return cluster_created


def ensure_namespace(api: Any, namespace: str) -> None:
    """Create the smoke namespace if it does not exist."""
    try:
        api.read_namespace(name=namespace)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if _exception_status(exc) != 404:
            raise
        body = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": namespace},
        }
        api.create_namespace(body=body)


def build_executor_config(config: SmokeConfig, run_id: str) -> KubernetesExecutorConfig:
    """Build the executor config used for one smoke run."""
    return KubernetesExecutorConfig(
        namespace=config.namespace,
        image=config.image,
        image_pull_policy=config.image_pull_policy,
        resource_pool=LOCAL_RESOURCE_POOL,
        labels={RUN_LABEL_KEY: run_id},
        active_pod_budget=config.active_pod_budget,
        capacity_poll_interval=config.capacity_poll_interval,
    )


def build_smoke_executor(
    api: Any, config: SmokeConfig, run_id: str
) -> tuple[str, KubernetesExecutor]:
    """Build the selector and executor used for one smoke run."""
    executor_config = build_executor_config(config, run_id)
    selector = _capacity_label_selector(executor_config)
    return selector, KubernetesExecutor(
        client=CoreV1ApiAdapter(api), config=executor_config
    )


def build_execution_spec(
    config: SmokeConfig, run_id: str, task_offset: int = 0
) -> ExecutionSpec:
    """Build the local smoke execution spec."""
    task_id = (uuid.UUID(run_id).int % 900_000_000) + task_offset + 1
    return ExecutionSpec(
        task_type=TaskType.SERVER_APP,
        appio_api_address=config.appio_api_address,
        token=f"smoke-token-{run_id}-{task_offset}",
        insecure=True,
        root_certificates_path=None,
        runtime_dependency_install=False,
        parent_pid=None,
        suppress_output=True,
        task_id=task_id,
    )


def launch_smoke_pods(
    executor: KubernetesExecutor, config: SmokeConfig, run_id: str
) -> None:
    """Launch enough smoke Pods to reach the configured active Pod budget."""
    for task_offset in range(config.active_pod_budget):
        spec = build_execution_spec(config, run_id, task_offset)
        result = executor.launch(spec)
        if result.status != LaunchResultStatus.ACCEPTED:
            message = result.message or "No message."
            raise SmokeFailure(f"KubernetesExecutor.launch was not accepted: {message}")


def prove_wait_for_capacity_blocks_and_unblocks(
    *,
    executor: KubernetesExecutor,
    cleanup: Callable[[], None],
    timeout: float,
    block_check_timeout: float = 1.0,
) -> None:
    """Prove wait_for_capacity blocks at budget and unblocks after cleanup."""
    finished = threading.Event()
    errors: list[BaseException] = []

    def _wait() -> None:
        try:
            executor.wait_for_capacity()
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=_wait, daemon=True)
    thread.start()

    if finished.wait(block_check_timeout):
        if errors:
            raise SmokeFailure(f"wait_for_capacity failed before blocking: {errors[0]}")
        raise SmokeFailure("wait_for_capacity returned before smoke Pod cleanup.")

    cleanup()
    if not finished.wait(timeout):
        raise SmokeFailure(
            "wait_for_capacity did not unblock before the harness timeout "
            f"({timeout}s)."
        )
    if errors:
        raise SmokeFailure(f"wait_for_capacity failed after cleanup: {errors[0]}")

    print("Bounded wait_for_capacity proof passed.")


def cleanup_labeled_objects(api: Any, namespace: str, label_selector: str) -> None:
    """Delete smoke Pods and Secrets matching the run selector."""
    delete_labeled_pods(api, namespace, label_selector)
    delete_labeled_secrets(api, namespace, label_selector)


def delete_labeled_pods(api: Any, namespace: str, label_selector: str) -> None:
    """Delete smoke Pods matching the run selector."""
    pods = _list_items(
        api.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
    )
    for pod in pods:
        name = _object_name(pod)
        if name is None:
            continue
        try:
            api.delete_namespaced_pod(
                name=name, namespace=namespace, grace_period_seconds=0
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _raise_unless_not_found(exc)


def delete_labeled_secrets(api: Any, namespace: str, label_selector: str) -> None:
    """Delete smoke Secrets matching the run selector."""
    secrets = _list_items(
        api.list_namespaced_secret(namespace=namespace, label_selector=label_selector)
    )
    for secret in secrets:
        name = _object_name(secret)
        if name is None:
            continue
        try:
            api.delete_namespaced_secret(name=name, namespace=namespace)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _raise_unless_not_found(exc)


def wait_for_no_labeled_objects(
    api: Any,
    namespace: str,
    label_selector: str,
    *,
    timeout: float,
    poll_interval: float = 0.5,
) -> None:
    """Wait until no smoke Pods or Secrets remain."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pod_count = len(
            _list_items(
                api.list_namespaced_pod(
                    namespace=namespace, label_selector=label_selector
                )
            )
        )
        secret_count = len(
            _list_items(
                api.list_namespaced_secret(
                    namespace=namespace, label_selector=label_selector
                )
            )
        )
        if pod_count == 0 and secret_count == 0:
            return
        time.sleep(poll_interval)

    raise SmokeFailure(
        "Smoke cleanup did not finish before timeout. Remaining objects can be "
        f"inspected with: kubectl get pods,secrets -n {namespace} -l "
        f"'{label_selector}'"
    )


def _check_local_tools() -> None:
    """Skip when optional local smoke-test tools are missing."""
    for tool in ("docker", "k3d", "kubectl"):
        if shutil.which(tool) is None:
            raise SkipSmoke(f"{tool} is required for the optional k3d smoke harness.")
    result = subprocess.run(
        ["docker", "info"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise SkipSmoke("Docker is installed, but the Docker daemon is not reachable.")


def _load_kubernetes() -> tuple[Any, Any]:
    """Load the optional Kubernetes Python client lazily."""
    try:
        from kubernetes import client, config  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SkipSmoke(
            "Optional Python package 'kubernetes' is missing. Run through "
            "`uv run --no-dev --with kubernetes ...` or install it locally."
        ) from exc
    return client, config


def _k3d_cluster_names() -> set[str]:
    """Return known local k3d cluster names."""
    result = _run_command(["k3d", "cluster", "list", "--no-headers"])
    names: set[str] = set()
    for line in result.stdout.splitlines():
        columns = line.split()
        if columns:
            names.add(columns[0])
    return names


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a local smoke-harness command."""
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SmokeFailure(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )
    return result


def _list_items(value: object) -> list[Any]:
    """Return items from a Kubernetes list response."""
    if isinstance(value, dict):
        items = value.get("items")
    else:
        items = getattr(value, "items", None)
    if items is None:
        return []
    return list(items)


def _object_name(value: object) -> str | None:
    """Return metadata.name from a Kubernetes dict or model object."""
    if isinstance(value, dict):
        metadata = value.get("metadata")
    else:
        metadata = getattr(value, "metadata", None)
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        name = metadata.get("name")
    else:
        name = getattr(metadata, "name", None)
    if isinstance(name, str) and name.strip():
        return name
    return None


def _exception_status(exc: Exception) -> int | None:
    """Return an HTTP-like status from Kubernetes client exceptions."""
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    if isinstance(status, str) and status.isdigit():
        return int(status)
    return None


def _raise_unless_not_found(exc: Exception) -> None:
    """Ignore cleanup races where Kubernetes already removed an object."""
    if _exception_status(exc) != 404:
        raise exc


def _env(name: str, default: str) -> str:
    """Return a smoke harness environment override."""
    return os.getenv(f"{ENV_PREFIX}{name}", default)


def _env_int(name: str, default: int) -> int:
    """Return an integer smoke harness environment override."""
    value = os.getenv(f"{ENV_PREFIX}{name}")
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    """Return a float smoke harness environment override."""
    value = os.getenv(f"{ENV_PREFIX}{name}")
    if value is None:
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    """Return a boolean smoke harness environment override."""
    value = os.getenv(f"{ENV_PREFIX}{name}")
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    sys.exit(main())
