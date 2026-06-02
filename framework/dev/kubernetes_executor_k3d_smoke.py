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
import ast
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
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
DEFAULT_PROBE_IMAGE = "flwr-kubernetes-executor-k3d-probe:dev"
PROBE_ROOT_CERTIFICATE_COMMON_NAME = "flwr-k8s-executor-smoke-ca"
PROBE_ROOT_CERTIFICATE = """-----BEGIN CERTIFICATE-----
MIIDKzCCAhOgAwIBAgIUYtO398ah/BPXfS9K+brpsqbASW4wDQYJKoZIhvcNAQEL
BQAwJTEjMCEGA1UEAwwaZmx3ci1rOHMtZXhlY3V0b3Itc21va2UtY2EwHhcNMjYw
NjAyMjAzNzU1WhcNMzYwNTMwMjAzNzU1WjAlMSMwIQYDVQQDDBpmbHdyLWs4cy1l
eGVjdXRvci1zbW9rZS1jYTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEB
ALYGcPOoD1Gz/mBIVU+SL6C8+1I7d0dq20txhbhqaV4kftxv+2Wxw/dUjrurjDQe
/97/BKLXgBNwUam26CeFcwnsMyqWFXGCnNNfLToMop1bMUAev06PScPaRJ0RnjIF
4+Ov5cAVNPkFxR+Nw6ayOpkwjSXTIiqD7fug+wqxEKJRfg/QEc4zFa2V6zaQFOJa
vb0WGUFm7MzPa+32tEbH1pN6VW5wAT0JghLVx5FRA2cpH8sCieov8L7IVXM5aMu9
upcne7FQ99sSErER9sEjqFNK7byTP9xZIIjGT4Ep4DqWzORphwwEEzC4BrvTzLJm
HgskFwljSadTOg/dZu/wXo0CAwEAAaNTMFEwHQYDVR0OBBYEFNB+hF1kkiZ2RraA
/yN44RPK3v+RMB8GA1UdIwQYMBaAFNB+hF1kkiZ2RraA/yN44RPK3v+RMA8GA1Ud
EwEB/wQFMAMBAf8wDQYJKoZIhvcNAQELBQADggEBABgKx51NWCvPFHN0eGyfZ+Yo
3a85eBT9xPhnmBViffZcRb9OOlaz9+5mFTpQqVH+E7SLuB44gyWiHw4DSNj3cidh
vW2+dhHxfi/1zAnP32fHM8CKd8qYooX1nCK+yXwePNX0k/Fl5AdjmHbVn5M71ivx
Qd2OwBmMat8bHy3nKML7zU3T2kkycfrUxxfxXVjnBWHjGHZs9jaecoL3nid8Iy3C
SFu6f/R8MZ63Y8fw8Wx+CEo7OZC8mm+/Fcx9dvB+TsjW+xDkHTUcjBdKhMPQ791b
4D1rmzythc7HsGmlE2RiXxZY5VTk02cxASo31+zWia4erpwGY3yWcZFe62Tjw6k=
-----END CERTIFICATE-----
"""
DEFAULT_APPIO_API_ADDRESS = "127.0.0.1:9092"
DEFAULT_ACTIVE_POD_BUDGET = 1
DEFAULT_CAPACITY_TIMEOUT = 30.0
DEFAULT_CAPACITY_POLL_INTERVAL = 0.25
DEFAULT_POD_SUCCEEDED_TIMEOUT = 60.0
DEFAULT_POD_SUCCEEDED_POLL_INTERVAL = 0.5
DIAGNOSTIC_POD_STATUS_TIMEOUT = 3.0
DIAGNOSTIC_POD_STATUS_POLL = 0.50
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
    probe_image: bool = False
    probe_image_tag: str = DEFAULT_PROBE_IMAGE
    require_pod_succeeded: bool = False
    pod_succeeded_timeout: float = DEFAULT_POD_SUCCEEDED_TIMEOUT
    pod_succeeded_poll_interval: float = DEFAULT_POD_SUCCEEDED_POLL_INTERVAL


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


def _print_run_overview(config: SmokeConfig) -> None:
    """Print a concise overview of the smoke proof before doing work."""
    _print_section("KubernetesExecutor k3d smoke harness")
    print(
        "Purpose: exercise KubernetesExecutor against a real local k3d "
        "Kubernetes API without adding a runtime dependency on the Kubernetes "
        "Python client."
    )
    print(
        "Expected proof: create/reuse a k3d cluster and namespace, submit "
        "TaskExecutor Secret/Pod objects through KubernetesExecutor.launch(), "
        "confirm the same capacity selector sees those Pods, prove "
        "wait_for_capacity() blocks at budget, then delete only this run's "
        "labeled Pods and Secrets."
    )
    print(
        "Note: the Pod may stay Pending or fail image startup. This harness is "
        "checking Kubernetes API interaction and capacity behavior, not a "
        "complete TaskExecutor runtime."
    )
    print("")
    print("Configuration:")
    _print_detail("cluster", config.cluster_name)
    _print_detail("namespace", config.namespace)
    _print_detail("image", config.image)
    _print_detail("image pull policy", config.image_pull_policy)
    _print_detail("probe image mode", str(config.probe_image))
    if config.probe_image:
        _print_detail("probe image tag", config.probe_image_tag)
        _print_detail(
            "probe root certificate commonName",
            PROBE_ROOT_CERTIFICATE_COMMON_NAME,
        )
    _print_detail("AppIo API address", config.appio_api_address)
    _print_detail("active Pod budget", str(config.active_pod_budget))
    _print_detail("capacity timeout", f"{config.capacity_timeout}s")
    _print_detail("capacity poll interval", f"{config.capacity_poll_interval}s")
    _print_detail("require Pod succeeded", str(config.require_pod_succeeded))
    if config.require_pod_succeeded:
        _print_detail("Pod succeeded timeout", f"{config.pod_succeeded_timeout}s")
        _print_detail(
            "Pod succeeded poll interval",
            f"{config.pod_succeeded_poll_interval}s",
        )
    _print_detail("keep resources", str(config.keep_resources))
    _print_detail("delete cluster", str(config.delete_cluster))


def _print_section(title: str) -> None:
    """Print a visible section header for command-line smoke output."""
    print("")
    print(f"=== {title} ===")


def _print_detail(name: str, value: str) -> None:
    """Print one indented name/value detail."""
    print(f"  {name}: {value}")


def _print_objects(
    title: str, objects: Sequence[Any], *, include_status: bool = False
) -> None:
    """Print Kubernetes object names returned by a list call."""
    print(f"{title}: {len(objects)}")
    for value in objects:
        name = _object_name(value) or "<missing metadata.name>"
        details = [name]
        if include_status:
            phase = _object_phase(value)
            if phase is not None:
                details.append(f"phase={phase}")
            if _object_deletion_timestamp(value) is not None:
                details.append("deleting=true")
        print(f"  - {' '.join(details)}")


def _print_pod_conditions(pod: Any) -> None:
    """Print Pod status conditions."""
    conditions = _pod_conditions(pod)
    print(f"Pod conditions: {len(conditions)}")
    for condition in conditions:
        condition_type = _field_as_str(condition, "type") or "<unknown>"
        status = _field_as_str(condition, "status") or "<unknown>"
        reason = _field_as_str(condition, "reason")
        message = _field_as_str(condition, "message")
        last_transition_time = _field_as_str(
            condition, "last_transition_time", "lastTransitionTime"
        )
        details = [f"type={condition_type}", f"status={status}"]
        if reason:
            details.append(f"reason={reason}")
        if last_transition_time:
            details.append(f"lastTransitionTime={last_transition_time}")
        print(f"  - {' '.join(details)}")
        if message:
            print(f"    message: {message}")


def _print_container_statuses(container_statuses: Sequence[Any]) -> None:
    """Print container lifecycle state from Pod status."""
    print(f"Container statuses: {len(container_statuses)}")
    for container_status in container_statuses:
        name = _field_as_str(container_status, "name") or "<unknown>"
        ready = _field_as_str(container_status, "ready")
        restart_count = _field_as_str(container_status, "restart_count", "restartCount")
        image = _field_as_str(container_status, "image")
        details = [f"name={name}"]
        if ready is not None:
            details.append(f"ready={ready}")
        if restart_count is not None:
            details.append(f"restartCount={restart_count}")
        if image:
            details.append(f"image={image}")
        print(f"  - {' '.join(details)}")
        print(f"    state: {_container_state_summary(container_status)}")
        last_state = _container_last_state_summary(container_status)
        if last_state is not None:
            print(f"    last state: {last_state}")


def _print_pod_events(api: Any, namespace: str, pod_name: str) -> None:
    """Print Kubernetes Events for one Pod without failing the smoke."""
    field_selector = f"involvedObject.kind=Pod,involvedObject.name={pod_name}"
    print("Pod events:")
    try:
        events = _list_items(
            api.list_namespaced_event(
                namespace=namespace, field_selector=field_selector
            )
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"  WARN: could not list Pod events: {exc}")
        return

    if not events:
        print("  (none)")
        return

    for event in events:
        event_type = _field_as_str(event, "type") or "<unknown>"
        reason = _field_as_str(event, "reason") or "<unknown>"
        count = _field_as_str(event, "count")
        timestamp = _event_timestamp(event)
        message = _field_as_str(event, "message")
        details = [f"type={event_type}", f"reason={reason}"]
        if count is not None:
            details.append(f"count={count}")
        if timestamp is not None:
            details.append(f"time={timestamp}")
        print(f"  - {' '.join(details)}")
        if message:
            print(f"    message: {message}")


def _print_started_container_logs(
    api: Any, namespace: str, pod_name: str, container_statuses: Sequence[Any]
) -> None:
    """Print logs for containers that reached running or terminated state."""
    started_containers = [
        container_status
        for container_status in container_statuses
        if _container_started(container_status)
    ]
    if not started_containers:
        print("Container logs: no container has started yet; skipping log read.")
        return

    print("Container logs:")
    for container_status in started_containers:
        container_name = _field_as_str(container_status, "name")
        if container_name is None:
            print("  WARN: skipping started container without a name.")
            continue
        print(f"  Reading logs from container `{container_name}`.")
        try:
            logs = api.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                container=container_name,
                tail_lines=80,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"  WARN: could not read logs for `{container_name}`: {exc}")
            continue

        logs_text = _command_output_text(logs)
        if not logs_text.strip():
            print("  (empty)")
            continue
        for line in logs_text.strip().splitlines():
            print(f"  {line}")


def _print_command_output(label: str, output: str) -> None:
    """Print captured command output with indentation."""
    print(f"{label}:")
    for line in output.strip().splitlines():
        print(f"  {line}")


def _command_output_text(output: object) -> str:
    """Return command or log output as printable text."""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    text = str(output)
    if text.startswith(("b'", 'b"')):
        try:
            literal = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
        if isinstance(literal, bytes):
            return literal.decode("utf-8", errors="replace")
    return text


def _format_command(command: Sequence[str]) -> str:
    """Return a shell-readable command line for smoke output."""
    return shlex.join(command)


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
        "--probe-image",
        action="store_true",
        default=_env_bool("PROBE_IMAGE", False),
        help=(
            "Build and import the dev-only probe image into k3d, use it as "
            "the TaskExecutor image with imagePullPolicy=Never, and require "
            "the executor-created Pod to reach Succeeded."
        ),
    )
    parser.add_argument(
        "--probe-image-tag",
        default=_env("PROBE_IMAGE_TAG", DEFAULT_PROBE_IMAGE),
        help="Local Docker image tag used when --probe-image is enabled.",
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
        "--require-pod-succeeded",
        action="store_true",
        default=_env_bool("REQUIRE_POD_SUCCEEDED", False),
        help=(
            "Require smoke Pods selected by the run label to reach Succeeded "
            "within the bounded Pod completion timeout."
        ),
    )
    parser.add_argument(
        "--pod-succeeded-timeout",
        type=float,
        default=_env_float("POD_SUCCEEDED_TIMEOUT", DEFAULT_POD_SUCCEEDED_TIMEOUT),
        help="Harness-level timeout for the required Pod Succeeded proof.",
    )
    parser.add_argument(
        "--pod-succeeded-poll-interval",
        type=float,
        default=_env_float(
            "POD_SUCCEEDED_POLL_INTERVAL", DEFAULT_POD_SUCCEEDED_POLL_INTERVAL
        ),
        help="Polling interval used while waiting for smoke Pods to succeed.",
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
    image = args.probe_image_tag if args.probe_image else args.image
    image_pull_policy = (
        DEFAULT_IMAGE_PULL_POLICY if args.probe_image else args.image_pull_policy
    )

    return SmokeConfig(
        cluster_name=args.cluster_name,
        namespace=args.namespace,
        image=image,
        image_pull_policy=image_pull_policy,
        appio_api_address=args.appio_api_address,
        active_pod_budget=args.active_pod_budget,
        capacity_timeout=args.capacity_timeout,
        capacity_poll_interval=args.capacity_poll_interval,
        keep_resources=args.keep_resources,
        delete_cluster=args.delete_cluster,
        probe_image=args.probe_image,
        probe_image_tag=args.probe_image_tag,
        require_pod_succeeded=args.require_pod_succeeded or args.probe_image,
        pod_succeeded_timeout=args.pod_succeeded_timeout,
        pod_succeeded_poll_interval=args.pod_succeeded_poll_interval,
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
    _print_run_overview(config)

    _print_section("Validate smoke configuration")
    validate_smoke_config(config)
    print("OK: configuration values are valid.")

    _check_local_tools()
    client_module, kube_config = _load_kubernetes()

    api: Any | None = None
    selector: str | None = None
    cleanup_completed = False
    cluster_created = ensure_k3d_cluster(config.cluster_name)
    try:
        prepare_probe_image(config)

        _print_section("Load kubeconfig and create Kubernetes API client")
        print(
            "Loading kubeconfig for the current context selected by "
            "`k3d kubeconfig merge --kubeconfig-switch-context`."
        )
        kube_config.load_kube_config()
        print("OK: kubeconfig loaded.")
        print(
            "Creating kubernetes.client.CoreV1Api for namespace, Pod, and Secret calls."
        )
        api = client_module.CoreV1Api()
        print("OK: CoreV1Api client created.")

        ensure_namespace(api, config.namespace)

        run_id = uuid.uuid4().hex
        selector, executor = build_smoke_executor(api, config, run_id)

        _print_section("Build smoke executor")
        _print_detail("run id", run_id)
        _print_detail("run label", f"{RUN_LABEL_KEY}={run_id}")
        _print_detail("capacity selector", selector)
        _print_detail("resource pool", LOCAL_RESOURCE_POOL)
        _print_detail("TaskExecutor image", config.image)
        _print_detail("image pull policy", config.image_pull_policy)
        _print_detail("AppIo API address", config.appio_api_address)
        if config.probe_image:
            _print_detail(
                "AppIo root certificate commonName",
                PROBE_ROOT_CERTIFICATE_COMMON_NAME,
            )
        _print_detail("active Pod budget", str(config.active_pod_budget))
        _print_detail("capacity poll interval", f"{config.capacity_poll_interval}s")
        print(
            "The executor creates one Secret and one Pod per launch. "
            "Task tokens are generated for the smoke run but are not printed."
        )

        launch_smoke_pods(executor, config, run_id)

        _print_section("Verify launched Pods are visible to the capacity selector")
        print(
            "Listing Pods with the same selector path used by "
            "KubernetesExecutor.wait_for_capacity()."
        )
        _print_detail("namespace", config.namespace)
        _print_detail("label selector", selector)
        pods = _list_items(
            api.list_namespaced_pod(namespace=config.namespace, label_selector=selector)
        )
        _print_objects("Pods returned by capacity selector", pods, include_status=True)
        if len(pods) < config.active_pod_budget:
            raise SmokeFailure(
                "KubernetesExecutor.launch returned accepted, but fewer smoke Pods "
                "than the configured active Pod budget were found by the capacity "
                f"selector: {len(pods)} of {config.active_pod_budget}."
            )
        print(
            "OK: capacity selector found at least the configured active Pod "
            f"budget ({len(pods)} of {config.active_pod_budget})."
        )
        if config.require_pod_succeeded:
            pods = wait_for_pods_succeeded(
                api,
                config.namespace,
                selector,
                expected_count=config.active_pod_budget,
                timeout=config.pod_succeeded_timeout,
                poll_interval=config.pod_succeeded_poll_interval,
            )
        report_pod_lifecycle_diagnostics(api, config.namespace, pods)

        if config.keep_resources:
            _print_section("Skip cleanup-sensitive wait proof")
            print(
                "Skipping bounded wait proof because --keep-resources preserves "
                "the active smoke Pods and Secrets for manual inspection."
            )
            print(
                "Inspect preserved resources with: "
                f"kubectl get pods,secrets -n {config.namespace} -l '{selector}'"
            )
        elif config.require_pod_succeeded:
            _print_section("Skip active-capacity wait proof")
            print(
                "Skipping wait_for_capacity() blocking proof because the probe "
                "mode requires the smoke Pod to complete first. Terminal Pods "
                "do not count against active capacity."
            )
            cleanup_labeled_objects(api, config.namespace, selector)
            wait_for_no_labeled_objects(
                api, config.namespace, selector, timeout=config.capacity_timeout
            )
            cleanup_completed = True
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
            cleanup_completed = True

        print("Kubernetes executor k3d smoke harness passed.")
        print(
            "Final manual cleanup check command: "
            f"kubectl get pods,secrets -n {config.namespace} -l '{selector}'"
        )
    finally:
        if not config.keep_resources and not cleanup_completed:
            try:
                if api is not None and selector is not None:
                    cleanup_labeled_objects(api, config.namespace, selector)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"WARN: cleanup failed: {exc}", file=sys.stderr)
        if config.delete_cluster and cluster_created:
            _print_section("Delete smoke k3d cluster")
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
        ("Probe image tag", config.probe_image_tag),
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
    if config.pod_succeeded_timeout <= 0:
        raise SmokeFailure("Pod succeeded timeout must be positive.")
    if config.pod_succeeded_poll_interval <= 0:
        raise SmokeFailure("Pod succeeded poll interval must be positive.")
    if config.keep_resources and config.delete_cluster:
        raise SmokeFailure("--keep-resources cannot be combined with --delete-cluster.")
    if config.probe_image and config.image != config.probe_image_tag:
        raise SmokeFailure(
            "Probe image mode must use the probe image tag as the "
            "TaskExecutor image."
        )
    if config.probe_image and config.image_pull_policy != DEFAULT_IMAGE_PULL_POLICY:
        raise SmokeFailure("Probe image mode requires imagePullPolicy=Never.")


def ensure_k3d_cluster(cluster_name: str) -> bool:
    """Create or reuse the local k3d cluster and select its kube context."""
    _print_section("Prepare local k3d cluster")
    print("Discovering existing local k3d clusters.")
    cluster_names = _k3d_cluster_names()
    if cluster_names:
        _print_detail("existing clusters", ", ".join(sorted(cluster_names)))
    else:
        _print_detail("existing clusters", "(none)")

    cluster_created = cluster_name not in cluster_names
    if cluster_created:
        print(f"Cluster `{cluster_name}` was not found; creating it now.")
        _run_command(["k3d", "cluster", "create", cluster_name])
        print(f"OK: created local k3d cluster `{cluster_name}`.")
    else:
        print(f"OK: reusing existing local k3d cluster `{cluster_name}`.")

    print("Selecting the cluster kube context for subsequent Kubernetes API calls.")
    _run_command(
        ["k3d", "kubeconfig", "merge", cluster_name, "--kubeconfig-switch-context"]
    )
    print(f"OK: kubeconfig now points at k3d cluster `{cluster_name}`.")
    return cluster_created


def prepare_probe_image(config: SmokeConfig) -> None:
    """Build and import the dev-only probe image when probe mode is enabled."""
    if not config.probe_image:
        return

    _print_section("Build and import controlled probe image")
    probe_dir = _probe_image_dir()
    print(
        "Building the dev-only probe image locally. The image exposes the "
        "TaskExecutor command names expected by KubernetesExecutor and exits "
        "successfully only after validating mounted credentials and rendered "
        "arguments."
    )
    _print_detail("probe image directory", str(probe_dir))
    _print_detail("probe image tag", config.probe_image_tag)
    _run_command(
        [
            "docker",
            "build",
            "--tag",
            config.probe_image_tag,
            str(probe_dir),
        ]
    )
    print(
        f"Importing `{config.probe_image_tag}` into k3d cluster "
        f"`{config.cluster_name}`."
    )
    _run_command(
        [
            "k3d",
            "image",
            "import",
            config.probe_image_tag,
            "--cluster",
            config.cluster_name,
        ]
    )


def _probe_image_dir() -> Path:
    """Return the dev-only probe image build directory."""
    return Path(__file__).resolve().parent / "kubernetes_executor_k3d_probe"


def ensure_namespace(api: Any, namespace: str) -> None:
    """Create the smoke namespace if it does not exist."""
    _print_section("Prepare smoke namespace")
    print(f"Checking whether namespace `{namespace}` already exists.")
    try:
        api.read_namespace(name=namespace)
        print(f"OK: namespace `{namespace}` already exists.")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if _exception_status(exc) != 404:
            raise
        print(f"Namespace `{namespace}` was not found; creating it now.")
        body = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": namespace},
        }
        api.create_namespace(body=body)
        print(f"OK: created namespace `{namespace}`.")


def build_executor_config(config: SmokeConfig, run_id: str) -> KubernetesExecutorConfig:
    """Build the executor config used for one smoke run."""
    return KubernetesExecutorConfig(
        namespace=config.namespace,
        image=config.image,
        appio_root_certificates=(
            PROBE_ROOT_CERTIFICATE if config.probe_image else None
        ),
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
        insecure=not config.probe_image,
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
    _print_section("Launch smoke TaskExecutor Pods")
    print(
        "Calling KubernetesExecutor.launch() until the configured active Pod "
        f"budget is reached ({config.active_pod_budget} launch attempt(s))."
    )
    print(
        "Each accepted launch proves the executor could submit the per-task "
        "Secret and Pod to the Kubernetes API. Pod startup is not required for "
        "this smoke test."
    )
    for task_offset in range(config.active_pod_budget):
        spec = build_execution_spec(config, run_id, task_offset)
        print("")
        print(
            f"Launch attempt {task_offset + 1} of {config.active_pod_budget}: "
            "TaskType.SERVER_APP"
        )
        _print_detail("task id", str(spec.task_id))
        _print_detail("AppIo API address", spec.appio_api_address)
        _print_detail("insecure AppIo", str(spec.insecure))
        if not spec.insecure:
            _print_detail(
                "root certificates",
                f"mounted dev certificate CN={PROBE_ROOT_CERTIFICATE_COMMON_NAME}",
            )
        _print_detail(
            "runtime dependency install", str(spec.runtime_dependency_install)
        )
        print("Submitting through KubernetesExecutor.launch().")
        result = executor.launch(spec)
        _print_detail("launch result status", result.status.value)
        if result.message:
            _print_detail("launch result message", result.message)
        if result.status != LaunchResultStatus.ACCEPTED:
            message = result.message or "No message."
            raise SmokeFailure(f"KubernetesExecutor.launch was not accepted: {message}")
        print("OK: launch was accepted by KubernetesExecutor.")


def report_pod_lifecycle_diagnostics(
    api: Any, namespace: str, pods: Sequence[Any]
) -> None:
    """Print diagnostic-only Pod lifecycle details for smoke Pods."""
    _print_section("Report smoke Pod lifecycle diagnostics")
    print(
        "These diagnostics are informational. Image pull failures, Pending "
        "Pods, and missing logs do not fail the default smoke test."
    )
    if not pods:
        print("No Pods were provided for lifecycle diagnostics.")
        return

    pods = _refresh_pods_for_lifecycle_diagnostics(api, namespace, pods)

    for pod in pods:
        name = _object_name(pod)
        if name is None:
            print("")
            print("Skipping lifecycle diagnostics for Pod without metadata.name.")
            continue

        print("")
        print(f"Pod `{name}`")
        _print_detail("phase", _object_phase(pod) or "<unknown>")
        deletion_timestamp = _object_deletion_timestamp(pod)
        _print_detail(
            "deletion timestamp",
            str(deletion_timestamp) if deletion_timestamp is not None else "<none>",
        )
        print("Manual follow-up commands:")
        print(f"  kubectl describe pod -n {namespace} {name}")
        container_statuses = _pod_container_statuses(pod)
        container_names = _container_names(container_statuses)
        if container_names:
            for container_name in container_names:
                print(
                    f"  kubectl logs -n {namespace} {name} "
                    f"-c {container_name} --tail=80"
                )
        else:
            print(f"  kubectl logs -n {namespace} {name} --tail=80")

        _print_pod_conditions(pod)
        _print_container_statuses(container_statuses)
        _print_pod_events(api, namespace, name)
        _print_started_container_logs(api, namespace, name, container_statuses)


def wait_for_pods_succeeded(
    api: Any,
    namespace: str,
    label_selector: str,
    *,
    expected_count: int,
    timeout: float,
    poll_interval: float,
) -> list[Any]:
    """Wait until all smoke Pods selected by the run label reach Succeeded."""
    _print_section("Require smoke Pods to reach Succeeded")
    print(
        "Probe mode requires the executor-created Pod to start and complete. "
        "The probe container validates the mounted token file and rendered "
        "TaskExecutor arguments without printing token contents."
    )
    _print_detail("namespace", namespace)
    _print_detail("label selector", label_selector)
    _print_detail("expected Pod count", str(expected_count))
    _print_detail("timeout", f"{timeout}s")

    deadline = time.monotonic() + timeout
    last_summary: str | None = None
    pods: list[Any] = []
    while time.monotonic() < deadline:
        pods = _list_items(
            api.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
        )
        summary = _pod_phase_summary(pods)
        if summary != last_summary:
            print(f"Current selected Pods: {summary}")
            last_summary = summary

        if len(pods) >= expected_count and all(
            _object_phase(pod) == "Succeeded" for pod in pods
        ):
            print("OK: all selected smoke Pods reached Succeeded.")
            return pods

        failed = [pod for pod in pods if _object_phase(pod) == "Failed"]
        if failed:
            report_pod_lifecycle_diagnostics(api, namespace, pods)
            failed_names = ", ".join(_object_name(pod) or "<unknown>" for pod in failed)
            raise SmokeFailure(f"Smoke Pod(s) failed before Succeeded: {failed_names}.")

        time.sleep(poll_interval)

    report_pod_lifecycle_diagnostics(api, namespace, pods)
    raise SmokeFailure(
        "Smoke Pod(s) did not reach Succeeded before timeout. Inspect with: "
        f"kubectl get pods -n {namespace} -l '{label_selector}'"
    )


def _pod_phase_summary(pods: Sequence[Any]) -> str:
    """Return a compact Pod phase summary for completion polling output."""
    if not pods:
        return "0 Pod(s)"
    values = []
    for pod in pods:
        name = _object_name(pod) or "<unknown>"
        phase = _object_phase(pod) or "<unknown>"
        values.append(f"{name}:{phase}")
    return ", ".join(values)


def _refresh_pods_for_lifecycle_diagnostics(
    api: Any, namespace: str, pods: Sequence[Any]
) -> list[Any]:
    """Refresh Pods briefly so diagnostic status fields can populate."""
    print(
        "Refreshing Pod status before diagnostics "
        f"(up to {DIAGNOSTIC_POD_STATUS_TIMEOUT}s)."
    )
    current_pods = list(pods)
    with_status = sum(1 for pod in current_pods if _pod_container_statuses(pod))
    if with_status == len(current_pods):
        print(
            "Pod status refresh: "
            f"{with_status} of {len(current_pods)} Pod(s) already have "
            "container status."
        )
        return current_pods

    deadline = time.monotonic() + DIAGNOSTIC_POD_STATUS_TIMEOUT
    while True:
        refreshed_pods = []
        for pod in current_pods:
            name = _object_name(pod)
            if name is None:
                refreshed_pods.append(pod)
                continue
            try:
                refreshed_pods.append(
                    api.read_namespaced_pod(name=name, namespace=namespace)
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"WARN: could not refresh Pod `{name}`: {exc}")
                refreshed_pods.append(pod)

        current_pods = refreshed_pods
        with_status = sum(1 for pod in current_pods if _pod_container_statuses(pod))
        print(
            "Pod status refresh: "
            f"{with_status} of {len(current_pods)} Pod(s) have container status."
        )
        if with_status == len(current_pods) or time.monotonic() >= deadline:
            return current_pods
        time.sleep(DIAGNOSTIC_POD_STATUS_POLL)


def prove_wait_for_capacity_blocks_and_unblocks(
    *,
    executor: KubernetesExecutor,
    cleanup: Callable[[], None],
    timeout: float,
    block_check_timeout: float = 1.0,
) -> None:
    """Prove wait_for_capacity blocks at budget and unblocks after cleanup."""
    _print_section("Prove wait_for_capacity blocks and unblocks")
    print(
        "Starting wait_for_capacity() in a helper thread while smoke Pods are "
        "still active. It should remain blocked because the active Pod count is "
        "at the configured budget."
    )
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

    print(f"OK: wait_for_capacity() was still blocked after {block_check_timeout}s.")
    print("Deleting smoke Pods to free capacity and unblock the helper thread.")
    cleanup()
    print(
        "Waiting for wait_for_capacity() to return after cleanup "
        f"(timeout {timeout}s)."
    )
    if not finished.wait(timeout):
        raise SmokeFailure(
            "wait_for_capacity did not unblock before the harness timeout "
            f"({timeout}s)."
        )
    if errors:
        raise SmokeFailure(f"wait_for_capacity failed after cleanup: {errors[0]}")

    print("OK: wait_for_capacity() unblocked after smoke Pod cleanup.")


def cleanup_labeled_objects(api: Any, namespace: str, label_selector: str) -> None:
    """Delete smoke Pods and Secrets matching the run selector."""
    _print_section("Clean up smoke Pods and Secrets")
    print(
        "Cleanup is scoped to the unique smoke-run label selector, so it does "
        "not delete unrelated namespace objects."
    )
    _print_detail("namespace", namespace)
    _print_detail("label selector", label_selector)
    delete_labeled_pods(api, namespace, label_selector)
    delete_labeled_secrets(api, namespace, label_selector)


def delete_labeled_pods(api: Any, namespace: str, label_selector: str) -> None:
    """Delete smoke Pods matching the run selector."""
    print("Listing smoke Pods selected for deletion.")
    pods = _list_items(
        api.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
    )
    _print_objects("Pods selected for deletion", pods, include_status=True)
    for pod in pods:
        name = _object_name(pod)
        if name is None:
            print("Skipping Pod without metadata.name.")
            continue
        try:
            print(f"Deleting Pod `{name}` with grace_period_seconds=0.")
            api.delete_namespaced_pod(
                name=name, namespace=namespace, grace_period_seconds=0
            )
            print(f"OK: delete requested for Pod `{name}`.")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _raise_unless_not_found(exc)
            print(f"OK: Pod `{name}` was already gone.")


def delete_labeled_secrets(api: Any, namespace: str, label_selector: str) -> None:
    """Delete smoke Secrets matching the run selector."""
    print("Listing smoke Secrets selected for deletion.")
    secrets = _list_items(
        api.list_namespaced_secret(namespace=namespace, label_selector=label_selector)
    )
    _print_objects("Secrets selected for deletion", secrets)
    for secret in secrets:
        name = _object_name(secret)
        if name is None:
            print("Skipping Secret without metadata.name.")
            continue
        try:
            print(f"Deleting Secret `{name}`.")
            api.delete_namespaced_secret(name=name, namespace=namespace)
            print(f"OK: delete requested for Secret `{name}`.")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _raise_unless_not_found(exc)
            print(f"OK: Secret `{name}` was already gone.")


def wait_for_no_labeled_objects(
    api: Any,
    namespace: str,
    label_selector: str,
    *,
    timeout: float,
    poll_interval: float = 0.5,
) -> None:
    """Wait until no smoke Pods or Secrets remain."""
    _print_section("Verify smoke cleanup completed")
    print(
        "Polling Kubernetes until no Pods or Secrets remain for the smoke-run "
        f"selector (timeout {timeout}s)."
    )
    deadline = time.monotonic() + timeout
    last_counts: tuple[int, int] | None = None
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
        counts = (pod_count, secret_count)
        if counts != last_counts:
            print(
                "Remaining labeled objects: "
                f"{pod_count} Pod(s), {secret_count} Secret(s)."
            )
            last_counts = counts
        if pod_count == 0 and secret_count == 0:
            print("OK: no smoke Pods or Secrets remain for this run selector.")
            return
        time.sleep(poll_interval)

    raise SmokeFailure(
        "Smoke cleanup did not finish before timeout. Remaining objects can be "
        f"inspected with: kubectl get pods,secrets -n {namespace} -l "
        f"'{label_selector}'"
    )


def _check_local_tools() -> None:
    """Skip when optional local smoke-test tools are missing."""
    _print_section("Check local prerequisites")
    for tool in ("docker", "k3d", "kubectl"):
        path = shutil.which(tool)
        if path is None:
            raise SkipSmoke(f"{tool} is required for the optional k3d smoke harness.")
        _print_detail(tool, path)
    print("Checking Docker daemon reachability with `docker info`.")
    result = subprocess.run(
        ["docker", "info"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise SkipSmoke("Docker is installed, but the Docker daemon is not reachable.")
    print("OK: Docker daemon is reachable.")


def _load_kubernetes() -> tuple[Any, Any]:
    """Load the optional Kubernetes Python client lazily."""
    _print_section("Load optional Kubernetes Python client")
    print("Importing `kubernetes.client` and `kubernetes.config` lazily.")
    try:
        from kubernetes import client, config  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SkipSmoke(
            "Optional Python package 'kubernetes' is missing. Run through "
            "`uv run --no-dev --with kubernetes ...` or install it locally."
        ) from exc
    print("OK: optional Kubernetes Python client imported.")
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
    print(f"Running command: {_format_command(command)}")
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        _print_command_output("stdout", result.stdout)
    if result.stderr.strip():
        _print_command_output("stderr", result.stderr)
    if result.returncode != 0:
        raise SmokeFailure(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )
    print(f"OK: command exited 0: {_format_command(command)}")
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


def _field(value: object, *names: str) -> Any | None:
    """Return the first named field from a Kubernetes dict or model object."""
    if value is None:
        return None
    for name in names:
        if isinstance(value, dict):
            if name in value:
                return value[name]
        elif hasattr(value, name):
            return getattr(value, name)
    return None


def _field_as_str(value: object, *names: str) -> str | None:
    """Return a Kubernetes object field formatted for diagnostic output."""
    field = _field(value, *names)
    if field is None:
        return None
    return str(field)


def _pod_conditions(pod: object) -> list[Any]:
    """Return Pod status conditions."""
    status = _field(pod, "status")
    conditions = _field(status, "conditions")
    if conditions is None:
        return []
    return list(conditions)


def _pod_container_statuses(pod: object) -> list[Any]:
    """Return Pod container statuses."""
    status = _field(pod, "status")
    container_statuses = _field(status, "container_statuses", "containerStatuses")
    if container_statuses is None:
        return []
    return list(container_statuses)


def _container_names(container_statuses: Sequence[Any]) -> list[str]:
    """Return named containers from container statuses."""
    names = []
    for container_status in container_statuses:
        name = _field_as_str(container_status, "name")
        if name is not None:
            names.append(name)
    return names


def _container_state_summary(container_status: object) -> str:
    """Return a concise current container state summary."""
    state = _field(container_status, "state")
    return _state_summary(state) or "<unknown>"


def _container_last_state_summary(container_status: object) -> str | None:
    """Return a concise previous container state summary, if present."""
    last_state = _field(container_status, "last_state", "lastState")
    return _state_summary(last_state)


def _state_summary(state: object) -> str | None:
    """Return a concise waiting/running/terminated state summary."""
    if state is None:
        return None
    for state_name in ("waiting", "running", "terminated"):
        state_detail = _field(state, state_name)
        if state_detail is not None:
            detail_summary = _state_detail_summary(state_detail)
            if detail_summary:
                return f"{state_name} {detail_summary}"
            return state_name
    return None


def _state_detail_summary(state_detail: object) -> str:
    """Return key fields from a Kubernetes container state detail."""
    field_names = (
        ("reason", "reason"),
        ("message", "message"),
        ("exitCode", "exit_code", "exitCode"),
        ("signal", "signal"),
        ("startedAt", "started_at", "startedAt"),
        ("finishedAt", "finished_at", "finishedAt"),
    )
    details = []
    for output_name, *lookup_names in field_names:
        value = _field_as_str(state_detail, *lookup_names)
        if value is not None:
            details.append(f"{output_name}={value}")
    return " ".join(details)


def _container_started(container_status: object) -> bool:
    """Return whether a container reached running or terminated state."""
    state = _field(container_status, "state")
    return (
        _field(state, "running") is not None or _field(state, "terminated") is not None
    )


def _event_timestamp(event: object) -> str | None:
    """Return a useful Kubernetes Event timestamp for diagnostics."""
    return _field_as_str(
        event,
        "event_time",
        "eventTime",
        "last_timestamp",
        "lastTimestamp",
        "first_timestamp",
        "firstTimestamp",
    )


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


def _object_phase(value: object) -> str | None:
    """Return status.phase from a Kubernetes dict or model object."""
    if isinstance(value, dict):
        status = value.get("status")
    else:
        status = getattr(value, "status", None)
    if status is None:
        return None
    if isinstance(status, dict):
        phase = status.get("phase")
    else:
        phase = getattr(status, "phase", None)
    if isinstance(phase, str) and phase.strip():
        return phase
    return None


def _object_deletion_timestamp(value: object) -> object | None:
    """Return metadata.deletionTimestamp from a Kubernetes dict or model object."""
    if isinstance(value, dict):
        metadata = value.get("metadata")
    else:
        metadata = getattr(value, "metadata", None)
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata.get("deletionTimestamp") or metadata.get("deletion_timestamp")
    return getattr(metadata, "deletion_timestamp", None)


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
