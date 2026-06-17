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
"""Dev-maintained in-cluster smoke script for the Kubernetes executor.

This is not a stable ``flwr-*`` command. It uses the production in-cluster
Kubernetes auth path, so run it inside a Kubernetes Pod with the optional
``kubernetes`` Python package installed. The runner ServiceAccount needs scoped
``pods`` and ``secrets`` list/create/delete permissions in the configured
namespace.

The smoke only proves config loading, client construction, RBAC/admission, and
create/list/delete for executor-rendered Kubernetes objects. It does not require
TaskExecutor completion or AppIo connectivity. Real in-cluster runs are optional
deployment evidence, not a Flower PR merge requirement.

Usage:
    python dev/kubernetes_executor_smoke.py --executor-config executor.yaml
    python dev/kubernetes_executor_smoke.py --executor-config executor.yaml --json
"""

# pylint: disable=protected-access

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from textwrap import dedent
from typing import Any
from uuid import uuid4

from flwr.supercore.constant import ExecutorType, TaskType
from flwr.supercore.superexec.executor.config import load_executor_config
from flwr.supercore.superexec.executor.factory import get_executor
from flwr.supercore.superexec.executor.kubernetes_executor import (
    KubernetesExecutor,
    KubernetesExecutorConfig,
    _build_appio_credentials_secret,
    _build_taskexecutor_pod,
    _label_selector,
    _object_name,
    _pod_items,
    _raise_unless_not_found,
    _secret_items,
    _taskexecutor_pool_label_selector,
)
from flwr.supercore.superexec.executor.types import ExecutionSpec

_SMOKE_LABEL = "flower.ai/smoke-run"
_SMOKE_TOKEN = "flower-kubernetes-executor-smoke-token"


@dataclass
class SmokeSummary:
    """Machine-readable summary for one smoke run."""

    status: str
    config_path: str
    smoke_run: str
    namespace: str | None = None
    resource_pool: str | None = None
    pool_selector: str | None = None
    smoke_selector: str | None = None
    secret_name: str | None = None
    pod_name: str | None = None
    secret_created: bool = False
    pod_created: bool = False
    secret_visible: bool = False
    pod_visible: bool = False
    secret_deleted: bool = False
    pod_deleted: bool = False
    cleanup_errors: list[str] = field(default_factory=list)
    error: str | None = None


def run_smoke(
    executor_config_path: str,
    *,
    smoke_run: str | None = None,
    log: Callable[[str], None] = print,
) -> SmokeSummary:
    """Run a bounded Kubernetes executor smoke check."""
    smoke_run = smoke_run or f"smoke-{uuid4().hex[:12]}"
    summary = SmokeSummary(
        status="failed",
        config_path=executor_config_path,
        smoke_run=smoke_run,
    )
    client: Any | None = None
    config: KubernetesExecutorConfig | None = None

    try:
        log(f"Loading executor config: {executor_config_path}")
        executor_config = load_executor_config(
            executor_config_path, ExecutorType.KUBERNETES
        )
        executor_config = _executor_config_with_smoke_label(executor_config, smoke_run)

        log("Constructing Kubernetes executor with in-cluster client")
        executor = get_executor(
            ExecutorType.KUBERNETES, executor_config=executor_config
        )
        if not isinstance(executor, KubernetesExecutor):
            raise RuntimeError("Factory did not return a KubernetesExecutor.")

        client = executor._client
        config = executor._config
        summary.namespace = config.namespace
        summary.resource_pool = config.resource_pool
        summary.pool_selector = _taskexecutor_pool_label_selector(config)
        summary.smoke_selector = _label_selector({_SMOKE_LABEL: smoke_run})

        log(f"Namespace: {config.namespace}")
        log(f"Resource pool: {config.resource_pool or '<none>'}")
        log(f"Pool selector: {summary.pool_selector}")
        log(f"Smoke selector: {summary.smoke_selector}")

        _check_scoped_list_access(client, config, summary, log)
        _create_smoke_objects(client, config, smoke_run, summary, log)
        _check_created_objects_visible(client, config, summary, log)
        summary.status = "passed"
        log("Kubernetes executor smoke passed")
    except Exception as err:  # pylint: disable=broad-exception-caught
        summary.status = "failed"
        summary.error = _safe_error_message(err)
        log(f"Kubernetes executor smoke failed: {summary.error}")
    finally:
        if client is not None and config is not None:
            _cleanup_created_objects(client, config, summary, log)

    if summary.cleanup_errors and summary.status == "passed":
        summary.status = "failed"
        summary.error = "Cleanup failed for one or more smoke objects."
    return summary


def _executor_config_with_smoke_label(
    executor_config: dict[object, object], smoke_run: str
) -> dict[object, object]:
    """Return executor config with a unique per-run smoke label."""
    updated_config = dict(executor_config)
    labels = updated_config.get("labels")
    if labels is None:
        updated_labels: dict[object, object] = {}
    elif isinstance(labels, Mapping):
        updated_labels = dict(labels)
    else:
        raise ValueError("Kubernetes executor config field 'labels' must be a mapping.")
    updated_labels[_SMOKE_LABEL] = smoke_run
    updated_config["labels"] = updated_labels
    return updated_config


def _check_scoped_list_access(
    client: Any,
    config: KubernetesExecutorConfig,
    summary: SmokeSummary,
    log: Callable[[str], None],
) -> None:
    """Check scoped Pod and Secret list access before creating objects."""
    if summary.smoke_selector is None:
        raise RuntimeError("Smoke selector was not initialized.")
    log("Checking scoped Pod and Secret list access")
    client.list_namespaced_pod(
        config.namespace,
        label_selector=summary.smoke_selector,
    )
    client.list_namespaced_secret(
        config.namespace,
        label_selector=summary.smoke_selector,
    )


def _create_smoke_objects(
    client: Any,
    config: KubernetesExecutorConfig,
    smoke_run: str,
    summary: SmokeSummary,
    log: Callable[[str], None],
) -> None:
    """Create one smoke Secret and Pod using executor object rendering."""
    spec = ExecutionSpec(
        task_type=TaskType.SERVER_APP,
        appio_api_address="127.0.0.1:9091",
        token=_SMOKE_TOKEN,
        insecure=True,
        root_certificates_path=None,
        runtime_dependency_install=False,
        parent_pid=None,
        suppress_output=True,
        task_id=1,
    )
    secret = _build_appio_credentials_secret(
        spec=spec,
        config=config,
        appio_root_certificates=None,
        launch_attempt_id=smoke_run,
    )
    pod = _build_taskexecutor_pod(
        spec=spec,
        config=config,
        appio_root_certificates=None,
        launch_attempt_id=smoke_run,
    )
    secret_name = _metadata_name(secret)
    pod_name = _metadata_name(pod)
    summary.secret_name = secret_name
    summary.pod_name = pod_name

    log(f"Creating smoke Secret: {secret_name}")
    client.create_namespaced_secret(config.namespace, secret)
    summary.secret_created = True

    log(f"Creating smoke Pod: {pod_name}")
    client.create_namespaced_pod(config.namespace, pod)
    summary.pod_created = True


def _check_created_objects_visible(
    client: Any,
    config: KubernetesExecutorConfig,
    summary: SmokeSummary,
    log: Callable[[str], None],
) -> None:
    """Check that created smoke objects are visible through the scoped selector."""
    if summary.smoke_selector is None:
        raise RuntimeError("Smoke selector was not initialized.")
    log("Checking created objects through scoped selectors")
    pods = _pod_items(
        client.list_namespaced_pod(
            config.namespace,
            label_selector=summary.smoke_selector,
        )
    )
    secrets = _secret_items(
        client.list_namespaced_secret(
            config.namespace,
            label_selector=summary.smoke_selector,
        )
    )
    summary.pod_visible = summary.pod_name in {_object_name(pod) for pod in pods}
    summary.secret_visible = summary.secret_name in {
        _object_name(secret) for secret in secrets
    }
    if not summary.pod_visible or not summary.secret_visible:
        raise RuntimeError("Created smoke objects were not visible through selector.")


def _cleanup_created_objects(
    client: Any,
    config: KubernetesExecutorConfig,
    summary: SmokeSummary,
    log: Callable[[str], None],
) -> None:
    """Delete smoke objects created by this run."""
    if summary.pod_created and summary.pod_name is not None:
        log(f"Deleting smoke Pod: {summary.pod_name}")
        try:
            client.delete_namespaced_pod(
                name=summary.pod_name,
                namespace=config.namespace,
                grace_period_seconds=0,
            )
            summary.pod_deleted = True
        except Exception as err:  # pylint: disable=broad-exception-caught
            try:
                _raise_unless_not_found(err)
            except Exception as cleanup_err:  # pylint: disable=broad-exception-caught
                message = _safe_error_message(cleanup_err)
                summary.cleanup_errors.append(f"Pod {summary.pod_name}: {message}")
                log(f"Failed to delete smoke Pod {summary.pod_name}: {message}")
            else:
                summary.pod_deleted = True

    if summary.secret_created and summary.secret_name is not None:
        log(f"Deleting smoke Secret: {summary.secret_name}")
        try:
            client.delete_namespaced_secret(
                name=summary.secret_name,
                namespace=config.namespace,
            )
            summary.secret_deleted = True
        except Exception as err:  # pylint: disable=broad-exception-caught
            try:
                _raise_unless_not_found(err)
            except Exception as cleanup_err:  # pylint: disable=broad-exception-caught
                message = _safe_error_message(cleanup_err)
                summary.cleanup_errors.append(
                    f"Secret {summary.secret_name}: {message}"
                )
                log(f"Failed to delete smoke Secret {summary.secret_name}: {message}")
            else:
                summary.secret_deleted = True


def _metadata_name(value: object) -> str:
    """Return a Kubernetes object metadata name or fail clearly."""
    name = _object_name(value)
    if name is None:
        raise RuntimeError("Rendered Kubernetes object is missing metadata.name.")
    return name


def _safe_error_message(err: Exception) -> str:
    """Return a redacted error message safe for smoke output."""
    return f"{type(err).__name__}: {str(err).replace(_SMOKE_TOKEN, '<redacted>')}"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test Kubernetes executor config and scoped API access from "
            "inside a Kubernetes Pod."
        ),
        epilog=dedent(
            """
            This dev-maintained script is not a stable flwr-* command. It uses
            production in-cluster auth, requires the optional kubernetes Python
            package, and needs runner ServiceAccount permissions to list,
            create, and delete Pods and Secrets in the configured namespace.
            It proves executor config/client/RBAC/admission/create/list/delete
            only; TaskExecutor completion and AppIo connectivity are out of
            scope. Real in-cluster runs are optional deployment evidence, not
            a Flower PR merge requirement.
            """
        ).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--executor-config",
        required=True,
        help="Path to the trusted Kubernetes executor YAML config.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit compact JSON summary on stdout; human logs go to stderr.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Kubernetes executor smoke script."""
    args = _parse_args(argv)
    log_stream = sys.stderr if args.json else sys.stdout
    summary = run_smoke(
        args.executor_config,
        log=lambda message: print(message, file=log_stream),
    )
    if args.json:
        print(json.dumps(asdict(summary), sort_keys=True, separators=(",", ":")))
    return 0 if summary.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
