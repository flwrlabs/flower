#!/bin/bash

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
#
# Operator-facing local k8s release-testing helper.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
tmp_root="${TMPDIR:-/tmp}"
tmp_root="${tmp_root%/}"

image_tag="${IMAGE_TAG:-dev}"
cluster_name="${CLUSTER_NAME:-flower-local-k8s}"
kubectl_context="${KUBECTL_CONTEXT:-k3d-${cluster_name}}"
namespace="${NAMESPACE:-flower-local-k8s}"
output_dir="${OUTPUT_DIR:-${tmp_root}/flower-local-k8s-harnessctl}"
timeout_seconds="${TIMEOUT_SECONDS:-600}"
superlink_image="${SUPERLINK_IMAGE:-flwr/superlink:${image_tag}}"
superexec_image="${SUPEREXEC_IMAGE:-flwr/superexec:${image_tag}}"
taskexecutor_image="${TASKEXECUTOR_IMAGE:-${superexec_image}}"
image_pull_policy="${IMAGE_PULL_POLICY:-IfNotPresent}"
runtime_image_pull_policy="${RUNTIME_IMAGE_PULL_POLICY:-IfNotPresent}"
resource_pool="${RESOURCE_POOL:-generic-k3d}"
active_pod_budget="${ACTIVE_POD_BUDGET:-}"
capacity_poll_interval="${CAPACITY_POLL_INTERVAL:-}"
capacity_log_interval="${CAPACITY_LOG_INTERVAL:-}"
superlink_name="${SUPERLINK_NAME:-flower-superlink}"
superexec_name="${SUPEREXEC_NAME:-flower-superexec}"
seed_job_name="${SEED_JOB_NAME:-flower-local-k8s-seed-run}"
seed_config_name="${SEED_CONFIG_NAME:-flower-local-k8s-seed-run}"
executor_config_name="${EXECUTOR_CONFIG_NAME:-flower-local-k8s-executor-config}"
superexec_service_account="${SUPEREXEC_SERVICE_ACCOUNT:-flower-superexec}"
rbac_name="${RBAC_NAME:-flower-superexec-kubernetes-executor}"
create_cluster="${CREATE_CLUSTER:-true}"
import_images="${IMPORT_IMAGES:-true}"

usage() {
  cat <<EOF
Local k8s release-testing toolkit for the Flower Kubernetes executor harness.

Usage:
  framework/dev/k8s/harnessctl.sh <command> [options]

Commands:
  start-superlink
      Create the k3d cluster if needed, import the SuperLink image, and apply
      the local SuperLink Service/Pod.
  start-superexec [--active-pod-budget N]
      Apply SuperExec RBAC, executor config, and the SuperExec Pod.
  seed --count N [--hold-seconds X]
      Seed deterministic ServerApp tasks through the local Control API.
  seed --count N --crash
      Seed deterministic ServerApp tasks whose probe ServerApp fails.
  kill-superexec
      Delete the current local SuperExec Pod.
  stop-taskexecutors --count N
      Delete N currently observed TaskExecutor Pods for the active harness run.
  cleanup
      Delete the local harness namespace.
  watch-pods
      Watch all Pods in the harness namespace.
  watch-taskexecutors
      Watch only TaskExecutor Pods for the active harness run.
  logs-superexec [kubectl logs flags]
      Print or follow SuperExec logs.
  logs-taskexecutors [kubectl logs flags]
      Print or follow TaskExecutor logs for the active harness run.

Common environment:
  CLUSTER_NAME=${cluster_name}
  KUBECTL_CONTEXT=${kubectl_context}
  NAMESPACE=${namespace}
  IMAGE_TAG=${image_tag}
  SUPERLINK_IMAGE=${superlink_image}
  SUPEREXEC_IMAGE=${superexec_image}
  TASKEXECUTOR_IMAGE=${taskexecutor_image}
  ACTIVE_POD_BUDGET=${active_pod_budget:-<unset>}
  CREATE_CLUSTER=${create_cluster}
  IMPORT_IMAGES=${import_images}
  OUTPUT_DIR=${output_dir}

Set CREATE_CLUSTER=false or IMPORT_IMAGES=false when using an existing
non-default cluster or externally available images.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    die "$1 is required. Install it or adjust the command environment."
  fi
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    die "${option} requires a value"
  fi
}

is_true() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

kubectl_cmd() {
  kubectl --context "${kubectl_context}" "$@"
}

ensure_cluster() {
  if ! is_true "${create_cluster}"; then
    return
  fi
  require_command k3d
  if ! k3d cluster list "${cluster_name}" >/dev/null 2>&1; then
    echo "Creating k3d cluster ${cluster_name}"
    k3d cluster create "${cluster_name}" --wait
  fi
}

maybe_import_images() {
  if ! is_true "${import_images}"; then
    return
  fi
  require_command k3d
  local images=()
  local seen_images=" "
  local image
  for image in "$@"; do
    [[ -n "${image}" ]] || continue
    case "${seen_images}" in
      *" ${image} "*) ;;
      *)
        images+=("${image}")
        seen_images="${seen_images}${image} "
        ;;
    esac
  done
  if [[ "${#images[@]}" -gt 0 ]]; then
    echo "Importing images into k3d cluster ${cluster_name}: ${images[*]}"
    k3d image import "${images[@]}" -c "${cluster_name}"
  fi
}

manifest_path() {
  mkdir -p "${output_dir}/objects"
  echo "${output_dir}/objects/harnessctl-$1.yaml"
}

new_run_id() {
  if [[ -n "${HARNESS_RUN_ID:-}" ]]; then
    echo "${HARNESS_RUN_ID}"
  else
    echo "harnessctl-$(date +%Y%m%d%H%M%S)"
  fi
}

get_superlink_run_id() {
  kubectl_cmd get pod "${superlink_name}" -n "${namespace}" \
    -o "jsonpath={.metadata.labels.flower\\.ai/harness-run}" 2>/dev/null || true
}

get_existing_run_id() {
  local run_id
  run_id="$(get_superlink_run_id)"
  if [[ -n "${run_id}" ]]; then
    echo "${run_id}"
    return
  fi
  kubectl_cmd get pods -n "${namespace}" \
    -l "app.kubernetes.io/component=taskexecutor" \
    -o "jsonpath={.items[0].metadata.labels.flower\\.ai/harness-run}" \
    2>/dev/null || true
}

require_superlink_run_id() {
  local run_id
  run_id="$(get_superlink_run_id)"
  if [[ -z "${run_id}" ]]; then
    die "no SuperLink harness run was found. Run start-superlink first."
  fi
  echo "${run_id}"
}

require_existing_run_id() {
  local run_id
  run_id="$(get_existing_run_id)"
  if [[ -z "${run_id}" ]]; then
    die "no harness run label was found on SuperLink or TaskExecutor Pods."
  fi
  echo "${run_id}"
}

taskexecutor_selector() {
  local run_id="$1"
  echo "app.kubernetes.io/component=taskexecutor,flower.ai/harness-run=${run_id}"
}

render_manifest() {
  local target="$1"
  local run_id="$2"
  local output_file="$3"
  local seed_count="${4:-1}"
  local hold_seconds="${5:-0.0}"
  local probe_crash="${6:-false}"

  python - "${script_dir}" "${target}" "${run_id}" "${output_file}" \
    "${namespace}" "${cluster_name}" "${kubectl_context}" \
    "${taskexecutor_image}" "${superlink_image}" "${superexec_image}" \
    "${image_pull_policy}" "${runtime_image_pull_policy}" "${resource_pool}" \
    "${active_pod_budget}" "${capacity_poll_interval}" "${capacity_log_interval}" \
    "${timeout_seconds}" "${superlink_name}" "${superexec_name}" \
    "${seed_job_name}" "${seed_config_name}" "${executor_config_name}" \
    "${superexec_service_account}" "${rbac_name}" "${seed_count}" \
    "${hold_seconds}" "${probe_crash}" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
sys.path.insert(0, str(script_dir))

import yaml
from common import HarnessProfile
from manifests import (
    render_appio_seed_manifests,
    render_namespace_manifest,
    render_real_launch_manifests,
    render_superexec_rbac_manifests,
)


def optional_int(value: str) -> int | None:
    return int(value) if value else None


def optional_float(value: str) -> float | None:
    return float(value) if value else None


(
    _script_dir,
    target,
    run_id,
    output_file,
    namespace,
    cluster_name,
    kubectl_context,
    taskexecutor_image,
    superlink_image,
    superexec_image,
    image_pull_policy,
    runtime_image_pull_policy,
    resource_pool,
    active_pod_budget,
    capacity_poll_interval,
    capacity_log_interval,
    timeout_seconds,
    superlink_name,
    superexec_name,
    seed_job_name,
    seed_config_name,
    executor_config_name,
    superexec_service_account,
    rbac_name,
    seed_count,
    hold_seconds,
    probe_crash,
) = sys.argv[1:]

profile = HarnessProfile(
    name="generic-k3d",
    cluster_name=cluster_name,
    namespace=namespace,
    resource_pool=resource_pool,
    image=taskexecutor_image,
    image_pull_policy=image_pull_policy,
    labels={
        "flower.ai/harness": "local-k8s-launch-path",
        "flower.ai/profile": "generic-k3d",
    },
    kubectl_context=kubectl_context,
    superlink_image=superlink_image,
    superexec_image=superexec_image,
    runtime_image_pull_policy=runtime_image_pull_policy,
    timeout_seconds=int(timeout_seconds),
    superlink_name=superlink_name,
    superexec_name=superexec_name,
    seed_job_name=seed_job_name,
    seed_config_name=seed_config_name,
    executor_config_name=executor_config_name,
    superexec_service_account=superexec_service_account,
    rbac_name=rbac_name,
    seed_run_count=int(seed_count),
    probe_hold_seconds=float(hold_seconds),
    active_pod_budget=optional_int(active_pod_budget),
    capacity_poll_interval=optional_float(capacity_poll_interval),
    capacity_log_interval=optional_float(capacity_log_interval),
)

runtime = render_real_launch_manifests(profile, run_id)
if target == "superlink":
    documents = [render_namespace_manifest(profile), *runtime[:2]]
elif target == "superexec":
    documents = [
        render_namespace_manifest(profile),
        *render_superexec_rbac_manifests(profile),
        *runtime[2:],
    ]
elif target == "seed":
    documents = render_appio_seed_manifests(
        profile, run_id, probe_crash=probe_crash == "true"
    )
else:
    raise ValueError(f"unknown manifest target: {target}")

path = Path(output_file)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(yaml.safe_dump_all(documents, sort_keys=False), encoding="utf-8")
PY
}

start_superlink() {
  require_command kubectl
  require_command python
  ensure_cluster
  local run_id
  local manifest
  run_id="$(new_run_id)"
  manifest="$(manifest_path superlink)"
  render_manifest superlink "${run_id}" "${manifest}"
  maybe_import_images "${superlink_image}"
  echo "Deleting existing SuperLink Pod ${superlink_name}, if present"
  kubectl_cmd delete pod "${superlink_name}" -n "${namespace}" \
    --ignore-not-found=true --wait=true >/dev/null 2>&1 || true
  echo "Applying SuperLink resources from ${manifest}"
  kubectl_cmd apply -f "${manifest}"
  kubectl_cmd wait --for=condition=Ready "pod/${superlink_name}" \
    -n "${namespace}" "--timeout=${timeout_seconds}s"
  echo "SuperLink is ready in namespace ${namespace}; harness run id: ${run_id}"
}

start_superexec() {
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --active-pod-budget)
        require_value "$1" "${2:-}"
        active_pod_budget="$2"
        shift 2
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        die "unknown start-superexec argument: $1"
        ;;
    esac
  done

  require_command kubectl
  require_command python
  ensure_cluster
  local run_id
  local manifest
  run_id="$(require_superlink_run_id)"
  manifest="$(manifest_path superexec)"
  render_manifest superexec "${run_id}" "${manifest}"
  maybe_import_images "${superexec_image}" "${taskexecutor_image}"
  echo "Deleting existing SuperExec Pod ${superexec_name}, if present"
  kubectl_cmd delete pod "${superexec_name}" -n "${namespace}" \
    --ignore-not-found=true --wait=true >/dev/null 2>&1 || true
  echo "Applying SuperExec resources from ${manifest}"
  kubectl_cmd apply -f "${manifest}"
  kubectl_cmd wait --for=condition=Ready "pod/${superexec_name}" \
    -n "${namespace}" "--timeout=${timeout_seconds}s"
  echo "SuperExec is ready in namespace ${namespace}; harness run id: ${run_id}"
}

seed_runs() {
  local seed_count=""
  local hold_seconds="0.0"
  local probe_crash="false"
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --count)
        require_value "$1" "${2:-}"
        seed_count="$2"
        shift 2
        ;;
      --hold-seconds)
        require_value "$1" "${2:-}"
        hold_seconds="$2"
        shift 2
        ;;
      --crash)
        probe_crash="true"
        shift
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        die "unknown seed argument: $1"
        ;;
    esac
  done
  [[ -n "${seed_count}" ]] || die "seed requires --count N"
  [[ "${seed_count}" =~ ^[0-9]+$ ]] || die "--count must be a positive integer"
  [[ "${seed_count}" -ge 1 ]] || die "--count must be at least 1"

  require_command kubectl
  require_command python
  local run_id
  local manifest
  run_id="$(require_superlink_run_id)"
  manifest="$(manifest_path seed)"
  render_manifest seed "${run_id}" "${manifest}" "${seed_count}" \
    "${hold_seconds}" "${probe_crash}"
  echo "Deleting previous seed Job ${seed_job_name}, if present"
  kubectl_cmd delete job "${seed_job_name}" -n "${namespace}" \
    --ignore-not-found=true --wait=true >/dev/null 2>&1 || true
  echo "Applying seed Job from ${manifest}"
  kubectl_cmd apply -f "${manifest}"
  kubectl_cmd wait --for=condition=Complete "job/${seed_job_name}" \
    -n "${namespace}" "--timeout=${timeout_seconds}s"
  kubectl_cmd logs "job/${seed_job_name}" -n "${namespace}"
  if [[ "${probe_crash}" == "true" ]]; then
    echo "Seeded ${seed_count} intentionally crashing probe ServerApp task(s)."
  else
    echo "Seeded ${seed_count} probe ServerApp task(s) with hold ${hold_seconds}s."
  fi
}

kill_superexec() {
  require_command kubectl
  echo "Deleting Kubernetes Pod ${superexec_name}; this does not send an in-pod signal."
  kubectl_cmd delete pod "${superexec_name}" -n "${namespace}" \
    --ignore-not-found=true --wait=false
}

stop_taskexecutors() {
  local count=""
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --count)
        require_value "$1" "${2:-}"
        count="$2"
        shift 2
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        die "unknown stop-taskexecutors argument: $1"
        ;;
    esac
  done
  [[ -n "${count}" ]] || die "stop-taskexecutors requires --count N"
  [[ "${count}" =~ ^[0-9]+$ ]] || die "--count must be a positive integer"
  [[ "${count}" -ge 1 ]] || die "--count must be at least 1"

  require_command kubectl
  local run_id
  local selector
  local pods=()
  local pod
  local stopped=0
  run_id="$(require_existing_run_id)"
  selector="$(taskexecutor_selector "${run_id}")"
  while IFS= read -r pod; do
    [[ -n "${pod}" ]] && pods+=("${pod}")
  done < <(kubectl_cmd get pods -n "${namespace}" -l "${selector}" \
    --sort-by=.metadata.creationTimestamp \
    -o 'jsonpath={range .items[*]}{.metadata.name}{"\n"}{end}')
  if [[ "${#pods[@]}" -eq 0 ]]; then
    echo "No TaskExecutor Pods matched selector ${selector}"
    return
  fi
  for pod in "${pods[@]}"; do
    if [[ "${stopped}" -ge "${count}" ]]; then
      break
    fi
    echo "Deleting TaskExecutor Pod ${pod}"
    kubectl_cmd delete pod "${pod}" -n "${namespace}" --wait=false
    stopped=$((stopped + 1))
  done
  echo "Requested deletion for ${stopped} TaskExecutor Pod(s)."
}

cleanup() {
  require_command kubectl
  echo "Deleting namespace ${namespace} in context ${kubectl_context}"
  kubectl_cmd delete namespace "${namespace}" --ignore-not-found=true --wait=true
}

watch_pods() {
  require_command kubectl
  if command -v watch >/dev/null 2>&1; then
    watch -n 1 "kubectl --context ${kubectl_context} get pods -n ${namespace} -o wide --sort-by=.metadata.creationTimestamp"
  else
    while true; do
      clear
      date
      kubectl_cmd get pods -n "${namespace}" -o wide \
        --sort-by=.metadata.creationTimestamp
      sleep 1
    done
  fi
}

watch_taskexecutors() {
  require_command kubectl
  local run_id
  local selector
  run_id="$(require_existing_run_id)"
  selector="$(taskexecutor_selector "${run_id}")"
  if command -v watch >/dev/null 2>&1; then
    watch -n 1 "kubectl --context ${kubectl_context} get pods -n ${namespace} -l ${selector} -L flower.ai/resource-pool,flower.ai/superexec-task-id,flower.ai/launch-attempt --sort-by=.metadata.creationTimestamp"
  else
    while true; do
      clear
      date
      kubectl_cmd get pods -n "${namespace}" -l "${selector}" \
        -L flower.ai/resource-pool,flower.ai/superexec-task-id,flower.ai/launch-attempt \
        --sort-by=.metadata.creationTimestamp
      sleep 1
    done
  fi
}

logs_superexec() {
  require_command kubectl
  if [[ "$#" -eq 0 ]]; then
    set -- --tail=200
  fi
  kubectl_cmd logs "pod/${superexec_name}" -n "${namespace}" "$@"
}

logs_taskexecutors() {
  require_command kubectl
  local run_id
  local selector
  run_id="$(require_existing_run_id)"
  selector="$(taskexecutor_selector "${run_id}")"
  if [[ "$#" -eq 0 ]]; then
    set -- --tail=200 --prefix
  fi
  kubectl_cmd logs -n "${namespace}" -l "${selector}" "$@"
}

if [[ "$#" -lt 1 ]]; then
  usage
  exit 1
fi

command_name="$1"
shift

case "${command_name}" in
  start-superlink)
    start_superlink "$@"
    ;;
  start-superexec)
    start_superexec "$@"
    ;;
  seed)
    seed_runs "$@"
    ;;
  kill-superexec)
    kill_superexec "$@"
    ;;
  stop-taskexecutors)
    stop_taskexecutors "$@"
    ;;
  cleanup)
    cleanup "$@"
    ;;
  watch-pods)
    watch_pods "$@"
    ;;
  watch-taskexecutors)
    watch_taskexecutors "$@"
    ;;
  logs-superexec)
    logs_superexec "$@"
    ;;
  logs-taskexecutors)
    logs_taskexecutors "$@"
    ;;
  -h | --help | help)
    usage
    ;;
  *)
    die "unknown command: ${command_name}"
    ;;
esac
