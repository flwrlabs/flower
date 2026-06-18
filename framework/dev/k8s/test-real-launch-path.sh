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
# User-facing documentation:
#   framework/dev/k8s/README.md

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
tmp_root="${TMPDIR:-/tmp}"
tmp_root="${tmp_root%/}"

image_tag="${IMAGE_TAG:-dev}"
base_image="${BASE_IMAGE:-flwr/base:${image_tag}}"
superlink_image="${SUPERLINK_IMAGE:-flwr/superlink:${image_tag}}"
superexec_image="${SUPEREXEC_IMAGE:-flwr/superexec:${image_tag}}"
cluster_name="${CLUSTER_NAME:-flower-local-k8s}"
namespace="${NAMESPACE:-flower-local-k8s}"
timeout_seconds="${TIMEOUT_SECONDS:-600}"
output_dir="${OUTPUT_DIR:-${tmp_root}/flower-local-k8s-$(date +%Y%m%d-%H%M%S)}"
platform="${PLATFORM:-}"
python_image="${PYTHON_IMAGE:-}"
kubernetes_package="${KUBERNETES_PACKAGE:-}"
build_images=true
cleanup=true
capacity_cleanup_proof=false
active_pod_budget="${ACTIVE_POD_BUDGET:-}"
seed_run_count="${SEED_RUN_COUNT:-1}"
probe_hold_seconds="${PROBE_HOLD_SECONDS:-0.0}"
tls_enabled="${APPIO_TLS:-false}"
tls_secret_name="${TLS_SECRET_NAME:-flower-local-k8s-appio-tls}"
tls_dir="${TLS_DIR:-}"
tls_pod_ca_path="/etc/flower/tls/ca.crt"
tls_ca_cert=""
tls_ca_key=""
tls_server_cert=""
tls_server_key=""
tls_server_csr=""
tls_san_config=""

usage() {
  cat <<EOF
Build local runtime images, configure k3d, run the local k8s launch-path
harness, and verify the resulting evidence.

Documentation:
  framework/dev/k8s/README.md

Usage:
  framework/dev/k8s/test-real-launch-path.sh [options]

Options:
  --output-dir DIR          Evidence output directory
                            (default: ${output_dir})
  --tag TAG                 Tag for default image names (default: ${image_tag})
  --base-image IMAGE        Base image to build (default: ${base_image})
  --superlink-image IMAGE   SuperLink image to build and run
                            (default: ${superlink_image})
  --superexec-image IMAGE   SuperExec image to build and run; also used as the
                            TaskExecutor runtime image (default: ${superexec_image})
  --cluster-name NAME       k3d cluster name (default: ${cluster_name})
  --namespace NAME          Kubernetes namespace (default: ${namespace})
  --timeout-seconds SECS    Harness wait timeout (default: ${timeout_seconds})
  --capacity-cleanup-proof  Run the capacity and cleanup proof
                            instead of the one-task launch-path proof
  --active-pod-budget COUNT Kubernetes executor active Pod budget for
                            --capacity-cleanup-proof (default: 1)
  --seed-run-count COUNT    Deterministic ServerApp runs to seed
                            (default: ${seed_run_count})
  --probe-hold-seconds SECS Seconds each probe ServerApp should stay active
                            (default: ${probe_hold_seconds})
  --demo                    Demo preset: --capacity-cleanup-proof,
                            --active-pod-budget 4, --seed-run-count 8,
                            --probe-hold-seconds 45, and --skip-cleanup
  --tls                     Enable local server-auth TLS for SuperLink,
                            SuperExec, seed Job, and TaskExecutor AppIo
  --tls-secret-name NAME    Kubernetes Secret name for local TLS material
                            (default: ${tls_secret_name})
  --tls-dir DIR             Directory for generated local TLS material
                            (default: <output-dir>/tls)
  --platform PLATFORM       Optional docker build platform, for example linux/arm64
  --python-image IMAGE      Optional Python base image passed to the image builder
  --kubernetes-package SPEC Optional Kubernetes package spec passed to the image
                            builder (default builder value: kubernetes)
  --skip-build              Reuse existing local images
  --skip-cleanup            Leave namespace resources in place for inspection
  -h, --help                Show this help

Prerequisites:
  docker, k3d, kubectl, uv, and python must be installed and on PATH.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
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
  kubectl --context "k3d-${cluster_name}" "$@"
}

prepare_tls_paths() {
  if [[ -z "${tls_dir}" ]]; then
    tls_dir="${output_dir}/tls"
  fi
  tls_ca_cert="${tls_dir}/ca.crt"
  tls_ca_key="${tls_dir}/ca.key"
  tls_server_cert="${tls_dir}/tls.crt"
  tls_server_key="${tls_dir}/tls.key"
  tls_server_csr="${tls_dir}/tls.csr"
  tls_san_config="${tls_dir}/openssl-san.cnf"
}

generate_tls_material() {
  mkdir -p "${tls_dir}"
  if [[ -f "${tls_ca_cert}" && -f "${tls_server_cert}" && -f "${tls_server_key}" ]]; then
    echo "Reusing AppIo TLS material in ${tls_dir}"
    return
  fi

  rm -f "${tls_ca_cert}" "${tls_ca_key}" "${tls_server_cert}" \
    "${tls_server_key}" "${tls_server_csr}" "${tls_san_config}" \
    "${tls_dir}/ca.srl"

  cat >"${tls_san_config}" <<EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = ${cluster_name}

[v3_req]
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = flower-superlink
DNS.2 = flower-superlink.${namespace}
DNS.3 = flower-superlink.${namespace}.svc
DNS.4 = flower-superlink.${namespace}.svc.cluster.local
DNS.5 = localhost
IP.1 = 127.0.0.1
EOF

  echo "Generating local AppIo TLS material in ${tls_dir}"
  openssl genrsa -out "${tls_ca_key}" 2048 >/dev/null 2>&1
  openssl req -x509 -new -nodes -key "${tls_ca_key}" -sha256 -days 7 \
    -out "${tls_ca_cert}" -subj "/CN=flower-local-k8s-ca" >/dev/null 2>&1
  openssl genrsa -out "${tls_server_key}" 2048 >/dev/null 2>&1
  openssl req -new -key "${tls_server_key}" -out "${tls_server_csr}" \
    -subj "/CN=flower-superlink" -config "${tls_san_config}" >/dev/null 2>&1
  openssl x509 -req -in "${tls_server_csr}" -CA "${tls_ca_cert}" \
    -CAkey "${tls_ca_key}" -CAcreateserial -out "${tls_server_cert}" \
    -days 7 -sha256 -extensions v3_req -extfile "${tls_san_config}" \
    >/dev/null 2>&1
}

ensure_tls_secret() {
  if ! k3d cluster list "${cluster_name}" >/dev/null 2>&1; then
    echo "Creating k3d cluster ${cluster_name} for TLS Secret setup"
    k3d cluster create "${cluster_name}" --wait
  fi
  generate_tls_material
  kubectl_cmd create namespace "${namespace}" --dry-run=client -o yaml \
    | kubectl_cmd apply -f -
  kubectl_cmd create secret generic "${tls_secret_name}" \
    -n "${namespace}" \
    --from-file=ca.crt="${tls_ca_cert}" \
    --from-file=tls.crt="${tls_server_cert}" \
    --from-file=tls.key="${tls_server_key}" \
    --dry-run=client -o yaml \
    | kubectl_cmd apply -f -
  echo "AppIo TLS Secret ${tls_secret_name} is ready in namespace ${namespace}"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --output-dir)
      require_value "$1" "${2:-}"
      output_dir="$2"
      shift 2
      ;;
    --tag)
      require_value "$1" "${2:-}"
      image_tag="$2"
      base_image="flwr/base:${image_tag}"
      superlink_image="flwr/superlink:${image_tag}"
      superexec_image="flwr/superexec:${image_tag}"
      shift 2
      ;;
    --base-image)
      require_value "$1" "${2:-}"
      base_image="$2"
      shift 2
      ;;
    --superlink-image)
      require_value "$1" "${2:-}"
      superlink_image="$2"
      shift 2
      ;;
    --superexec-image)
      require_value "$1" "${2:-}"
      superexec_image="$2"
      shift 2
      ;;
    --cluster-name)
      require_value "$1" "${2:-}"
      cluster_name="$2"
      shift 2
      ;;
    --namespace)
      require_value "$1" "${2:-}"
      namespace="$2"
      shift 2
      ;;
    --timeout-seconds)
      require_value "$1" "${2:-}"
      timeout_seconds="$2"
      shift 2
      ;;
    --capacity-cleanup-proof)
      capacity_cleanup_proof=true
      shift
      ;;
    --active-pod-budget)
      require_value "$1" "${2:-}"
      active_pod_budget="$2"
      shift 2
      ;;
    --seed-run-count)
      require_value "$1" "${2:-}"
      seed_run_count="$2"
      shift 2
      ;;
    --probe-hold-seconds)
      require_value "$1" "${2:-}"
      probe_hold_seconds="$2"
      shift 2
      ;;
    --demo)
      capacity_cleanup_proof=true
      active_pod_budget="4"
      seed_run_count="8"
      probe_hold_seconds="45"
      cleanup=false
      shift
      ;;
    --tls)
      tls_enabled=true
      shift
      ;;
    --tls-secret-name)
      require_value "$1" "${2:-}"
      tls_secret_name="$2"
      shift 2
      ;;
    --tls-dir)
      require_value "$1" "${2:-}"
      tls_dir="$2"
      shift 2
      ;;
    --platform)
      require_value "$1" "${2:-}"
      platform="$2"
      shift 2
      ;;
    --python-image)
      require_value "$1" "${2:-}"
      python_image="$2"
      shift 2
      ;;
    --kubernetes-package)
      require_value "$1" "${2:-}"
      kubernetes_package="$2"
      shift 2
      ;;
    --skip-build)
      build_images=false
      shift
      ;;
    --skip-cleanup)
      cleanup=false
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ "${capacity_cleanup_proof}" == true ]]; then
  if [[ -z "${active_pod_budget}" ]]; then
    active_pod_budget="1"
  fi
  if [[ "${seed_run_count}" == "1" ]]; then
    seed_run_count="2"
  fi
  if [[ "${probe_hold_seconds}" == "0.0" ]]; then
    probe_hold_seconds="5.0"
  fi
fi

if is_true "${tls_enabled}"; then
  prepare_tls_paths
fi

for command in docker k3d kubectl uv python; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    die "${command} is required. Install dependencies, then rerun this script."
  fi
done
if is_true "${tls_enabled}" && ! command -v openssl >/dev/null 2>&1; then
  die "openssl is required for --tls. Install dependencies, then rerun this script."
fi

echo "=== local k8s launch-path test ==="
echo "Cluster: ${cluster_name}"
echo "Namespace: ${namespace}"
echo "Evidence: ${output_dir}"
echo "SuperLink image: ${superlink_image}"
echo "SuperExec/TaskExecutor image: ${superexec_image}"
echo "Capacity cleanup proof: ${capacity_cleanup_proof}"
echo "AppIo TLS: ${tls_enabled}"
if is_true "${tls_enabled}"; then
  echo "TLS Secret: ${tls_secret_name}"
  echo "TLS material: ${tls_dir}"
fi
if [[ "${capacity_cleanup_proof}" == true ]]; then
  echo "Active Pod budget: ${active_pod_budget}"
  echo "Seed run count: ${seed_run_count}"
  echo "Probe hold seconds: ${probe_hold_seconds}"
fi
echo

if [[ "${build_images}" == true ]]; then
  build_args=(
    --base-image "${base_image}"
    --superlink-image "${superlink_image}"
    --superexec-image "${superexec_image}"
  )
  if [[ -n "${platform}" ]]; then
    build_args+=(--platform "${platform}")
  fi
  if [[ -n "${python_image}" ]]; then
    build_args+=(--python-image "${python_image}")
  fi
  if [[ -n "${kubernetes_package}" ]]; then
    build_args+=(--kubernetes-package "${kubernetes_package}")
  fi
  echo "=== Building local runtime images ==="
  "${script_dir}/build-local-runtime-images.sh" "${build_args[@]}"
  echo
else
  echo "=== Skipping image build; harness image preflight will inspect local images ==="
  echo
fi

if is_true "${tls_enabled}"; then
  echo "=== Preparing local AppIo TLS material ==="
  ensure_tls_secret
  echo
fi

harness_args=(
  --mode local-k8s-launch-path
  --output-dir "${output_dir}"
  --execute
  --create-cluster
  --apply-manifests
  --import-images
  --cluster-name "${cluster_name}"
  --namespace "${namespace}"
  --image "${superexec_image}"
  --superlink-image "${superlink_image}"
  --superexec-image "${superexec_image}"
  --timeout-seconds "${timeout_seconds}"
  --seed-run-count "${seed_run_count}"
  --probe-hold-seconds "${probe_hold_seconds}"
)
verify_args=("${output_dir}")

if is_true "${tls_enabled}"; then
  harness_args+=(--appio-root-certificates-path "${tls_pod_ca_path}")
  harness_args+=(--appio-root-certificates-local-path "${tls_ca_cert}")
  harness_args+=(--tls-secret-name "${tls_secret_name}")
  verify_args+=(--require-tls)
fi

if [[ "${capacity_cleanup_proof}" == true ]]; then
  harness_args[1]="capacity-cleanup-proof"
  harness_args+=(--active-pod-budget "${active_pod_budget}")
  harness_args+=(--capacity-poll-interval "1.0")
  harness_args+=(--capacity-log-interval "1.0")
  verify_args+=(--expected-result "local-k8s-capacity-cleanup-proof")
  verify_args+=(--expected-active-pod-budget "${active_pod_budget}")
  verify_args+=(--expected-seed-run-count "${seed_run_count}")
fi

if [[ "${cleanup}" == true ]]; then
  harness_args+=(--cleanup)
else
  verify_args+=(--no-require-cleanup)
fi

echo "=== Running local k8s launch-path harness ==="
set +e
python "${script_dir}/harness.py" "${harness_args[@]}"
harness_status="$?"
set -e
echo

echo "=== Verifying local k8s launch-path evidence ==="
set +e
python "${script_dir}/verify_evidence.py" "${verify_args[@]}"
verify_status="$?"
set -e

if [[ "${harness_status}" -ne 0 || "${verify_status}" -ne 0 ]]; then
  echo
  echo "local k8s launch-path test failed. Evidence is available at ${output_dir}" >&2
  exit 1
fi

echo
echo "local k8s launch-path test passed. Evidence is available at ${output_dir}"
