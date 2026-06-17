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

image_tag="${IMAGE_TAG:-dev}"
base_image="${BASE_IMAGE:-flwr/base:${image_tag}}"
superlink_image="${SUPERLINK_IMAGE:-flwr/superlink:${image_tag}}"
superexec_image="${SUPEREXEC_IMAGE:-flwr/superexec:${image_tag}}"
cluster_name="${CLUSTER_NAME:-flower-local-k8s}"
namespace="${NAMESPACE:-flower-local-k8s}"
timeout_seconds="${TIMEOUT_SECONDS:-600}"
output_dir="${OUTPUT_DIR:-${TMPDIR:-/tmp}/flower-local-k8s-$(date +%Y%m%d-%H%M%S)}"
platform="${PLATFORM:-}"
python_image="${PYTHON_IMAGE:-}"
kubernetes_package="${KUBERNETES_PACKAGE:-}"
build_images=true
cleanup=true

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

for command in docker k3d kubectl uv python; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    die "${command} is required. Install dependencies, then rerun this script."
  fi
done

echo "=== local k8s launch-path test ==="
echo "Cluster: ${cluster_name}"
echo "Namespace: ${namespace}"
echo "Evidence: ${output_dir}"
echo "SuperLink image: ${superlink_image}"
echo "SuperExec/TaskExecutor image: ${superexec_image}"
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
)
verify_args=("${output_dir}")

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
