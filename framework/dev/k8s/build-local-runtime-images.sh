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

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
framework_dir="$(cd "${script_dir}/../.." >/dev/null 2>&1 && pwd)"

image_tag="${IMAGE_TAG:-dev}"
base_image="${BASE_IMAGE:-flwr/base:${image_tag}}"
superlink_image="${SUPERLINK_IMAGE:-flwr/superlink:${image_tag}}"
superexec_image="${SUPEREXEC_IMAGE:-flwr/superexec:${image_tag}}"
python_image="${PYTHON_IMAGE:-python:3.11-slim-bookworm}"
pip_version="${PIP_VERSION:-26.0.1}"
setuptools_version="${SETUPTOOLS_VERSION:-82.0.0}"
kubernetes_package="${KUBERNETES_PACKAGE:-kubernetes}"
platform="${PLATFORM:-}"
verify_images=true

usage() {
  cat <<EOF
Build local Flower runtime images from the current framework checkout.

This builds:
  ${base_image}
  ${superlink_image}
  ${superexec_image}

A separate TaskExecutor image is intentionally out of scope. Use the built
SuperExec image as the TaskExecutor runtime image for the local k3d harness.

Usage:
  framework/dev/k8s/build-local-runtime-images.sh [options]

Options:
  --tag TAG                 Tag used for default image names (default: ${image_tag})
  --base-image IMAGE        Base image to build (default: ${base_image})
  --superlink-image IMAGE   SuperLink image to build (default: ${superlink_image})
  --superexec-image IMAGE   SuperExec image to build (default: ${superexec_image})
  --python-image IMAGE      Python base image (default: ${python_image})
  --pip-version VERSION     pip version installed in the image (default: ${pip_version})
  --setuptools-version VERSION
                            setuptools version installed in the image
                            (default: ${setuptools_version})
  --kubernetes-package SPEC
                            Optional Kubernetes Python package spec installed
                            for SuperExec's kubernetes executor
                            (default: ${kubernetes_package})
  --skip-kubernetes-package
                            Do not install the Kubernetes Python package
  --platform PLATFORM       Optional docker build platform, for example linux/arm64
  --skip-verify             Do not run command/import checks in built images
  -h, --help                Show this help
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
    --python-image)
      require_value "$1" "${2:-}"
      python_image="$2"
      shift 2
      ;;
    --pip-version)
      require_value "$1" "${2:-}"
      pip_version="$2"
      shift 2
      ;;
    --setuptools-version)
      require_value "$1" "${2:-}"
      setuptools_version="$2"
      shift 2
      ;;
    --kubernetes-package)
      require_value "$1" "${2:-}"
      kubernetes_package="$2"
      shift 2
      ;;
    --skip-kubernetes-package)
      kubernetes_package=""
      shift
      ;;
    --platform)
      require_value "$1" "${2:-}"
      platform="$2"
      shift 2
      ;;
    --skip-verify)
      verify_images=false
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

command -v docker >/dev/null 2>&1 || die "docker is required"
command -v uv >/dev/null 2>&1 || die "uv is required"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/flower-runtime-images.XXXXXX")"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

wheel_dir="${tmp_dir}/dist"
context_dir="${tmp_dir}/context"
mkdir -p "${wheel_dir}" "${context_dir}"

echo "Building local framework wheel..."
(cd "${framework_dir}" && uv build --wheel --out-dir "${wheel_dir}")

shopt -s nullglob
wheels=("${wheel_dir}"/*.whl)
shopt -u nullglob

if [[ "${#wheels[@]}" -ne 1 ]]; then
  die "expected exactly one wheel in ${wheel_dir}, found ${#wheels[@]}"
fi

cp "${wheels[0]}" "${context_dir}/"

cat >"${context_dir}/Dockerfile.base" <<'EOF'
ARG PYTHON_IMAGE=python:3.11-slim-bookworm
FROM ${PYTHON_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get -y --no-install-recommends install \
      ca-certificates \
      libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd \
      --no-create-home \
      --home-dir /app \
      --uid 49999 \
      app \
    && mkdir -p /app \
    && chown -R app:app /app

ARG PIP_VERSION=26.0.1
ARG SETUPTOOLS_VERSION=82.0.0
ARG KUBERNETES_PACKAGE=kubernetes
COPY *.whl /tmp/
RUN python -m pip install -U --no-cache-dir \
      pip==${PIP_VERSION} \
      setuptools==${SETUPTOOLS_VERSION} \
    && python -m pip install --no-cache-dir /tmp/*.whl \
    && if [ -n "${KUBERNETES_PACKAGE}" ]; then \
      python -m pip install --no-cache-dir "${KUBERNETES_PACKAGE}"; \
    fi \
    && rm /tmp/*.whl

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    HOME=/app

WORKDIR /app
USER app
EOF

cat >"${context_dir}/Dockerfile.superlink" <<'EOF'
ARG BASE_IMAGE=flwr/base:dev
FROM ${BASE_IMAGE}

ENTRYPOINT ["flower-superlink"]
EOF

cat >"${context_dir}/Dockerfile.superexec" <<'EOF'
ARG BASE_IMAGE=flwr/base:dev
FROM ${BASE_IMAGE}

ENTRYPOINT ["flower-superexec"]
EOF

docker_build() {
  if [[ -n "${platform}" ]]; then
    docker build --platform "${platform}" "$@"
  else
    docker build "$@"
  fi
}

echo "Building ${base_image} from ${python_image}..."
docker_build \
  --build-arg "PYTHON_IMAGE=${python_image}" \
  --build-arg "PIP_VERSION=${pip_version}" \
  --build-arg "SETUPTOOLS_VERSION=${setuptools_version}" \
  --build-arg "KUBERNETES_PACKAGE=${kubernetes_package}" \
  -f "${context_dir}/Dockerfile.base" \
  -t "${base_image}" \
  "${context_dir}"

echo "Building ${superlink_image}..."
docker_build \
  --build-arg "BASE_IMAGE=${base_image}" \
  -f "${context_dir}/Dockerfile.superlink" \
  -t "${superlink_image}" \
  "${context_dir}"

echo "Building ${superexec_image}..."
docker_build \
  --build-arg "BASE_IMAGE=${base_image}" \
  -f "${context_dir}/Dockerfile.superexec" \
  -t "${superexec_image}" \
  "${context_dir}"

if [[ "${verify_images}" == true ]]; then
  echo "Verifying ${superlink_image}..."
  docker run --rm "${superlink_image}" --help >/dev/null
  echo "Verifying ${superexec_image}..."
  docker run --rm "${superexec_image}" --help >/dev/null
  echo "Verifying TaskExecutor commands in ${superexec_image}..."
  docker run --rm --entrypoint flwr-serverapp "${superexec_image}" --help >/dev/null
  docker run --rm --entrypoint flwr-clientapp "${superexec_image}" --help >/dev/null
  if [[ -n "${kubernetes_package}" ]]; then
    echo "Verifying Kubernetes Python client in ${superexec_image}..."
    docker run --rm --entrypoint python "${superexec_image}" \
      -c "import kubernetes" >/dev/null
  fi
fi

cat <<EOF
Built local runtime images:
  ${base_image}
  ${superlink_image}
  ${superexec_image}

For the current k3d harness, use ${superexec_image} as both --superexec-image
and --image.
EOF
