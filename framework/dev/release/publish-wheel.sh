#!/bin/bash

# Copyright 2026 Flower Labs GmbH. All Rights Reserved.

set -euo pipefail

missing=0
for var in PYPI_REPOSITORY_USERNAME PYPI_REPOSITORY_PASSWORD; do
  if [[ -z "${!var:-}" ]]; then
    echo "Missing required configuration: ${var}" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi

if [[ -n "${PACKAGE_VERSION:-}" ]]; then
  tag_name="${PACKAGE_VERSION}"
elif [[ "${GITHUB_REF_NAME:-}" == framework-* ]]; then
  tag_name="${GITHUB_REF_NAME#framework-}"
else
  tag_name=$(cd framework && python -m poetry version --short)
fi

echo "flwr-version=${tag_name}" >> "${GITHUB_OUTPUT}"
wheel_name="flwr-${tag_name}-py3-none-any.whl"
tar_name="flwr-${tag_name}.tar.gz"
wheel_url="https://artifact.flower.ai/py/release/v${tag_name}/${wheel_name}"
tar_url="https://artifact.flower.ai/py/release/v${tag_name}/${tar_name}"

mkdir -p framework/dist
curl --fail --location --silent --show-error "${wheel_url}" --output "framework/dist/${wheel_name}"
curl --fail --location --silent --show-error "${tar_url}" --output "framework/dist/${tar_name}"

(cd framework && python -m poetry publish -u "${PYPI_REPOSITORY_USERNAME}" -p "${PYPI_REPOSITORY_PASSWORD}")
