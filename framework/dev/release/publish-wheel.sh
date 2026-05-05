#!/bin/bash

# Copyright 2026 Flower Labs GmbH. All Rights Reserved.

set -euo pipefail

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
mkdir -p framework/dist
curl "https://artifact.flower.ai/py/release/v${tag_name}/${wheel_name}" --output "framework/dist/${wheel_name}"
curl "https://artifact.flower.ai/py/release/v${tag_name}/${tar_name}" --output "framework/dist/${tar_name}"
(cd framework && python -m poetry publish -u __token__ -p "${PYPI_REPOSITORY_PASSWORD}")
