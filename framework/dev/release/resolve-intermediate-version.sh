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

pipeline_id="${PIPELINE_ID:-${GITHUB_RUN_ID:-}}"
if [[ -z "${pipeline_id}" ]]; then
  echo "Missing required configuration: PIPELINE_ID or GITHUB_RUN_ID" >&2
  exit 1
fi

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"/../../

python_bin="${PYTHON_BIN:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="python3"
fi

base_version="$("${python_bin}" -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
short_sha="${GITHUB_SHA:-$(git rev-parse HEAD)}"
short_sha="${short_sha:0:7}"

package_version="${base_version}.dev${pipeline_id}+g${short_sha}"
docker_image_tag="${DOCKER_IMAGE_TAG:-${base_version}-main.${pipeline_id}.g${short_sha}}"

{
  echo "flwr-version=${package_version}"
  echo "docker-image-tag=${docker_image_tag}"
} >> "${GITHUB_OUTPUT}"
