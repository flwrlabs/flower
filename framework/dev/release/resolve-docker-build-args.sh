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

# Start from build args supplied by the workflow input (can be empty).
build_args="${INPUT_BUILD_ARGS:-}"

# If requested, replace the __FLWR_WHEEL__ placeholder with the local wheel name
# produced under framework/dist/.
if [[ "${BUILD_LOCAL_WHEEL:-false}" == "true" ]]; then
  # Treat an unmatched glob as empty so we can validate count explicitly.
  shopt -s nullglob
  wheels=(framework/dist/*.whl)
  shopt -u nullglob

  # Enforce a single wheel to avoid ambiguous or missing substitutions.
  if [[ "${#wheels[@]}" -ne 1 ]]; then
    echo "Expected exactly one wheel in framework/dist, found ${#wheels[@]}" >&2
    exit 1
  fi

  # Extract only the filename and substitute it into the build args.
  flwr_wheel=$(basename "${wheels[0]}")
  build_args=${build_args//__FLWR_WHEEL__/${flwr_wheel}}
fi

# Indent multiline build args for the nested docker/build-push-action input.
build_args=${build_args//$'\n'/$'\n  '}
# Emit a multiline output variable for subsequent GitHub Actions steps.
{
  echo "build-args<<EOF"
  echo "${build_args}"
  echo "EOF"
} >> "${GITHUB_OUTPUT}"
