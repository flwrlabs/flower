#!/bin/bash

# Copyright 2026 Flower Labs GmbH. All Rights Reserved.

set -euo pipefail

build_args="${INPUT_BUILD_ARGS:-}"
if [[ "${BUILD_LOCAL_WHEEL:-false}" == "true" ]]; then
  flwr_wheel=$(basename framework/dist/*.whl)
  build_args=${build_args//__FLWR_WHEEL__/${flwr_wheel}}
fi

# Indent multiline build args for the nested docker/build-push-action input.
build_args=${build_args//$'\n'/$'\n  '}
{
  echo "build-args<<EOF"
  echo "${build_args}"
  echo "EOF"
} >> "${GITHUB_OUTPUT}"
