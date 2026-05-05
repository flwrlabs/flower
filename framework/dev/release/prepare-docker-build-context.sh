#!/bin/bash

# Copyright 2026 Flower Labs GmbH. All Rights Reserved.

set -euo pipefail

if [[ "${PREPARE_FRAMEWORK:-false}" == "true" ]]; then
  echo "No Flower repository sync is required in flwrlabs/flower."
fi

if [[ "${PATCH_EE_BASE_DOCKERFILES:-false}" == "true" ]]; then
  echo "No repository-specific Dockerfile patches are required in flwrlabs/flower."
fi

if [[ "${BUILD_EE_WHEEL:-false}" == "true" ]]; then
  (cd framework && ./dev/build.sh)
fi
