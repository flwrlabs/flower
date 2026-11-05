#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"

"$repo_root/.github/act/run-framework.sh" push-main changes

ACT_MATRIX="${ACT_MATRIX:-python:3.10}" \
  "$repo_root/.github/act/run-framework.sh" push-main test_core
