#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"

event_fixture="${1:-push-main}"
job="changes"

if [[ $# -ge 1 ]]; then
  shift
fi

if [[ $# -gt 0 && "${1:-}" != "--" ]]; then
  job="$1"
  shift
fi

exec "$repo_root/.github/act/run.sh" \
  "$event_fixture" \
  ".github/workflows/framework-test.yml" \
  "$job" \
  "$@"
