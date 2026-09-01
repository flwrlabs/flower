#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  .github/act/run-publish.sh <release|docker-main|nightly> <profile> [job|all] [-- <act args>...]

Examples:
  .github/act/run-publish.sh release testpypi-gitlab publish-wheel -- --dryrun
  .github/act/run-publish.sh docker-main testpypi-gitlab prepare-docker-build-matrix
  .github/act/run-publish.sh docker-main testpypi-gitlab all

Profiles:
  The profile argument maps to:
    .github/act/profiles/<profile>.env.local
    .github/act/profiles/<profile>.vars.local
    .github/act/profiles/<profile>.secrets.local

  Start from one of the committed *.example files in .github/act/profiles/.

  Python package publish jobs require act --dryrun by default. Set
  ACT_ALLOW_PACKAGE_PUBLISH=1 only when intentionally publishing to a
  disposable package repository.
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 64
fi

target="$1"
profile="$2"
job=""

if [[ -z "$profile" ]]; then
  echo "Profile must not be empty." >&2
  usage
  exit 64
fi

shift 2
if [[ $# -gt 0 && "${1:-}" != "--" ]]; then
  job="$1"
  shift
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

repo_root="$(git rev-parse --show-toplevel)"

export ACT_UBUNTU_22_04_IMAGE="${ACT_UBUNTU_22_04_IMAGE:-catthehacker/ubuntu:full-22.04}"
export ACT_UBUNTU_4_CORE_ARM64_IMAGE="${ACT_UBUNTU_4_CORE_ARM64_IMAGE:-$ACT_UBUNTU_22_04_IMAGE}"

has_act_dryrun_arg() {
  local arg

  for arg in "$@"; do
    if [[ "$arg" == "--dryrun" || "$arg" == "-n" ]]; then
      return 0
    fi
  done

  return 1
}

case "$target" in
  release)
    event_fixture="${ACT_EVENT_FIXTURE:-framework-release-dispatch}"
    workflow=".github/workflows/framework-release.yml"
    default_job="publish-wheel"
    export ACT_EVENT_NAME="${ACT_EVENT_NAME:-workflow_dispatch}"
    ;;
  docker-main)
    event_fixture="${ACT_EVENT_FIXTURE:-framework-docker-main-dispatch}"
    workflow=".github/workflows/framework-docker-build-main.yml"
    default_job="prepare-docker-build-matrix"
    export ACT_EVENT_NAME="${ACT_EVENT_NAME:-workflow_dispatch}"
    ;;
  nightly)
    event_fixture="${ACT_EVENT_FIXTURE:-schedule-nightly}"
    workflow=".github/workflows/framework-release-nightly.yml"
    default_job="release-nightly"
    export ACT_EVENT_NAME="${ACT_EVENT_NAME:-schedule}"
    ;;
  *)
    echo "Unknown publish target: $target" >&2
    usage
    exit 64
    ;;
esac

if [[ -z "$job" ]]; then
  job="$default_job"
fi

if [[ "$target" == "release" || "$target" == "nightly" ]]; then
  if [[ "${ACT_ALLOW_PACKAGE_PUBLISH:-0}" != "1" ]] && ! has_act_dryrun_arg "$@"; then
    echo "The '$target' target runs a Python package publish job." >&2
    echo "Pass act --dryrun, for example:" >&2
    echo "  .github/act/run-publish.sh $target $profile $job -- --dryrun" >&2
    echo "Set ACT_ALLOW_PACKAGE_PUBLISH=1 only for an intentional publish to a disposable package repository." >&2
    exit 64
  fi
fi

ACT_PROFILE="$profile" ACT_REQUIRE_PROFILE=1 \
  exec "$repo_root/.github/act/run.sh" "$event_fixture" "$workflow" "$job" "$@"
