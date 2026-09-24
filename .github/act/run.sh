#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  .github/act/run.sh <event-fixture|event-json-path> <workflow-path> [job|all] [-- <act args>...]

Examples:
  .github/act/run.sh push-main .github/workflows/framework-test.yml changes
  ACT_MATRIX=python:3.10 .github/act/run.sh push-main .github/workflows/framework-test.yml test_core
  .github/act/run.sh pull-request-main .github/workflows/repo-check-pr-title.yml all

Optional local files:
  .github/act/env.local      passed with --env-file
  .github/act/vars.local     passed with --var-file
  .github/act/secrets.local  passed with --secret-file

Set ACT_PROFILE=<name> to use profile-specific files instead:
  .github/act/profiles/<name>.env.local
  .github/act/profiles/<name>.vars.local
  .github/act/profiles/<name>.secrets.local
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 64
fi

event_fixture="$1"
workflow_path="$2"
job=""

shift 2
if [[ $# -gt 0 && "${1:-}" != "--" ]]; then
  job="$1"
  shift
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

repo_root="$(git rev-parse --show-toplevel)"
act_dir="$repo_root/.github/act"

event_path="$event_fixture"
if [[ "$event_path" != /* ]]; then
  if [[ -f "$act_dir/events/${event_path}.json" ]]; then
    event_path="$act_dir/events/${event_path}.json"
  else
    event_path="$repo_root/$event_path"
  fi
fi

if [[ ! -f "$event_path" ]]; then
  echo "Event fixture not found: $event_fixture" >&2
  exit 66
fi

if [[ "$workflow_path" != /* ]]; then
  workflow_path="$repo_root/$workflow_path"
fi

if [[ ! -f "$workflow_path" ]]; then
  echo "Workflow not found: $workflow_path" >&2
  exit 66
fi

event_name="${ACT_EVENT_NAME:-}"
if [[ -z "$event_name" ]]; then
  case "$(basename "$event_path" .json)" in
    push*) event_name="push" ;;
    pull-request* | pr*) event_name="pull_request" ;;
    workflow-dispatch* | *dispatch*) event_name="workflow_dispatch" ;;
    schedule*) event_name="schedule" ;;
    *) event_name="push" ;;
  esac
fi

args=(
  "--container-architecture" "${ACT_CONTAINER_ARCHITECTURE:-linux/amd64}"
  "-P" "ubuntu-22.04=${ACT_UBUNTU_22_04_IMAGE:-catthehacker/ubuntu:act-22.04}"
  "-P" "ubuntu-4-core-arm64=${ACT_UBUNTU_4_CORE_ARM64_IMAGE:-${ACT_UBUNTU_22_04_IMAGE:-catthehacker/ubuntu:act-22.04}}"
  "-W" "$workflow_path"
  "-e" "$event_path"
  "--rm"
)

if [[ -n "$job" && "$job" != "all" ]]; then
  args+=("-j" "$job")
fi

profile="${ACT_PROFILE:-}"
if [[ -n "$profile" ]]; then
  env_file="${ACT_ENV_FILE:-$act_dir/profiles/${profile}.env.local}"
  vars_file="${ACT_VARS_FILE:-$act_dir/profiles/${profile}.vars.local}"
  secrets_file="${ACT_SECRETS_FILE:-$act_dir/profiles/${profile}.secrets.local}"
else
  env_file="${ACT_ENV_FILE:-$act_dir/env.local}"
  vars_file="${ACT_VARS_FILE:-$act_dir/vars.local}"
  secrets_file="${ACT_SECRETS_FILE:-$act_dir/secrets.local}"
fi

loaded_profile_files=0

if [[ -f "$env_file" ]]; then
  args+=("--env-file" "$env_file")
  loaded_profile_files=$((loaded_profile_files + 1))
elif [[ -n "${ACT_ENV_FILE:-}" ]]; then
  echo "ACT_ENV_FILE does not exist: $env_file" >&2
  exit 66
fi

if [[ -f "$vars_file" ]]; then
  args+=("--var-file" "$vars_file")
  loaded_profile_files=$((loaded_profile_files + 1))
elif [[ -n "${ACT_VARS_FILE:-}" ]]; then
  echo "ACT_VARS_FILE does not exist: $vars_file" >&2
  exit 66
fi

if [[ -f "$secrets_file" ]]; then
  args+=("--secret-file" "$secrets_file")
  loaded_profile_files=$((loaded_profile_files + 1))
elif [[ -n "${ACT_SECRETS_FILE:-}" ]]; then
  echo "ACT_SECRETS_FILE does not exist: $secrets_file" >&2
  exit 66
fi

if [[ -n "$profile" && "${ACT_REQUIRE_PROFILE:-0}" == "1" && "$loaded_profile_files" -eq 0 ]]; then
  echo "ACT_PROFILE was set to '$profile', but no matching profile files were found in $act_dir/profiles." >&2
  echo "Create one or more of: ${profile}.env.local, ${profile}.vars.local, ${profile}.secrets.local" >&2
  exit 66
fi

if [[ -n "${ACT_MATRIX:-}" ]]; then
  read -r -a matrix_entries <<< "$ACT_MATRIX"
  for matrix_entry in "${matrix_entries[@]}"; do
    args+=("--matrix" "$matrix_entry")
  done
fi

cd "$repo_root"
exec act "$event_name" "${args[@]}" "$@"
