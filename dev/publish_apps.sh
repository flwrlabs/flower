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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
APPS_DIR="${ROOT_DIR}/hub/apps"

usage() {
    cat <<'EOF'
Usage:
  publish_apps.sh [--apps "app1,app2"] [--apps /path/to/apps.txt]

Description:
  Make sure that you have run `flwr login` before executing this script.
  Run `flwr app publish <app-path>` for selected apps.

Options:
  --apps VALUE   Either:
                 - a comma-separated list of app names
                 - a file path where each line is an app name
  -h, --help     Show this help message

Examples:
  publish_apps.sh
  publish_apps.sh --apps "quickstart-catboost,quickstart-xgboost"
  publish_apps.sh --apps hub/dev/apps.txt
EOF
}

APPS_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --apps)
            if [[ $# -lt 2 ]]; then
                echo "Error: --apps requires a value" >&2
                exit 1
            fi
            APPS_ARG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "${APPS_DIR}" ]]; then
    echo "Error: apps directory does not exist: ${APPS_DIR}" >&2
    exit 1
fi

selected_apps=()

load_apps_from_arg() {
    local value="$1"

    if [[ -z "${value}" ]]; then
        return
    fi

    if [[ -f "${value}" ]]; then
        while IFS= read -r line || [[ -n "${line:-}" ]]; do
            # Remove UTF-8 BOM on first line if present
            line="${line#$'\ufeff'}"
            # Remove Windows CR if present
            line="${line%$'\r'}"
            # Trim leading/trailing whitespace
            line="${line#"${line%%[![:space:]]*}"}"
            line="${line%"${line##*[![:space:]]}"}"

            [[ -z "${line}" ]] && continue
            [[ "${line}" == \#* ]] && continue

            selected_apps+=("${line}")
        done < "${value}"
    else
        IFS=',' read -r -a raw_apps <<< "${value}"
        for app in "${raw_apps[@]}"; do
            # Trim leading/trailing whitespace
            app="${app#"${app%%[![:space:]]*}"}"
            app="${app%"${app##*[![:space:]]}"}"
            [[ -z "${app}" ]] && continue
            selected_apps+=("${app}")
        done
    fi
}

load_apps_from_arg "${APPS_ARG}"

publish_count=0
skip_count=0
fail_count=0

publish_app() {
    local app_name="$1"
    local app_path="${APPS_DIR}/${app_name}"

    if [[ ! -d "${app_path}" ]]; then
        echo "SKIP  ${app_name}: app directory does not exist"
        ((skip_count+=1))
        return
    fi

    echo "RUN   flwr app publish ${app_path}"
    if flwr app publish "${app_path}"; then
        echo "OK    ${app_name}"
        ((publish_count+=1))
    else
        echo "ERROR ${app_name}: publish failed" >&2
        ((fail_count+=1))
    fi
}

if [[ ${#selected_apps[@]} -eq 0 ]]; then
    while IFS= read -r app_dir; do
        publish_app "$(basename "${app_dir}")"
    done < <(find "${APPS_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)
else
    for app_name in "${selected_apps[@]}"; do
        publish_app "${app_name}"
    done
fi

echo
echo "Done. published=${publish_count}, skipped=${skip_count}, failed=${fail_count}"

if [[ ${fail_count} -gt 0 ]]; then
    exit 1
fi
