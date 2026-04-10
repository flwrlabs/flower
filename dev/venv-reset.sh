#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
FRAMEWORK_ROOT="${REPO_ROOT}/framework"

version=${1:-3.10.19}

# Delete caches, venv, and lock file
"${FRAMEWORK_ROOT}/dev/rm-caches.sh"
"${SCRIPT_DIR}/venv-delete.sh" "$version"
[ ! -e "${FRAMEWORK_ROOT}/poetry.lock" ] || rm "${FRAMEWORK_ROOT}/poetry.lock"

# Recreate
"${SCRIPT_DIR}/venv-create.sh" "$version"
"${SCRIPT_DIR}/bootstrap.sh"
