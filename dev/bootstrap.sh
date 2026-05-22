#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
FRAMEWORK_ROOT="${REPO_ROOT}/framework"
version=${1:-3.11.14}

# Setup environment variables for development
"${SCRIPT_DIR}/setup-envs.sh"

# Remove caches
"${FRAMEWORK_ROOT}/dev/rm-caches.sh"

cd "${FRAMEWORK_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

# Use `uv` to install project dependencies from the lockfile
uv sync --python="${version}" --locked --all-extras --all-groups
