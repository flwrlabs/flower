#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
FRAMEWORK_ROOT="${REPO_ROOT}/framework"

# Setup environment variables for development
"${SCRIPT_DIR}/setup-envs.sh"

# Remove caches
"${FRAMEWORK_ROOT}/dev/rm-caches.sh"

cd "${FRAMEWORK_ROOT}"

# Upgrade/install specific versions of `pip`, `setuptools`, and `poetry`
python -m pip install -U pip==26.0.1
python -m pip install -U setuptools==82.0.0
python -m pip install -U poetry==2.3.2

# Use `poetry` to install project dependencies
python -m poetry install --all-extras
