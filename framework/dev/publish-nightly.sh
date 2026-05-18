#!/bin/bash

# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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

# Fail as soon as a command errors, an unset variable is used, or a pipeline fails.
set -euo pipefail

# The password/token is required. The username is optional because token-based
# repositories commonly use "__token__", which this script provides by default.
if [[ -z "${PYPI_REPOSITORY_PASSWORD:-}" ]]; then
    echo "Missing required configuration: PYPI_REPOSITORY_PASSWORD" >&2
    exit 1
fi

# Move from framework/dev to framework so Poetry can find pyproject.toml.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"/../

# Nightlies are only published if there was repository activity in the last 24
# hours. The CI checkout is disposable, so it is safe to rewrite pyproject.toml
# before building: "flwr" becomes "flwr-nightly" and the current date is appended
# to the version, for example "1.2.3.dev20260514".
if [[ $(git log --since="24 hours ago" --pretty=oneline) ]]; then
    sed -i -E "s/^name = \"(.+)\"/name = \"\1-nightly\"/" pyproject.toml
    sed -i -E "s/^version = \"(.+)\"/version = \"\1.dev$(date '+%Y%m%d')\"/" pyproject.toml

    # Build both publishable Python artifacts from the rewritten metadata.
    python -m poetry build

    # Store publish options in an array so usernames, passwords, and URLs are
    # passed to Poetry as separate arguments even if they contain shell metacharacters.
    publish_args=(-u "${PYPI_REPOSITORY_USERNAME:-__token__}" -p "${PYPI_REPOSITORY_PASSWORD}")
    if [[ -n "${PYPI_REPOSITORY_URL:-}" ]]; then
        # When a repository URL is configured, register it with Poetry and
        # publish there instead of using Poetry's default PyPI endpoint.
        repository_name="${PYPI_REPOSITORY_NAME:-act}"
        python -m poetry config "repositories.${repository_name}" "${PYPI_REPOSITORY_URL}"
        publish_args=(-r "${repository_name}" "${publish_args[@]}")
    fi

    # Poetry only builds when --build is passed, so this publishes the artifacts
    # from the explicit build step above instead of rebuilding them.
    python -m poetry publish --dist-dir dist "${publish_args[@]}"
else
    echo "There were no commits in the last 24 hours."
fi
