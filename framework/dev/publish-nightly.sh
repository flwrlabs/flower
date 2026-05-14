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

set -euo pipefail

if [[ -z "${PYPI_REPOSITORY_PASSWORD:-}" ]]; then
    echo "Missing required configuration: PYPI_REPOSITORY_PASSWORD" >&2
    exit 1
fi

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"/../

# This script will build and publish a nightly release of Flower under the condition
# that at least one commit was made in the last 24 hours.
# It will rename the package name in the pyproject.toml to from "flwr" to "flwr-nightly".
# The version name in the pyproject.toml will be appended with "-dev" and the current date.
# The result will be a release on PyPi of the package "flwr-nightly" of version e.g.
# "0.1.1.dev20200716" as seen at https://pypi.org/project/flwr-nightly/

if [[ $(git log --since="24 hours ago" --pretty=oneline) ]]; then
    sed -i -E "s/^name = \"(.+)\"/name = \"\1-nightly\"/" pyproject.toml
    sed -i -E "s/^version = \"(.+)\"/version = \"\1.dev$(date '+%Y%m%d')\"/" pyproject.toml
    python -m poetry build
    python -m poetry publish -u __token__ -p "${PYPI_REPOSITORY_PASSWORD}"
else
    echo "There were no commits in the last 24 hours."
fi
