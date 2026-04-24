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

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Purpose of this script is to evaluate if the user changed the schema
# definitions but did not generate and commit a corresponding migration.

TEMP_DIR=""
cleanup() {
    if [ -n "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}
trap cleanup EXIT

TEMP_DIR=$(mktemp -d)
TEMP_CONFIG="$TEMP_DIR/alembic.ini"
TEMP_DB="$TEMP_DIR/state.db"
FRAMEWORK_DIR=$(pwd)

# Use a temporary Alembic config so this check can run against an isolated
# database while keeping the copied config's path placeholders anchored here.
awk -v db_url="sqlite:///$TEMP_DB" -v framework_dir="$FRAMEWORK_DIR" '
    {
        gsub(/%\(here\)s/, framework_dir)
    }
    /^sqlalchemy\.url = / {
        print "sqlalchemy.url = " db_url
        next
    }
    { print }
' alembic.ini > "$TEMP_CONFIG"

# Upgrade a temporary database to the latest migration
alembic -c "$TEMP_CONFIG" upgrade heads

# Check if schema matches the current models (exit code non-zero if not)
alembic -c "$TEMP_CONFIG" check
