#!/bin/bash
# Build Flower framework docs.
# Usage: ./build-docs.sh [full [DOC_VERSION]]
# - No args: build default English docs via `make html` in `build/html/`.
# - `full`: build all available languages for one docs version.
# - Optional second arg is passed as DOC_VERSION to
#   `framework/docs/build-single-version-docs.sh`; if omitted, DOC_VERSION
#   must be set in the environment for that script to succeed.
set -e

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../
ROOT=$(pwd)
cd "$ROOT/docs"

if [ "${1:-}" = "full" ]; then
    ./build-single-version-docs.sh "${2:-}"
else
    make html
fi
