#!/bin/sh
# Build Flower framework docs for a single docs version.
# Usage: ./build-single-version-docs.sh [DOC_VERSION]
# - Uses DOC_VERSION from environment, or from first positional argument.
# - Builds English plus all languages found under `locales/`.
# - Writes output to `build/html/${DOC_VERSION}/<language>/`.
set -e

if [ -n "$1" ]; then
  DOC_VERSION="$1"
fi

if [ -z "$DOC_VERSION" ]; then
  echo "DOC_VERSION is required (e.g. main or 1.26)" >&2
  exit 1
fi

# Move to the docs directory
cd "$(git rev-parse --show-toplevel)/framework/docs"

current_version="$DOC_VERSION"
export current_version

# Clean previous output for this version only
rm -rf "build/html/${DOC_VERSION}"

# Generate autosummary sources once. Each locale uses the shared source tree,
# so generating these files during every concurrent Sphinx build would race.
rm -rf source/ref-api
rm -rf "build/autosummary/${DOC_VERSION}"
sphinx-build \
  -b dummy \
  source/ \
  "build/autosummary/${DOC_VERSION}" \
  -A lang=True \
  -D language=en

# Get a list of languages based on the folders in locales
languages="en"
for lang_dir in locales/*; do
  if [ -d "$lang_dir" ]; then
    languages="$languages $(basename "$lang_dir")"
  fi
done

# Each language has its own output and doctree directory, so the builds can run
# concurrently without sharing mutable Sphinx state.
build_language() {
  current_language="$1"
  export current_language

  echo "Building ${current_language} docs"
  FLWR_DOCS_AUTOSUMMARY_READY=1 sphinx-build \
    -b html \
    source/ \
    "build/html/${current_version}/${current_language}" \
    -A lang=True \
    -D "language=${current_language}"
}

pids=""
for current_language in $languages; do
  build_language "${current_language}" &
  pids="${pids} $!"
done

status=0
for pid in ${pids}; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
