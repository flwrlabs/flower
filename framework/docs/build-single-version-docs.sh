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
rm -rf "build/autosummary/${DOC_VERSION}"
rm -rf "build/doctrees/${DOC_VERSION}"
rm -rf "build/source/${DOC_VERSION}"

# Generate docs from an isolated source tree for each locale. Autosummary
# templates translate headings such as "Methods" and "Attributes", so one
# English pre-build cannot be shared by localized builds.
rm -rf source/ref-api
mkdir -p "build/source/${DOC_VERSION}"
ln -s "$(pwd)/locales" "build/source/${DOC_VERSION}/locales"

# Get a list of languages based on the folders in locales
languages="en"
for lang_dir in locales/*; do
  if [ -d "$lang_dir" ]; then
    languages="$languages $(basename "$lang_dir")"
  fi
done

for current_language in $languages; do
  source_tree="build/source/${current_version}/${current_language}"
  rsync -a --delete --exclude ref-api/ source/ "${source_tree}/"
done

# Each language has its own source, output, and doctree directory, so
# autosummary generation and HTML builds can run concurrently without sharing
# mutable Sphinx state.
build_language() {
  current_language="$1"
  export current_language
  source_tree="build/source/${current_version}/${current_language}"
  doctree_dir="build/doctrees/${current_version}/${current_language}"

  echo "Building ${current_language} docs"
  sphinx-build \
    -c source \
    -b html \
    -d "${doctree_dir}" \
    "${source_tree}/" \
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
