#!/bin/bash

set -euo pipefail

cd framework

files=(
  "pyproject.toml"
  "devtool/pyproject.toml"
)

comment_taplo() {
  sed -i "s/^\(\s*taplo\s*=.*\)$/#\1/" "$1"
}

uncomment_taplo() {
  sed -i "s/^#\(\s*taplo\s*=.*\)/\1/" "$1"
}

# Comment out taplo from pyproject.toml and devtool/pyproject.toml.
# This prevents version conflicts with taplo built from maturin and
# is only required for devcontainer builds.
for f in "${files[@]}"; do
  comment_taplo "$f"
done

sudo poetry install --all-extras

# Restore taplo lines in "files"
for f in "${files[@]}"; do
  uncomment_taplo "$f"
done
