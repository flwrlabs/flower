#!/bin/bash

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

version="${1:-${GITHUB_REF_NAME#framework-}}"
if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Usage: $0 X.Y.Z" >&2
    exit 1
fi

changelog_file="docs/source/changelog/v${version}.md"
if [[ ! -f "${changelog_file}" ]]; then
    echo "Changelog not found: ${changelog_file}" >&2
    exit 1
fi

awk '
    NR == 1 && /^## v[0-9]+\.[0-9]+\.[0-9]+([[:space:]]|\()/ {next}
    {sub(/<!--.*-->/, "")}
    !printed && /^$/ {next}
    {printed=1; print}
    END {print ""}
' "${changelog_file}"
