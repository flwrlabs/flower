#!/bin/bash

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Extract the latest release notes from the changelog.
# We use the first two released headings in ref-changelog.md:
# - top heading: current release
# - second heading: previous release
versions=$(
    grep -E '^## v[0-9]+\.[0-9]+\.[0-9]+([[:space:]]|\()' docs/source/ref-changelog.md \
        | sed -E 's/^## v([0-9]+\.[0-9]+\.[0-9]+).*/\1/' \
        | head -n 2
)
new_version=$(echo "$versions" | sed -n '1p')
old_version=$(echo "$versions" | sed -n '2p')

if [[ -z "${new_version}" || -z "${old_version}" ]]; then
    echo "Could not determine latest/previous versions from ref-changelog.md." >&2
    exit 1
fi

awk '{sub(/<!--.*-->/, ""); print}' docs/source/ref-changelog.md | awk -v start="v${new_version}" -v end="v${old_version}" '
    $0 ~ "^## " start "([[:space:]]|\\()" {flag=1; next}
    $0 ~ "^## " end "([[:space:]]|\\()" {flag=0}
    flag && !printed && /^$/ {next}  # skip the first blank line
    flag && !printed {printed=1}
    flag'
