#!/bin/bash
# Test benchmarks and examples
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

echo "=== test.sh ==="

echo "- Start Python checks"

echo "- isort: start"
pids=()
for dir in ../hub/apps/*/; do
    src_args=$(find "$dir" -maxdepth 1 -type d | sed 's/^/--src /' | tr '\n' ' ')
    python -m isort --check-only $dir $src_args --settings-path . &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        exit 1   # Fail CI if any `isort` job fails
    fi
done
python -m isort --check-only ../benchmarks
echo "- isort: done"

echo "- black: start"
python -m black --check ../benchmarks ../hub
echo "- black: done"

echo "- All Python checks passed"

echo "- Start Markdown checks"

echo "- mdformat: start"
python -m mdformat --check --number ../hub
echo "- mdformat: done"

echo "- All Markdown checks passed"

echo "- Start TOML checks"

echo "- taplo: start"
taplo fmt --check ../benchmarks ../hub
echo "- taplo: done"

echo "- All TOML checks passed"
