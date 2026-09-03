#!/bin/bash
# Test benchmarks and examples
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

echo "=== test.sh ==="

echo "- Start Python checks"

echo "- isort: start"
pids=()
for dir in ../examples/*/; do
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
python -m black --check ../benchmarks ../examples
echo "- black: done"

echo "- All Python checks passed"

echo "- Start Markdown checks"

echo "- mdformat: start"
python -m mdformat --check --number ../examples
echo "- mdformat: done"

echo "- All Markdown checks passed"

echo "- Start TOML checks"

echo "- taplo: start"
taplo fmt --check ../benchmarks ../examples
echo "- taplo: done"

echo "- All TOML checks passed"

if [ "${RUN_SPLIT_LEARNING_TESTS}" = "1" ]; then
    echo "- split_learning example: installing test dependencies"
    pip install -e ../examples/split_learning[dev]
    echo "- split_learning example: running pytest"
    python -m pytest ../examples/split_learning/tests
    echo "- split_learning example tests passed"
fi
