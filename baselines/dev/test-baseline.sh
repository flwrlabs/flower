#!/bin/bash

set -e
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <baseline-name=directory-of-the-baseline>"
    exit 1
fi

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../$1

echo "=== test.sh ==="

echo "- Start Python checks"

echo "- isort: start"
python -m isort --check-only .
echo "- isort: done"

echo "- black: start"
python -m black --check .
echo "- black: done"

echo "- ruff: start"
python -m ruff check .
echo "- ruff: done"

echo "- mypy: start"
python -m mypy .
echo "- mypy: done"

echo "- pylint: start"
python -m pylint ./$1
echo "- pylint: done"

echo "- pytest: start"
python -m pytest . || ([ $? -eq 5 ] || [ $? -eq 0 ])
echo "- pytest: done"

echo "- All Python checks passed"
