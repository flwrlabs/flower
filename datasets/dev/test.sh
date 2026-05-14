#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test.sh ==="

echo "- Start Python checks"

echo "- isort: start"
python -m isort --check-only flwr_datasets/
echo "- isort: done"

echo "- black: start"
python -m black --check flwr_datasets/
echo "- black: done"

echo "- init_py_check: start"
python -m devtool.init_py_check flwr_datasets/
echo "- init_py_check: done"

echo "- copyright: start"
python -m devtool.check_copyright flwr_datasets/
echo "- copyright: done"

echo "- ruff: start"
python -m ruff check flwr_datasets/
echo "- ruff: done"

echo "- mypy: start"
python -m mypy flwr_datasets/
echo "- mypy: done"

echo "- pylint: start"
python -m pylint flwr_datasets/
echo "- pylint: done"

echo "- taplo: start"
taplo fmt --check
echo "- taplo: done"

echo "- pytest: start"
python -m pytest flwr_datasets/
echo "- pytest: done"

echo "- All Python checks passed"
