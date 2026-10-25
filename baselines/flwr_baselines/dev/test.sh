#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test.sh ==="

python -m isort --check-only .                      && echo "- isort:         done" &&
python -m black --check .                           && echo "- black:         done" &&
python -m mypy --explicit-package-bases flwr_baselines                       && echo "- mypy:          done" &&
python -m pylint flwr_baselines                     && echo "- pylint:        done" &&
python -m pytest --durations=0 -v flwr_baselines    && echo "- pytest:        done" &&
echo "- All Python checks passed"
