#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# pip 26.2.2 requires Python 3.10 or newer.
python -c 'import sys; sys.exit("Bootstrap requires Python 3.10 or newer.") if sys.version_info < (3, 10) else None'

# Remove caches
./dev/rm-caches.sh

# Upgrade/install specific versions of `pip`, `setuptools`, and `poetry`
python -m pip install -U --no-cache-dir 'pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70'
python -m pip install -U setuptools==80.9.0
python -m pip install -U poetry==2.3.1

# Use `poetry` to install project dependencies
python -m poetry install
