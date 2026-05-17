#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

version=${1:-3.11.14}

# Check if the directory for the Python version does not exist and if so, 
# install the right Python version through pyenv
[[ ! -d $PYENV_ROOT/versions/$version ]] && pyenv install $version

pyenv virtualenv $version flower-$version
echo flower-$version > .python-version
