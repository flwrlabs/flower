#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../model/docs
if [[ -d build ]]; then
  rm -r build
fi
make html
