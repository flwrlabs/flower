#!/bin/bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"

cd ..
cd dev && pwd && uv lock && cd ..
cd datasets && uv lock && cd ..
cd framework && uv lock && cd ..
