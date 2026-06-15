#!/bin/bash

set -euo pipefail

cd framework

uv sync --locked --all-extras --all-groups --no-install-package taplo
