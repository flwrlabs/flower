#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

flower-super-dnode \
  --execution-mode simulation \
  --sim-config configs/simulation_dynamic_graph.yaml \
  --nodeapps-pyproject pyproject.toml
