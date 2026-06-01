#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

flower-super-dnode \
  --execution-mode simulation \
  --sim-config configs/simulation_static_graph_sampling.yaml \
  --network-config-mode csr \
  --enable-sampling \
  --nodeapps-pyproject pyproject.toml
