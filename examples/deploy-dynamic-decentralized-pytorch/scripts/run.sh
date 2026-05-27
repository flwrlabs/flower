#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

flower-super-dnode \
  --execution-mode deploy \
  --config configs/deploy_dynamic.yaml \
  --nodeapps-pyproject pyproject.toml \
  --port "${1:-9100}" \
  --node-data-config-json '{"partition-id": 0, "num-partitions": 1}'
