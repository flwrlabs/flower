#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NODE_NAME="${1:-node_1}"
PORT="${2:-9200}"

flower-super-dnode \
  --execution-mode deploy \
  --config configs/deploy_static.yaml \
  --nodeapps-pyproject pyproject.toml \
  --node-name "${NODE_NAME}" \
  --port "${PORT}" \
  --node-data-config-json '{"partition-id": 0, "num-partitions": 4}'
