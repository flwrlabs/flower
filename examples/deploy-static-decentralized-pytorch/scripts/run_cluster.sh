#!/usr/bin/env bash
# Launch N nodes with static ring topology on localhost.
#
# Usage:
#   ./scripts/run_cluster.sh                  # 4 nodes starting at port 9200
#   ./scripts/run_cluster.sh 6                # 6 nodes starting at port 9200
#   ./scripts/run_cluster.sh 8 9300           # 8 nodes starting at port 9300
#
# What this does:
#   1. (Re)generates the static ring topology YAML for N nodes.
#   2. Starts nodes node_1..node_N each on BASE_PORT..BASE_PORT+N-1.
#   All processes run in the background.
#   Ctrl-C kills the whole cluster.

set -euo pipefail

cd "$(dirname "$0")/.."

N="${1:-4}"
BASE_PORT="${2:-9200}"
TOPO_FILE="configs/generated_static_topology.yaml"
LOG_DIR="$(mktemp -d /tmp/flwr-static-cluster.XXXXX)"

cleanup() {
    echo ""
    echo "Stopping cluster (PIDs: ${PIDS[*]:-})"
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "Cluster stopped. Logs in: ${LOG_DIR}"
}
trap cleanup INT TERM EXIT

declare -a PIDS=()

# ── Step 1: generate the topology ────────────────────────────────────────────
echo "Generating static ring topology for ${N} nodes → ${TOPO_FILE}"
python3 - <<PYEOF
from flwr.decentralized.common.graph.api import generate_deploy_topology_yaml
generate_deploy_topology_yaml(
    node_count=${N},
    kind="ring",
    output_path="${TOPO_FILE}",
)
print("  Topology written to ${TOPO_FILE}")
PYEOF

echo ""
echo "Starting static cluster: ${N} nodes, base port ${BASE_PORT}"
echo "Logs will be written to: ${LOG_DIR}"
echo ""

# ── Step 2: start all nodes in parallel ──────────────────────────────────────
for i in $(seq 1 "${N}"); do
    NODE_NAME="node_${i}"
    PORT=$((BASE_PORT + i - 1))
    PARTITION_ID=$((i - 1))
    NODE_LOG="${LOG_DIR}/${NODE_NAME}.log"

    flower-super-dnode \
        --execution-mode deploy \
        --config configs/deploy_static_cluster.yaml \
        --nodeapps-pyproject pyproject.toml \
        --node-name "${NODE_NAME}" \
        --port "${PORT}" \
        --node-data-config-json "{\"partition-id\": ${PARTITION_ID}, \"num-partitions\": ${N}}" \
        > "${NODE_LOG}" 2>&1 &
    PIDS+=($!)
    echo "  [${NODE_NAME}] port=${PORT} pid=${PIDS[-1]}  log=${NODE_LOG}"
done

echo ""
echo "Cluster running (${N} nodes). Press Ctrl-C to stop."

# Tail all logs to stdout so the cluster output is visible.
tail -qF "${LOG_DIR}"/node_*.log &
TAIL_PID=$!

# Wait for any child to exit (crash or SIGTERM).
wait "${PIDS[0]}"
kill "${TAIL_PID}" 2>/dev/null || true
