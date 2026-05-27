#!/usr/bin/env bash
# Launch N virtual nodes with dynamic topology on localhost.
#
# Usage:
#   ./scripts/run_cluster.sh                  # 4 nodes starting at port 9100
#   ./scripts/run_cluster.sh 8                # 8 nodes starting at port 9100
#   ./scripts/run_cluster.sh 6 9200           # 6 nodes starting at port 9200
#
# What this does:
#   1. Starts node-0 (bootstrap node) on BASE_PORT, no bootnodes.
#   2. Starts nodes 1..N-1 each on BASE_PORT+i, pointing to node-0 as bootnode.
#   All processes run in the background.
#   Ctrl-C kills the whole cluster.

set -euo pipefail

cd "$(dirname "$0")/.."

N="${1:-4}"
BASE_PORT="${2:-9100}"
LOG_DIR="$(mktemp -d /tmp/flwr-dynamic-cluster.XXXXX)"

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

echo "Starting dynamic cluster: ${N} nodes, base port ${BASE_PORT}"
echo "Logs will be written to: ${LOG_DIR}"
echo ""

# ── Node 0: bootstrap node ───────────────────────────────────────────────────
NODE0_PORT="${BASE_PORT}"
NODE0_LOG="${LOG_DIR}/node_0.log"

flower-super-dnode \
    --execution-mode deploy \
    --config configs/deploy_dynamic.yaml \
    --nodeapps-pyproject pyproject.toml \
    --port "${NODE0_PORT}" \
    --node-data-config-json "{\"partition-id\": 0, \"num-partitions\": ${N}}" \
    > "${NODE0_LOG}" 2>&1 &
PIDS+=($!)
echo "  [node_0] port=${NODE0_PORT} pid=${PIDS[-1]}  log=${NODE0_LOG}"

# Give node-0 a moment to bind its port before others try to connect.
sleep 1

# ── Nodes 1..N-1: connect to node 0 as bootnode ─────────────────────────────
for i in $(seq 1 $((N - 1))); do
    PORT=$((BASE_PORT + i))
    NODE_LOG="${LOG_DIR}/node_${i}.log"

    flower-super-dnode \
        --execution-mode deploy \
        --config configs/deploy_dynamic.yaml \
        --nodeapps-pyproject pyproject.toml \
        --port "${PORT}" \
        --node-data-config-json "{\"partition-id\": ${i}, \"num-partitions\": ${N}}" \
        > "${NODE_LOG}" 2>&1 &
    PIDS+=($!)
    echo "  [node_${i}] port=${PORT} pid=${PIDS[-1]}  log=${NODE_LOG}"
done

echo ""
echo "Cluster running (${N} nodes). Press Ctrl-C to stop."

# Tail all logs to stdout so the cluster output is visible.
tail -qF "${LOG_DIR}"/node_*.log &
TAIL_PID=$!

# Wait for any child to exit (crash or SIGTERM).
wait "${PIDS[0]}"
kill "${TAIL_PID}" 2>/dev/null || true
