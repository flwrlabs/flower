#!/usr/bin/env bash
set -euo pipefail

# Launch N Flower Super DNodes in background (dynamic mode by default).
#
# Usage:
#   ./scripts/run_n_nodes.sh <N> [BASE_PORT] [CONFIG]
#
# Example:
#   ./scripts/run_n_nodes.sh 3 9100 configs/node_dynamic.yaml
#
# Notes:
# - Node #1 is started without bootnode.
# - Nodes #2..N use node #1 as bootnode.
# - Logs are written under ./logs/run_n_nodes
# - Stop all spawned nodes with Ctrl+C

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <N> [BASE_PORT] [CONFIG]"
  exit 1
fi

N="$1"
BASE_PORT="${2:-9100}"
CONFIG="${3:-configs/node_dynamic.yaml}"
RUN_TIMEOUT_MS="${RUN_TIMEOUT_MS:-15000}"

if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
  echo "Error: N must be an integer >= 1"
  exit 1
fi

LOG_DIR="logs/run_n_nodes"
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EXAMPLE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NODEAPPS_PYPROJECT="$EXAMPLE_ROOT/pyproject.toml"

if [[ -n "${SUPER_DNODE_CMD:-}" ]]; then
  read -r -a SUPER_DNODE_CMD_ARR <<< "${SUPER_DNODE_CMD}"
elif command -v flower-super-dnode >/dev/null 2>&1; then
  SUPER_DNODE_CMD_ARR=(flower-super-dnode)
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
      PYTHON_BIN="python"
    else
      echo "Error: neither 'flower-super-dnode' nor Python interpreter found."
      exit 1
    fi
  fi

  export PYTHONPATH="$REPO_ROOT/framework/py${PYTHONPATH:+:$PYTHONPATH}"
  SUPER_DNODE_CMD_ARR=(
    "$PYTHON_BIN"
    -c
    "from flwr.decentralized.superdnode.cli.flower_super_dnode import flower_super_dnode; flower_super_dnode()"
  )
fi

PIDS=()
FIRST_PORT="$BASE_PORT"

cleanup() {
  echo "Stopping ${#PIDS[@]} node(s)..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
    fi
  done
}
trap cleanup EXIT INT TERM

for ((i=0; i<N; i++)); do
  port=$((BASE_PORT + i))
  node_name="node_$i"
  log_file="$LOG_DIR/node_${i}.log"

  # Build data_config: each node gets partition-id=$i with num-partitions=$N
  data_config_json="{\"partition-id\": $i, \"num-partitions\": $N}"

  cmd=(
    "${SUPER_DNODE_CMD_ARR[@]}"
    --execution-mode deploy
    --config "$CONFIG"
    --nodeapps-pyproject "$NODEAPPS_PYPROJECT"
    --timeout "$RUN_TIMEOUT_MS"
    --port "$port"
    --node-name "$node_name"
    --node-data-config-json "$data_config_json"
  )

#   if [[ "$i" -gt 0 ]]; then
#     cmd+=(--bootnodes "127.0.0.1:${FIRST_PORT}")
#   fi

  echo "Starting node $i on port $port (log: $log_file)"
  "${cmd[@]}" >"$log_file" 2>&1 &
  PIDS+=("$!")
done

echo "Started ${#PIDS[@]} node(s). Press Ctrl+C to stop all."
wait
