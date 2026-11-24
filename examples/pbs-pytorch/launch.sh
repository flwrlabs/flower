#!/usr/bin/env bash
set -Eeuo pipefail

: "${PBS_JOBID:?Run this launcher inside a PBS allocation}"
: "${PBS_NODEFILE:?PBS must supply a nodefile}"
: "${PYTHON_BIN:?Set an absolute virtualenv interpreter path}"
: "${DATA_ROOT:?Set a shared CIFAR-10 directory}"
: "${OUTPUT_BASE:?Set a shared results directory}"
[[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]]
APP_ROOT="${PBS_O_WORKDIR:?Submit from this example directory}"
cd "$APP_ROOT"
mapfile -t hosts < <(awk '!seen[$0]++' "$PBS_NODEFILE")
[[ "${#hosts[@]}" -eq 4 && "$(wc -l < "$PBS_NODEFILE")" -eq 4 ]] || {
    echo "Request four distinct nodes with mpiprocs=1" >&2
    exit 1
}
[[ "$(hostname -s)" = "${hosts[0]%%.*}" ]] || {
    echo "Launch from the first allocated compute node" >&2
    exit 1
}
MODE="${1:?Specify deployment or simulation}"
[[ "$MODE" = deployment || "$MODE" = simulation ]]
export OUTPUT_ROOT="$OUTPUT_BASE/$PBS_JOBID/$MODE"
mkdir -p "$OUTPUT_ROOT/events" "$DATA_ROOT"
cp "$PBS_NODEFILE" "$OUTPUT_ROOT/pbs-nodefile"
export PYTHONPATH="$APP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_USAGE_STATS_ENABLED=0
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
"$PYTHON_BIN" prepare_data.py > "$OUTPUT_ROOT/data.log" 2>&1
job_number="${PBS_JOBID%%.*}"
export MASTER_ADDR="${hosts[0]}"
export FLEET_PORT=$((20000 + job_number % 10000))
export RAY_PORT=$((30000 + job_number % 10000))

# Pass workload variables as arguments while preserving module-provided MPI settings.
# Keep scheduler-provided GPU visibility local to each physical node.
mpirun --map-by ppr:1:node --bind-to none -np 4 \
    /usr/bin/env "APP_ROOT=$APP_ROOT" "PYTHON_BIN=$PYTHON_BIN" \
    "PYTHONPATH=$PYTHONPATH" "DATA_ROOT=$DATA_ROOT" "OUTPUT_ROOT=$OUTPUT_ROOT" \
    "MASTER_ADDR=$MASTER_ADDR" "FLEET_PORT=$FLEET_PORT" "RAY_PORT=$RAY_PORT" \
    "PBS_JOBID=$PBS_JOBID" "OMP_NUM_THREADS=2" "OPENBLAS_NUM_THREADS=1" \
    "RAY_ENABLE_UV_RUN_RUNTIME_ENV=0" "RAY_USAGE_STATS_ENABLED=0" \
    "FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1" \
    bash -lc '
        set -Eeuo pipefail
        cd "$APP_ROOT"
        awk -v host="$(hostname -s)" '\''{split($0, parts, "."); if (parts[1] == host) found=1} END {exit !found}'\'' "$OUTPUT_ROOT/pbs-nodefile"
        export PATH="$(dirname "$PYTHON_BIN"):$PATH"
        exec "$PYTHON_BIN" launch.py "$1"
    ' bash "$MODE" 2>&1 | tee "$OUTPUT_ROOT/launcher.log"
