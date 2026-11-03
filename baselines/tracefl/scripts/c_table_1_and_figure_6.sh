#!/usr/bin/env bash

# TraceFL baseline: Faulty client detection (Table-1 and Figure-6)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Running TraceFL baseline faulty client detection experiment..."

RUN_CONFIG="num-server-rounds=3 \
tracefl.dataset='mnist' \
tracefl.model='resnet18' \
tracefl.num-clients=4 \
tracefl.dirichlet-alpha=0.7 \
tracefl.max-per-client-data-size=2048 \
tracefl.max-server-data-size=2048 \
tracefl.batch-size=32 \
tracefl.provenance-rounds='1,2,3' \
tracefl.faulty-clients-ids='[0]' \
tracefl.label2flip='{1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0}' \
tracefl.use-deterministic-sampling=true \
tracefl.random-seed=42 \
tracefl.client-weights-normalization=false \
tracefl.output-dir='results/experiment_c' \
min-train-nodes=4 \
fraction-train=1.0 \
local-epochs=2"

echo "Running TraceFL baseline faulty client detection..."
flwr run . --federation-config "options.num-supernodes=4" --run-config "$RUN_CONFIG"

echo "Generating plots for faulty client detection..."
python -m scripts.generate_graphs \
  --output-dir "results/experiment_c/graphs" \
  --pattern "results/experiment_c/prov_*.csv" \
  --title "TraceFL Faulty Client Detection"
