#!/usr/bin/env bash

# TraceFL baseline: Single alpha=0.3 experiment (matches original TraceFL)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Running TraceFL baseline localization experiment (alpha=0.3 only)..."

RUN_CONFIG="num-server-rounds=2 \
tracefl.dataset='mnist' \
tracefl.model='resnet18' \
tracefl.num-clients=10 \
tracefl.dirichlet-alpha=0.3 \
tracefl.max-per-client-data-size=2048 \
tracefl.max-server-data-size=2048 \
tracefl.batch-size=32 \
tracefl.provenance-rounds='1,2' \
tracefl.use-deterministic-sampling=true \
tracefl.random-seed=42 \
tracefl.output-dir='results/experiment_a' \
min-train-nodes=4 \
fraction-train=0.4"

echo "Running TraceFL baseline with Dirichlet alpha=0.3..."
flwr run . --run-config "$RUN_CONFIG"

echo "Generating accuracy plots..."
python -m scripts.generate_graphs \
  --output-dir "results/experiment_a/graphs" \
  --pattern "results/experiment_a/prov_*.csv" \
  --title "TraceFL Localization Accuracy (MNIST, 10 Clients, α=0.3)"
