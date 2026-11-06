#!/usr/bin/env bash
set -euo pipefail

flwr run . --run-config "num-server-rounds=5 random-seed=2026 verify-aggregation=true"
python benchmark_report.py --report-path artifacts/report.json
