#!/usr/bin/env bash
# Generic paper-scale launchers (no cluster/Slurm specifics).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./run_experiments.sh [demo|eval-smoke|short|cifar10|cifar100]

  demo         4-client CPU demo (default)
  eval-smoke   4 clients, 3 rounds, centralized eval
  short        100-client / 3-round CIFAR-10 check (GPU federation)
  cifar10      paper-scale CIFAR-10
  cifar100     paper-scale CIFAR-100
EOF
}

cmd="${1:-demo}"
case "$cmd" in
  demo)
    flwr run . --stream
    ;;
  eval-smoke)
    flwr run . --stream --run-config conf/cifar10_eval_smoke.toml
    ;;
  short)
    flwr run . gpu-simulation --stream --run-config conf/cifar10_short.toml
    ;;
  cifar10)
    flwr run . gpu-simulation --stream --run-config conf/cifar10_paper.toml
    ;;
  cifar100)
    flwr run . gpu-simulation --stream --run-config conf/cifar100_paper.toml
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
