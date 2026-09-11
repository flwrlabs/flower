#!/usr/bin/env bash
# Portable Flower simulation launchers (no scheduler/cluster assumptions).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CLIENT_CPUS="${SSFL_CLIENT_CPUS:-1}"
CLIENT_GPUS="${SSFL_CLIENT_GPUS:-0.125}"
DEMO_FEDERATION="num-supernodes=2 client-resources-num-cpus=${CLIENT_CPUS} client-resources-num-gpus=0"
EVAL_FEDERATION="num-supernodes=4 client-resources-num-cpus=${CLIENT_CPUS} client-resources-num-gpus=0"
GPU_FEDERATION="num-supernodes=100 client-resources-num-cpus=${CLIENT_CPUS} client-resources-num-gpus=${CLIENT_GPUS}"

usage() {
  cat <<'EOF'
Usage: ./run_experiments.sh [demo|eval-smoke|short|cifar10|cifar100]

  demo         2-client CPU demo (default)
  eval-smoke   4 clients, 3 rounds, centralized eval
  short        100-client / 3-round CIFAR-10 GPU check
  cifar10      paper-scale CIFAR-10
  cifar100     paper-scale CIFAR-100

GPU runs allocate 0.125 GPU per concurrent ClientApp by default. Override
this for your hardware, for example:
  SSFL_CLIENT_GPUS=0.25 ./run_experiments.sh cifar100
EOF
}

cmd="${1:-demo}"
case "$cmd" in
  demo)
    flwr run . --stream --federation-config "${DEMO_FEDERATION}"
    ;;
  eval-smoke)
    flwr run . --stream \
      --federation-config "${EVAL_FEDERATION}" \
      --run-config conf/cifar10_eval_smoke.toml
    ;;
  short)
    flwr run . --stream \
      --federation-config "${GPU_FEDERATION}" \
      --run-config conf/cifar10_short.toml
    ;;
  cifar10)
    flwr run . --stream \
      --federation-config "${GPU_FEDERATION}" \
      --run-config conf/cifar10_paper.toml
    ;;
  cifar100)
    flwr run . --stream \
      --federation-config "${GPU_FEDERATION}" \
      --run-config conf/cifar100_paper.toml
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
