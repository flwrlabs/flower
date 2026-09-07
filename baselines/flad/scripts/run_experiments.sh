#!/bin/bash
set -euo pipefail

# Run a number of experiments with different random seeds

# List of seeds to run experiments with
SEEDS=(8506 6369 5111 2697 3078 409 752 165 1752 8132)

for seed in "${SEEDS[@]}"; do
    echo "Starting experiment with seed $seed"
    if ! flwr run . --federation-config="num-supernodes=13 client-resources-num-cpus=2" \
      --run-config rn_seed=$seed --stream; then
        echo "Experiment with seed $seed failed" >&2
        exit 1
    fi

    echo "Experiment with seed $seed done"
    sleep 10
done

echo "All experiments completed"