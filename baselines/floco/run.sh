#!/bin/bash

# Exit script on first error
set -e

echo "Starting Fold experiments..."
flwr run . --federation-config "options.num-supernodes=100" --run-config conf/cifar10_fold_fedavg.toml
flwr run . --federation-config "options.num-supernodes=100" --run-config conf/cifar10_fold_floco.toml
flwr run . --federation-config "options.num-supernodes=100" --run-config conf/cifar10_fold_floco_p.toml

echo "Starting Dirichlet experiments..."
flwr run . --federation-config "options.num-supernodes=100" --run-config conf/cifar10_dir_fedavg.toml
flwr run . --federation-config "options.num-supernodes=100" --run-config conf/cifar10_dir_floco.toml
flwr run . --federation-config "options.num-supernodes=100" --run-config conf/cifar10_dir_floco_p.toml

echo "All experiments completed successfully."
