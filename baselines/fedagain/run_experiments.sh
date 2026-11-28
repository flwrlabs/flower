#!/bin/bash
# Reproduce the experiments of Fig. 11 of the paper (Sec. 6.6).
#
#   ./run_experiments.sh [cifar10|mnist] [seed]
#
# Corruption scenarios (10 clients):  light  = 10% corrupted clients, severity 3
#                                     medium = 30% corrupted clients, severity 4
#                                     heavy  = 40% corrupted clients, severity 5
# Scalability (30% corrupted clients, severity 3): 5, 10 and 20 clients.
#
# Each run writes a JSON file to outputs/; figures go to _static/.
# Set STRATEGIES="fedavg fedmedian fedagain" to also run FedMedian.
set -e
DATASET=${1:-cifar10}
SEED=${2:-42}
STRATEGIES=${STRATEGIES:-"fedavg fedagain"}
CPUS_PER_CLIENT=${CPUS_PER_CLIENT:-3}
GPUS_PER_CLIENT=${GPUS_PER_CLIENT:-0.0}
OUT_DIR="$(pwd)/outputs"
mkdir -p "${OUT_DIR}/configs"

# "<num-clients> <corrupted-fraction> <severity>"
SCENARIOS=("10 0.1 3" "10 0.3 4" "10 0.4 5" "5 0.3 3" "10 0.3 3" "20 0.3 3")

for SCENARIO in "${SCENARIOS[@]}"; do
  read -r CLIENTS RHO SEVERITY <<< "${SCENARIO}"
  for STRATEGY in ${STRATEGIES}; do
    echo "=== ${DATASET} | ${CLIENTS} clients | rho=${RHO} | sev=${SEVERITY} | ${STRATEGY} | seed=${SEED} ==="
    # `flwr run` does not accept a TOML file and key-value pairs together,
    # so the experiment profile and the overrides are merged into one file.
    CFG="${OUT_DIR}/configs/${DATASET}_n${CLIENTS}_rho${RHO}_sev${SEVERITY}_${STRATEGY}_seed${SEED}.toml"
    cat "conf/${DATASET}.toml" > "${CFG}"
    {
      echo "strategy = \"${STRATEGY}\""
      echo "corrupted-fraction = ${RHO}"
      echo "corruption-severity = ${SEVERITY}"
      echo "seed = ${SEED}"
      echo "output-dir = \"${OUT_DIR}\""
    } >> "${CFG}"
    flwr run . --stream \
      --federation-config "num-supernodes=${CLIENTS} client-resources-num-cpus=${CPUS_PER_CLIENT} client-resources-num-gpus=${GPUS_PER_CLIENT}" \
      --run-config "${CFG}"
  done
done

python -m fedagain.plot --dataset "${DATASET}" --seed "${SEED}" --outputs outputs --static _static
