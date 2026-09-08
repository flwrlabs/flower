#!/bin/bash
#
# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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