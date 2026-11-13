#!/bin/bash

FS_VALUES=(0.2)
M_VALUES=(14 16 18 20)
PROX_VALUES=(0.0 0.2)
P_PARTITIONER=('iid')
N_EXECUTIONS=1

for part in "${P_PARTITIONER[@]}"; do        
    for fs in "${FS_VALUES[@]}"; do
        # FedSaSync executions
        for m in "${M_VALUES[@]}"; do
            for prox in "${PROX_VALUES[@]}"; do
                for ((run=1; run<=N_EXECUTIONS; run++)); do
                    flwr run . --federation-config 'num-supernodes=20' --run-config "name=\"FedProxSaSync\" proximal-mu=${prox} \
                      fraction-slow=${fs} run-id=${run} semiasync-deg=${m} num-server-rounds=25 data-distribution=\"${part}\"" --stream
                done
            done
        done
    done
done