FS_VALUES=(0.5 0.4 0.3 0.2 0.1 0.0)
M_VALUES=(8 10 12 14 16 18 20)
N_EXECUTIONS=5
P_PARTITIONER=('dirichlet' 'iid')

for part in "${P_PARTITIONER[@]}"; do        
    for fs in "${FS_VALUES[@]}"; do
        # FedAvg executions
        for ((run=1; run<=N_EXECUTIONS; run++)); do
            flwr run . --federation-config 'num-supernodes=20' \
              --run-config "dataset-name=\"ylecun/mnist\" name=\"FedAvg\" fraction-slow=${fs} run-id=${run} num-server-rounds=25 data-distribution=\"${part}\"" --stream
        done

        # FedSaSync executions
        for m in "${M_VALUES[@]}"; do
            for ((run=1; run<=N_EXECUTIONS; run++)); do
                flwr run . --federation-config 'num-supernodes=20' \
                  --run-config "dataset-name=\"ylecun/mnist\" fraction-slow=${fs} run-id=${run} semiasync-deg=${m} num-server-rounds=25 data-distribution=\"${part}\"" --stream
            done
        done
    done
done