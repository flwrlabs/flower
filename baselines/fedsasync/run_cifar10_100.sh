FS_VALUES=(0.2)
M_VALUES=(65 75 80 85 95)
N_EXECUTIONS=3
P_PARTITIONER=('iid' 'dirichlet')

for part in "${P_PARTITIONER[@]}"; do        
    for fs in "${FS_VALUES[@]}"; do
        # FedSaSync executions
        for m in "${M_VALUES[@]}"; do
            for ((run=1; run<=N_EXECUTIONS; run++)); do
                flwr run . --federation-config 'num-supernodes=100' \
                  --run-config "name=\"FedSaSync_100\" fraction-evaluate=0.0 local-epochs=3 fraction-slow=${fs} run-id=${run} \
                  semiasync-deg=${m} num-server-rounds=40 data-distribution=\"${part}\"" --stream
            done
        done
    done
done