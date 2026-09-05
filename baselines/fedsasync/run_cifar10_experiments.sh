#!/bin/bash

FS_VALUES=(0 1 2)
M_VALUES=(7 8 9 10)

for fs in "${FS_VALUES[@]}"; do
    flwr run . --federation-config 'num-supernodes=10' --run-config "name=\"FedAvg\" number-slow=${fs}" --stream

    for m in "${M_VALUES[@]}"; do
        flwr run . --federation-config 'num-supernodes=10' --run-config "number-slow=${fs} semiasync-deg=${m}" --stream
    done
done
