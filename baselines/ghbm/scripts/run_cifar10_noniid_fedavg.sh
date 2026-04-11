#!/usr/bin/env bash

set -euo pipefail

LOCAL_BYPASS="127.0.0.1,localhost,::1"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$LOCAL_BYPASS"
export no_proxy="${no_proxy:+$no_proxy,}$LOCAL_BYPASS"
export WANDB__SERVICE_WAIT=300

RUN_CONFIG="num-server-rounds=10000 \
fraction-train=0.1 \
local-epochs=1 \
learning-rate=0.1 \
weight-decay=1e-5 \
ghbm-beta=0.0 \
ghbm-tau=0 \
dataset-name='cifar10' \
dirichlet-alpha=0.0 \
batch-size=64 \
evaluate-every=100 \
algorithm-name='fedavg' \
wandb-enabled=true \
wandb-project='ghbm-comparison' \
wandb-group='cifar10-noniid' \
wandb-run-name='fedavg-cifar10-noniid' \
model-name='resnet' \
resnet-version=20 \
norm-layer='group'"

flwr run . --stream \
  --run-config "$RUN_CONFIG" \
  --federation-config "num-supernodes=100" \
  --federation-config "client-resources-num-cpus=2" \
  --federation-config "client-resources-num-gpus=1.0" \
  --federation-config "init-args-num-cpus=16" \
  --federation-config "init-args-num-gpus=4"
