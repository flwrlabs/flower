#!/usr/bin/env bash

set -Eeuo pipefail

: "${FLWR_SUPERLINK_ADDRESS:?Set FLWR_SUPERLINK_ADDRESS to the Fleet API address.}"

export FLWR_HOME="${FLUX_JOB_TMPDIR:-/tmp}/flower-${FLUX_JOB_ID:-manual}"

supernode_args=(
    --superlink "${FLWR_SUPERLINK_ADDRESS}"
)

if [[ "${FLWR_INSECURE:-false}" == "true" ]]; then
    supernode_args+=(--insecure)
else
    : "${FLWR_SUPERNODE_PRIVATE_KEY:?Set FLWR_SUPERNODE_PRIVATE_KEY to the registered private key.}"
    supernode_args+=(
        --auth-supernode-private-key "${FLWR_SUPERNODE_PRIVATE_KEY}"
    )
fi

if [[ -n "${FLWR_NODE_CONFIG:-}" ]]; then
    supernode_args+=(--node-config "${FLWR_NODE_CONFIG}")
fi

exec flower-supernode "${supernode_args[@]}" "$@"
