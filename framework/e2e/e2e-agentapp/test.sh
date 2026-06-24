#!/bin/bash
set -e

server_arg="--insecure"
server_app_address="127.0.0.1:9091"
db_arg="--database :flwr-in-memory:"
runtime_dependency_install_arg="--disable-runtime-dependency-installation"

# Install Flower app
pip install -e . --no-deps

echo -e $"\n[tool.flwr.federations.e2e]\naddress = \"127.0.0.1:9093\"\ninsecure = true" >> pyproject.toml

timeout 5m flower-superlink \
$server_arg $db_arg $runtime_dependency_install_arg \
--control-api-address 127.0.0.1:9093 \
--serverappio-api-address "$server_app_address" &
sl_pid=$!
sleep 3

timeout 5m flower-superexec \
$server_arg \
--appio-api-address "$server_app_address" \
--plugin-type serverapp &
sx_pid=$!
sleep 3

# Trigger migration
flwr ls "." e2e || true

timeout 1m flwr run "." e2e \
--run-config 'agent.input="What is the Flower federated learning framework? Answer in one sentence."'

found_success=false
timeout=120
elapsed=0

cleanup_and_exit() {
    kill $sx_pid;
    sleep 1; kill $sl_pid;
    exit $1
}

while [ "$found_success" = false ] && [ $elapsed -lt $timeout ]; do
    output=$(flwr ls e2e --format=json)
    status=$(echo "$output" | jq -r '.runs[0].status')

    echo "Current status: $status"

    if [ "$status" == "finished:completed" ]; then
    found_success=true
    echo "AgentApp worked correctly!"
    cleanup_and_exit 0
    else
    echo "⏳ Not completed yet, retrying in 2s..."
    sleep 2
    elapsed=$((elapsed + 2))
    fi
done

if [ "$found_success" = false ]; then
    echo "AgentApp had an issue and timed out."
    cleanup_and_exit 1
fi