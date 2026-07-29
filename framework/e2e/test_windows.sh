#!/usr/bin/env bash

# Create and install Flower app
flwr new @flwrlabs/numpy-ci
cd numpy-ci

# Modify the config file
echo -e $"\n[tool.flwr.federations.e2e]\naddress = \"127.0.0.1:9093\"\ninsecure = true" >> pyproject.toml

# Start Flower processes in the background
flower-superlink --insecure &
sleep 2

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address localhost:9094 \
    --max-retries 0 &
cl1_pid=$!
sleep 2

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address localhost:9095 \
    --max-retries 0 &
sleep 2

flwr run --run-config num-server-rounds=1 . e2e

# Trap to clean up on exit
cleanup() {
    echo "Stopping Flower processes..."
    taskkill //F //FI "IMAGENAME eq flower*" //T
}
trap cleanup EXIT

# Initialize a flag to track if training is successful
found_success=false
timeout=120  # Timeout after 120 seconds
elapsed=0

# Define a cleanup function
cleanup_and_exit() {
    kill $cl1_pid; kill $cl2_pid;
    sleep 1; kill $sl_pid;
    exit $1
}

# Check for "finished:completed" status in a loop with a timeout
while [ "$found_success" = false ] && [ $elapsed -lt $timeout ]; do
    # Run the command and capture output
    output=$(flwr ls e2e --format=json)

    # Extract status from the first run (or loop over all if needed)
    status=$(echo "$output" | jq -r '.runs[0].status')

    echo "Current status: $status"

    if [ "$status" == "finished:completed" ]; then
      found_success=true
      echo "Training worked correctly!"
      cleanup_and_exit 0
    else
      echo "⏳ Not completed yet, retrying in 2s..."
      sleep 2
    fi
done

if [ "$found_success" = false ]; then
    echo "Training had an issue and timed out."
    cleanup_and_exit 1
fi
