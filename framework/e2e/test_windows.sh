#!/usr/bin/env bash
set -euo pipefail

background_pids=()

cleanup() {
    local exit_code=$?
    trap - EXIT

    echo "Stopping Flower processes..."
    if command -v taskkill >/dev/null 2>&1; then
        for pid in "${background_pids[@]}"; do
            taskkill //F //PID "$pid" //T >/dev/null 2>&1 || true
        done
    else
        for pid in "${background_pids[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        if ((${#background_pids[@]} > 0)); then
            wait "${background_pids[@]}" 2>/dev/null || true
        fi
    fi

    exit "$exit_code"
}
trap cleanup EXIT

wait_for_port() {
    local pid="$1"
    local port="$2"
    local name="$3"

    for _ in {1..30}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$name exited before port $port became available."
            return 1
        fi

        if python -c 'import socket, sys; s = socket.create_connection(("127.0.0.1", int(sys.argv[1])), 1); s.close()' "$port" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done

    echo "$name did not open port $port in time."
    return 1
}

# Create and install Flower app
flwr new @flwrlabs/numpy-ci
cd numpy-ci

# Modify the config file
printf '\n[tool.flwr.federations.e2e]\naddress = "127.0.0.1:9093"\ninsecure = true\n' >> pyproject.toml

# Start the SuperLink and wait until its control API is ready.
flower-superlink --insecure &
sl_pid=$!
background_pids+=("$sl_pid")
wait_for_port "$sl_pid" 9092 "SuperLink"

# Start two SuperNodes and wait until their ClientApp APIs are ready.
flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address localhost:9094 \
    --max-retries 0 &
cl1_pid=$!
background_pids+=("$cl1_pid")
wait_for_port "$cl1_pid" 9094 "SuperNode 1"

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address localhost:9095 \
    --max-retries 0 &
cl2_pid=$!
background_pids+=("$cl2_pid")
wait_for_port "$cl2_pid" 9095 "SuperNode 2"

flwr run --run-config num-server-rounds=1 . e2e

training_timeout=120
deadline=$((SECONDS + training_timeout))

while [ "$SECONDS" -lt "$deadline" ]; do
    if ! kill -0 "$sl_pid" 2>/dev/null; then
        echo "SuperLink exited before training completed."
        exit 1
    fi

    if ! output=$(flwr ls e2e --format=json); then
        echo "flwr ls failed."
        exit 1
    fi
    if ! echo "$output" | jq -e '.success == true' >/dev/null; then
        echo "Unexpected flwr ls output:"
        echo "$output"
        exit 1
    fi

    status=$(echo "$output" | jq -r '.runs[0].status')
    echo "Current status: $status"

    case "$status" in
        finished:completed)
            echo "Training worked correctly!"
            exit 0
            ;;
        finished:*)
            status_details=$(echo "$output" | jq -r '.runs[0]["status-details"]')
            echo "Training failed: ${status_details}"
            exit 1
            ;;
    esac

    echo "⏳ Not completed yet, retrying in 2s..."
    sleep 2
done

echo "Training did not complete within ${training_timeout} seconds."
exit 1
