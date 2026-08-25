#!/bin/bash
set -eoux pipefail

# Set connectivity parameters
case "$1" in
    secure)
      ./generate.sh
      server_arg='--ssl-ca-certfile ../certificates/ca.crt
                  --ssl-certfile    ../certificates/server.pem
                  --ssl-keyfile     ../certificates/server.key'
      client_arg='--root-certificates ../certificates/ca.crt'
      ;;
    insecure)
      server_arg='--insecure'
      client_arg=$server_arg
    ;;
esac

# Set authentication parameters
case "$2" in
    client-auth)
      server_auth='--enable-supernode-auth'
      client_auth_1='--auth-supernode-private-key ../keys/client_credentials_1'
      client_auth_2='--auth-supernode-private-key ../keys/client_credentials_2'
      server_address='127.0.0.1:9092'
      ;;
    *)
    server_auth=''
    client_auth_1=''
    client_auth_2=''
    server_address='127.0.0.1:9092'
    ;;
esac

# Set engine
case "$3" in
    deployment-engine)
      simulation_arg=""
      ;;
    simulation-engine)
      simulation_arg="--simulation"
      ;;
esac


# Create and install Flower app
flwr new @flwrlabs/numpy-ci
cd numpy-ci
# Remove flwr dependency from `pyproject.toml`. Seems necessary so that it does
# not override the wheel dependency
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS (Darwin) system
    sed -i '' '/flwr\[simulation\]/d' pyproject.toml
else
    # Non-macOS system (Linux)
    sed -i '/flwr\[simulation\]/d' pyproject.toml
fi
pip install -e . --no-deps

# Check if the first argument is 'insecure'
if [ "$1" == "insecure" ]; then
  # If $1 is 'insecure', append the first line
  echo -e $"\n[tool.flwr.federations.e2e]\naddress = \"127.0.0.1:9093\"\ninsecure = true" >> pyproject.toml
else
  # Otherwise, append the second line
  echo -e $"\n[tool.flwr.federations.e2e]\naddress = \"127.0.0.1:9093\"\nroot-certificates = \"../certificates/ca.crt\"" >> pyproject.toml
fi

if [ "$3" = "simulation-engine" ]; then
  echo -e $"options.num-supernodes = 10" >> pyproject.toml
fi

# Combine the arguments into a single command for flower-superlink
combined_args="$server_arg $server_auth $simulation_arg"

background_pids=()
cleanup() {
  if [ "${#background_pids[@]}" -gt 0 ]; then
    kill "${background_pids[@]}" 2>/dev/null || true

    # Give the services a short grace period to run their shutdown handlers. A
    # stuck child must not hold the whole CI job until the workflow timeout.
    cleanup_timeout=15
    cleanup_deadline=$((SECONDS + cleanup_timeout))
    while [ "$SECONDS" -lt "$cleanup_deadline" ]; do
      running=false
      for pid in "${background_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          running=true
          break
        fi
      done

      if [ "$running" = false ]; then
        break
      fi
      sleep 1
    done

    if [ "$running" = true ]; then
      echo "Timed out after ${cleanup_timeout}s; force-killing background services."
      kill -KILL "${background_pids[@]}" 2>/dev/null || true
    fi

    wait "${background_pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

flower-superlink $combined_args &
sl_pid=$!
background_pids+=("$sl_pid")
sleep 1
sleep 2

# Trigger migration
flwr ls ../numpy-ci e2e || true

if [ "$2" = "client-auth" ] && [ "$3" = "deployment-engine" ]; then
  # Register two SuperNodes using the Flower CLI
  flwr supernode register ../keys/client_credentials_1.pub e2e
  flwr supernode register ../keys/client_credentials_2.pub e2e
fi

if [ "$3" = "deployment-engine" ]; then
  flower-supernode $client_arg \
      --superlink $server_address $client_auth_1 \
      --host localhost --port 9094 \
      --node-config "partition-id=0 num-partitions=2" --max-retries 0 &
  background_pids+=("$!")
  sleep 2

  flower-supernode $client_arg \
      --superlink $server_address $client_auth_2 \
      --host localhost --port 9095 \
      --node-config "partition-id=1 num-partitions=2" --max-retries 0 &
  background_pids+=("$!")
  sleep 2
fi

timeout 1m flwr run --run-config num-server-rounds=1 ../numpy-ci e2e

# Keep the services alive for the entire training run. The workflow-level
# timeout remains the final guard if cleanup or a command hangs.
training_timeout=300
deadline=$((SECONDS + training_timeout))
status_query_timeout=10

while [ "$SECONDS" -lt "$deadline" ]; do
    if ! kill -0 "$sl_pid" 2>/dev/null; then
      echo "SuperLink exited before training completed."
      exit 1
    fi

    # Run the command and capture output
    if ! output=$(timeout "${status_query_timeout}s" flwr ls e2e --format=json); then
      echo "flwr ls failed or timed out after ${status_query_timeout} seconds."
      exit 1
    fi

    # Extract status from the first run (or loop over all if needed)
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
