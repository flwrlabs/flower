#!/bin/bash
set -e

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

timeout 2m flower-superlink $combined_args &
sl_pid=$(pgrep -f "flower-superlink")
sleep 2

# Trigger migration
flwr ls ../numpy-ci e2e || true

if [ "$2" = "client-auth" ] && [ "$3" = "deployment-engine" ]; then
  # Register two SuperNodes using the Flower CLI
  flwr supernode register ../keys/client_credentials_1.pub e2e
  flwr supernode register ../keys/client_credentials_2.pub e2e
fi

if [ "$3" = "deployment-engine" ]; then
  timeout 2m flower-supernode $client_arg \
      --superlink $server_address $client_auth_1 \
      --clientappio-api-address localhost:9094 \
      --node-config "partition-id=0 num-partitions=2" --max-retries 0 &
  cl1_pid=$!
  sleep 2

  timeout 2m flower-supernode $client_arg \
      --superlink $server_address $client_auth_2 \
      --clientappio-api-address localhost:9095 \
      --node-config "partition-id=1 num-partitions=2" --max-retries 0 &
  cl2_pid=$!
  sleep 2
fi

timeout 1m flwr run --run-config num-server-rounds=1 ../numpy-ci e2e

# Initialize a flag to track if training is successful
found_success=false
timeout=120  # Timeout after 120 seconds
elapsed=0
engine="$3"

# Define a cleanup function
cleanup_and_exit() {
    if [ "$engine" = "deployment-engine" ]; then
      kill $cl1_pid; kill $cl2_pid;
    fi
    sleep 1; kill $sl_pid;
    exit $1
}

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
