#!/bin/bash
set -e

case "$1" in
  e2e-bare-https | e2e-bare-auth)
    ./generate.sh
    server_arg="--ssl-ca-certfile certificates/ca.crt --ssl-certfile certificates/server.pem --ssl-keyfile certificates/server.key"
    client_arg="--root-certificates certificates/ca.crt"
    server_dir="./"
    ;;
  *)
    server_arg="--insecure"
    client_arg="--insecure"
    server_dir="./"
    ;;
esac

case "$2" in
  rest)
    rest_arg_superlink="--fleet-api-type rest"
    rest_arg_supernode="--rest"
    server_address="http://localhost:9095"
    server_app_address="127.0.0.1:9091"
    db_arg="--database :flwr-in-memory:"
    server_auth=""
    client_auth_1=""
    client_auth_2=""
    ;;
  sqlite)
    rest_arg_superlink=""
    rest_arg_supernode=""
    server_address="127.0.0.1:9092"
    server_app_address="127.0.0.1:9091"
    db_arg="--database $(date +%s).db"
    server_auth=""
    client_auth_1=""
    client_auth_2=""
    ;;
  client-auth)
    rest_arg_superlink=""
    rest_arg_supernode=""
    server_address="127.0.0.1:9092"
    server_app_address="127.0.0.1:9091"
    db_arg="--database :flwr-in-memory:"
    server_auth="--enable-supernode-auth"
    client_auth_1="--auth-supernode-private-key keys/client_credentials_1 --auth-supernode-public-key keys/client_credentials_1.pub"
    client_auth_2="--auth-supernode-private-key keys/client_credentials_2 --auth-supernode-public-key keys/client_credentials_2.pub"
    ;;
  *)
    rest_arg_superlink=""
    rest_arg_supernode=""
    server_address="127.0.0.1:9092"
    server_app_address="127.0.0.1:9091"
    db_arg="--database :flwr-in-memory:"
    server_auth=""
    client_auth_1=""
    client_auth_2=""
    ;;
esac

# Install Flower app
pip install -e . --no-deps

# revert changes if any in pyproject.toml
# This is needed for multi-stage CI tests that 
# perform migrations more than once. A conflict
# arise when toml-federations are migrated and then 
# re-injected with commands below. It's safer to
# start from a clean slate.
git checkout pyproject.toml

# Remove any duplicates
sed -i '/^\[tool\.flwr\.federations\.e2e\]/,/^$/d' pyproject.toml

# Check if the first argument is 'insecure'
if [ "$server_arg" = "--insecure" ]; then
  # If $server_arg is '--insecure', append the first line
  echo -e $"\n[tool.flwr.federations.e2e]\naddress = \"127.0.0.1:9093\"\ninsecure = true" >> pyproject.toml
else
  # Otherwise, append the second line
  echo -e $"\n[tool.flwr.federations.e2e]\naddress = \"127.0.0.1:9093\"\nroot-certificates = \"certificates/ca.crt\"" >> pyproject.toml
fi

timeout 5m flower-superlink $server_arg $db_arg $rest_arg_superlink $server_auth &
sl_pid=$(pgrep -f "flower-superlink")
sleep 3

# Trigger migration
flwr ls "." e2e || true

if [ "$2" = "client-auth" ]; then
  # Register two SuperNodes using the Flower CLI
  flwr supernode register keys/client_credentials_1.pub e2e
  flwr supernode register keys/client_credentials_2.pub e2e
fi

timeout 5m flower-supernode $client_arg $rest_arg_supernode \
  --superlink $server_address $client_auth_1 \
  --clientappio-api-address "localhost:9094" \
  --max-retries 0 &
cl1_pid=$!
sleep 3

timeout 5m flower-supernode $client_arg $rest_arg_supernode \
  --superlink $server_address $client_auth_2 \
  --clientappio-api-address "localhost:9096" \
  --max-retries 0 &
cl2_pid=$!
sleep 3

timeout 1m flwr run "." e2e

# Initialize a flag to track if training is successful
found_success=false
timeout=240  # Timeout after 240 seconds
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
