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
  sqlite)
    server_address="127.0.0.1:9092"
    server_app_address="127.0.0.1:9091"
    db_arg="--database $(date +%s).db"
    server_auth=""
    client_auth_1=""
    client_auth_2=""
    ;;
  client-auth)
    server_address="127.0.0.1:9092"
    server_app_address="127.0.0.1:9091"
    db_arg="--database :flwr-in-memory:"
    server_auth="--enable-supernode-auth"
    client_auth_1="--auth-supernode-private-key keys/client_credentials_1 --auth-supernode-public-key keys/client_credentials_1.pub"
    client_auth_2="--auth-supernode-private-key keys/client_credentials_2 --auth-supernode-public-key keys/client_credentials_2.pub"
    ;;
  *)
    server_address="127.0.0.1:9092"
    server_app_address="127.0.0.1:9091"
    db_arg="--database :flwr-in-memory:"
    server_auth=""
    client_auth_1=""
    client_auth_2=""
    ;;
esac

# These e2e apps are preinstalled by CI; keep the SuperLink harness from creating
# per-run dependency environments.
runtime_dependency_install_arg="--disable-runtime-dependency-installation"

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

background_pids=()
cleanup() {
  local exit_code=$?
  trap - EXIT

  if [ "${#background_pids[@]}" -gt 0 ]; then
    kill "${background_pids[@]}" 2>/dev/null || true
    sleep 1
    kill -9 "${background_pids[@]}" 2>/dev/null || true
    wait "${background_pids[@]}" 2>/dev/null || true
  fi

  exit "$exit_code"
}
trap cleanup EXIT

flower-superlink \
  $server_arg $db_arg $server_auth \
  $runtime_dependency_install_arg &
sl_pid=$!
background_pids+=("$sl_pid")
sleep 3

# Trigger migration
flwr ls "." e2e || true

if [ "$2" = "client-auth" ]; then
  # Register two SuperNodes using the Flower CLI
  flwr supernode register keys/client_credentials_1.pub e2e
  flwr supernode register keys/client_credentials_2.pub e2e
fi

flower-supernode $client_arg \
  --superlink $server_address $client_auth_1 \
  --clientappio-api-address "localhost:9094" \
  --max-retries 0 &
cl1_pid=$!
background_pids+=("$cl1_pid")
sleep 3

flower-supernode $client_arg \
  --superlink $server_address $client_auth_2 \
  --clientappio-api-address "localhost:9096" \
  --max-retries 0 &
cl2_pid=$!
background_pids+=("$cl2_pid")
sleep 3

timeout 1m flwr run "." e2e

training_timeout=240
deadline=$((SECONDS + training_timeout))
status_query_timeout=10

check_process() {
  local pid=$1
  local name=$2
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "$name exited before training completed."
    return 1
  fi
}

while [ "$SECONDS" -lt "$deadline" ]; do
    check_process "$sl_pid" "SuperLink"
    check_process "$cl1_pid" "SuperNode 1"
    check_process "$cl2_pid" "SuperNode 2"

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
        status_details=$(echo "$output" | jq -r '.runs[0]["status-details"] // empty')
        if [ -n "$status_details" ]; then
          echo "Training failed: ${status_details}"
        else
          echo "Training failed with status ${status}:"
          echo "$output"
        fi
        exit 1
        ;;
    esac

    echo "⏳ Not completed yet, retrying in 2s..."
    sleep 2
done

echo "Training did not complete within ${training_timeout} seconds."
exit 1
