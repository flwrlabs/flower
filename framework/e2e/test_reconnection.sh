#!/bin/bash
set -e

case "$1" in
  rest)
    rest_arg="--rest"
    server_app_address="http://localhost:9091"
    server_address="http://localhost:9093"
    db_arg="--database :flwr-in-memory:"
    ;;
  sqlite)
    rest_arg=""
    server_address="127.0.0.1:9092"
    server_app_address="127.0.0.1:9091"
    db_arg="--database $(date +%s).db"
    ;;
  *)
    rest_arg=""
    server_address="127.0.0.1:9092"
    server_app_address="127.0.0.1:9091"
    db_arg="--database :flwr-in-memory:"
    ;;
esac

# Define the function
check_and_kill() {
  local pids=$1  # Get the PID as the first argument to the function
  for pid in $pids; do
    echo "Attempting to kill process ID: $pid"
    if kill "$pid" 2>/dev/null; then
        echo "Process $pid successfully killed."
    else
        echo "Failed to kill process $pid or it may have already terminated."
    fi
  done
}

# Remove any duplicates
sed -i '/^\[tool\.flwr\.federations\.e2e\]/,/^$/d' pyproject.toml

# Append the federations config to pyproject.toml
echo -e $"\n[tool.flwr.federations.e2e]\naddress = \"127.0.0.1:9093\"\ninsecure = true" >> pyproject.toml
sleep 1

timeout 10m flower-superlink --insecure $db_arg $rest_arg &
sl_pids=$(pgrep -f "flower-superlink")
echo "Starting SuperLink"
sleep 3

timeout 10m flower-supernode --insecure $rest_arg --superlink $server_address \
  --clientappio-api-address="localhost:9094" &
cl1_pid=$!
echo "Starting first client"
sleep 3

timeout 10m flower-supernode --insecure $rest_arg --superlink $server_address \
  --clientappio-api-address="localhost:9095" &
cl2_pid=$!
echo "Starting second client"
sleep 3

# Kill superlink, this should send the clients into their retry loops
check_and_kill "$sl_pids"
echo "Killing Superlink"
sleep 3

# Restart superlink, the clients should now be able to reconnect to it
timeout 10m flower-superlink --insecure $db_arg $rest_arg &
sl_pids=$(pgrep -f "flower-superlink")
echo "Restarting Superlink"
sleep 20

# Kill second client, this should send a DeleteNode message to the Superlink
kill $cl1_pid
echo "Killing second client"
sleep 5

# Starting new client, this is so we have enough clients to execute `flwr run`
timeout 10m flower-supernode --insecure $rest_arg --superlink $server_address \
  --clientappio-api-address "localhost:9094" &
cl1_pid=$!
echo "Starting new client"
sleep 5

# We execute `flwr run` to begin the training
timeout 2m flwr run "." e2e &
echo "Executing flwr run to start training"
sleep 8

# Kill first client as soon as the training starts, the flwr-serverapp should just 
# receive a failure in this case and continue the rounds when enough clients are 
# connected
kill $cl1_pid
echo "Killing first client"
sleep 3

# Restart first client so enough clients are connected to continue the FL rounds
timeout 5m flower-supernode --insecure $rest_arg --superlink $server_address \
  --clientappio-api-address "localhost:9094" &
cl1_pid=$!
echo "Starting new client"
sleep 5

# Initialize a flag to track if training is successful
found_success=false
timeout=120  # Timeout after 120 seconds
elapsed=0

# Define a cleanup function
cleanup_and_exit() {
    kill $cl1_pid; kill $cl2_pid
    sleep 2  # Allow some time for SuperNodes to terminate
    check_and_kill "$sl_pids"
    sleep 2  # Allow some time for SuperLink to terminate
    exit $1
}

# Check for "finished:completed" status in a loop with a timeout
while [ "$found_success" = false ] && [ $elapsed -lt $timeout ]; do
    # Run the command and capture output
    output=$(flwr ls . e2e --format=json)

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
