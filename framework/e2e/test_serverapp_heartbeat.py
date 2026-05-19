"""Test run heartbeat functionality."""

import json
import os
import subprocess
import sys
import time

from flwr.common.constant import (
    HEARTBEAT_DEFAULT_INTERVAL,
    HEARTBEAT_PATIENCE,
    SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS,
    Status,
    SubStatus,
)

use_sim = sys.argv[1] == "simulation" if len(sys.argv) > 1 else False
superlink_connection = "e2e-sim" if use_sim else "e2e"
plugin_type_arg = "simulation" if use_sim else "serverapp"
app_cmd = "flwr-simulation" if use_sim else "flwr-serverapp"
SUPEREXEC_AUTH_SECRET_FILE = "_e2e_superexec_secret.bin"


def run_superlink() -> subprocess.Popen:
    """Run the SuperLink."""
    cmd = ["flower-superlink", "--insecure"]
    cmd += ["--database", "tmp.db"]
    cmd += ["--isolation", "process"]
    cmd += ["--superexec-auth-secret-file", SUPEREXEC_AUTH_SECRET_FILE]
    if use_sim:
        cmd += ["--simulation"]

    return subprocess.Popen(cmd)


def run_superexec() -> subprocess.Popen:
    """Run the SuperExec."""
    cmd = ["flower-superexec", "--insecure"]
    cmd += ["--appio-api-address", SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS]
    cmd += ["--plugin-type", plugin_type_arg]
    cmd += ["--superexec-auth-secret-file", SUPEREXEC_AUTH_SECRET_FILE]
    return subprocess.Popen(cmd)


def flwr_run() -> str:
    """Run the `flwr run` command and return `run_id`."""
    # Run the command
    result = subprocess.run(
        ["flwr", "run", ".", superlink_connection, "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Parse JSON output and ensure the command succeeded
    data = json.loads(result.stdout)
    assert data["success"], "flwr run failed\n" + str(data)

    # Return the run ID
    return data["run-id"]


def flwr_ls(max_retries: int = 5, retry_delay: float = 0.5) -> dict[str, str]:
    """Run `flwr ls` command and return a mapping of run_id to status.

    Parameters
    ----------
    max_retries : int
        Maximum number of `flwr ls` attempts before failing.
    retry_delay : float
        Delay in seconds between retry attempts.

    Returns
    -------
    dict[str, str]
        A dictionary where keys are run IDs and values are their statuses.
    """
    last_error: str = ""
    for attempt in range(max_retries):
        result = subprocess.run(
            ["flwr", "ls", superlink_connection, "--format", "json"],
            capture_output=True,
            text=True,
            check=False,
        )

        data = json.loads(result.stdout)  # fail immediately on invalid JSON

        if data["success"]:
            return {entry["run-id"]: entry["status"] for entry in data["runs"]}

        last_error = data["error-message"]

        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    raise AssertionError(f"flwr ls failed after retries: {last_error}")


def get_pids(command: str) -> list[int]:
    """Get the PIDs of a running command."""
    result = subprocess.run(
        ["pgrep", "-f", command],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = result.stdout.strip().split("\n")
    return [int(pid) for pid in pids if pid]


def main() -> None:
    """."""
    # Trigger migration to Flower configuration
    subprocess.run(
        ["flwr", "ls"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    with open(SUPEREXEC_AUTH_SECRET_FILE, "wb") as secret_file:
        secret_file.write(b"e2e-superexec-shared-secret")

    # Start the SuperLink
    print("Starting SuperLink...")
    superlink_proc = run_superlink()

    # Allow time for SuperLink to start
    time.sleep(3)

    # Start the SuperExec
    print("Starting SuperExec...")
    superexec_proc = run_superexec()
    time.sleep(1)

    # Submit the first run
    print("Starting the first run...")
    run_id1 = flwr_run()

    # Get the PID of the first app process
    while True:
        if pids := get_pids(app_cmd):
            app_pid = pids[0]
            break
        time.sleep(0.1)

    # Submit the second run
    print("Starting the second run...")
    run_id2 = flwr_run()

    # Wait up to 6 seconds for both runs to reach RUNNING status
    tic = time.time()
    is_running = False
    while (time.time() - tic) < 6:
        run_status = flwr_ls()
        if (
            run_status.get(run_id1) == Status.RUNNING
            and run_status.get(run_id2) == Status.RUNNING
        ):
            is_running = True
            break
        time.sleep(1)
    assert is_running, "Run IDs did not start within 6 seconds"
    print("Both runs are running.")

    # Kill SuperLink process first to simulate restart scenario
    # This prevents ServerApp from notifying SuperLink, isolating the heartbeat test
    print("Terminating SuperLink process...")
    superlink_proc.terminate()
    superlink_proc.wait()

    # Kill the first ServerApp process
    print("Terminating the first ServerApp process...")
    os.kill(app_pid, 9)  # SIGKILL to ensure it stops immediately

    # Restart the SuperLink
    print("Restarting SuperLink...")
    superlink_proc = run_superlink()

    # Allow time for SuperLink to start
    time.sleep(1)

    # Allow enough time for token expiry based heartbeat detection:
    # HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL (+ buffer for restart/retries)
    heartbeat_timeout = HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL + 30

    # Allow time for SuperLink to detect heartbeat failures and update statuses
    tic = time.time()
    is_valid = False
    while (time.time() - tic) < heartbeat_timeout:
        run_status = flwr_ls()
        if (
            run_status[run_id1] == f"{Status.FINISHED}:{SubStatus.FAILED}"
            and run_status[run_id2] == f"{Status.FINISHED}:{SubStatus.COMPLETED}"
        ):
            is_valid = True
            break
        time.sleep(1)
    assert is_valid, f"Run statuses are not updated correctly:\n{run_status}"
    print("Run statuses are updated correctly.")

    # Clean up
    superexec_proc.terminate()
    superexec_proc.wait()
    superlink_proc.terminate()
    superlink_proc.wait()
    os.remove(SUPEREXEC_AUTH_SECRET_FILE)


if __name__ == "__main__":
    main()
