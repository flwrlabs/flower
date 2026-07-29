"""Test run heartbeat functionality."""

import json
import os
import subprocess
import sys
import tempfile
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
COMMAND_TIMEOUT = 30
PROCESS_START_TIMEOUT = 15
PROCESS_STOP_TIMEOUT = 10


def run_superlink(database_path: str, secret_path: str) -> subprocess.Popen:
    """Run the SuperLink."""
    cmd = ["flower-superlink", "--insecure"]
    cmd += ["--database", database_path]
    cmd += ["--isolation", "process"]
    cmd += ["--superexec-auth-secret-file", secret_path]
    if use_sim:
        cmd += ["--simulation"]

    return subprocess.Popen(cmd)


def run_superexec(secret_path: str) -> subprocess.Popen:
    """Run the SuperExec."""
    cmd = ["flower-superexec", "--insecure"]
    cmd += ["--appio-api-address", SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS]
    cmd += ["--plugin-type", plugin_type_arg]
    cmd += ["--superexec-auth-secret-file", secret_path]
    return subprocess.Popen(cmd)


def flwr_run() -> str:
    """Run the `flwr run` command and return `run_id`."""
    # Run the command
    result = subprocess.run(
        ["flwr", "run", ".", superlink_connection, "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
        timeout=COMMAND_TIMEOUT,
    )

    # Parse JSON output and ensure the command succeeded
    data = json.loads(result.stdout)
    assert data["success"], "flwr run failed\n" + str(data)

    # Return the run ID
    return data["run-id"]


def flwr_ls(
    deadline: float, max_retries: int = 5, retry_delay: float = 0.5
) -> dict[str, str]:
    """Run `flwr ls` command and return a mapping of run_id to status.

    Parameters
    ----------
    deadline : float
        Monotonic deadline shared by all attempts.
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
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError("flwr ls exceeded its polling deadline")

        result = subprocess.run(
            ["flwr", "ls", superlink_connection, "--format", "json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=min(COMMAND_TIMEOUT, remaining),
        )

        data = json.loads(result.stdout)  # fail immediately on invalid JSON

        if data["success"]:
            return {entry["run-id"]: entry["status"] for entry in data["runs"]}

        last_error = data["error-message"]

        if attempt < max_retries - 1:
            time.sleep(min(retry_delay, max(0.0, deadline - time.monotonic())))

    raise AssertionError(f"flwr ls failed after retries: {last_error}")


def get_pids(command: str, timeout: float) -> list[int]:
    """Get the PIDs of a running command."""
    result = subprocess.run(
        ["pgrep", "-f", command],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    pids = result.stdout.strip().split("\n")
    return [int(pid) for pid in pids if pid]


def wait_for_pid(
    command: str, superlink_proc: subprocess.Popen, superexec_proc: subprocess.Popen
) -> int:
    """Wait for a process matching the command and return its PID."""
    deadline = time.monotonic() + PROCESS_START_TIMEOUT
    while time.monotonic() < deadline:
        if superlink_proc.poll() is not None:
            raise AssertionError("SuperLink exited before the app process started")
        if superexec_proc.poll() is not None:
            raise AssertionError("SuperExec exited before the app process started")

        remaining = deadline - time.monotonic()
        try:
            pids = get_pids(command, remaining)
        except subprocess.TimeoutExpired:
            continue
        if pids:
            return pids[0]
        time.sleep(0.1)
    raise AssertionError(
        f"{command} did not start within {PROCESS_START_TIMEOUT} seconds"
    )


def stop_process(process: subprocess.Popen | None) -> None:
    """Terminate a process without waiting indefinitely."""
    if process is None or process.poll() is not None:
        return

    try:
        process.terminate()
    except ProcessLookupError:
        process.wait(timeout=PROCESS_STOP_TIMEOUT)
        return
    try:
        process.wait(timeout=PROCESS_STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            process.wait(timeout=PROCESS_STOP_TIMEOUT)
            return
        try:
            process.wait(timeout=PROCESS_STOP_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise AssertionError("Process did not stop after SIGKILL") from exc


def main() -> None:
    """Test heartbeat handling across a SuperLink restart."""
    with tempfile.TemporaryDirectory(prefix="flwr-e2e-heartbeat-") as temp_dir:
        database_path = os.path.join(temp_dir, "tmp.db")
        secret_path = os.path.join(temp_dir, SUPEREXEC_AUTH_SECRET_FILE)
        superlink_proc: subprocess.Popen | None = None
        superexec_proc: subprocess.Popen | None = None

        try:
            # Trigger migration to Flower configuration
            subprocess.run(
                ["flwr", "ls"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=COMMAND_TIMEOUT,
            )

            with open(secret_path, "wb") as secret_file:
                secret_file.write(b"e2e-superexec-shared-secret")

            # Start the SuperLink
            print("Starting SuperLink...")
            superlink_proc = run_superlink(database_path, secret_path)

            # Allow time for SuperLink to start
            time.sleep(3)

            # Start the SuperExec
            print("Starting SuperExec...")
            superexec_proc = run_superexec(secret_path)
            time.sleep(1)

            # Submit the first run
            print("Starting the first run...")
            run_id1 = flwr_run()

            # Get the PID of the first app process
            app_pid = wait_for_pid(app_cmd, superlink_proc, superexec_proc)

            # Submit the second run
            print("Starting the second run...")
            run_id2 = flwr_run()

            # Wait up to 6 seconds for both runs to reach RUNNING status
            running_deadline = time.monotonic() + 6
            is_running = False
            while time.monotonic() < running_deadline:
                run_status = flwr_ls(running_deadline)
                if (
                    run_status.get(run_id1) == Status.RUNNING
                    and run_status.get(run_id2) == Status.RUNNING
                ):
                    is_running = True
                    break
                time.sleep(min(1.0, max(0.0, running_deadline - time.monotonic())))
            assert is_running, "Run IDs did not start within 6 seconds"
            print("Both runs are running.")

            # Kill SuperLink process first to simulate restart scenario
            # Prevent ServerApp from notifying SuperLink, isolating the
            # heartbeat test.
            print("Terminating SuperLink process...")
            stop_process(superlink_proc)

            # Kill the first ServerApp process
            print("Terminating the first ServerApp process...")
            try:
                os.kill(app_pid, 9)  # SIGKILL to ensure it stops immediately
            except ProcessLookupError:
                pass

            # Restart the SuperLink
            print("Restarting SuperLink...")
            superlink_proc = run_superlink(database_path, secret_path)

            # Allow time for SuperLink to start
            time.sleep(1)

            # Allow enough time for token expiry based heartbeat detection:
            # HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL (+ buffer for retries)
            heartbeat_timeout = HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL + 30

            # Allow time for SuperLink to detect heartbeat failures and update statuses
            heartbeat_deadline = time.monotonic() + heartbeat_timeout
            is_valid = False
            while time.monotonic() < heartbeat_deadline:
                run_status = flwr_ls(heartbeat_deadline)
                if (
                    run_status[run_id1] == f"{Status.FINISHED}:{SubStatus.FAILED}"
                    and run_status[run_id2]
                    == f"{Status.FINISHED}:{SubStatus.COMPLETED}"
                ):
                    is_valid = True
                    break
                time.sleep(min(1.0, max(0.0, heartbeat_deadline - time.monotonic())))
            assert is_valid, f"Run statuses are not updated correctly:\n{run_status}"
            print("Run statuses are updated correctly.")
        finally:
            stop_process(superexec_proc)
            stop_process(superlink_proc)


if __name__ == "__main__":
    main()
