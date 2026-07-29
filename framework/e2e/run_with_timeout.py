"""Run a command with a timeout and terminate its process tree."""

import os
import signal
import subprocess
import sys


def terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate a process and all processes that it spawned."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/PID", str(process.pid), "/T"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        os.killpg(process.pid, signal.SIGKILL)


def main() -> int:
    """Run the requested command, returning 124 if it times out."""
    if len(sys.argv) < 3:
        print(
            f"Usage: {sys.argv[0]} TIMEOUT_SECONDS COMMAND [ARG ...]",
            file=sys.stderr,
        )
        return 2

    timeout = float(sys.argv[1])
    command = sys.argv[2:]
    process = subprocess.Popen(
        command,
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            if os.name == "nt"
            else 0
        ),
        start_new_session=os.name != "nt",
    )

    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_process_tree(process)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print(
            f"Command timed out after {timeout:g} seconds: {' '.join(command)}",
            file=sys.stderr,
        )
        return 124


if __name__ == "__main__":
    sys.exit(main())
