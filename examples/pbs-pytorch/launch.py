"""One Open MPI supervisor per PBS node; Flower communicates independently of MPI."""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def wait_for(predicate, description, processes, seconds=180):
    """Bound startup waits and fail when a supervised service exits."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Service exited with {process.returncode}: {process.args}"
                )
        if predicate():
            return
        time.sleep(2)
    raise TimeoutError(description)


def port_ready(host, port):
    """Check connectivity without treating a scheduler RUNNING state as readiness."""
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def check_ray_nodes(ray, output):
    """Refresh retained Ray liveness evidence and require four live nodes."""
    ray.init(address=os.environ["RAY_ADDRESS"])
    try:
        nodes = [node for node in ray.nodes() if node["Alive"]]
        (output / "ray-nodes.json").write_text(
            json.dumps(nodes, indent=2), encoding="utf-8"
        )
        if len(nodes) != 4:
            raise RuntimeError(f"Expected four live Ray nodes, got {len(nodes)}")
    finally:
        ray.shutdown()


def main():
    """Launch a real SuperNode or a Ray service according to MPI rank and mode."""
    import flwr
    import ray
    import torch
    import torchvision

    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    mode = sys.argv[1]
    if mode not in ("deployment", "simulation"):
        raise ValueError("Mode must be deployment or simulation")
    output = Path(os.environ["OUTPUT_ROOT"])
    python = Path(sys.executable)
    binaries = python.parent
    master = socket.gethostbyname(os.environ["MASTER_ADDR"])
    local_ip = socket.gethostbyname(socket.gethostname())
    fleet_port = int(os.environ["FLEET_PORT"])
    ray_port = int(os.environ["RAY_PORT"])
    app_root = Path(__file__).absolute().parent
    processes, logs = [], []
    ray_started = False

    def start(command, name):
        log = (output / f"rank-{rank}-{name}.log").open("w", encoding="utf-8")
        logs.append(log)
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        processes.append(process)
        return process

    def terminate(_signum, _frame):
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    with tempfile.TemporaryDirectory(
        prefix=f"flower-{os.environ['PBS_JOBID']}-{rank}-"
    ) as temporary:
        os.environ["FLWR_HOME"] = temporary
        Path(temporary, "config.toml").write_text(
            '[superlink]\ndefault = "pbs"\n[superlink.pbs]\naddress = "127.0.0.1:8000"\ninsecure = true\n',
            encoding="utf-8",
        )
        placement = {
            "rank": rank,
            "host": socket.gethostname(),
            "ip": local_ip,
            "job_id": os.environ["PBS_JOBID"],
            "python": sys.executable,
            "prefix": sys.prefix,
            "flwr": flwr.__version__,
            "flwr_source": flwr.__file__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "ray": ray.__version__,
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
        if rank:
            if not torch.cuda.is_available():
                raise RuntimeError("Follower has no usable CUDA device")
            placement["gpu"] = torch.cuda.get_device_name(0)
        (output / f"placement-{rank}.json").write_text(
            json.dumps(placement, indent=2), encoding="utf-8"
        )
        print(json.dumps(placement), flush=True)
        try:
            if mode == "simulation":
                # These scripts request entire nodes, so all Ray services here belong to this job.
                ray_args = [
                    str(binaries / "ray"),
                    "start",
                    f"--node-ip-address={local_ip}",
                    "--disable-usage-stats",
                ]
                if rank == 0:
                    ray_args += [
                        "--head",
                        f"--port={ray_port}",
                        "--num-cpus=0",
                        "--num-gpus=0",
                        "--include-dashboard=false",
                    ]
                else:
                    wait_for(
                        lambda: (output / "head-ready").exists(),
                        "Ray head did not start",
                        processes,
                    )
                    ray_args += [
                        f"--address={master}:{ray_port}",
                        "--num-cpus=8",
                        "--num-gpus=1",
                    ]
                ray_started = True
                with (output / f"rank-{rank}-ray.log").open(
                    "w", encoding="utf-8"
                ) as log:
                    subprocess.run(
                        ray_args,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=180,
                    )
                if rank == 0:
                    os.environ["RAY_ADDRESS"] = f"{master}:{ray_port}"
                    (output / "head-ready").touch()

            if rank == 0:
                command = [
                    str(binaries / "flower-superlink"),
                    "--insecure",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                    "--disable-runtime-dependency-installation",
                ]
                if mode == "simulation":
                    command.append("--simulation")
                else:
                    command += ["--fleet-api-address", f"{master}:{fleet_port}"]
                start(command, "superlink")
                wait_for(
                    lambda: port_ready("127.0.0.1", 8000),
                    "SuperLink HTTP API did not start",
                    processes,
                )
            elif mode == "deployment":
                wait_for(
                    lambda: port_ready(master, fleet_port),
                    "Fleet API did not start",
                    processes,
                )
                start(
                    [
                        str(binaries / "flower-supernode"),
                        "--insecure",
                        "--superlink",
                        f"{master}:{fleet_port}",
                        "--node-config",
                        f"partition-id={rank - 1} num-partitions=3",
                    ],
                    "supernode",
                )

            (output / f"ready-{rank}").touch()
            if rank:
                wait_for(
                    lambda: (output / "finished").exists(),
                    "Master did not complete",
                    processes,
                    seconds=1500,
                )
                return

            wait_for(
                lambda: all((output / f"ready-{i}").exists() for i in range(4)),
                "Followers did not start",
                processes,
            )
            if mode == "simulation":
                check_ray_nodes(ray, output)

            count = 3 if mode == "deployment" else 10
            command = [
                str(binaries / "flwr"),
                "run",
                str(app_root),
                "--run-config",
                f"num-partitions={count}",
                "--format",
                "json",
            ]
            if mode == "simulation":
                command += [
                    "--federation-config",
                    "num-supernodes=10 client-resources-num-cpus=2 client-resources-num-gpus=0.25",
                ]
            submitted = subprocess.run(
                command, text=True, capture_output=True, check=True, timeout=120
            )
            (output / "submit.json").write_text(submitted.stdout, encoding="utf-8")
            (output / "submit.log").write_text(submitted.stderr, encoding="utf-8")
            run_id = json.loads(submitted.stdout)["run-id"]

            def completed():
                response = subprocess.run(
                    [
                        str(binaries / "flwr"),
                        "list",
                        "--run-id",
                        str(run_id),
                        "--format",
                        "json",
                    ],
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=60,
                )
                status = json.loads(response.stdout)
                (output / "status.json").write_text(
                    json.dumps(status, indent=2), encoding="utf-8"
                )
                state = status["runs"][0]["status"].lower()
                if state.startswith("finished") and state != "finished:completed":
                    raise RuntimeError(f"Flower run failed: {status}")
                return state == "finished:completed"

            wait_for(completed, "Flower run did not complete", processes, seconds=1350)
            if mode == "simulation":
                check_ray_nodes(ray, output)
            subprocess.run(
                [str(python), str(app_root / "verify.py"), str(output), mode],
                check=True,
            )
            (output / "finished").touch()
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process in reversed(processes):
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if ray_started:
                with (output / f"rank-{rank}-ray-stop.log").open(
                    "w", encoding="utf-8"
                ) as log:
                    subprocess.run(
                        [str(binaries / "ray"), "stop", "--force"],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=60,
                    )
            for log in logs:
                log.close()


if __name__ == "__main__":
    main()
