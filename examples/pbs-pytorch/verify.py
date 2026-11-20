"""Verify complete training, evaluation and physical placement from retained evidence."""

import json
import math
import sys
from pathlib import Path


def verify(output, mode):
    """Refuse partial rounds, CPU fallback, missing clients or unused follower hosts."""
    result = json.loads((output / "result.json").read_text())
    placements = [
        json.loads((output / f"placement-{rank}.json").read_text()) for rank in range(4)
    ]
    hosts = {placement["host"] for placement in placements}
    if len(hosts) != 4 or result["host"] != placements[0]["host"]:
        raise ValueError("Expected one master and three distinct follower hosts")
    count = 3 if mode == "deployment" else 10
    if result["num_partitions"] != count or result["rounds"] != 3:
        raise ValueError("Expected the full cohort in three rounds")
    if result["initial_sha256"] == result["final_sha256"]:
        raise ValueError("Global model did not change")
    status = json.loads((output / "status.json").read_text())
    if status["runs"][0]["status"].lower() != "finished:completed":
        raise ValueError("Flower did not report successful completion")
    events = [
        json.loads(path.read_text()) for path in (output / "events").glob("*.json")
    ]
    followers = {placement["host"] for placement in placements[1:]}
    for phase, size in (("train", 50000), ("evaluate", 10000)):
        for server_round in range(1, 4):
            selected = [
                event
                for event in events
                if event["phase"] == phase and event["round"] == server_round
            ]
            if len(selected) != count or {
                event["partition"] for event in selected
            } != set(range(count)):
                raise ValueError(
                    f"Missing or duplicate clients: {phase} round {server_round}"
                )
            if {event["host"] for event in selected} != followers:
                raise ValueError(
                    f"Not all follower hosts participated: {phase} round {server_round}"
                )
            if sum(event["partition_size"] for event in selected) != size:
                raise ValueError("The full CIFAR-10 split was not processed")
            for event in selected:
                epochs = (
                    int(result["config"]["local-epochs"]) if phase == "train" else 1
                )
                batch_size = int(result["config"]["batch-size"])
                if (
                    not event["gpu"]
                    or event["examples_processed"] != event["partition_size"] * epochs
                    or event["steps"]
                    != math.ceil(event["partition_size"] / batch_size) * epochs
                    or not math.isfinite(event["loss"])
                    or not 0 <= event["accuracy"] <= 1
                ):
                    raise ValueError(f"Invalid training/evaluation evidence: {event}")
                if (
                    mode == "deployment"
                    and event["host"] != placements[event["partition"] + 1]["host"]
                ):
                    raise ValueError("Real SuperNode moved between physical hosts")
    if mode == "simulation":
        nodes = json.loads((output / "ray-nodes.json").read_text())
        if len(nodes) != 4 or not all(node["Alive"] for node in nodes):
            raise ValueError("Ray did not use four live physical nodes")
    summary = {
        "verified": True,
        "mode": mode,
        "hosts": sorted(hosts),
        "clients": count,
        "rounds": 3,
        "events": len(events),
        "train_examples_per_round": 50000,
        "test_examples_per_round": 10000,
    }
    (output / "verification.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    verify(Path(sys.argv[1]), sys.argv[2])
