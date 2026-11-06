"""Regressions for false-positive multi-node validation reports."""

import json
import math
from unittest.mock import Mock

import pytest

from launch import check_ray_nodes
from verify import verify


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def simulation(tmp_path):
    """Create one complete run, with all ten clients spread across three hosts."""
    hosts = ["master", "worker1", "worker2", "worker3"]
    result = {
        "host": hosts[0],
        "num_partitions": 10,
        "rounds": 3,
        "initial_sha256": "initial",
        "final_sha256": "trained",
        "config": {"batch-size": 64, "local-epochs": 1},
    }
    write_json(tmp_path / "result.json", result)
    write_json(tmp_path / "status.json", {"runs": [{"status": "finished:completed"}]})
    write_json(tmp_path / "ray-nodes.json", [{"Alive": True} for _ in hosts])
    for rank, host in enumerate(hosts):
        write_json(tmp_path / f"placement-{rank}.json", {"host": host})
    events = tmp_path / "events"
    events.mkdir()
    for phase, size in (("train", 5000), ("evaluate", 1000)):
        for server_round in range(1, 4):
            for partition in range(10):
                event = {
                    "phase": phase,
                    "round": server_round,
                    "partition": partition,
                    "host": hosts[1 + partition % 3],
                    "partition_size": size,
                    "examples_processed": size,
                    "steps": math.ceil(size / 64),
                    "gpu": "GPU",
                    "loss": 1.0,
                    "accuracy": 0.25,
                }
                write_json(events / f"{phase}-{server_round}-{partition}.json", event)
    return tmp_path


def test_complete_simulation(simulation):
    verify(simulation, "simulation")
    assert json.loads((simulation / "verification.json").read_text())["events"] == 60


def test_rejects_clients_all_on_one_host(simulation):
    for path in (simulation / "events").glob("*.json"):
        event = json.loads(path.read_text())
        event["host"] = "worker1"
        write_json(path, event)
    with pytest.raises(ValueError, match="Not all follower hosts"):
        verify(simulation, "simulation")


def test_rejects_incomplete_local_epoch(simulation):
    path = simulation / "events/train-1-0.json"
    event = json.loads(path.read_text())
    event["examples_processed"] = 64
    event["steps"] = 1
    write_json(path, event)
    with pytest.raises(ValueError, match="Invalid training/evaluation evidence"):
        verify(simulation, "simulation")


def test_rejects_failed_flower_status(simulation):
    write_json(simulation / "status.json", {"runs": [{"status": "finished:failed"}]})
    with pytest.raises(ValueError, match="successful completion"):
        verify(simulation, "simulation")


def test_rejects_duplicate_client(simulation):
    path = simulation / "events/train-1-0.json"
    event = json.loads(path.read_text())
    event["partition"] = 1
    write_json(path, event)
    with pytest.raises(ValueError, match="Missing or duplicate clients"):
        verify(simulation, "simulation")


def test_rejects_ray_worker_lost_after_startup(simulation, monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "ray-head:6379")
    ray = Mock()
    ray.nodes.side_effect = [
        [{"Alive": True} for _ in range(4)],
        [{"Alive": True} for _ in range(3)] + [{"Alive": False}],
    ]
    check_ray_nodes(ray, simulation)
    with pytest.raises(RuntimeError, match="Expected four live Ray nodes, got 3"):
        check_ray_nodes(ray, simulation)
    with pytest.raises(ValueError, match="Ray did not use four live physical nodes"):
        verify(simulation, "simulation")
    assert ray.shutdown.call_count == 2
