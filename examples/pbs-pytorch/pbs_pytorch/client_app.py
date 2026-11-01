"""The same GPU ClientApp runs on real SuperNodes and simulated SuperNodes."""

import json
import os
import socket
import time
from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pbs_pytorch.task import execute, load_data, make_model

app = ClientApp()


def process(msg: Message, context: Context, train: bool) -> Message:
    """Execute a complete partition and retain evidence of its physical placement."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This GPU example requires a CUDA-enabled PyTorch environment"
        )
    torch.set_num_threads(2)
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.run_config["num-partitions"])
    if "num-partitions" in context.node_config:
        if int(context.node_config["num-partitions"]) != num_partitions:
            raise ValueError("SuperNode and ServerApp partition counts differ")
    server_round = int(msg.content["config"]["server-round"])
    torch.manual_seed(42 + partition_id + server_round * num_partitions)
    model = make_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    loader = load_data(
        partition_id, num_partitions, int(context.run_config["batch-size"]), train
    )
    epochs = int(context.run_config["local-epochs"]) if train else 0
    loss, accuracy, examples, steps = execute(
        model,
        loader,
        "cuda:0",
        epochs,
        float(msg.content["config"].get("lr", context.run_config["learning-rate"])),
    )
    phase = "train" if train else "evaluate"
    evidence = {
        "phase": phase,
        "round": server_round,
        "partition": partition_id,
        "node_id": context.node_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "partition_size": len(loader.dataset),
        "examples_processed": examples,
        "steps": steps,
        "epochs": epochs,
        "loss": loss,
        "accuracy": accuracy,
    }
    path = Path(os.environ["OUTPUT_ROOT"]) / "events"
    path.mkdir(exist_ok=True)
    (path / f"{phase}-{server_round}-{partition_id}-{time.time_ns()}.json").write_text(
        json.dumps(evidence, indent=2),
        encoding="utf-8",
    )
    metrics = MetricRecord(
        {"num-examples": len(loader.dataset), "loss": loss, "accuracy": accuracy}
    )
    content = RecordDict({"metrics": metrics})
    if train:
        content["arrays"] = ArrayRecord(model.cpu().state_dict())
    print(json.dumps(evidence), flush=True)
    return Message(content=content, reply_to=msg)


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train a ResNet-18 on one CIFAR-10 partition."""
    return process(msg, context, True)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate global weights on one held-out CIFAR-10 test partition."""
    return process(msg, context, False)
