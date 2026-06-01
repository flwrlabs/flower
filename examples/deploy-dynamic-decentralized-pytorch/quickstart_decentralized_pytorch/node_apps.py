"""Concrete PyTorch NodeApp used by deploy and simulation scenarios."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from flwr.app.message_type import MessageType
from flwr.common import Context, Message
from flwr.common.constant import NUM_PARTITIONS_KEY, PARTITION_ID_KEY
from flwr.app.message.arrayrecord import ArrayRecord
from flwr.app.message.configrecord import ConfigRecord
from flwr.app.message.metricrecord import MetricRecord
from flwr.app.message.recorddict import RecordDict
from flwr.decentralized.common.run_config import DLRunConfig
from flwr.decentralized.nodeapp import NodeApp
from torch.utils.data import DataLoader, TensorDataset

INPUT_DIM = 16
TRAIN_SAMPLES_PER_NODE = 512
TEST_SAMPLES_PER_NODE = 128


class TinyNet(nn.Module):
    """Small MLP used for local synthetic training."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def _make_partition_dataset(
    partition_id: int,
    num_partitions: int,
) -> tuple[TensorDataset, TensorDataset]:
    generator = torch.Generator().manual_seed(9000 + partition_id)
    total_samples = TRAIN_SAMPLES_PER_NODE + TEST_SAMPLES_PER_NODE
    features = torch.randn(total_samples, INPUT_DIM, generator=generator)

    base = torch.linspace(-0.8, 0.8, INPUT_DIM)
    shift = (partition_id / max(num_partitions, 1)) * 0.4
    logits = features @ base + shift
    logits += 0.2 * torch.randn(total_samples, generator=generator)
    labels = (logits > 0.0).long()

    train_x = features[:TRAIN_SAMPLES_PER_NODE]
    train_y = labels[:TRAIN_SAMPLES_PER_NODE]
    test_x = features[TRAIN_SAMPLES_PER_NODE:]
    test_y = labels[TRAIN_SAMPLES_PER_NODE:]
    return TensorDataset(train_x, train_y), TensorDataset(test_x, test_y)


def _partition_info(context: Context) -> tuple[int, int, str]:
    partition_id = int(context.node_config.get(PARTITION_ID_KEY, 0))
    num_partitions = int(context.node_config.get(NUM_PARTITIONS_KEY, 1))
    node_id = str(context.node_config.get("node-id", partition_id))
    return partition_id, num_partitions, node_id


_NODE_STATE: dict[str, dict[str, Any]] = {}


def _get_or_create_state(context: Context) -> dict[str, Any]:
    partition_id, num_partitions, node_id = _partition_info(context)
    batch_size = int(context.run_config.get("batch-size", 32))

    if node_id not in _NODE_STATE:
        train_ds, test_ds = _make_partition_dataset(partition_id, num_partitions)
        _NODE_STATE[node_id] = {
            "model": TinyNet(),
            "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            "test_loader": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            "round": 0,
        }

    return _NODE_STATE[node_id]


def _train_one_round(
    model: nn.Module,
    train_loader: DataLoader,
    local_epochs: int,
    learning_rate: float,
) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train()

    total_loss = 0.0
    for _ in range(local_epochs):
        for features, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

    return total_loss / max(local_epochs * len(train_loader), 1)


def _evaluate(model: nn.Module, test_loader: DataLoader) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            total_loss += float(criterion(outputs, labels).item())
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total += labels.shape[0]

    return total_loss / max(len(test_loader), 1), correct / max(total, 1)


app = NodeApp(
    subject="trainer-pytorch-modes",
    initial_arrays=ArrayRecord(TinyNet().state_dict()),
    data_config=ConfigRecord({PARTITION_ID_KEY: 0, NUM_PARTITIONS_KEY: 1}),
    run_config=DLRunConfig(rounds=5, n_aggregation_steps=2),
    train_config=ConfigRecord({"local-epochs": 1, "lr": 0.05, "batch-size": 32}),
    eval_config=ConfigRecord({"batch-size": 64}),
    timeout=2,
)


@app.train()
def train(message: Message, context: Context) -> Message:
    """Train for one local round and return arrays + metrics."""
    state = _get_or_create_state(context)
    model: TinyNet = state["model"]
    train_loader: DataLoader = state["train_loader"]

    incoming_arrays = message.content.array_records.get(app.strategy.arrayrecord_key)
    if incoming_arrays is not None:
        model.load_state_dict(incoming_arrays.to_torch_state_dict())

    local_epochs = int(context.run_config.get("local-epochs", 1))
    learning_rate = float(context.run_config.get("lr", 0.05))
    loss = _train_one_round(model, train_loader, local_epochs, learning_rate)

    state["round"] += 1
    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(
                {
                    "train_loss": loss,
                    "round": state["round"],
                    "num-examples": len(train_loader.dataset),
                }
            ),
        }
    )

    return Message(
        content=content,
        message_type=MessageType.TRAIN,
        dst_node_id=0,
        ttl=2,
        group_id="trainer-pytorch-modes",
    )


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    """Evaluate local model and return metrics."""
    state = _get_or_create_state(context)
    model: TinyNet = state["model"]
    test_loader: DataLoader = state["test_loader"]

    incoming_arrays = message.content.array_records.get(app.strategy.arrayrecord_key)
    if incoming_arrays is not None:
        model.load_state_dict(incoming_arrays.to_torch_state_dict())

    eval_loss, eval_acc = _evaluate(model, test_loader)
    content = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "num-examples": len(test_loader.dataset),
                }
            )
        }
    )

    return Message(
        content=content,
        message_type=MessageType.EVALUATE,
        dst_node_id=0,
        ttl=2,
        group_id="trainer-pytorch-modes",
    )
