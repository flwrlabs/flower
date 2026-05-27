"""NodeApp definition for the PyTorch simulation example."""

from __future__ import annotations

from typing import Any

from flwr.common import Context, Message
from flwr.common.constant import NUM_PARTITIONS_KEY, PARTITION_ID_KEY
from flwr.common.record.arrayrecord import ArrayRecord
from flwr.common.record.configrecord import ConfigRecord
from flwr.common.record.metricrecord import MetricRecord
from flwr.common.record.recorddict import RecordDict
from flwr.decentralized.common.run_config import DLRunConfig
from flwr.decentralized.nodeapp import NodeApp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

INPUT_DIM = 20
TRAIN_SAMPLES_PER_NODE = 1024
TEST_SAMPLES_PER_NODE = 256


class TinyNet(nn.Module):
    """Small MLP used for local training in simulation."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.layers(x)


def _make_partition_dataset(
    partition_id: int,
    num_partitions: int,
) -> tuple[TensorDataset, TensorDataset]:
    """Create deterministic synthetic binary-classification data for one partition."""
    generator = torch.Generator().manual_seed(2026 + partition_id)

    total_samples = TRAIN_SAMPLES_PER_NODE + TEST_SAMPLES_PER_NODE
    features = torch.randn(total_samples, INPUT_DIM, generator=generator)

    base_weights = torch.linspace(-1.0, 1.0, INPUT_DIM)
    partition_shift = (partition_id / max(num_partitions, 1)) * 0.5
    logits = features @ base_weights + partition_shift
    logits += 0.15 * torch.randn(total_samples, generator=generator)
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


def _get_or_create_state(context: Context) -> tuple[dict[str, Any], int]:
    partition_id, num_partitions, node_id = _partition_info(context)
    batch_size = int(context.run_config.get("batch-size", 32))

    state = _NODE_STATE.get(node_id)
    if state is None:
        train_ds, test_ds = _make_partition_dataset(partition_id, num_partitions)
        state = {
            "model": TinyNet(),
            "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            "test_loader": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            "round": 0,
        }
        _NODE_STATE[node_id] = state

    return state, batch_size


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
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

    n_steps = max(local_epochs * len(train_loader), 1)
    return total_loss / n_steps


def _evaluate(model: nn.Module, test_loader: DataLoader) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += float(loss.item())
            predictions = outputs.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.shape[0]

    avg_loss = total_loss / max(len(test_loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


_DEFAULT_DATA_CONFIG = ConfigRecord(
    {
        PARTITION_ID_KEY: 0,
        NUM_PARTITIONS_KEY: 1,
    }
)

_INITIAL_MODEL = TinyNet()

RUN_CONFIG = DLRunConfig(
    rounds=8,
    n_aggregation_steps=2,
)

app = NodeApp(
    subject="trainer-pytorch-sim",
    initial_arrays=ArrayRecord(_INITIAL_MODEL.state_dict()),
    data_config=ConfigRecord(_DEFAULT_DATA_CONFIG),
    run_config=RUN_CONFIG,
    train_config=ConfigRecord({"local-epochs": 1, "lr": 0.05, "batch-size": 32}),
    eval_config=ConfigRecord({"batch-size": 64}),
    timeout=2,
)


@app.train()
def train(message: Message, context: Context) -> Message:
    """Train one local round and return updated arrays + train metrics."""
    state, _ = _get_or_create_state(context)
    model: TinyNet = state["model"]
    train_loader: DataLoader = state["train_loader"]

    local_epochs = int(context.run_config.get("local-epochs", 1))
    learning_rate = float(context.run_config.get("lr", 0.05))

    train_loss = _train_one_round(
        model=model,
        train_loader=train_loader,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
    )

    state["round"] += 1

    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(
                {
                    "train_loss": train_loss,
                    "num-examples": len(train_loader.dataset),
                    "round": state["round"],
                }
            ),
        }
    )
    return Message(content=content, reply_to=message)


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    """Evaluate local model and return evaluation metrics."""
    state, _ = _get_or_create_state(context)
    model: TinyNet = state["model"]
    test_loader: DataLoader = state["test_loader"]

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
    return Message(content=content, reply_to=message)
