"""User-defined NodeApp instances (app1/app2) using PyTorch and partitioned data."""

from __future__ import annotations

import logging
from typing import Any

from flwr.client.mod.comms_mods import message_size_mod
from flwr.common.record.metricrecord import MetricRecord
from flwr.common.record.recorddict import RecordDict
from flwr.decentralized.common.run_config import DLRunConfig
import torch
import torch.nn as nn
import torch.nn.functional as func
from datasets import load_dataset
from flwr.common import Context, Message
from flwr.common.constant import NUM_PARTITIONS_KEY, PARTITION_ID_KEY
from flwr.common.record.arrayrecord import ArrayRecord
from flwr.common.record.configrecord import ConfigRecord
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from flwr.decentralized.nodeapp import NodeApp

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
# if not LOGGER.handlers:
#     handler = logging.StreamHandler()
#     handler.setFormatter(logging.Formatter("%(levelname)s:      %(message)s"))
#     LOGGER.addHandler(handler)


class Net(nn.Module):
    """Model architecture for CIFAR-10."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        activations = self.pool(func.relu(self.conv1(images)))
        activations = self.pool(func.relu(self.conv2(activations)))
        activations = activations.view(-1, 16 * 5 * 5)
        activations = func.relu(self.fc1(activations))
        activations = func.relu(self.fc2(activations))
        return self.fc3(activations)


fds: FederatedDataset | None = None
transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
_NODE_STATE: dict[str, dict[str, Any]] = {}

_DEFAULT_DATA_CONFIG = ConfigRecord(
    {
        PARTITION_ID_KEY: 0,
        NUM_PARTITIONS_KEY: 1,
    }
)

_INITIAL_MODEL = Net()

RUN_CONFIG = DLRunConfig(
    rounds=25,
    n_aggregation_steps=2,
)

app1 = NodeApp(
    subject="trainer-pytorch",
    initial_arrays=ArrayRecord(_INITIAL_MODEL.state_dict()),
    data_config=ConfigRecord(_DEFAULT_DATA_CONFIG),
    run_config=RUN_CONFIG,
    train_config=ConfigRecord({"local-epochs": 1, "lr": 0.01, "batch-size": 32}),
    eval_config=ConfigRecord({"batch-size": 32}),
    timeout=1,
)

app2 = NodeApp(
    subject="analytics-pytorch",
    initial_arrays=ArrayRecord(_INITIAL_MODEL.state_dict()),
    data_config=ConfigRecord(_DEFAULT_DATA_CONFIG),
    run_config={"rounds": 10},
    train_config=ConfigRecord({"window": 5}),
    eval_config=ConfigRecord(),
    timeout=1,
)


def apply_transforms(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Apply transforms to image batches."""
    batch["img"] = [transform(image) for image in batch["img"]]
    return batch


def load_partition_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Load partitioned train/test dataloaders."""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    split_partition = partition.train_test_split(test_size=0.2, seed=42)
    transformed_partition = split_partition.with_transform(apply_transforms)
    train_loader = DataLoader(
        transformed_partition["train"], batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(transformed_partition["test"], batch_size=batch_size)
    return train_loader, test_loader


def _get_partition_info(context: Context) -> tuple[int, int, str]:
    partition_id_value = context.node_config.get(PARTITION_ID_KEY, 0)
    num_partitions_value = context.node_config.get(NUM_PARTITIONS_KEY, 1)
    node_id_value = context.node_config.get("node-id", partition_id_value)

    partition_id = int(partition_id_value)
    num_partitions = int(num_partitions_value)
    node_id = str(node_id_value)

    return partition_id, num_partitions, node_id


def _get_or_init_state(context: Context, batch_size: int) -> tuple[str, dict[str, Any]]:
    partition_id, num_partitions, node_id = _get_partition_info(context)

    if node_id not in _NODE_STATE:
        train_loader, test_loader = load_partition_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
            batch_size=batch_size,
        )

        model = Net()
        _NODE_STATE[node_id] = {
            "model": model,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "cycle": 0,
            "history": [],
        }

    return node_id, _NODE_STATE[node_id]


def train_one_cycle(
    model: Net,
    train_loader: DataLoader,
    local_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> float:
    """Train model locally for one cycle."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    model.train()

    cumulative_loss = 0.0
    for _ in range(local_epochs):
        for batch in train_loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()

    normalizer = max(local_epochs * len(train_loader), 1)
    return cumulative_loss / normalizer


def evaluate_model(
    model: Net,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on test partition."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = len(test_loader.dataset)

    with torch.no_grad():
        for batch in test_loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)

            total_loss += criterion(outputs, labels).item()
            total_correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    average_loss = total_loss / max(len(test_loader), 1)
    accuracy = total_correct / max(total_examples, 1)
    return average_loss, accuracy


@app1.train(mods=[message_size_mod])
def train_app1(message: Message, context: Context) -> Message:
    """Train the local PyTorch model on a partition."""
    run_config = context.run_config
    batch_size = int(run_config.get("batch-size", 32))
    local_epochs = int(run_config.get("local-epochs", 1))
    learning_rate = float(run_config.get("lr", 0.1))

    node_id, state = _get_or_init_state(context, batch_size=batch_size)
    model: Net = state["model"]
    train_loader: DataLoader = state["train_loader"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loss = train_one_cycle(
        model=model,
        train_loader=train_loader,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        device=device,
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(train_loader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    # return Message(content=content, reply_to=msg)

    state["cycle"] += 1
    state["history"].append({"cycle": state["cycle"], "train_loss": train_loss})

    return Message(content=content, reply_to=message)


@app1.evaluate(mods=[message_size_mod])
def evaluate_app1(message: Message, context: Context) -> Message:
    """Evaluate the local PyTorch model on held-out partition data."""
    batch_size = int(context.run_config.get("batch-size", 32))
    node_id, state = _get_or_init_state(context, batch_size=batch_size)

    if state["cycle"] == 0:
        return Message(content=message.content, reply_to=message)

    model: Net = state["model"]
    test_loader: DataLoader = state["test_loader"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loss, accuracy = evaluate_model(
        model=model, test_loader=test_loader, device=device
    )

    state["history"][-1]["test_loss"] = test_loss
    state["history"][-1]["accuracy"] = accuracy

    metrics = {
        "eval_loss": test_loss,
        "eval_acc": accuracy,
        "num-examples": len(test_loader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=message)


@app2.train()
def train_app2(message: Message, context: Context) -> Message:
    """Report rolling metrics from PyTorch training history."""
    window = int(context.run_config.get("window", 5))
    partition_id, _, node_id = _get_partition_info(context)
    history = _NODE_STATE.get(node_id, {}).get("history", [])
    recent = history[-window:]

    if not recent:
        # LOGGER.info(
        #     "[NodeApp:%s][analytics] node=%s partition=%d no training history yet",
        #     app2.name,
        #     node_id,
        #     partition_id,
        # )
        return Message(content=message.content, reply_to=message)

    recent_losses = [entry["train_loss"] for entry in recent]
    recent_accuracies = [entry.get("accuracy", 0.0) for entry in recent]

    LOGGER.info(
        "[NodeApp:%s][analytics] node=%s window=%d avg_train_loss=%.4f avg_acc=%.4f",
        app2.name,
        node_id,
        len(recent),
        float(sum(recent_losses) / len(recent_losses)),
        float(sum(recent_accuracies) / len(recent_accuracies)),
    )
    return Message(content=message.content, reply_to=message)


@app2.evaluate()
def evaluate_app2(message: Message, context: Context) -> Message:
    """Report best local accuracy observed for this node."""
    _, _, node_id = _get_partition_info(context)
    history = _NODE_STATE.get(node_id, {}).get("history", [])

    history_with_accuracy = [entry for entry in history if "accuracy" in entry]
    if not history_with_accuracy:
        # LOGGER.info("[NodeApp:%s][evaluate] node=%s no evaluated cycles", app2.name, node_id)
        return Message(content=message.content, reply_to=message)

    best_entry = max(history_with_accuracy, key=lambda entry: float(entry["accuracy"]))
    LOGGER.info(
        "[NodeApp:%s][evaluate] node=%s best_acc=%.4f at cycle=%d",
        app2.name,
        node_id,
        float(best_entry["accuracy"]),
        int(best_entry["cycle"]),
    )
    return Message(content=message.content, reply_to=message)


def load_centralized_dataset(batch_size: int = 128) -> DataLoader:
    """Optional helper kept for parity with quickstart-pytorch."""
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    transformed_dataset = test_dataset.with_format("torch").with_transform(
        apply_transforms
    )
    return DataLoader(transformed_dataset, batch_size=batch_size)
