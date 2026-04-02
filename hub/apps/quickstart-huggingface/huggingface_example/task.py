"""huggingface_example: A Flower / Hugging Face app."""

from typing import Any

import torch
from datasets.utils.logging import disable_progress_bar
from evaluate import load as load_metric
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

disable_progress_bar()
fds = None  # Cache FederatedDataset


def load_data(
    partition_id: int, num_partitions: int, model_name: str
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Load IMDB data (training and eval)"""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # Partition the IMDB dataset into N partitions
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="stanfordnlp/imdb", partitioners={"train": partitioner}
        )
    partition = fds.load_partition(partition_id)
    # Divide data: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, add_special_tokens=True)

    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns("text")
    partition_train_test = partition_train_test.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        partition_train_test["test"], batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader


def get_model(model_name):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


def train_fn(net, trainloader, epochs, device) -> None:
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test_fn(net, testloader, device) -> tuple[Any | float, Any]:
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy
