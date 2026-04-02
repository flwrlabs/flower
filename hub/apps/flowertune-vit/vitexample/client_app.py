"""vitexample: A Flower / PyTorch app with Vision Transformers."""

import warnings

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torch.utils.data import DataLoader

from vitexample.task import (
    apply_train_transforms,
    get_dataset_partition,
    get_model,
    trainer,
)

warnings.filterwarnings("ignore", category=UserWarning)


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    dataset_name = context.run_config["dataset-name"]
    trainpartition = get_dataset_partition(num_partitions, partition_id, dataset_name)

    batch_size = context.run_config["batch-size"]
    lr = context.run_config["learning-rate"]
    num_classes = context.run_config["num-classes"]

    # Load dataset
    trainset = trainpartition.with_transform(apply_train_transforms)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, num_workers=2, shuffle=True
    )

    # Load model
    model = get_model(num_classes)
    finetune_layers = model.heads
    finetune_layers.load_state_dict(
        msg.content["arrays"].to_torch_state_dict(), strict=True
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Train locally
    avg_train_loss = trainer(model, trainloader, optimizer, epochs=1, device=device)

    # Construct and return reply Message
    model_record = ArrayRecord(finetune_layers.state_dict())
    metrics = {
        "train_loss": avg_train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
