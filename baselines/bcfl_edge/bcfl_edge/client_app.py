"""bcfl_edge: An edge device training a keyword classifier."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from bcfl_edge.dataset import load_partition
from bcfl_edge.model import Net
from bcfl_edge.model import train as train_fn

# Flower ClientApp
app = ClientApp()


def _trainloader(context: Context):
    return load_partition(
        partition_id=int(context.node_config["partition-id"]),
        num_partitions=int(context.node_config["num-partitions"]),
        batch_size=int(context.run_config["batch-size"]),
        dataset=str(context.run_config["dataset"]),
    )


@app.query()
def query(msg: Message, context: Context):
    """Report the device's partition id and local dataset size D_n."""
    profile = MetricRecord(
        {
            "partition-id": int(context.node_config["partition-id"]),
            "num-examples": len(_trainloader(context).dataset),
        }
    )
    return Message(content=RecordDict({"profile": profile}), reply_to=msg)


@app.train()
def train(msg: Message, context: Context):
    """Train the received global model on local data."""
    model = Net()
    model.load_state_dict(msg.content.array_records["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainloader = _trainloader(context)

    train_loss = train_fn(
        model,
        trainloader,
        int(context.run_config["local-epochs"]),
        float(context.run_config["learning-rate"]),
        device,
    )

    metrics = MetricRecord(
        {"train_loss": train_loss, "num-examples": len(trainloader.dataset)}
    )
    content = RecordDict(
        {"arrays": ArrayRecord(model.state_dict()), "metrics": metrics}
    )
    return Message(content=content, reply_to=msg)
