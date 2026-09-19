"""FedSaSync: Semi-asynchronous Federated Learning in Flower."""

import time

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedsasync.dataset import load_data
from fedsasync.model import Net
from fedsasync.model import test as test_fn
from fedsasync.model import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Init_time counter
    init_time = time.perf_counter()
    # Load the model and initialize it with the received weights
    dataset_name = str(context.run_config["dataset-name"])
    if dataset_name == "uoft-cs/cifar10":
        model = Net()
    elif dataset_name == "ylecun/mnist":
        model = Net(input_channels=1, pool_size=4)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    arrays = msg.content.array_records["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    trainloader, _ = load_data(partition_id, num_partitions, dataset_name)
    local_epochs = context.run_config["local-epochs"]

    # Probability of being a slow client, for simulating stragglers in FedSaSync
    number_slow = int(context.run_config["number-slow"])
    if partition_id < number_slow:
        time.sleep(5)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        local_epochs,
        device,
        dataset_name
    )

    # End_time counter
    end_time = time.perf_counter()
    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "train_time": end_time - init_time,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # Load the model and initialize it with the received weights
    dataset_name = str(context.run_config["dataset-name"])
    if dataset_name == "uoft-cs/cifar10":
        model = Net()
        image = "img"
    elif dataset_name == "ylecun/mnist":
        model = Net(input_channels=1, pool_size=4)
        image = "image"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    arrays = msg.content.array_records["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    _, valloader = load_data(partition_id, num_partitions, dataset_name)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(model, valloader, device, image)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
