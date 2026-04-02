"""huggingface_example: A Flower / Hugging Face app."""

import warnings

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from transformers import logging

from huggingface_example.task import get_model, load_data, test_fn, train_fn

warnings.filterwarnings("ignore", category=FutureWarning)

# To mute warnings reminding that we need to train the model to a downstream task
# This is something this example does.
logging.set_verbosity_error()

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the model on local data."""

    # Get this client's dataset partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    model_name = context.run_config["model-name"]
    trainloader, _ = load_data(partition_id, num_partitions, model_name)

    # Load model
    model = get_model(model_name)

    # Initialize it with the received weights
    arrays = msg.content["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict(), strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the model on local data
    train_fn(model, trainloader, epochs=1, device=device)

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = MetricRecord({"num-examples": len(trainloader)})
    # Construct RecordDict and add ArrayRecord and MetricRecord
    content = RecordDict({"arrays": model_record, "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the model on local data."""

    # Get this client's dataset partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    model_name = context.run_config["model-name"]
    _, testloader = load_data(partition_id, num_partitions, model_name)

    # Load model
    model = get_model(model_name)

    # Initialize it with the received weights
    arrays = msg.content["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict(), strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate the model on local data
    loss, accuracy = test_fn(model, testloader, device=device)

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = MetricRecord(
        {
            "num-examples": len(testloader),
            "loss": float(loss),
            "accuracy": float(accuracy),
        }
    )
    # Construct RecordDict and add ArrayRecord and MetricRecord
    content = RecordDict({"arrays": model_record, "metrics": metrics})
    return Message(content=content, reply_to=msg)
