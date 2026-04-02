"""fastai_example: A Flower / Fastai app."""

import warnings

from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat
from fastai.vision.all import error_rate, squeezenet1_1
from fastai.vision.data import DataLoaders
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import Context

from fastai_example.task import load_data

warnings.filterwarnings("ignore", category=UserWarning)


app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = squeezenet1_1()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloader, valloader, _ = load_data(partition_id, num_partitions)
    dls = DataLoaders(trainloader, valloader)

    # Intialize learner
    learn = Learner(
        dls,
        model,
        loss_func=CrossEntropyLossFlat(),
        metrics=error_rate,
    )

    # Fit model to the data
    with learn.no_bar(), learn.no_logging():
        learn.fit(1)

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = squeezenet1_1()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloader, valloader, _ = load_data(partition_id, num_partitions)
    dls = DataLoaders(trainloader, valloader)

    # Intialize learner
    learn = Learner(
        dls,
        model,
        loss_func=CrossEntropyLossFlat(),
        metrics=error_rate,
    )

    # Evaluate model on the data
    with learn.no_bar(), learn.no_logging():
        loss, e_rate = learn.validate()

    # Construct and return reply Message
    metrics = {
        "eval_loss": loss,
        "eval_acc": 1 - e_rate,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
