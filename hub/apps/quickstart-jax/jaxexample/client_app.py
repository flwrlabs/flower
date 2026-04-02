"""jaxexample: A Flower / JAX app."""

from typing import cast

import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from jaxexample.task import (
    apply_model,
    create_train_state,
    get_params,
    load_data,
    set_params,
)
from jaxexample.task import train as jax_train

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Create train state object (model + optimizer)
    lr = float(context.run_config["learning-rate"])
    train_state = create_train_state(lr)
    # Extract numpy arrays from ArrayRecord before applying
    ndarrays = cast(ArrayRecord, msg.content["arrays"]).to_numpy_ndarrays()
    train_state = set_params(train_state, ndarrays)

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    train_state, loss, acc = jax_train(train_state, trainloader)
    params = get_params(train_state.params)

    # Construct and return reply Message
    model_record = ArrayRecord(params)
    metrics = {
        "train_loss": float(loss),
        "train_acc": float(acc),
        "num-examples": int(len(trainloader) * batch_size),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Create train state object (model + optimizer)
    lr = float(context.run_config["learning-rate"])
    train_state = create_train_state(lr)
    ndarrays = cast(ArrayRecord, msg.content["arrays"]).to_numpy_ndarrays()
    train_state = set_params(train_state, ndarrays)

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    losses = []
    accs = []
    for batch in valloader:
        _, loss, accuracy = apply_model(train_state, batch["image"], batch["label"])
        losses.append(float(loss))
        accs.append(float(accuracy))

    # Construct and return reply Message
    metrics = {
        "eval_loss": float(np.mean(losses)),
        "eval_acc": float(np.mean(accs)),
        "num-examples": int(len(valloader) * batch_size),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
