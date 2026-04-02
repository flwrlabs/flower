"""mlxexample: A Flower / MLX app."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from mlxexample.task import (
    MLP,
    batch_iterate,
    eval_fn,
    get_params,
    load_data,
    loss_fn,
    set_params,
)

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Read config
    num_layers = context.run_config["num-layers"]
    input_dim = context.run_config["input-dim"]
    hidden_dim = context.run_config["hidden-dim"]
    batch_size = context.run_config["batch-size"]
    learning_rate = context.run_config["learning-rate"]
    num_epochs = context.run_config["local-epochs"]

    # Instantiate model and apply global parameters
    model = MLP(num_layers, input_dim, hidden_dim, output_dim=10)
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_params(model, ndarrays)

    # Define optimizer and loss function
    optimizer = optim.SGD(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Load data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_images, train_labels, _, _ = load_data(partition_id, num_partitions)

    # Train on local data
    for _ in range(num_epochs):
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            _, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

    # Compute train accuracy and loss
    accuracy = eval_fn(model, train_images, train_labels)
    loss = loss_fn(model, train_images, train_labels)
    # Construct and return reply Message
    model_record = ArrayRecord(get_params(model))
    metrics = {
        "num-examples": len(train_images),
        "accuracy": float(accuracy.item()),
        "loss": float(loss.item()),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Read config
    num_layers = context.run_config["num-layers"]
    input_dim = context.run_config["input-dim"]
    hidden_dim = context.run_config["hidden-dim"]

    # Instantiate model and apply global parameters
    model = MLP(num_layers, input_dim, hidden_dim, output_dim=10)
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_params(model, ndarrays)

    # Load data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, _, test_images, test_labels = load_data(partition_id, num_partitions)

    # Evaluate model on local data
    accuracy = eval_fn(model, test_images, test_labels)
    loss = loss_fn(model, test_images, test_labels)

    # Construct and return reply Message
    metrics = {
        "num-examples": len(test_images),
        "accuracy": float(accuracy.item()),
        "loss": float(loss.item()),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
