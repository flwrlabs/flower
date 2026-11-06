"""ClientApp for verifiable aggregation baseline."""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from verifiableagg.dataset import load_data
from verifiableagg.model import Net, evaluate, train

app = ClientApp()


@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    """Train local model and reply with model arrays and metrics."""
    num_features = int(context.run_config["num-features"])
    local_epochs = int(context.run_config["local-epochs"])
    learning_rate = float(context.run_config["learning-rate"])
    batch_size = int(context.run_config["batch-size"])
    num_train_examples = int(context.run_config["num-train-examples"])
    num_val_examples = int(context.run_config["num-val-examples"])
    base_seed = int(context.run_config["random-seed"])

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    trainloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        num_train_examples=num_train_examples,
        num_val_examples=num_val_examples,
        num_features=num_features,
        batch_size=batch_size,
        base_seed=base_seed,
    )

    model = Net(num_features=num_features)
    arrays = msg.content.array_records["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cpu")
    train_loss = train(
        model=model,
        trainloader=trainloader,
        epochs=local_epochs,
        learning_rate=learning_rate,
        device=device,
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = MetricRecord(
        {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
        }
    )
    content = RecordDict({"arrays": model_record, "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    """Evaluate global model on local validation split."""
    num_features = int(context.run_config["num-features"])
    batch_size = int(context.run_config["batch-size"])
    num_train_examples = int(context.run_config["num-train-examples"])
    num_val_examples = int(context.run_config["num-val-examples"])
    base_seed = int(context.run_config["random-seed"])

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    _, valloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        num_train_examples=num_train_examples,
        num_val_examples=num_val_examples,
        num_features=num_features,
        batch_size=batch_size,
        base_seed=base_seed,
    )

    model = Net(num_features=num_features)
    arrays = msg.content.array_records["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cpu")
    eval_loss, eval_acc = evaluate(model=model, valloader=valloader, device=device)

    metrics = MetricRecord(
        {
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "num-examples": len(valloader.dataset),
        }
    )
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)
