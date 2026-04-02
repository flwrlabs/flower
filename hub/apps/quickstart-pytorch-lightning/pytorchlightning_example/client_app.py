"""pytorchlightning_example: A Flower / PyTorch Lightning app."""

import pytorch_lightning as pl
from datasets.utils.logging import disable_progress_bar
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

disable_progress_bar()

from pytorchlightning_example.task import LitAutoEncoder, load_data

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = LitAutoEncoder()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_loader, val_loader, _ = load_data(partition_id, num_partitions)

    # Train the model on local data
    max_epochs = context.run_config["max-epochs"]
    trainer = pl.Trainer(max_epochs=max_epochs, enable_progress_bar=False)
    trainer.fit(model, train_loader, val_loader)

    # FIT does not return values, but you can read final epoch metrics here:
    fit_metrics = trainer.callback_metrics  # Dict[str, Tensor]
    loss = float(fit_metrics["train_loss"])

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": loss,
        "num-examples": len(train_loader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = LitAutoEncoder()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, _, test_loader = load_data(partition_id, num_partitions)

    # Call the evaluation function
    trainer = pl.Trainer(enable_progress_bar=False)
    results = trainer.test(model, test_loader)
    # Test returns a list[dict] with your logged test_* metrics:
    loss = results[0]["test_loss"]

    # Construct and return reply Message
    metrics = {
        "eval_loss": loss,
        "num-examples": len(test_loader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
