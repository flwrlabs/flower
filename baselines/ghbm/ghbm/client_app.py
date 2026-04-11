"""ClientApp for the GHBM Flower baseline."""

from dataclasses import dataclass
from typing import Final

import torch
from flwr.app import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp

from ghbm.algorithm import (
    TrainingModifiers,
    finalize_training_state,
    prepare_training_modifiers,
)
from ghbm.config import AlgorithmName, DatasetName, ModelName, NormLayer, parse_enum
from ghbm.dataset import get_num_classes, load_data
from ghbm.model import create_model
from ghbm.trainer import test as test_fn
from ghbm.trainer import train as train_fn
from ghbm.utils import MetricDict

# Flower ClientApp
app = ClientApp()


TRAIN_SUMMARY_KEY: Final[str] = "train_summary_printed"


@dataclass(frozen=True)
class ClientRunConfig:
    """Parsed client-side run configuration."""

    algorithm_name: AlgorithmName
    dataset_name: DatasetName
    dirichlet_alpha: float
    batch_size: int
    model_name: ModelName
    resnet_version: int
    norm_layer: NormLayer
    local_epochs: int
    fraction_train: float
    learning_rate: float
    weight_decay: float
    ghbm_beta: float


def _parse_client_run_config(context: Context) -> ClientRunConfig:
    """Parse typed client-side run config values from Flower context."""
    return ClientRunConfig(
        algorithm_name=parse_enum(context.run_config["algorithm-name"], AlgorithmName),
        dataset_name=parse_enum(context.run_config["dataset-name"], DatasetName),
        dirichlet_alpha=float(context.run_config["dirichlet-alpha"]),
        batch_size=int(context.run_config["batch-size"]),
        model_name=parse_enum(context.run_config["model-name"], ModelName),
        resnet_version=int(context.run_config["resnet-version"]),
        norm_layer=parse_enum(context.run_config["norm-layer"], NormLayer),
        local_epochs=int(context.run_config["local-epochs"]),
        fraction_train=float(context.run_config["fraction-train"]),
        learning_rate=float(context.run_config["learning-rate"]),
        weight_decay=float(context.run_config["weight-decay"]),
        ghbm_beta=float(context.run_config["ghbm-beta"]),
    )


def _create_configured_model(config: ClientRunConfig):
    """Create the configured model for this run."""
    return create_model(
        model_name=config.model_name,
        num_classes=get_num_classes(config.dataset_name),
        resnet_version=config.resnet_version,
        norm_layer=config.norm_layer,
    )


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    config = _parse_client_run_config(context)

    # Load the model and initialize it with the received weights
    model = _create_configured_model(config)
    arrays = msg.content.array_records["arrays"]
    current_state = arrays.to_torch_state_dict()
    model.load_state_dict(current_state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    trainloader, _, _ = load_data(
        partition_id,
        num_partitions,
        config.dataset_name,
        config.dirichlet_alpha,
        batch_size=config.batch_size,
    )

    modifiers: TrainingModifiers = prepare_training_modifiers(
        context=context,
        algorithm_name=config.algorithm_name,
        fraction_train=config.fraction_train,
        current_global_model=current_state,
        message=msg,
    )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        config.local_epochs,
        device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta=config.ghbm_beta,
        modifiers=modifiers,
    )
    finalize_training_state(context, config.algorithm_name, model.state_dict())

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics: MetricDict = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    config = _parse_client_run_config(context)

    # Load the model and initialize it with the received weights
    model = _create_configured_model(config)
    arrays = msg.content.array_records["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    _, _, testloader = load_data(
        partition_id,
        num_partitions,
        config.dataset_name,
        config.dirichlet_alpha,
        batch_size=config.batch_size,
    )

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(model, testloader, device)

    # Construct and return reply Message
    metrics: MetricDict = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(testloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
