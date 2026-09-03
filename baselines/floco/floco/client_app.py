"""floco: A Flower Baseline."""

import copy

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import bytes_to_ndarray

from .dataset import get_federated_dataloaders
from .model import SimplexModel, create_model, test, train

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

app = ClientApp()


@app.train()
def train_fn(msg: Message, context: Context) -> Message:
    """Train the model on local data."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    local_epochs = int(context.run_config["local-epochs"])
    pers_lamda = int(context.run_config["pers_lamda"])
    algorithm = str(context.run_config["algorithm"])

    # Load model from received arrays
    global_model = create_model(context).to(DEVICE)
    arrays = msg.content.array_records["arrays"]
    global_model.load_state_dict(arrays.to_torch_state_dict())

    # Load data
    trainloader, _ = get_federated_dataloaders(partition_id, num_partitions, context)

    # Set simplex params if present in config
    config = msg.content.config_records.get("config", {})
    if isinstance(global_model, SimplexModel):
        _apply_simplex_config(global_model, config, training=True)

    # Floco+ personalization: save reg params before training global model
    use_pers = pers_lamda != 0 and algorithm == "Floco"
    if use_pers:
        reg_parameters = copy.deepcopy(list(global_model.parameters()))

    # Train global model
    train_loss = train(global_model, trainloader, local_epochs, DEVICE)

    # Floco+ personalization: train personalized model
    if use_pers:
        endpoints = int(context.run_config["endpoints"])
        pers_model = SimplexModel(endpoints=endpoints).to(DEVICE)
        if "pers_parameters" not in context.state:
            context.state["pers_parameters"] = ArrayRecord()
        pers_record = context.state["pers_parameters"]
        if len(pers_record) > 0:
            pers_model.load_state_dict(pers_record.to_torch_state_dict())
        _apply_simplex_config(pers_model, config, training=True)
        train(pers_model, trainloader, local_epochs, DEVICE, reg_parameters, pers_lamda)
        context.state["pers_parameters"] = ArrayRecord(pers_model.state_dict())

    # Construct reply
    model_record = ArrayRecord(global_model.state_dict())
    metrics = MetricRecord({
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    })
    content = RecordDict({"arrays": model_record, "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    """Evaluate the model on local data."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    pers_lamda = int(context.run_config["pers_lamda"])
    algorithm = str(context.run_config["algorithm"])

    # Load model from received arrays
    model = create_model(context).to(DEVICE)
    arrays = msg.content.array_records["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())

    # Load data
    _, valloader = get_federated_dataloaders(partition_id, num_partitions, context)

    # Floco+ personalization: use personalized model if available
    if pers_lamda != 0 and algorithm == "Floco":
        if "pers_parameters" in context.state:
            pers_record = context.state["pers_parameters"]
            if len(pers_record) > 0:
                endpoints = int(context.run_config["endpoints"])
                model = SimplexModel(endpoints=endpoints).to(DEVICE)
                model.load_state_dict(pers_record.to_torch_state_dict())

    # Set simplex params for evaluation
    config = msg.content.config_records.get("config", {})
    if isinstance(model, SimplexModel):
        _apply_simplex_config(model, config, training=False)

    loss, accuracy = test(model, valloader, DEVICE)

    # Construct reply
    metrics = MetricRecord({
        "loss": loss,
        "accuracy": accuracy,
        "num-examples": len(valloader.dataset),
    })
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)


def _apply_simplex_config(model, config, training):
    """Apply simplex subregion parameters from a message config to a model."""
    model.training = training
    if "center" in config and "radius" in config:
        model.subregion_parameters = (
            bytes_to_ndarray(config["center"]),
            config["radius"],
        )
