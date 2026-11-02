"""FedAMP: Personalized Cross-Silo Federated Learning on Non-IID Data."""

import copy

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedamp.dataset import load_data
from fedamp.model import FashionNet, Net
from fedamp.model import test as test_fn
from fedamp.model import train as train_fn
from fedamp.utils import combine_aggregated_arrays

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Load common variables to all algorithms
    dataset = context.run_config["dataset"]
    if dataset == "cifar10":
        model = Net(n_channels=3)
    elif dataset == "fashion":
        model = FashionNet(n_channels=1)
    else:
         raise ValueError(f"Unsupported dataset: {dataset}")
         
    arrays = msg.content.array_records["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    partition_by: str = context.run_config["partition-by"]
    num_classes_per_partition: int = int(
        context.run_config["num-classes-per-partition"]
    )
    batch_size: int = int(context.run_config["batch-size"])

    trainloader, _ = load_data(
        dataset=dataset,
        num_classes_per_partition=num_classes_per_partition,
        partition_by=partition_by,
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
    )

    local_epochs = context.run_config["local-epochs"]
    lr = float(context.run_config["lr"])
    algorithm = context.run_config["algorithm"]

    # handle the algorithm spefici implementation directly in train function
    # for FedAMP: we use 2 models, one that will stay local (model) and the one from
    # aggregated neighbors (aggregated_model)
    if algorithm == "fedamp":
        if dataset == "cifar10":
            global_model = Net(n_channels=3)
        elif dataset == "fashion":
            global_model = FashionNet(n_channels=1)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # for FedAMP: we also recieve the aggregated weights and coef_self;
        # the aggregatedrecords_key is `mu`
        partial_aggregated_arrays: ArrayRecord = msg.content.array_records["mu"]
        coef_self: float = msg.content.config_records["config"]["coef_self"]

        aggregated_arrays: ArrayRecord = combine_aggregated_arrays(
            partial_aggregated_arrays, arrays, coef_self
        )
        global_model.load_state_dict(aggregated_arrays.to_torch_state_dict())
        alphaK = float(context.run_config["alphaK"])
        fedamp_lambda = float(context.run_config["fedamp-lambda"])
        proximal_weight = fedamp_lambda / alphaK

    elif algorithm in ("fedavg", "fedprox"):  # this works either for fedavg, fedprox
        global_model = copy.deepcopy(model)
        # FedProx will send the proximal-mu as part of config
        if algorithm == "fedprox":
            proximal_weight = msg.content.config_records.get(
                "proximal-mu", context.run_config["proximal_mu"]
            )
        else:
            proximal_weight = 0.0
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Call the training function: use aggregated_model for the regularization term.
    train_loss = train_fn(
        net=model,
        global_net=global_model,
        trainloader=trainloader,
        epochs=local_epochs,
        proximal_weight=proximal_weight,
        lr=lr,
        device=device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # Load the model and initialize it with the received weights
    dataset = context.run_config["dataset"]
    if dataset == "cifar10":
        model = Net(n_channels=3)
    elif dataset == "fashion":
        model = FashionNet(n_channels=1)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    arrays = msg.content.array_records["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    partition_by: str = context.run_config.get("partition-by", "label")
    num_classes_per_partition: int = int(
        context.run_config.get("num-classes-per-partition", 3)
    )
    batch_size = int(context.run_config["batch-size"])
    _, valloader = load_data(
        dataset=dataset,
        num_classes_per_partition=num_classes_per_partition,
        partition_by=partition_by,
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
    )

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(model, valloader, device)

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
