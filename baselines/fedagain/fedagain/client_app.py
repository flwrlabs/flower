"""fedagain: Flower ClientApp.

The training handler implements the client side of FedAgain (Algorithm 2, step
2): the benchmark error E_k of the received global model is computed on the local
validation set *before* local training, and sent to the server together with the
updated model.
"""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedagain.dataset import get_dataset_spec, load_client_dataloaders
from fedagain.model import benchmark_error, build_model
from fedagain.model import test as test_fn
from fedagain.model import train as train_fn
from fedagain.strategy import BENCHMARK_ERROR_KEY, CLIENT_ID_KEY
from fedagain.utils import get_device, set_seed

app = ClientApp()


def _load_model(msg: Message, run_config: dict):
    spec = get_dataset_spec(str(run_config["dataset"]))
    model = build_model(str(run_config["model"]), spec.num_classes, spec.in_channels)
    arrays = msg.content.array_records["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())
    return model


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Compute E_k with the global model, then train on local data."""
    run_config = dict(context.run_config)
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    config = msg.content.config_records["config"]
    server_round = int(config.get("server-round", 0))  # type: ignore[arg-type]
    # Distinct but reproducible seed for every (client, round) pair
    set_seed(int(run_config["seed"]) * 10_007 + 1_000 * server_round + partition_id)

    device = get_device()
    model = _load_model(msg, run_config)
    trainloader, valloader, _ = load_client_dataloaders(
        partition_id, num_partitions, run_config, server_round
    )

    metrics: dict[str, int | float | list[int] | list[float]] = {}
    if bool(config.get("compute-benchmark-error", False)):
        # E_k: global model, eval mode, BEFORE any local training step (Eq. 6)
        metrics[BENCHMARK_ERROR_KEY] = benchmark_error(model, valloader, device)

    train_loss = train_fn(
        model,
        trainloader,
        epochs=int(run_config["local-epochs"]),
        lr=float(run_config["learning-rate"]),
        momentum=float(run_config["momentum"]),
        device=device,
    )
    metrics["train-loss"] = train_loss
    metrics["num-examples"] = len(trainloader.dataset)  # type: ignore[arg-type]
    metrics[CLIENT_ID_KEY] = partition_id

    # partition-id lets the server relate trust weights to corrupted clients
    content = RecordDict(
        {
            "arrays": ArrayRecord(model.cpu().state_dict()),
            "metrics": MetricRecord(metrics),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the global model on the local validation set."""
    run_config = dict(context.run_config)
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    device = get_device()
    model = _load_model(msg, run_config)
    _, valloader, _ = load_client_dataloaders(partition_id, num_partitions, run_config)
    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval-loss": eval_loss,
        "eval-acc": eval_acc,
        "num-examples": len(valloader.dataset),  # type: ignore[arg-type]
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
