"""FedSCS CIFAR-10 client application."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedscs.dataset import load_data
from fedscs.model import Net, test, train

app = ClientApp()


def get_device() -> torch.device:
    """Return the available computation device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            f"Using CUDA: {torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
        return device

    print("Using CPU")
    return torch.device("cpu")


@app.train()
def train_client(msg: Message, context: Context) -> Message:
    """Train the global model on one client's local CIFAR-10 partition."""

    node_id = int(context.node_config["partition-id"])
    num_nodes = int(context.node_config["num-partitions"])

    batch_size = int(context.run_config.get("batch-size", 128))
    local_epochs = int(context.run_config.get("local-epochs", 2))
    learning_rate = float(context.run_config.get("learning-rate", 0.01))

    device = get_device()

    # ---------------------------------------------------------------
    # Load this client's local CIFAR-10 training partition
    # ---------------------------------------------------------------
    trainloader, _ = load_data(
        partition_id=node_id,
        num_partitions=num_nodes,
        batch_size=batch_size,
    )

    print(
        f"Client {node_id}: "
        f"training samples={len(trainloader.dataset)}, "
        f"epochs={local_epochs}"
    )

    # ---------------------------------------------------------------
    # Load global model received from server
    # ---------------------------------------------------------------
    arrays = msg.content["arrays"]

    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    model.to(device)

    # ---------------------------------------------------------------
    # Local training
    #
    # train() must return:
    #   train_loss, train_accuracy
    # ---------------------------------------------------------------
    train_loss, train_accuracy = train(
        model=model,
        trainloader=trainloader,
        epochs=local_epochs,
        device=device,
        lr=learning_rate,
    )

    print(
        f"Client {node_id}: "
        f"train_loss={train_loss:.4f}, "
        f"train_accuracy={train_accuracy:.4f}"
    )

    # ---------------------------------------------------------------
    # Return updated model + local training metrics
    # ---------------------------------------------------------------
    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(model.state_dict()),
                "metrics": MetricRecord(
                    {
                        "train-loss": float(train_loss),
                        "train-accuracy": float(train_accuracy),
                        "num-examples": len(trainloader.dataset),
                    }
                ),
            }
        ),
        reply_to=msg,
    )


@app.evaluate()
def evaluate_client(msg: Message, context: Context) -> Message:
    """Evaluate the global model on the CIFAR-10 test set."""

    node_id = int(context.node_config["partition-id"])
    num_nodes = int(context.node_config["num-partitions"])

    batch_size = int(context.run_config.get("batch-size", 128))

    device = get_device()

    # ---------------------------------------------------------------
    # Load CIFAR-10 test set
    # ---------------------------------------------------------------
    _, testloader = load_data(
        partition_id=node_id,
        num_partitions=num_nodes,
        batch_size=batch_size,
    )

    # ---------------------------------------------------------------
    # Load global model after FedSCS aggregation
    # ---------------------------------------------------------------
    arrays = msg.content["arrays"]

    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    model.to(device)

    # ---------------------------------------------------------------
    # Evaluate global model
    # ---------------------------------------------------------------
    test_loss, test_accuracy = test(
        model=model,
        testloader=testloader,
        device=device,
    )

    print(
        f"Client {node_id}: "
        f"test_loss={test_loss:.4f}, "
        f"test_accuracy={test_accuracy:.4f}"
    )

    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "test-loss": float(test_loss),
                        "test-accuracy": float(test_accuracy),
                        "num-examples": len(testloader.dataset),
                    }
                )
            }
        ),
        reply_to=msg,
    )
