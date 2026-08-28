"""FedSCS server application."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from fedscs.model import Net
from fedscs.strategy import FedSCS

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run FedSCS federated training."""

    num_rounds = int(
        context.run_config.get(
            "num-server-rounds",
            10,
        )
    )

    batch_size = int(
        context.run_config.get(
            "batch-size",
            128,
        )
    )

    local_epochs = int(
        context.run_config.get(
            "local-epochs",
            2,
        )
    )

    learning_rate = float(
        context.run_config.get(
            "learning-rate",
            0.01,
        )
    )

    num_clients = int(
        context.run_config.get(
            "num-supernodes",
            10,
        )
    )

    # ---------------------------------------------------------------
    # Initialize global model
    # ---------------------------------------------------------------
    global_model = Net()

    initial_arrays = ArrayRecord(
        global_model.state_dict()
    )

    # ---------------------------------------------------------------
    # FedSCS strategy
    #
    # ALL clients participate in training.
    # ALL clients participate in evaluation.
    # ---------------------------------------------------------------
    strategy = FedSCS(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=num_clients,
        min_evaluate_nodes=num_clients,
        min_available_nodes=num_clients,
        epsilon=1e-6,
    )

    strategy.summary()

    print(
        f"\nStarting FedSCS for {num_rounds} rounds..."
    )

    print(
        "\nExperiment configuration:"
    )
    print(
        f"  Number of clients: {num_clients}"
    )
    print(
        "  Client participation: 100%"
    )
    print(
        f"  Local epochs: {local_epochs}"
    )
    print(
        f"  Batch size: {batch_size}"
    )
    print(
        f"  Learning rate: {learning_rate}"
    )
    print(
        "  Training metrics: client-average"
    )
    print(
        "  Test metrics: client-average on CIFAR-10 test set"
    )
    print(
        "  Model aggregation: FedSCS"
    )

    # ---------------------------------------------------------------
    # Send training configuration to clients
    # ---------------------------------------------------------------
    train_config = ConfigRecord(
        {
            "batch-size": batch_size,
            "local-epochs": local_epochs,
            "learning-rate": learning_rate,
        }
    )

    evaluate_config = ConfigRecord(
        {
            "batch-size": batch_size,
        }
    )

    # ---------------------------------------------------------------
    # Start federated learning
    # ---------------------------------------------------------------
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=train_config,
        evaluate_config=evaluate_config,
        num_rounds=num_rounds,
    )

    # ---------------------------------------------------------------
    # Save final model
    # ---------------------------------------------------------------
    print(
        "\nSaving final FedSCS model..."
    )

    state_dict = result.arrays.to_torch_state_dict()

    torch.save(
        state_dict,
        "final_model.pt",
    )

    print(
        "Final FedSCS model saved as: "
        "final_model.pt"
    )
