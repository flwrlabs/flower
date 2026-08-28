
"""FedSCS server application."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from fedscs.model import Net
from fedscs.strategy import FedSCS

from fedscs.model import Net
from fedscs.strategy import FedSCS
from fedscs.utils import set_seed


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run FedSCS federated training."""
    seed = int(context.run_config.get("seed", 42))
    set_seed(seed)

    num_rounds = int(context.run_config.get("num-server-rounds", 2))
    batch_size = int(context.run_config.get("batch-size", 32))
    local_epochs = int(context.run_config.get("local-epochs", 2))
    learning_rate = float(context.run_config.get("learning-rate", 0.01))
    fraction_train = float(context.run_config.get("fraction-train", 1.0))
    fraction_evaluate = float(context.run_config.get("fraction-evaluate", 1.0))

    node_ids = list(grid.get_node_ids())
    num_clients = len(node_ids)

    if num_clients == 0:
        raise RuntimeError("No Flower nodes are available.")

    if not 0.0 < fraction_train <= 1.0:
        raise ValueError("fraction-train must be in the range (0, 1].")

    if not 0.0 < fraction_evaluate <= 1.0:
        raise ValueError("fraction-evaluate must be in the range (0, 1].")

    min_train_nodes = max(1, int(num_clients * fraction_train))
    min_evaluate_nodes = max(1, int(num_clients * fraction_evaluate))

    # Initialize global model.
    global_model = Net()
    initial_arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedSCS strategy.
    strategy = FedSCS(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=min_train_nodes,
        min_evaluate_nodes=min_evaluate_nodes,
        min_available_nodes=num_clients,
        epsilon=1e-6,
    )

    strategy.summary()

    print(f"\nStarting FedSCS for {num_rounds} rounds...")
    print("\nExperiment configuration:")
    print(f"  Number of clients: {num_clients}")
    print(f"  Training participation: {fraction_train:.0%}")
    print(f"  Evaluation participation: {fraction_evaluate:.0%}")
    print(f"  Minimum training clients: {min_train_nodes}")
    print(f"  Minimum evaluation clients: {min_evaluate_nodes}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print("  Training metrics: client-average")
    print("  Test metrics: client-average on CIFAR-10 test set")
    print("  Model aggregation: FedSCS")

    # Send training configuration to clients.
    train_config = ConfigRecord(
        {
            "batch-size": batch_size,
            "local-epochs": local_epochs,
            "learning-rate": learning_rate,
            "seed": seed,
        }
    )

    evaluate_config = ConfigRecord(
        {
            "batch-size": batch_size,
            "seed": seed,
        }
    )

    # Start federated learning.
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=train_config,
        evaluate_config=evaluate_config,
        num_rounds=num_rounds,
    )

    # Save final model.
    print("\nSaving final FedSCS model...")

    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

    print("Final FedSCS model saved as: final_model.pt")


