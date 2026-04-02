"""quickstart-pennylane: A Flower / Pennylane Quantum Federated Learning app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from quickstart_pennylane.task import QuantumNet, load_data, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the quantum federated learning ServerApp."""

    # Read run config
    fraction_fit: float = context.run_config.get("fraction-fit", 1.0)
    fraction_evaluate: float = context.run_config.get("fraction-evaluate", 1.0)
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    n_qubits: int = context.run_config.get("n-qubits", 4)
    n_layers: int = context.run_config.get("n-layers", 3)

    print("Server configuration:")
    print(f"  - Number of rounds: {num_rounds}")
    print(f"  - Fraction fit: {fraction_fit}")
    print(f"  - Fraction evaluate: {fraction_evaluate}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Number of qubits: {n_qubits}")
    print(f"  - Number of layers: {n_layers}")

    # Load global quantum model
    global_model = QuantumNet(num_classes=10, n_qubits=n_qubits, n_layers=n_layers)
    arrays = ArrayRecord(global_model.state_dict())

    print(
        f"Initialized quantum neural network with {sum(p.numel() for p in global_model.parameters())} parameters"
    )

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_train=fraction_fit,
        fraction_evaluate=fraction_evaluate,
    )

    # Create evaluation function with quantum parameters
    def make_global_evaluate(n_qubits: int, n_layers: int):
        def global_evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
            return global_evaluate(server_round, arrays, n_qubits, n_layers)

        return global_evaluate_fn

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=make_global_evaluate(n_qubits, n_layers),
    )

    # Save final model to disk
    print("\nSaving final quantum model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_quantum_model.pt")


def global_evaluate(
    server_round: int, arrays: ArrayRecord, n_qubits: int, n_layers: int
) -> MetricRecord:
    """Evaluate global quantum model on centralized test data."""

    # Load the model and initialize it with the received weights
    model = QuantumNet(num_classes=10, n_qubits=n_qubits, n_layers=n_layers)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load centralized test data (using partition 0 as test set)
    _, testloader = load_data(partition_id=0, num_partitions=1, batch_size=128)

    # Evaluate the global model on the test set
    test_loss, test_accuracy = test(model, testloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_accuracy, "loss": test_loss})
