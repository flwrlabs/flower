"""mlxexample: A Flower / MLX app."""

from flwr.app import ArrayRecord, Context
from flwr.server import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from mlxexample.task import MLP, get_params, set_params

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    num_layers = context.run_config["num-layers"]
    input_dim = context.run_config["input-dim"]
    hidden_dim = context.run_config["hidden-dim"]

    # Initialize global model
    model = MLP(num_layers, input_dim, hidden_dim, output_dim=10)
    params = get_params(model)
    arrays = ArrayRecord(params)

    # Initialize FedAvg strategy
    strategy = FedAvg()

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    ndarrays = result.arrays.to_numpy_ndarrays()
    set_params(model, ndarrays)
    model.save_weights("final_model.npz")
