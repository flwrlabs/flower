"""jaxexample: A Flower / JAX app."""

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from jax import random

from jaxexample.task import create_model, get_params

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = float(context.run_config["fraction-evaluate"])
    num_rounds: int = int(context.run_config["num-server-rounds"])
    lr: float = float(context.run_config["learning-rate"])

    rng = random.PRNGKey(0)
    rng, _ = random.split(rng)
    _, model_params = create_model(rng)
    params = get_params(model_params)

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_train=0.1,
        fraction_evaluate=fraction_evaluate,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(params),
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    ndarrays = result.arrays.to_numpy_ndarrays()
    np.savez("final_model.npz", *ndarrays)
