"""sklearnexample: A Flower / sklearn app."""

import joblib
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from sklearnexample.task import (
    create_log_reg_and_instantiate_parameters,
    get_model_params,
    set_model_params,
)

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    model = create_log_reg_and_instantiate_parameters(penalty)
    # Construct ArrayRecord representation
    arrays = ArrayRecord(get_model_params(model))

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model parameters
    print("\nSaving final model to disk...")
    ndarrays = result.arrays.to_numpy_ndarrays()
    set_model_params(model, ndarrays)
    joblib.dump(model, "logreg_model.pkl")
