"""MerkleTrim: Verifiable and Byzantine-Robust Federated Learning Framework."""

import joblib
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from merkleTrim.custom_strategy import VerifiableRobustStrategy

from merkleTrim.task import get_model, get_model_params, set_initial_params, set_model_params

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)
    set_initial_params(model)
    arrays = ArrayRecord(get_model_params(model))

    # ── Read experiment parameters from run_config, with safe fallbacks ────
    # CLI parameter overrides:
    #   flwr run . --run-config "beta=0.2 malicious-fraction=0.3"
    # Configured under [tool.flwr.app.config] in pyproject.toml
    beta = float(context.run_config.get("beta", 0.2))
    malicious_fraction = float(context.run_config.get("malicious-fraction", 0.0))

    print(f"[CONFIG] beta={beta} malicious_fraction={malicious_fraction}")

    # ── Both parameters passed explicitly — no silent fallback to class defaults ──
    strategy = VerifiableRobustStrategy(
        beta=beta,
        malicious_fraction=malicious_fraction,
    )

    # Start strategy, run for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model parameters
    print("\nSaving final model to disk...")
    ndarrays = result.arrays.to_numpy_ndarrays()
    if ndarrays:
        set_model_params(model, ndarrays)
        joblib.dump(model, "logreg_model.pkl")
        print("Model saved successfully.")
    else:
        print("WARNING: No verified weights were aggregated. Model NOT saved.")

    strategy.print_ledger()