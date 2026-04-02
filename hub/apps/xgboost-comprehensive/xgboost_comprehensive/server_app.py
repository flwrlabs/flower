"""xgboost_comprehensive: A Flower / XGBoost app."""

import numpy as np
import xgboost as xgb
from datasets import load_dataset
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging, FedXgbCyclic

from xgboost_comprehensive.task import replace_keys, transform_dataset_to_dmatrix

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Read from config

    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    train_method = context.run_config["train-method"]
    centralised_eval = context.run_config["centralised-eval"]

    # Flatted config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    if centralised_eval:
        # This is the exact same dataset as the one downloaded by the clients via
        # FlowerDatasets. However, we don't use FlowerDatasets for the server since
        # partitioning is not needed.
        # We make use of the "test" split only
        test_set = load_dataset("jxie/higgs")["test"]
        test_set.set_format("numpy")
        test_dmatrix = transform_dataset_to_dmatrix(test_set)

    # Init global model
    global_model = b""  # Init with an empty object; the XGBooster will be created and trained on the client side.
    # Note: we store the model as the first item in a list into ArrayRecord,
    # which can be accessed using index ["0"].
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])

    # Define strategy
    if train_method == "bagging":
        # Bagging training
        strategy = FedXgbBagging(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
        )
    else:
        # Cyclic training
        strategy = FedXgbCyclic()

    # Start strategy, run for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=(
            get_evaluate_fn(test_dmatrix, params) if centralised_eval else None
        ),
    )

    # Save final model to disk
    bst = xgb.Booster(params=params)
    global_model = bytearray(result.arrays["0"].numpy().tobytes())

    # Load global model into booster
    bst.load_model(global_model)

    # Save model
    print("\nSaving final model to disk...")
    bst.save_model("final_model.json")


def get_evaluate_fn(test_data, params):
    """Return a function for centralised evaluation."""

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:

        # Skip init eval
        if server_round == 0:
            return

        # Load global model
        global_model = bytearray(arrays["0"].numpy().tobytes())
        bst = xgb.Booster(params=params)
        bst.load_model(global_model)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(test_data, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        return MetricRecord({"AUC": auc})

    return evaluate_fn
