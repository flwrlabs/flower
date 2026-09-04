#
# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""flad: A Flower Baseline."""

from logging import INFO

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.common import log
from flwr.common.logger import update_console_handler
from flwr.serverapp import Grid, ServerApp

from .dataset import set_seed
from .model import load_model
from .strategy import Flad
from .utils import make_run_output_folder, save_training_history

# Create the ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Execute ServerApp."""
    # Configure logging
    update_console_handler(level=INFO, timestamps=True)

    # Load config (load all paramters for logging purposes)
    log(INFO, "Loading configuration for Flad strategy...")
    client_names = str(context.run_config["client_names"])
    min_epochs = int(context.run_config["min_epochs"])
    max_epochs = int(context.run_config["max_epochs"])
    min_steps = int(context.run_config["min_steps"])
    max_steps = int(context.run_config["max_steps"])
    rn_seed = int(context.run_config["rn_seed"])
    patience = int(context.run_config["patience"])
    output_folder = str(context.run_config["output_folder"])
    num_rounds = int(context.run_config["num-server-rounds"])

    # Create, if needed, the output folder for logs and models
    output_folder = make_run_output_folder(output_folder, rn_seed)

    # Set the seed for reproducibility
    set_seed(rn_seed)

    # Load initial model
    log(INFO, "Loading initial model...")
    model = load_model()
    arrays = ArrayRecord(model.get_weights())

    # Define and start Flad strategy
    strategy = Flad(client_names)

    # Create a train_config record
    train_config = ConfigRecord(
        {
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
            "min_steps": min_steps,
            "max_steps": max_steps,
            "patience": patience,
        }
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        train_config=train_config,
    )

    # Save the final model
    ndarrays = result.arrays.to_numpy_ndarrays()
    model.set_weights(ndarrays)
    final_model_name = "model.keras"
    log(INFO, "Saving final model to disk as '%s'", final_model_name)
    model.save(output_folder + "/" + final_model_name)

    # Save the training history
    log(INFO, "Saving training history for seed '%d'", rn_seed)
    save_training_history(output_folder, rn_seed, result.evaluate_metrics_clientapp)
