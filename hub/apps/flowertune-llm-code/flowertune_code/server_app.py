"""flowertune-code: A Flower / FlowerTune app."""

import os
from datetime import datetime

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from flowertune_code.dataset import replace_keys
from flowertune_code.models import get_model
from flowertune_code.strategy import FlowerTuneLlm

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Get initial model weights
    init_model = get_model(cfg.model)
    arrays = ArrayRecord(get_peft_model_state_dict(init_model))

    # Define strategy
    strategy = FlowerTuneLlm(
        fraction_train=cfg.strategy.fraction_train,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
    )

    # Start strategy, run FedAvg for `num_rounds`
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"save_path": save_path}),
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(
            cfg.model, cfg.train.save_every_round, num_rounds, save_path
        ),
    )


# Get function that will be executed by the strategy
# Here we use it to save global model checkpoints
def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path):
    """Return an evaluation function for saving global model."""

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        # Save model
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model = get_model(model_cfg)
            set_peft_model_state_dict(model, arrays.to_torch_state_dict())

            model.save_pretrained(f"{save_path}/peft_{server_round}")

        return MetricRecord()

    return evaluate
