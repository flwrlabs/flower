"""lerobot_example: A Flower / Hugging Face LeRobot app."""

from datetime import datetime
from pathlib import Path

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from lerobot_example.task import get_dataset, get_model, get_params, set_params


def get_evaluate_fn_callback(save_path: Path):

    def evaluate_fn(server_round: int, parameters, config):

        # Save model if round % 5 = 0 (exactly as frequently as model is evaluated)
        if server_round % 5 == 0:
            # Instantiate model
            dataset = get_dataset()
            model = get_model(dataset_stats=dataset.stats)
            # Apply current global model weights
            set_params(model, parameters)
            # Save checkpoint
            model.save_pretrained(
                str(save_path / "global_model" / f"round_{server_round}")
            )

    return evaluate_fn


def get_evaluate_config_callback(save_path: Path):
    """Return a function to configure an evaluate round."""

    def evaluate_config_fn(server_round: int) -> Metrics:
        eval_save_path = save_path / "evaluate" / f"round_{server_round}"
        return {"save_path": str(eval_save_path), "skip": server_round % 5 != 0}

    return evaluate_config_fn


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""
    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    # Create output directory given current timestamp
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path(f"outputs/{folder_name}")
    save_path.mkdir(parents=True)

    # Set global model initialization
    dataset = get_dataset()
    ndarrays = get_params(get_model(dataset_stats=dataset.stats))
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define strategy
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        initial_parameters=global_model_init,
        on_evaluate_config_fn=get_evaluate_config_callback(save_path),
        evaluate_fn=get_evaluate_fn_callback(save_path),
    )

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
