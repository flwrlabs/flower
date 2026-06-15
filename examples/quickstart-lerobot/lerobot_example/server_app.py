"""lerobot_example: A Flower / Hugging Face LeRobot app."""

from datetime import datetime
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.datasets.utils import get_safe_version

from lerobot_example.task import get_dataset_metadata, get_policy_components


class EvalEveryFedAvg(FedAvg):
    """FedAvg strategy that only runs evaluation every N rounds."""

    def __init__(self, eval_every: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eval_every = max(1, int(eval_every))

    def configure_evaluate(self, server_round, arrays, config, grid):
        if server_round % self.eval_every != 0:
            return []
        return super().configure_evaluate(server_round, arrays, config, grid)


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_fit = float(context.run_config["fraction-fit"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    eval_every = int(context.run_config["eval-every"])
    repo_id = context.run_config["dataset-name"]
    server_device = torch.device(context.run_config["server-device"])

    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path(f"outputs/{folder_name}")
    save_path.mkdir(parents=True, exist_ok=True)
    eval_root = save_path / "evaluate"
    eval_root.mkdir(parents=True, exist_ok=True)

    revision = get_safe_version(repo_id, CODEBASE_VERSION)
    meta = get_dataset_metadata(repo_id, revision)
    policy, preprocessor, postprocessor, _cfg = get_policy_components(
        meta, server_device
    )
    arrays = ArrayRecord(policy.state_dict())

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
        if server_round == 0 or server_round % eval_every != 0:
            return None
        round_dir = save_path / "global_model" / f"round_{server_round}"
        round_dir.mkdir(parents=True, exist_ok=True)
        policy.load_state_dict(arrays.to_torch_state_dict())
        policy.save_pretrained(round_dir)
        preprocessor.save_pretrained(round_dir)
        postprocessor.save_pretrained(round_dir)
        return None

    strategy = EvalEveryFedAvg(
        eval_every=eval_every,
        fraction_train=fraction_fit,
        fraction_evaluate=fraction_evaluate,
    )

    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord(),
        evaluate_config=ConfigRecord({"eval-root": str(eval_root)}),
        evaluate_fn=evaluate_fn,
    )
