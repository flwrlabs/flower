"""lerobot_example: A Flower / Hugging Face LeRobot app."""

import warnings
from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.datasets.utils import get_safe_version
from transformers import logging as hf_logging

from lerobot_example.task import (
    get_dataset_metadata,
    get_policy_components,
    load_data,
    test,
    train,
)

warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()

# Flower ClientApp
app = ClientApp()


def _build_policy(repo_id: str, device: torch.device):
    revision = get_safe_version(repo_id, CODEBASE_VERSION)
    meta = get_dataset_metadata(repo_id, revision)
    policy, preprocessor, postprocessor, _cfg = get_policy_components(meta, device)
    if device.type == "cpu":
        # Decrease the number of reverse-diffusion steps (trades off a bit of quality for 10x speed)
        policy.diffusion.num_inference_steps = 10
    return policy, preprocessor, postprocessor


@app.train()
def train_client(msg: Message, context: Context) -> Message:
    """Train the model on local data."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    repo_id = context.run_config["dataset-name"]
    local_epochs = int(context.run_config["local-epochs"])

    policy, preprocessor, _postprocessor = _build_policy(repo_id, device)
    policy.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader = load_data(partition_id, num_partitions, repo_id, device)

    train_loss = train(
        net=policy,
        trainloader=trainloader,
        epochs=local_epochs,
        preprocessor=preprocessor,
    )

    model_record = ArrayRecord(policy.state_dict())
    metrics = MetricRecord(
        {"train_loss": train_loss, "num-examples": len(trainloader.dataset)}
    )
    content = RecordDict({"arrays": model_record, "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_client(msg: Message, context: Context) -> Message:
    """Evaluate the model on local data."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    repo_id = context.run_config["dataset-name"]

    policy, preprocessor, postprocessor = _build_policy(repo_id, device)
    policy.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    partition_id = context.node_config["partition-id"]
    eval_root = Path(msg.content["config"]["eval-root"])
    server_round = int(msg.content["config"]["server-round"])
    eval_save_path = eval_root / f"round_{server_round}"
    eval_save_path.mkdir(parents=True, exist_ok=True)

    loss, accuracy = test(
        partition_id=partition_id,
        net=policy,
        output_dir=eval_save_path,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    metrics = MetricRecord(
        {"loss": float(loss), "accuracy": float(accuracy), "num-examples": 1}
    )
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)
