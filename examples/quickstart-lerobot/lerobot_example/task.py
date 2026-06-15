"""lerobot_example: A Flower / Hugging Face LeRobot app."""

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy as np
import torch
from datasets.utils.logging import disable_progress_bar
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import GroupedNaturalIdPartitioner
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import (
    CODEBASE_VERSION,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.datasets.utils import dataset_to_policy_features, get_safe_version
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE
from torch.utils.data import DataLoader

disable_progress_bar()
fds = None  # Cache FederatedDataset


@lru_cache(maxsize=None)
def get_dataset_metadata(repo_id: str, revision: str) -> LeRobotDatasetMetadata:
    """Load and cache LeRobot dataset metadata."""
    return LeRobotDatasetMetadata(repo_id, revision=revision)


def get_policy_config(
    meta: LeRobotDatasetMetadata, device: torch.device
) -> DiffusionConfig:
    """Build a Diffusion policy config from dataset metadata."""
    features = dataset_to_policy_features(meta.features)
    output_features = {
        k: ft for k, ft in features.items() if ft.type == FeatureType.ACTION
    }
    input_features = {k: ft for k, ft in features.items() if k not in output_features}
    return DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        device=str(device),
    )


def get_policy_components(meta: LeRobotDatasetMetadata, device: torch.device):
    """Build policy + preprocessors for training/inference."""
    cfg = get_policy_config(meta, device)
    policy = DiffusionPolicy(cfg)
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=meta.stats
    )
    return policy, preprocessor, postprocessor, cfg


def get_delta_timestamps(
    cfg: DiffusionConfig, meta: LeRobotDatasetMetadata
) -> dict[str, list[float]]:
    """Compute delta timestamps in seconds for observations/actions."""
    obs_delta = [i / meta.fps for i in cfg.observation_delta_indices]
    action_delta = [i / meta.fps for i in cfg.action_delta_indices]
    return {
        OBS_IMAGE: obs_delta,
        OBS_STATE: obs_delta,
        ACTION: action_delta,
    }


def get_partition_episodes(
    partition_id: int, num_partitions: int, repo_id: str, revision: str
) -> list[int]:
    """Load episode ids for a specific partition using Flower Datasets."""
    global fds
    if fds is None:
        meta = get_dataset_metadata(repo_id, revision)
        episodes_per_partition = max(1, meta.total_episodes // num_partitions)
        partitioner = GroupedNaturalIdPartitioner(
            partition_by="episode_index", group_size=episodes_per_partition
        )
        fds = FederatedDataset(
            dataset=repo_id,
            partitioners={"train": partitioner},
            revision=revision,
        )

    partition = fds.load_partition(partition_id)
    if hasattr(partition, "unique"):
        episode_ids = partition.unique("episode_index")
    else:
        episode_ids = sorted({int(x) for x in partition["episode_index"]})
    return sorted(int(x) for x in episode_ids)


def load_data(
    partition_id: int, num_partitions: int, repo_id: str, device: torch.device
) -> DataLoader[Any]:
    """Load pusht data (training only)."""
    revision = get_safe_version(repo_id, CODEBASE_VERSION)
    meta = get_dataset_metadata(repo_id, revision)
    cfg = get_policy_config(meta, device)
    delta_timestamps = get_delta_timestamps(cfg, meta)

    episodes = get_partition_episodes(partition_id, num_partitions, repo_id, revision)
    dataset = LeRobotDataset(
        repo_id,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        revision=revision,
    )

    trainloader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )
    return trainloader


def train(net=None, trainloader=None, epochs=None, preprocessor=None) -> float:
    """Train the policy for a given number of optimization steps."""
    log_freq = 250
    policy = net
    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    step = 0
    done = False
    loss_value = 0.0
    while not done:
        for batch in trainloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_value = float(loss.item())

            if step % log_freq == 0 and step > 0:
                print(f"train step: {step} train loss: {loss_value:.3f}")
            step += 1
            assert isinstance(
                epochs, int
            ), f"epochs value: {epochs} , type: {epochs.__class__}"
            if step >= epochs:
                done = True
                break
    return loss_value


def test(
    partition_id: int,
    net,
    output_dir: Path,
    preprocessor,
    postprocessor,
) -> tuple[Any | float, Any]:
    """Evaluate policy by running a rollout in the PushT env."""
    policy = net
    policy.eval()
    policy.reset()

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=300,
    )

    numpy_observation, _info = env.reset(seed=42)

    rewards = []
    frames = []
    frames.append(env.render())

    step = 0
    done = False
    while not done:
        state = torch.from_numpy(numpy_observation["agent_pos"]).to(torch.float32)
        image = torch.from_numpy(numpy_observation["pixels"]).to(torch.float32) / 255
        image = image.permute(2, 0, 1)

        observation = {
            OBS_STATE: state,
            OBS_IMAGE: image,
        }

        with torch.inference_mode():
            processed_obs = preprocessor(observation)
            action = policy.select_action(processed_obs)
            action = postprocessor(action)

        numpy_action = action.squeeze(0).to("cpu").numpy()

        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
        print(f"{step=} {reward=} {terminated=} {info=}")

        rewards.append(reward)
        frames.append(env.render())

        done = terminated | truncated | done
        step += 1

    if terminated:
        print("Success! Robot completed the task.")
    else:
        print("Failure! Robot did not complete the task.")

    fps = env.metadata["render_fps"]

    timestr = time.strftime("%Y%m%d-%H%M%S")
    video_dir = output_dir / f"client_{partition_id}"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"rollout_{timestr}.mp4"
    imageio.mimsave(str(video_path), np.stack(frames), fps=fps)

    env.close()

    print(f"Video of the evaluation is available in '{video_path}'.")

    accuracy = np.max(rewards)
    loss = 1 - accuracy
    return loss, accuracy
