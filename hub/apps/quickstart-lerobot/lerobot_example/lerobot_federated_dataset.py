"""lerobot_example: A Flower / Hugging Face LeRobot app."""

from pathlib import Path
from typing import Callable

from datasets import Dataset
from lerobot.common.datasets.lerobot_dataset import (
    CODEBASE_VERSION,
    DATA_DIR,
    LeRobotDataset,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    load_info,
    load_stats,
    load_videos,
    reset_episode_index,
)


class FilteredLeRobotDataset(LeRobotDataset):
    """Behaves like `LeRobotDataset` but using the dataset partition passed during
    construction."""

    def __init__(
        self,
        repo_id: str,
        hf_dataset: Dataset,
        root: Path | None = DATA_DIR,
        split: str = "train",
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        video_backend: str | None = None,
    ):
        self.repo_id = repo_id
        self.root = root
        self.split = split
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.video_backend = video_backend
        self.hf_dataset = hf_dataset

        # after filtering, the stored episode data index may not be the same
        # so let's calculate it on the filtered data
        self.episode_data_index = calculate_episode_data_index(self.hf_dataset)
        self.hf_dataset = reset_episode_index(self.hf_dataset)

        self.stats = load_stats(self.repo_id, CODEBASE_VERSION, self.root)
        self.info = load_info(self.repo_id, CODEBASE_VERSION, self.root)
        if self.video:
            self.videos_dir = load_videos(self.repo_id, CODEBASE_VERSION, self.root)
            self.video_backend = (
                self.video_backend if self.video_backend is not None else "pyav"
            )
