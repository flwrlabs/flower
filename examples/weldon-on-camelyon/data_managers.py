import math
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import Dataset


# Duplicated in substrafl/benchmark/camelyon/pure_substrafl/assets/opener.py
class Data:
    def __init__(self, paths: List[Path]):
        indexes = list()
        for path in paths:
            index_path = Path(path) / "index.csv"
            assert index_path.is_file(), "Wrong data sample, it must contain index.csv"
            ds_indexes = np.loadtxt(index_path, delimiter=",", dtype=object)
            ds_indexes[:, 0] = np.array([str(Path(path) / x) for x in ds_indexes[:, 0]])
            indexes.extend(ds_indexes)

        self._indexes = np.asarray(indexes, dtype=object)

    @property
    def indexes(self):
        return self._indexes

    def __len__(self):
        return len(self.indexes)


class CamelyonDataset(Dataset):
    """Torch Dataset for the Camelyon data.

    Padding is done on the fly.
    """

    def __init__(self, datasamples, is_inference=False) -> None:
        data_indexes = datasamples.indexes
        self.data_indexes = (
            data_indexes
            if len(data_indexes.shape) > 1
            else np.array(data_indexes).reshape(1, data_indexes.shape[0])
        )
        self.is_inference = is_inference

    def __len__(self):
        return len(self.data_indexes)

    def __getitem__(self, index):
        """Get the needed item from index and preprocess them on the fly."""
        sample_file_path, target = self.data_indexes[index]
        x = torch.from_numpy(np.load(sample_file_path).astype(np.float32))

        y = torch.tensor(int(target == "Tumor")).type(torch.float32)

        missing_tiles = 25000 - x.shape[0]
        assert (
            missing_tiles > 0
        ), f"The padding value is too low, got {x.shape[0]} in this sample."

        up = math.ceil(missing_tiles / 2)
        down = missing_tiles // 2

        x = F.pad(input=x, pad=(0, 0, up, down), mode="constant", value=0)

        if self.is_inference:
            return x

        return x, y
