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
"""Download and reconstruct FLAD dataset from Hugging Face Hub.

Download the FLAD Parquet dataset from the Hugging Face Hub and reconstruct
the per-client HDF5 layout.
"""

import argparse
import os

import h5py
import numpy as np
from datasets import Dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner

SPLITS = ("train", "val", "test")
# Map our local split naming to the split names HF's auto-detection assigned
# when the Parquet files were uploaded (it normalizes "val" to "validation").
HF_SPLIT_NAME = {"train": "train", "val": "validation", "test": "test"}
HDF5_FILENAME = "10t-10n-DOS2019-dataset-{split}.hdf5"
FEATURE_SHAPE = (10, 11)


def reconstruct_arrays(
    partition: Dataset, feature_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a partition's flattened `features` column back into (N, P, F)."""
    X_flat = np.array(partition["features"], dtype=np.float32)
    X = X_flat.reshape(-1, *feature_shape)
    Y = np.array(partition["label"], dtype=np.int64)
    return X, Y


def write_hdf5(path: str, X: np.ndarray, Y: np.ndarray) -> None:
    """Write features/labels to disk in the original `set_x`/`set_y` layout."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("set_x", data=X)
        f.create_dataset("set_y", data=Y)


def download_and_convert(
    repo_id: str, output_folder: str, feature_shape: tuple[int, int]
) -> None:
    """Download `repo_id` from the HF Hub and rebuild per-client HDF5 files."""
    partitioners = {
        HF_SPLIT_NAME[split]: NaturalIdPartitioner(partition_by="client_id")
        for split in SPLITS
    }
    fds = FederatedDataset(dataset=repo_id, partitioners=partitioners, shuffle=False)

    for split in SPLITS:
        hf_split = HF_SPLIT_NAME[split]
        partitioner = fds.partitioners[hf_split]
        for partition_id in range(partitioner.num_partitions):
            client_id = partitioner.partition_id_to_natural_id[partition_id]
            partition = fds.load_partition(partition_id, hf_split)
            X, Y = reconstruct_arrays(partition, feature_shape)

            out_path = os.path.join(
                output_folder, client_id, HDF5_FILENAME.format(split=split)
            )
            write_hdf5(out_path, X, Y)
            print(f"Wrote {out_path}: {X.shape[0]} samples")


def main() -> None:
    """Parse CLI arguments and run the dataset download/conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HF Hub dataset repo id",
    )
    parser.add_argument(
        "--output-folder",
        default="./dataset/DOS2019_highly_unbalanced",
        help="Root folder under which per-client subfolders are created",
    )
    args = parser.parse_args()
    download_and_convert(args.repo_id, args.output_folder, FEATURE_SHAPE)


if __name__ == "__main__":
    main()
