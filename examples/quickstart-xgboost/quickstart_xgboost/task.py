"""quickstart_xgboost: A Flower / XGBoost app."""

import numpy as np
import xgboost as xgb
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


def train_test_split(partition, test_fraction, seed):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data):
    """Transform dataset to DMatrix format for xgboost."""
    batch = data[:]
    x = np.asarray(batch["inputs"], dtype=np.float32)
    y = np.asarray(batch["label"], dtype=np.float32)
    return xgb.DMatrix(x, label=y)


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_clients):
    """Load partition HIGGS data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_clients)
        fds = FederatedDataset(
            dataset="jxie/higgs",
            partitioners={"train": partitioner},
        )

    # Load the partition for this `partition_id`
    partition = fds.load_partition(partition_id, split="train")
    partition.set_format("numpy")

    # Train/test splitting
    train_data, valid_data, num_train, num_val = train_test_split(
        partition, test_fraction=0.2, seed=42
    )

    # Reformat data to DMatrix for xgboost
    train_dmatrix = transform_dataset_to_dmatrix(train_data)
    valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

    return train_dmatrix, valid_dmatrix, num_train, num_val


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
