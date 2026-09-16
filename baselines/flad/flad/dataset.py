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

import glob
import os
import random
from typing import cast

import h5py
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

CLASSES = np.array([0, 1])


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data(client: dict, validation_only: bool = False) -> None:
    """Load the training and validation data for a given client."""
    if not validation_only:
        x_train, y_train = _load_set(
            client["dataset_folder"] + "/" + client["name"], "train", client["rn_seed"]
        )
        x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
        client["training"] = (x_train_tensor, y_train)
        client["training_samples"] = client["training"][1].shape[0]

    x_val, y_val = _load_set(
        client["dataset_folder"] + "/" + client["name"], "val", client["rn_seed"]
    )
    x_val_tensor = tf.convert_to_tensor(x_val, dtype=tf.float32)
    client["validation"] = (x_val_tensor, y_val)
    client["validation_samples"] = client["validation"][1].shape[0]


def _load_dataset(filename: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(filename, "r") as dataset:
        set_x = cast(h5py.Dataset, dataset["set_x"])
        set_y = cast(h5py.Dataset, dataset["set_y"])
        set_x_orig = np.array(set_x[:])  # features
        set_y_orig = np.array(set_y[:])  # labels

    if len(set_x_orig.shape) == 3:  # array-like data with no channels
        x_train = np.reshape(
            set_x_orig,
            (set_x_orig.shape[0], set_x_orig.shape[1], set_x_orig.shape[2], 1),
        )
    else:
        x_train = set_x_orig
    y_train = set_y_orig

    return x_train, y_train


def _load_set(
    data_folder: str, set_type: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    set_list = []

    # Assume only one folder where data files are stored.
    files = glob.glob(data_folder + "/*" + "-" + set_type + ".hdf5")

    if not files:
        raise ValueError(f"No '{set_type}' dataset files found in '{data_folder}' ")

    for file in files:
        X, Y = _load_dataset(file)
        if not np.array_equal(np.unique(Y), CLASSES):
            raise ValueError("Mismatching classes among datasets!")
        set_list.append((X, Y))

    # Concatenation of all the training and validation sets
    X = set_list[0][0]
    Y = set_list[0][1]
    for n in range(1, len(set_list)):
        X = np.concatenate((X, set_list[n][0]), axis=0)
        Y = np.concatenate((Y, set_list[n][1]), axis=0)

    X, Y = shuffle(X, Y, random_state=seed)
    return X, Y
