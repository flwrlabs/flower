"""Utility functions and data loading for the scikit‑learn Flower example.

This module defines helper functions for working with scikit‑learn
``LogisticRegression`` models in the context of Flower federated
learning. It exposes functions to get and set model parameters, create
a model with initialized weights, and load federated partitions of the
iris dataset. This file is largely unchanged from the original example
because it operates independently of Flower’s API changes.
"""

import numpy as np
from flwr.common import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression

# This information is needed to create a correct scikit‑learn model
UNIQUE_LABELS = [0, 1, 2]
FEATURES = ["petal_length", "petal_width", "sepal_length", "sepal_width"]


def get_model_parameters(model: LogisticRegression) -> NDArrays:
    """Return the parameters of a scikit‑learn ``LogisticRegression`` model."""

    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(model: LogisticRegression, params: NDArrays) -> LogisticRegression:
    """Set the parameters of a scikit‑learn ``LogisticRegression`` model."""

    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression, n_classes: int, n_features: int) -> None:
    """Set initial parameters to zeros on an untrained model."""

    model.classes_ = np.array([i for i in range(n_classes)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def create_log_reg_and_instantiate_parameters(penalty: str) -> LogisticRegression:
    """Create a ``LogisticRegression`` model and initialize its weights."""

    model = LogisticRegression(
        penalty=penalty,
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
        solver="saga",
    )
    # Setting initial parameters, akin to model.compile for Keras models
    set_initial_params(model, n_features=len(FEATURES), n_classes=len(UNIQUE_LABELS))
    return model


# Global variable to cache the FederatedDataset instance
fds = None  # type: FederatedDataset | None


def load_data(partition_id: int, num_partitions: int):
    """Load the data for a given federated partition."""
    
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="hitorilabs/iris", partitioners={"train": partitioner}
        )
    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    X = dataset[FEATURES]
    y = dataset["species"]
    # Split the on‑device data: 80% train, 20% test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, y_train, X_test, y_test
