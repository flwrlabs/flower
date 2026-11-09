"""MerkleTrim: Verifiable and Byzantine-Robust Federated Learning Framework."""

import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression

fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition MNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
        )

    dataset = fds.load_partition(partition_id, "train").with_format("numpy")

    X, y = dataset["image"][:].reshape((len(dataset), -1)), dataset["label"][:]

    # Split the on edge data: 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]

    return X_train, X_test, y_train, y_test


def get_model(penalty: str, local_epochs: int):

    return LogisticRegression(
        penalty=penalty,
        max_iter=local_epochs,
        warm_start=True,
    )


def get_model_params(model):
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [model.coef_]
    return params


def set_model_params(model, params):
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model):
    n_classes = 10  # MNIST has 10 classes
    n_features = 784  # Number of features in dataset
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
