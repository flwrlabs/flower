import numpy as np
from flwr.common import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression

# This information is needed to create a correct scikit-learn model
UNIQUE_LABELS = [0, 1, 2]
FEATURES = ["petal_length", "petal_width", "sepal_length", "sepal_width"]


def get_model_params(model: LogisticRegression) -> NDArrays:
    """Return the parameters of a sklearn LogisticRegression model."""
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
    """Set the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression, n_classes: int, n_features: int):
    """Set initial parameters as zeros.

    Required since model params are uninitialized until model.fit is called but server
    asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def create_log_reg_and_instantiate_parameters(penalty):
    model = LogisticRegression(
        penalty=penalty,
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting,
        solver="saga",
    )
    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model, n_features=len(FEATURES), n_classes=len(UNIQUE_LABELS))
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load the data for the given partition."""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="hitorilabs/iris", partitioners={"train": partitioner}
        )
    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    X = dataset[FEATURES]
    y = dataset["species"]
    # Split the on-edge data: 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]
    return X_train.values, y_train.values, X_test.values, y_test.values
