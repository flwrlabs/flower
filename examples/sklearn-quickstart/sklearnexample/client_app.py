"""sklearnexample: A Flower / sklearn app."""

import warnings
from typing import List

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import log_loss
from sklearnexample.task import (
    UNIQUE_LABELS,
    create_log_reg_and_instantiate_parameters,
    get_model_parameters,
    load_data,
    set_model_params,
)

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the model on local data."""

    # Read the node configuration to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, y_train, _, _ = load_data(partition_id, num_partitions)

    # Read hyperparameter from run configuration
    penalty = context.run_config["penalty"]

    # Create a fresh logistic regression model and load global weights
    model = create_log_reg_and_instantiate_parameters(penalty)
    arrays: ArrayRecord = msg.content["arrays"]  
    ndarrays = arrays.to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # Train the model on the local partition
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train.values, y_train.values)

    # Compute a simple training metric (accuracy)
    train_accuracy = model.score(X_train.values, y_train.values)

    # Construct reply Message: updated weights and training metrics
    params = get_model_parameters(model)
    arrays_record = ArrayRecord.from_numpy_ndarrays(params)  # type: ignore[arg-type]
    metrics = MetricRecord(
        {
            "train_accuracy": float(train_accuracy),
            # Include number of examples so the server can perform weighted averaging
            "num-examples": len(X_train),
        }
    )
    content = RecordDict({"arrays": arrays_record, "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the model on local data."""

    # Read the node configuration to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, _, X_test, y_test = load_data(partition_id, num_partitions)

    # Read hyperparameter from run configuration
    penalty = context.run_config["penalty"]

    # Create a fresh logistic regression model and load global weights
    model = create_log_reg_and_instantiate_parameters(penalty)
    arrays: ArrayRecord = msg.content["arrays"] 
    ndarrays = arrays.to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # Perform local evaluation
    y_pred = model.predict_proba(X_test.values)
    loss = log_loss(y_test.values, y_pred, labels=UNIQUE_LABELS)
    accuracy = model.score(X_test.values, y_test.values)

    # Construct reply Message: only evaluation metrics are returned
    metrics = MetricRecord(
        {
            "eval_loss": float(loss),
            "eval_accuracy": float(accuracy),
            # Include number of examples so the server can perform weighted averaging
            "num-examples": len(X_test),
        }
    )
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)
