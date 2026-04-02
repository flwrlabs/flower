"""sklearnexample: A Flower / sklearn app."""

import warnings

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import log_loss

from sklearnexample.task import (
    UNIQUE_LABELS,
    create_log_reg_and_instantiate_parameters,
    get_model_params,
    load_data,
    set_model_params,
)

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    # Create LogisticRegression Model
    model = create_log_reg_and_instantiate_parameters(penalty)

    # Apply received pararameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, y_train, _, _ = load_data(partition_id, num_partitions)

    # Ignore convergence failure due to low local epochs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Train the model on local data
        model.fit(X_train, y_train)

    # Let's compute train loss
    y_train_pred_proba = model.predict_proba(X_train)
    train_logloss = log_loss(y_train, y_train_pred_proba, labels=UNIQUE_LABELS)
    accuracy = model.score(X_train, y_train)

    # Construct and return reply Message
    ndarrays = get_model_params(model)
    model_record = ArrayRecord(ndarrays)
    metrics = {
        "num-examples": len(X_train),
        "train_logloss": train_logloss,
        "train_accuracy": accuracy,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    # Create LogisticRegression Model
    model = create_log_reg_and_instantiate_parameters(penalty)

    # Apply received pararameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, _, X_test, y_test = load_data(partition_id, num_partitions)

    # Evaluate the model on local data
    y_test_pred_proba = model.predict_proba(X_test)
    accuracy = model.score(X_test, y_test)
    loss = log_loss(y_test, y_test_pred_proba, labels=UNIQUE_LABELS)

    # Construct and return reply Message
    metrics = {
        "num-examples": len(X_test),
        "test_logloss": loss,
        "accuracy": accuracy,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
