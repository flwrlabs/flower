"""MerkleTrim: Verifiable and Byzantine-Robust Federated Learning Framework."""

import warnings

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from merkleTrim.task import (
    get_model,
    get_model_params,
    load_data,
    set_initial_params,
    set_model_params,
)

import hashlib
import os


# Flower ClientApp
app = ClientApp()


@app.train()
def fit(msg: Message, context: Context):
    """Train the model on local data."""

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)
    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)

    # Apply received parameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, _, y_train, _ = load_data(partition_id, num_partitions)

    # Read server_round and malicious_fraction from config (injected by VerifiableRobustStrategy)
    config = msg.content.get("config", {})
    server_round = int(config.get("server_round", 0))
    malicious_fraction = float(config.get("malicious_fraction", 0.0))

    # Determine malicious_type
    n_malicious = int(malicious_fraction * num_partitions)
    malicious_type = int(config.get("malicious_type", 0))

    if malicious_type == 0 and n_malicious > 0:
        # First n_malicious partitions (by id) are attackers (Type 2: label flipping)
        malicious_type = 2 if partition_id < n_malicious else 0
    elif malicious_type == 0:
        # Fallback simulation testing if malicious_fraction is 0:
        # - Partition 1: Type 1 attacker (tamper weights after committing) -> Caught by Strategy Layer 1
        # - Partition 2: Type 2 attacker (commit honestly to bad data)     -> Caught by Strategy Layer 2
        if partition_id == 1:
            malicious_type = 1
        elif partition_id == 2:
            malicious_type = 2

    # Ignore convergence failure due to low local epochs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Train the model on local data
        model.fit(X_train, y_train)

    # Get trained weights
    weights = get_model_params(model)

    # ── LAYER 1 (Commitment Phase): compute commitment over honest weights ────────
    nonce = os.urandom(32)
    weights_bytes = b''.join(w.tobytes() for w in weights)
    commitment = hashlib.sha256(
        weights_bytes + nonce + server_round.to_bytes(4, 'big')
    ).hexdigest()

    # ── ATTACK SIMULATION: Apply malicious behavior AFTER committing ───────────────────────────
    if malicious_type == 1:
        # Type 1: tamper after committing — server hash check (Layer 1) catches mismatch
        reveal_weights = [w * 50.0 for w in weights]
    elif malicious_type == 2:
        # Type 2: commit honestly to bad data (retrain on flipped labels)
        y_flipped = (y_train.max() - y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_flipped)
        bad_weights = get_model_params(model)
        
        # Compute commitment honestly over poisoned weights
        nonce = os.urandom(32)
        bad_bytes = b''.join(w.tobytes() for w in bad_weights)
        commitment = hashlib.sha256(
            bad_bytes + nonce + server_round.to_bytes(4, 'big')
        ).hexdigest()
        reveal_weights = bad_weights
    else:
        reveal_weights = weights

    # Compute train loss
    y_train_pred_proba = model.predict_proba(X_train)
    train_logloss = log_loss(y_train, y_train_pred_proba)

    # Construct and return reply Message with commitment + nonce
    model_record = ArrayRecord(reveal_weights)
    # MetricRecord only accepts int/float — strings go in ConfigRecord
    metrics = {
        "num-examples": len(X_train),
        "train_logloss": train_logloss,
        "malicious_type": malicious_type,
    }
    metric_record = MetricRecord(metrics)
    # Send commitment & nonce as strings via ConfigRecord
    commit_info = ConfigRecord({
        "commitment": commitment,
        "nonce": nonce.hex(),
    })
    content = RecordDict({"arrays": model_record, "metrics": metric_record, "commit_info": commit_info})
    return Message(content=content, reply_to=msg)



@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on test data."""

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)

    # Apply received pararameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, X_test, _, y_test = load_data(partition_id, num_partitions)

    # Evaluate the model on local data
    y_train_pred = model.predict(X_test)
    y_train_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_train_pred)
    loss = log_loss(y_test, y_train_pred_proba)
    precision = precision_score(y_test, y_train_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_train_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_train_pred, average="macro", zero_division=0)

    # Construct and return reply Message
    metrics = {
        "num-examples": len(X_test),
        "test_logloss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)


