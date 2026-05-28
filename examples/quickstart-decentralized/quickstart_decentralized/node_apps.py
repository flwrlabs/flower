"""User-defined NodeApp instances (app1/app2) with decorators.

Demonstrates federated learning with a real scikit-learn model:
- app1 (trainer)  : local LogisticRegression training on a node-specific data partition.
- app2 (analytics): computes running accuracy statistics across cycles.

These objects are auto-loaded by SuperDNode from:
[tool.flwr.app.components]
nodeapp1 = "quickstart_decentralized.node_apps:app1"
nodeapp2 = "quickstart_decentralized.node_apps:app2"
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from flwr.common import Context, Message
from flwr.common.constant import NUM_PARTITIONS_KEY, PARTITION_ID_KEY
from flwr.common.record.array import Array
from flwr.common.record.arrayrecord import ArrayRecord
from flwr.common.record.configrecord import ConfigRecord
from flwr.common.record.metricrecord import MetricRecord
from flwr.common.record.recorddict import RecordDict
from flwr.decentralized.nodeapp import NodeApp

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:      %(message)s"))
    LOGGER.addHandler(_handler)

# ---------------------------------------------------------------------------
# Per-node model state: keyed by node_id so each node maintains its own model.
# ---------------------------------------------------------------------------
_NODE_STATE: dict[str, dict[str, Any]] = {}

N_SAMPLES = 500   # samples per node partition
N_FEATURES = 20   # input features
N_CLASSES = 2


def _get_or_init_state(node_id: str) -> dict[str, Any]:
    """Return (creating if needed) the model state for this node."""
    if node_id not in _NODE_STATE:
        # Derive a deterministic seed from the node_id so each node gets a
        # unique but reproducible data partition.
        seed = int(hashlib.md5(node_id.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.default_rng(seed)
        X, y = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=10,
            n_classes=N_CLASSES,
            random_state=int(rng.integers(0, 2**31)),
        )
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        model = LogisticRegression(max_iter=1, warm_start=True, solver="lbfgs")
        _NODE_STATE[node_id] = {
            "model": model,
            "scaler": scaler,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "cycle": 0,
            "history": [],  # list of (cycle, train_acc, test_acc, loss)
        }
        LOGGER.info(
            "[state] node_id=%s  partition: %d train / %d test samples, seed=%d",
            node_id,
            len(X_train),
            len(X_test),
            seed,
        )
    return _NODE_STATE[node_id]


# ---------------------------------------------------------------------------
# NodeApp definitions
# ---------------------------------------------------------------------------

_DEFAULT_DATA_CONFIG = ConfigRecord(
    {
        PARTITION_ID_KEY: 0,
        NUM_PARTITIONS_KEY: 1,
    }
)

app1 = NodeApp(
    subject="trainer",
    initial_arrays=ArrayRecord(),
    data_config=ConfigRecord(_DEFAULT_DATA_CONFIG),
    timeout=1,
    train_config=ConfigRecord(
        {
            "local-epochs": 3,
            "lr": 0.1,
        }
    ),
    eval_config=ConfigRecord(
        {
            "metric-name": "accuracy",
        }
    ),
    run_config={
        "rounds": 25,
    },
)

app2 = NodeApp(
    subject="analytics",
    initial_arrays=ArrayRecord(),
    data_config=ConfigRecord(_DEFAULT_DATA_CONFIG),
    timeout=1,
    train_config=ConfigRecord(
        {
            "window": 5,
        }
    ),
    eval_config=ConfigRecord(),
    run_config={
        "rounds": 25,
    },
)


def _arrays_to_record(model: LogisticRegression) -> ArrayRecord:
    """Serialize fitted LogisticRegression weights into an ArrayRecord."""
    if not hasattr(model, "coef_"):
        return ArrayRecord()
    return ArrayRecord(
        {
            "coef": Array.from_numpy_ndarray(model.coef_),
            "intercept": Array.from_numpy_ndarray(model.intercept_),
        }
    )


def _load_arrays_into_model(record: ArrayRecord, model: LogisticRegression) -> None:
    """Load weights from ArrayRecord into a LogisticRegression, if present."""
    if "coef" not in record or "intercept" not in record:
        return
    model.coef_ = record["coef"].numpy()
    model.intercept_ = record["intercept"].numpy()
    # Ensure sklearn knows the model is fitted
    model.classes_ = np.arange(model.coef_.shape[0] + 1) if model.coef_.shape[0] > 1 else np.array([0, 1])


def _get_node_id(context: Context) -> str:
    """Get a stable node identifier from Flower Context."""
    node_id = context.node_config.get("node-id")
    if node_id is None:
        node_id = context.node_config.get("node_id")
    if node_id is None:
        node_id = context.node_config.get(PARTITION_ID_KEY)

    if isinstance(node_id, str) and node_id:
        return node_id
    if isinstance(node_id, int):
        return str(node_id)
    return str(context.node_id)


@app1.train()
def train_app1(
    message: Message,
    context: Context,
) -> Message:
    """Train a local LogisticRegression for `local-epochs` iterations."""
    nid = _get_node_id(context)
    run_config = context.run_config
    subject = app1.name
    state = _get_or_init_state(nid)

    # Load peer-averaged weights from incoming message content when present.
    incoming_arrays = message.content.array_records.get(app1.strategy.arrayrecord_key)
    if incoming_arrays:
        _load_arrays_into_model(incoming_arrays, state["model"])

    local_epochs: int = int(run_config.get("local-epochs", 3))
    model: LogisticRegression = state["model"]
    # warm_start=True accumulates training; bump max_iter each cycle
    model.max_iter = (state["cycle"] + 1) * local_epochs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress ConvergenceWarning on incremental steps
        model.fit(state["X_train"], state["y_train"])

    train_acc = accuracy_score(state["y_train"], model.predict(state["X_train"]))
    train_loss = log_loss(state["y_train"], model.predict_proba(state["X_train"]))

    state["cycle"] += 1
    state["history"].append((state["cycle"], round(train_acc, 4), train_loss))

    LOGGER.info(
        "[NodeApp:%s][train] node=%.12s  cycle=%d  epochs=%d  "
        "train_acc=%.4f  train_loss=%.4f",
        subject,
        nid,
        state["cycle"],
        local_epochs,
        train_acc,
        train_loss,
    )
    return Message(
        content=RecordDict(
            {
                app1.strategy.arrayrecord_key: _arrays_to_record(model),
                "metrics": MetricRecord(
                    {
                        "train_acc": train_acc,
                        "train_loss": train_loss,
                        "num-examples": len(state["X_train"]),
                        "cycle": state["cycle"],
                    }
                ),
            }
        ),
        reply_to=message,
    )


@app1.evaluate()
def evaluate_app1(
    message: Message,
    context: Context,
) -> Message:
    """Evaluate the local model on the held-out test split."""
    nid = _get_node_id(context)
    run_config = context.run_config
    subject = app1.name
    state = _get_or_init_state(nid)
    model: LogisticRegression = state["model"]

    # Load peer-averaged weights so evaluation reflects the gossiped model.
    incoming_arrays = message.content.array_records.get(app1.strategy.arrayrecord_key)
    if incoming_arrays:
        _load_arrays_into_model(incoming_arrays, model)

    if state["cycle"] == 0:
        LOGGER.info("[NodeApp:%s][evaluate] node=%.12s  model not yet trained", subject, nid)
        return message

    test_acc = accuracy_score(state["y_test"], model.predict(state["X_test"]))
    test_loss = log_loss(state["y_test"], model.predict_proba(state["X_test"]))
    metric = run_config.get("metric-name", "accuracy")

    LOGGER.info(
        "[NodeApp:%s][evaluate] node=%.12s  cycle=%d  %s=%.4f  test_loss=%.4f",
        subject,
        nid,
        state["cycle"],
        metric,
        test_acc,
        test_loss,
    )
    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        metric: test_acc,
                        "eval_loss": test_loss,
                        "num-examples": len(state["X_test"]),
                        "cycle": state["cycle"],
                    }
                ),
            }
        ),
        reply_to=message,
    )


@app2.train()
def train_app2(
    message: Message,
    context: Context,
) -> Message:
    """Analytics app: report a rolling summary of app1's training history."""
    nid = _get_node_id(context)
    run_config = context.run_config
    subject = app2.name
    window: int = int(run_config.get("window", 5))

    # Read app1's history if available for this node
    history = _NODE_STATE.get(nid, {}).get("history", [])
    recent = history[-window:]

    if not recent:
        LOGGER.info("[NodeApp:%s][analytics] node=%.12s  no training history yet", subject, nid)
        return message

    cycles = [r[0] for r in recent]
    accs   = [r[1] for r in recent]
    losses = [r[2] for r in recent]

    LOGGER.info(
        "[NodeApp:%s][analytics] node=%.12s  last %d cycles=%s  "
        "avg_acc=%.4f  avg_loss=%.4f  trend=%s",
        subject,
        nid,
        len(recent),
        cycles,
        float(np.mean(accs)),
        float(np.mean(losses)),
        "↑" if len(accs) > 1 and accs[-1] > accs[0] else "→" if len(accs) < 2 else "↓",
    )
    return message


@app2.evaluate()
def evaluate_app2(
    message: Message,
    context: Context,
) -> Message:
    """Analytics evaluate: print best accuracy seen so far."""
    nid = _get_node_id(context)
    subject = app2.name

    history = _NODE_STATE.get(nid, {}).get("history", [])
    if not history:
        LOGGER.info("[NodeApp:%s][evaluate] node=%.12s  no history", subject, nid)
        return message

    best_cycle, best_acc, best_loss = max(history, key=lambda r: r[1])
    LOGGER.info(
        "[NodeApp:%s][evaluate] node=%.12s  best_acc=%.4f at cycle=%d  loss=%.4f",
        subject,
        nid,
        best_acc,
        best_cycle,
        best_loss,
    )
    return message
