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

import ctypes
import os
import platform
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# pylint: disable=wrong-import-position
import numpy as np
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp
from sklearn.metrics import f1_score

from .dataset import load_data, set_seed
from .model import compile_model, load_model

# glibc's default malloc() retains freed memory in its own arena rather
# than returning it to the OS. If you hit OOM errors, run with
# tcmalloc preloaded instead (see README).
_GLOBAL_SCOPE = ctypes.CDLL(None)
_USE_TCMALLOC = hasattr(_GLOBAL_SCOPE, "MallocExtension_ReleaseFreeMemory")


def _trim_memory() -> None:
    """Release freed memory back to the OS."""
    if platform.system() == "Linux":
        if _USE_TCMALLOC:
            _GLOBAL_SCOPE.MallocExtension_ReleaseFreeMemory()
        else:
            _GLOBAL_SCOPE.malloc_trim(0)


# Flower ClientApp
app = ClientApp()

# Map partition-id to client name
client_names = {
    0: "00-WebDDoS",
    1: "01-LDAP",
    2: "02-Portmap",
    3: "03-DNS",
    4: "04-UDPLag",
    5: "05-NTP",
    6: "06-SNMP",
    7: "07-SSDP",
    8: "08-Syn",
    9: "09-TFTP",
    10: "10-UDP",
    11: "11-NetBIOS",
    12: "12-MSSQL",
}


def _as_int(value: Any) -> int:
    """Narrow a ConfigRecord value to int before conversion."""
    assert isinstance(value, int | float)
    return int(value)


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # FLAD clients are pets not cattle, so we need to name them.
    client: dict[str, Any] = {}
    client_id = int(context.node_config["partition-id"])
    client["name"] = client_names[client_id]

    client["rn_seed"] = int(context.run_config["rn_seed"])
    client["optimizer"] = str(context.run_config["optimizer"])
    client["dataset_folder"] = str(context.run_config["dataset_folder"])

    client["epochs"] = _as_int(msg.content["config"]["epochs"])
    client["steps_per_epoch"] = _as_int(msg.content["config"]["steps_per_epoch"])
    client["server_round"] = _as_int(msg.content["config"]["server_round"])

    # Set the seed for reproducibility
    set_seed(client["rn_seed"])

    # Load the data
    load_data(client)

    # Load the model and set its weights to the ones received from the server
    model = load_model()
    arrays = msg.content["arrays"]
    assert isinstance(arrays, ArrayRecord)
    model.set_weights(arrays.to_numpy_ndarrays())
    compile_model(model, client["optimizer"], "binary_crossentropy")

    if client["steps_per_epoch"] > 0:
        client["batch_size"] = max(
            int(len(client["training"][1]) / client["steps_per_epoch"]), 1
        )
    else:
        raise ValueError("Steps per epoch must be greater than zero.")

    # Train the model
    history = model.fit(
        x=client["training"][0],
        y=client["training"][1],
        validation_data=(client["validation"][0], client["validation"][1]),
        epochs=client["epochs"],
        batch_size=client["batch_size"],
        verbose=0,
        callbacks=[],
    )

    # Get training metrics
    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    # Pack and send the model weights and metrics back as a message.
    model_record = ArrayRecord(model.get_weights())
    metrics = MetricRecord({"train_loss": train_loss, "val_loss": val_loss})
    content = RecordDict({"arrays": model_record, "metrics": metrics})

    _trim_memory()
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # FLAD clients are pets not cattle, so we need to name them.
    client: dict[str, Any] = {}
    client_id = int(context.node_config["partition-id"])
    client["name"] = client_names[client_id]
    client["rn_seed"] = int(context.run_config["rn_seed"])
    client["dataset_folder"] = str(context.run_config["dataset_folder"])

    # Set the seed for reproducibility
    set_seed(client["rn_seed"])

    # Load the data
    load_data(client, validation_only=True)

    # Load the model and set its weights to the ones received from the server
    model = load_model()
    arrays = msg.content["arrays"]
    assert isinstance(arrays, ArrayRecord)
    model.set_weights(arrays.to_numpy_ndarrays())

    # Evaluate the model
    x_val, y_val = client["validation"]
    y_pred = np.squeeze(model.predict(x_val, batch_size=2048, verbose=0) > 0.5)
    client_f1 = f1_score(y_val, y_pred)

    # Pack and send the model weights and metrics as a message
    metrics = MetricRecord({"f1_score": float(client_f1)})
    content = RecordDict({"metrics": metrics})

    _trim_memory()
    return Message(content=content, reply_to=msg)


@app.query("info")
def info(msg: Message, context: Context) -> Message:
    """Return the client name."""
    # Return the client name
    client_id = int(context.node_config["partition-id"])
    client_name = client_names[client_id]
    content = RecordDict(
        {
            "config": ConfigRecord({"name": client_name}),
        }
    )

    return Message(content=content, reply_to=msg)
