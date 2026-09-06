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

import gc

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam

# Disable GPUs for test reproducibility
tf.config.set_visible_devices([], "GPU")

MLP_UNITS = 32
N = 11
P = 10


# MPL model
def fc_model(
    model_name: str, input_shape: tf.TensorShape, units: int, classes: int = 1
) -> Model:
    """Create FLAD fully connected model."""
    K.clear_session()
    gc.collect()

    model = Sequential(name=model_name)

    model.add(Input(shape=input_shape))
    model.add(Flatten())
    model.add(Dense(units, activation="relu", name="fc0"))
    model.add(Dense(units, activation="relu", name="fc1"))
    model.add(Dense(classes, activation="sigmoid", name="fc3"))

    return model


def compile_model(
    model: Model, optimizer_type: str = "SGD", loss: str = "binary_crossentropy"
) -> None:
    """Compile the given model with the specified optimizer and loss."""
    if optimizer_type == "Adam":
        optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    else:
        optimizer = SGD(learning_rate=0.1, momentum=0.0, nesterov=False)

    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])


def load_model() -> Model:
    """Load the FLAD model."""
    # Input shape is fixed and predefined by the server.
    input_shape = tf.TensorShape([P, N, 1])
    model = fc_model("mlp", input_shape, units=MLP_UNITS)

    return model
