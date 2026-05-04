import os

import numpy as np
import tensorflow as tf
from datasets import load_dataset

from flwr.app import Context
from flwr.client import NumPyClient, start_client
from flwr.clientapp import ClientApp

# Set subset sizes
TRAIN_SUBSET_SIZE = 100
TEST_SUBSET_SIZE = 10

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_cifar10():
    trainset = load_dataset("uoft-cs/cifar10", split=f"train[:{TRAIN_SUBSET_SIZE}]")
    testset = load_dataset("uoft-cs/cifar10", split=f"test[:{TEST_SUBSET_SIZE}]")
    x_train = np.array([item["img"] for item in trainset])
    y_train = np.array([item["label"] for item in trainset])
    x_test = np.array([item["img"] for item in testset])
    y_test = np.array([item["label"] for item in testset])
    return (x_train, y_train), (x_test, y_test)


# Load CIFAR-10 from Hugging Face
(x_train, y_train), (x_test, y_test) = load_cifar10()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

ds_train = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)
ds_test = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)


# Load model (MobileNetV2, CIFAR-10)
model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 3), classes=10, weights=None
)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(ds_train, epochs=1, batch_size=32)
        return model.get_weights(), len(ds_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(ds_test)
        return loss, len(ds_test), {"accuracy": accuracy}


def client_fn(context: Context):
    return FlowerClient().to_client()


app = ClientApp(
    client_fn=client_fn,
)

if __name__ == "__main__":
    # Start Flower client
    start_client(server_address="127.0.0.1:8080", client=FlowerClient().to_client())
