from os import path

import numpy as np
import pandas as pd
import tensorflow as tf

from .. import *
from .model import ToyRegressionModel

DIR = path.dirname(__file__)


def get_training_data():
    data = pd.read_csv(
        f"{DIR}/data.csv", header=0, names=["step", "calorie", "distance"]
    ).dropna()

    x_train, y_train = data.iloc[:, 0:2].astype("float32"), data["distance"].to_frame(
        "distance"
    ).astype("float32")
    assert x_train.shape == (11, 2)
    assert y_train.shape == (11, 1)

    shapes = str({"x": x_train.shape, "y": y_train.shape})
    print(f"Data shapes: {red(shapes)}")
    return x_train, y_train


def train_model(x_train, y_train):
    NUM_EPOCHS = 30
    BATCH_SIZE = 1
    losses = np.zeros([NUM_EPOCHS])
    model = ToyRegressionModel(0.00000000003)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.batch(BATCH_SIZE)
    for i in range(NUM_EPOCHS):
        result = {}
        for x, y in train_ds:
            assert model.train is not None
            result = model.train(x, y)

        losses[i] = result["loss"]
        if (i + 1) % 10 == 0:
            print(f"Finished {i+1} epochs")
            print(f"  loss: {losses[i]:.3f}")

    h = model.model.predict(x_train)
    assert h.shape == (11, 1)

    return model


def test_tflite_file(tflite_file, x_train):
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()

    infer = interpreter.get_signature_runner("infer")
    h1 = infer(x=x_train).get("logits")
    print(f"Inference result logits before training: {h1}.")

    parameters = interpreter.get_signature_runner("parameters")
    raw_old_params = parameters()
    print(f"Raw parameters before training: {raw_old_params}.")
    old_params = parameters_from_raw_dict(raw_old_params)
    print(f"Parameters before training: {old_params}.")

    train = interpreter.get_signature_runner("train")
    result = train(
        x=np.array([[1837, 72.332]], dtype="float32"),
        y=np.array([[1311]], dtype="float32"),
    )
    print(f"Training loss: {result['loss']}.")

    infer = interpreter.get_signature_runner("infer")
    h1 = infer(x=x_train).get("logits")
    print(f"Inference result logits after training: {h1}.")

    raw_new_params = parameters()
    print(f"Raw parameters after training: {raw_new_params}.")
    new_params = parameters_from_raw_dict(raw_new_params)
    print(f"Parameters after training: {new_params}.")

    restore = interpreter.get_signature_runner("restore")
    print(f"Raw parameters after restoring old parameters: {restore(**raw_old_params)}")
    h1 = infer(x=x_train).get("logits")
    print(f"Inference result logits after restoring old parameters: {h1}.")

    print(f"Raw parameters after restoring new parameters: {restore(**raw_new_params)}")
    h1 = infer(x=x_train).get("logits")
    print(f"Inference result logits after restoring new parameters: {h1}.")


TFLITE_FILE = f"toy_regression.tflite"


def main():
    x_train, y_train = get_training_data()
    model = train_model(x_train, y_train)
    save_model(model, SAVED_MODEL_DIR)
    tflite_model = convert_saved_model(SAVED_MODEL_DIR)
    save_tflite_model(tflite_model, TFLITE_FILE)
    test_tflite_file(TFLITE_FILE, x_train)


main() if __name__ == "__main__" else None
