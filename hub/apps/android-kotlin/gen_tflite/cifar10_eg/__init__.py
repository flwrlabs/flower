import tensorflow as tf

from .. import *


@tflite_model_class
class CIFAR10Model(BaseTFLiteModel):
    X_SHAPE = [32, 32, 3]
    Y_SHAPE = [10]

    def __init__(self):
        self.model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=tuple(self.X_SHAPE)),
                tf.keras.layers.Conv2D(6, 5, activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(16, 5, activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=120, activation="relu"),
                tf.keras.layers.Dense(units=84, activation="relu"),
                tf.keras.layers.Dense(units=10, activation="softmax"),
            ]
        )

        self.model.compile(loss="categorical_crossentropy", optimizer="sgd")
