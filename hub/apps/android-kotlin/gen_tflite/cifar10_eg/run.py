from os import path

from .. import *
from . import CIFAR10Model

DIR = path.dirname(__file__)


TFLITE_FILE = f"cifar10.tflite"


def main():
    model = CIFAR10Model()
    save_model(model, SAVED_MODEL_DIR)
    tflite_model = convert_saved_model(SAVED_MODEL_DIR)
    save_tflite_model(tflite_model, TFLITE_FILE)


main() if __name__ == "__main__" else None
