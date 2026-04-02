import tensorflow as tf

SAVED_MODEL_DIR = "saved_model"


def red(string: str) -> str:
    return f"\033[91m{string}\033[0m"


class BaseTFLiteModel(tf.Module):
    """Base TFLite model class to inherit from. # Usage Inherent from this class and
    then annotate with `@tflite_model_class`.

    Override these attributes:
    - `X_SHAPE`: Shape of the input to the model.
    - `Y_SHAPE`: Shape of the output from the model.
    - `model`: A `tf.keras.Model` initialized in `__init__`.
    # Functionality
    Provides default implementation of `train`, `infer`, `parameters`, `restore`.
    These methods are not annotated with `@tf.function`;
    they are supposed to be converted by `@tflite_model_class`.
    """

    X_SHAPE: list[int]
    Y_SHAPE: list[int]
    model: tf.keras.Model

    def train(self, x, y):
        return self.model.train_step((x, y))

    def infer(self, x):
        return {"logits": self.model(x)}

    def parameters(self):
        return {
            f"a{index}": weight.read_value()
            for index, weight in enumerate(self.model.weights)
        }

    def restore(self, **parameters):
        for index, weight in enumerate(self.model.weights):
            parameter = parameters[f"a{index}"]
            weight.assign(parameter)
        assert self.parameters is not None
        return self.parameters()


def tflite_model_class(cls):
    """Convert `cls` that inherits from `BaseTFLiteModel` to a TFLite model class.

    Convert `cls`'s methods using `@tf.function` with proper `input_signature`
    according to `X_SHAPE` and `Y_SHAPE`.
    The converted methods are `train`, `infer`, `parameters`, `restore`.
    Only `restore`'s `input_signature` is not specified because it need to be
    determined after examples of parameters are given.
    """
    cls.x_spec = tf.TensorSpec([None] + cls.X_SHAPE, tf.float32)  # type: ignore
    cls.y_spec = tf.TensorSpec([None] + cls.Y_SHAPE, tf.float32)  # type: ignore
    cls.train = tf.function(
        cls.train,
        input_signature=[cls.x_spec, cls.y_spec],
    )
    cls.infer = tf.function(
        cls.infer,
        input_signature=[cls.x_spec],
    )
    cls.parameters = tf.function(cls.parameters, input_signature=[])
    cls.restore = tf.function(cls.restore)
    return cls


def save_model(model, saved_model_dir):
    parameters = model.parameters.get_concrete_function()
    init_params = parameters()
    print(f"Initial parameters is {init_params}.")
    restore = model.restore.get_concrete_function(**init_params)
    restore_test = restore(**init_params)
    print(f"Restore test result: {restore_test}.")
    tf.saved_model.save(
        model,
        saved_model_dir,
        signatures={
            "train": model.train.get_concrete_function(),
            "infer": model.infer.get_concrete_function(),
            "parameters": parameters,
            "restore": restore,
        },
    )

    converted_params = [
        param.numpy() for param in parameters_from_raw_dict(init_params)
    ]
    shape = f"{[list(param.shape) for param in converted_params]}"
    print(f"Model parameter shape: {red(shape)}.")
    byte_sizes = f"{[param.size * param.itemsize for param in converted_params]}"
    print(f"Model parameter sizes in bytes: {red(byte_sizes)}.")
    return converted_params


def convert_saved_model(saved_model_dir):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]

    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    return tflite_model


def parameters_from_raw_dict(raw_dict):
    parameters = []
    index = 0
    while True:
        parameter = raw_dict.get(f"a{index}")
        if parameter is None:
            break
        parameters.append(parameter)
        index += 1
    return parameters


def save_tflite_model(tflite_model, tflite_file):
    with open(tflite_file, "wb") as model_file:
        return model_file.write(tflite_model)
