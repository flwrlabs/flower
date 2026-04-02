# Generate TFLite model files

This module provides infrastructure to generate `.tflite` files that are compatible with the Android package.

*Note* that this module currently doesn't work on macOS due to TensorFlow Lite issues.

For example implementations, check out `cifar10_eg/` and `toy_regression_eg/`.

## Dependencies installation

Using [Poetry](https://python-poetry.org/docs/):

```sh
poetry install
poetry shell
```

<details>
<summary>Alternatively, with Pip.</summary>

```sh
python3 -m pip install -r requirements.txt
```

</details>

## Model declaration

Inherit from `BaseTFLiteModel`, annotate with `@tflite_model_class`, and override `X_SHAPE`, `Y_SHAPE`, and `__init__` to assign `self.model`.
For example:

```python
@tflite_model_class
class MyModel(BaseTFLiteModel):
    X_SHAPE = [WIDTH, HEIGHT]
    Y_SHAPE = [N_CLASSES]

    def __init__(self):
        self.model = tf.keras.Sequential([
            # Layers.
        ])
        self.model.compile()
```

If you are not content with the default implementation of `train`, `infer`, `parameters`, `restore` provided by `BaseTFLiteModel`, override them.
If you want the change how the `tf.function` conversions are done, though, you would need to write the whole class yourself (exactly like below):

```python
class CustomModel(tf.Module):
    X_SHAPE = …
    Y_SHAPE = …

    def __init__(self, …):
        self.model = …
        self.model.compile()
        # …

    @tf.function(input_signature=[
        # `TensorSpec` of `x` and `y`.
    ])
    def train(self, x, y):
        return {"loss": …}

    @tf.function(input_signature=[
        # `TensorSpec` of `y`.
    ])
    def infer(self, x):
        return {"logits": …}

    @tf.function(input_signature=[])
    def parameters(self):
        return {"a0": …, "a1": …, …}

    @tf.function()
    def restore(self, **parameters):
        # …
        return self.parameters()
```

## TFLite file generation

```python
model = MyModel()
save_model(model, SAVED_MODEL_DIR)
tflite_model = convert_saved_model(SAVED_MODEL_DIR)
save_tflite_model(tflite_model, "my_model.tflite")
```

The script prints the `Model parameter sizes in bytes` in <span style="color: red;">red</span>. Copy that red list for later specifying the parameter layer sizes in the Android app.
That list is `layersSizes` for your `FlowerClient`.

The above script generates the `.tflite` file at `../my_model.tflite`. Move that file to `../client/app/src/main/assets/model/a_more_proper_name.tflite`.

Load the model into a `FlowerClient` in the Android app like this:

```kotlin
val buffer = loadMappedAssetFile(this, "a_more_proper_name.tflite")
// Replace with your "model parameter sizes in bytes."
val layersSizes = intArrayOf(1800, 24, 9600, 64, 768000, 480, 40320, 336, 3360, 40)
val sampleSpec = …
flowerClient = FlowerClient(buffer, layersSizes, sampleSpec) // Attribute of your `Activity`.
```

For validation, see the `toy_regression_eg/` example.

## Running demo

Please see `cifar10_eg/README.md` and `toy_regression_eg/README.md`.
