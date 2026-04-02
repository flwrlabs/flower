---
tags: [mods, monitoring]
dataset: [CIFAR-10]
framework: [wandb, tensorboard]
---

# Using custom mods 🧪

> 🧪 = This example covers experimental features that might change in future versions of Flower
> Please consult the regular PyTorch code examples ([quickstart](https://flower.ai/apps/flwrlabs/quickstart-pytorch/), [advanced](https://github.com/flwrlabs/flower/tree/main/examples/advanced-pytorch)) to learn how to use Flower with PyTorch.

The following steps describe how to write custom Flower Mods and use them in a simple example.

## Writing custom Flower Mods

As described [in the documentation](https://flower.ai/docs/framework/how-to-use-built-in-mods.html#what-are-mods), Flower Mods in their simplest form can be described as:

```python
def basic_mod(msg: Message, context: Context, app: ClientApp) -> Message:
    # Do something with incoming Message (or Context)
    # before passing to the inner ``ClientApp``
    reply = app(msg, context)
    # Do something with outgoing Message (or Context)
    # before returning
    return reply
```

and used when defining the `ClientApp` as:

```python
app = ClientApp(
    mods=[basic_mod],
)
```

The mods in this example do not modify the `Message` object that the `ClientApp` is communicating to the `ServerApp`. Instead, the mods only log the _metrics_ returned by the `ClientApp`'s `train()` method to Weight & Biases or into TensorBoard .

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/custom-mods
```

This will create a new directory called `custom-mods` with the following structure:

```shell
custom-mods
├── README.md
├── custom_mods
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── mods.py         # Defines a Weights & Biases and TensorBoard mod
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
└── pyproject.toml      # Project metadata like dependencies and configs
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `custom_mods` package.

```bash
pip install -e .
```

## Run the project

> [!TIP]
> By default the `ClientApp` uses the TensorBoard mod, if you would like to enable the Weight & Biases mod, please edit `custom_mods/client_app.py` and pass the `get_wandb_mod` mod to the constructor of your `ClientApp`.

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!NOTE]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"
```

### Check the results

For W&B, you will need to login to the [website](https://wandb.ai).

For TensorBoard, you will need to run the following command in your terminal:

```bash
tensorboard --logdir <LOG_DIR>
```

Where `<LOG_DIR>` needs to be replaced by the directory passed as an argument to the wrapper function (`.runs_history/` by default).

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
