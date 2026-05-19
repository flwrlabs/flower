---
tags: [DP, DP-SGD, basic, vision, fds, privacy]
dataset: [CIFAR-10]
framework: [opacus, torch]
---

# Training with Sample-Level Differential Privacy using Opacus Privacy Engine

In this example, we demonstrate how to train a model with differential privacy (DP) using Flower. We employ PyTorch and integrate the Opacus Privacy Engine to achieve sample-level differential privacy. This setup ensures robust privacy guarantees during the client training phase. The code is adapted from the [PyTorch Quickstart example](https://flower.ai/apps/flwrlabs/quickstart-pytorch/).

For more information about DP in Flower please refer to the [tutorial](https://flower.ai/docs/framework/how-to-use-differential-privacy.html). For additional information about Opacus, visit the official [website](https://opacus.ai/).

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/opacus-fl
```

This will create a new directory called `opacus` containing the following files:

```shell
opacus
├── opacus_fl
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training, and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `opacus_fl` package. From a new python environment, run:

```shell
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!NOTE]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
flwr run .  --stream
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "max-grad-norm=1.0 num-server-rounds=5"  --stream
```

> [!NOTE]
> Please note that, at the current state, users cannot set `NodeConfig` for simulated `ClientApp`s. For this reason, the hyperparameter `noise_multiplier` is set in the `client_fn` method based on a condition check on `partition_id`. This will be modified in a future version of Flower to allow users to set `NodeConfig` for simulated `ClientApp`s.
