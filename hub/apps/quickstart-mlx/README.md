---
title: Federated Learning with MLX and Flower (Quickstart Example)
tags: [quickstart, vision]
dataset: [MNIST]
framework: [MLX]
---

# Federated Learning with MLX and Flower (Quickstart Example)

This introductory example to Flower uses [MLX](https://ml-explore.github.io/mlx/build/html/index.html), but you don't need deep knowledge of MLX to run it. The example will help you understand how to adapt Flower to your specific use case, and running it is quite straightforward.

[MLX](https://ml-explore.github.io/mlx/build/html/index.html) is a NumPy-like array framework designed for efficient and flexible machine learning on Apple Silicon. In this example, we will train a simple 2-layer MLP on the [MNIST](https://huggingface.co/datasets/ylecun/mnist) dataset (handwritten digits recognition). The data will be downloaded and partitioned using [Flower Datasets](https://flower.ai/docs/datasets/).

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/quickstart-mlx
```

This will create a new directory called `quickstart-mlx` with the following structure:

```shell
quickstart-mlx
├── mlxexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `mlxexample` package.

```bash
pip install -e .
```

## Run the project

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

> [!TIP]
> For a more detailed walk-through check our [quickstart MLX tutorial](https://flower.ai/docs/framework/tutorial-quickstart-mlx.html)

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
