---
tags: [quickstart, medical, vision]
dataset: [MedNIST]
framework: [MONAI]
---

# Federated Learning with MONAI and Flower (Quickstart Example)

This introductory example to Flower uses MONAI, but deep knowledge of MONAI is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case.
Running this example in itself is quite easy. [MONAI](https://docs.monai.io/en/latest/index.html)(Medical Open Network for AI) is a PyTorch-based, open-source framework for deep learning in healthcare imaging, part of the PyTorch Ecosystem. This example uses a subset of the [MedMNIST](https://medmnist.com/) dataset including 6 classes, as done in [MONAI's classification demo](https://colab.research.google.com/drive/1wy8XUSnNWlhDNazFdvGBHLfdkGvOHBKe). Each client trains am [DenseNet121](https://docs.monai.io/en/stable/networks.html#densenet121) from MONAI.

> [!NOTE]
> This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to partition the MedMNIST dataset. Its a good example to show how to bring any dataset into Flower and partition it using any of the built-in [partitioners](https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html) (e.g. `DirichletPartitioner`, `PathologicalPartitioner`). Learn [how to use partitioners](https://flower.ai/docs/datasets/tutorial-use-partitioners.html) in a step-by-step tutorial.

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/quickstart-monai
```

This will create a new directory called `quickstart-monai` with the following structure:

```shell
quickstart-monai
├── monaiexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `monaiexample` package.

```bash
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

```bash
# Run with the default federation (CPU only)
flwr run .  --stream
```

> [!TIP]
> This example runs faster when the `ClientApp`s have access to a GPU. If your system has one, adjust the settings of your simulations as shown below. Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
flwr federation simulation-config \
    --client-resources-num-cpus=4 \ # each ClientApp assumes to use 4CPUs
--client-resources-num-gpus=0.25  # at most 4 ClientApp will run in a given GPU
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 batch-size=32" --stream
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
