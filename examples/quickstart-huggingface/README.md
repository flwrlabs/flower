---
tags: [quickstart, llm, nlp, sentiment]
dataset: [IMDB]
framework: [transformers]
---

# Federated Learning with HuggingFace Transformers and Flower (Quickstart Example)

This introductory example to using [🤗Transformers](https://huggingface.co/docs/transformers/en/index) with Flower. The training script closely follows the [HuggingFace course](https://huggingface.co/course/chapter3?fw=pt), so you are encouraged to check that out for a detailed explanation of the transformer pipeline.

In this example, we will federated the training of a [BERT-tiny](https://huggingface.co/prajjwal1/bert-tiny) model on the [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) dataset. The data will be downloaded and partitioned using [Flower Datasets](https://flower.ai/docs/datasets/). This example runs best when a GPU is available.

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/quickstart-huggingface
```

This will create a new directory called `quickstart-huggingface` containing the following files:

```shell
quickstart-huggingface
├── huggingface_example
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `huggingface_example` package.

```bash
pip install -e .
```

## Run the Example

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!TIP]
> This example runs faster when the `ClientApp`s have access to a GPU. If your system has one, you can make use of it by configuring the `backend.client-resources` component in your Flower Configuration. Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
# Run with the default federation (CPU only)
flwr run .
```

You can add a new connection in your Flower Configuration (find if via `flwr config list`):

```TOML
[superlink.local-gpu]
options.num-supernodes = 100
options.backend.client-resources.num-cpus = 4 # each ClientApp assumes to use 4CPUs
options.backend.client-resources.num-gpus = 0.25 # at most 4 ClientApp will run in a given GPU (lower it to increase parallelism)
```

And then run the app

```bash
# Run with the `local-gpu` settings
flwr run . local-gpu
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example

```bash
flwr run --run-config "num-server-rounds=5 fraction-train=0.1"
```

> [!TIP]
> For a more detailed walk-through check our [quickstart 🤗Transformers tutorial](https://flower.ai/docs/framework/tutorial-quickstart-huggingface.html)

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
