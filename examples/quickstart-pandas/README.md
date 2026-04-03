---
tags: [quickstart, tabular, federated analytics]
dataset: [Iris]
framework: [pandas]
---

# Federated Learning with Pandas and Flower (Quickstart Example)

> [!CAUTION]
> This example uses Flower's low-level API which remains a preview feature and subject to change. Both `ClientApp` and `ServerApp` operate directly on [Message](https://flower.ai/docs/framework/ref-api/flwr.common.Message.html) and [RecordDict](https://flower.ai/docs/framework/ref-api/flwr.common.RecordDict.html) objects.

This introductory example to Flower uses [Pandas](https://pandas.pydata.org/), but deep knowledge of Pandas is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to
download, partition and preprocess the [Iris dataset](https://huggingface.co/datasets/scikit-learn/iris).
Running this example in itself is quite easy.

This example implements a form of Federated Analyics by which instead of training a model using locally available data, the nodes run a query on the data they own. In this example the query is to compute the histogram on specific columns of the dataset. These metrics are sent to the `ServerApp` for aggregation.

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/quickstart-pandas
```

This will create a new directory called `quickstart-pandas` with the following structure:

```shell
quickstart-pandas
├── pandas_example
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   └── server_app.py   # Defines your ServerApp
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `pandas_example` package.

```bash
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!NOTE]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
flwr run . --stream
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example

```bash
flwr run . --run-config num-server-rounds=5  --stream
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
