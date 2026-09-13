---
tags: [quickstart, classification, tabular]
dataset: [adult-census-income]
framework: [catboost]
---

# Federated Learning with CatBoost and Flower (Quickstart Example)

This example demonstrates how to perform [CatBoost](https://catboost.ai) within Flower using the `catboost` package.
We use [adult-census-income](https://huggingface.co/datasets/scikit-learn/adult-census-income) dataset for this example to perform a binary classification task.
Tree-based with bagging method is used for aggregation on the server.

## Tree-based bagging aggregation

Bagging (bootstrap) aggregation is an ensemble meta-algorithm in machine learning, used for enhancing the stability and accuracy of machine learning algorithms. Here, we leverage this algorithm for learning CatBoost trees in a federated learning environment.

Specifically, each client is treated as a bootstrap by random sub-sampling (data partitioning in FL). At each FL round, all clients boost a number of trees (in this example, 1 tree) based on the local bootstrap samples.
Then, the clients' trees are aggregated on the server, and concatenates them to the global model from previous round. The aggregated tree ensemble is regarded as the new global model.

For instance, if we consider a scenario with M clients, then at any given federation round R, the bagging models consist of (M\*R) trees in total.

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/quickstart-catboost
```

This will create a new directory called `quickstart-catboost` with the following structure:

```shell
quickstart-catboost
├── quickstart_catboost
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your utilities and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `quickstart_catboost` package.

```bash
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
flwr run . --run-config "num-server-rounds=3 depth=5"  --stream
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
