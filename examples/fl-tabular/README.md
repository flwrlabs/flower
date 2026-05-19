---
tags: [basic, tabular, fds]
dataset: [Adult Census Income]
framework: [scikit-learn, torch]
---

# Flower Example on Adult Census Income Tabular Dataset

This code exemplifies a federated learning setup using the Flower framework on the ["Adult Census Income"](https://huggingface.co/datasets/scikit-learn/adult-census-income) tabular dataset. The "Adult Census Income" dataset contains demographic information such as age, education, occupation, etc., with the target attribute being income level (\<=50K or >50K). The dataset is partitioned into subsets, simulating a federated environment with 5 clients, each holding a distinct portion of the data. Categorical variables are one-hot encoded, and the data is split into training and testing sets. A simple deep neural network is used for this task, and federated learning is performed using the FedAvg strategy over 5 rounds.

This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the dataset.

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/fl-tabular
```

This will create a new directory called `fl-tabular` containing the following files:

```shell
fl-tabular
├── fltabular
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `fltabular` package.

```shell
# From a new python environment, run:
pip install -e .
```

## Run the Example

You can run your `ClientApp` and `ServerApp` in both _simulation_ and
_deployment_ mode without making changes to the code. If you are starting
with Flower, we recommend you using the _simulation_ model as it requires
fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!NOTE]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

This example is designed to run with 5 virtual `SuperNodes`. First we need to change the configuration of the Simulation Runtime (which by default uses 10 nodes). This guide assumes your default `SuperLink` connection points to one ready for simulations. If you aren't sure, please refer to the [How-to run Flower locally](https://flower.ai/docs/framework/how-to-run-flower-locally.html) guide.

```bash
flwr federation simulation-config --num-supernodes=5
```

Finally, let's run the app:

```bash
flwr run .  --stream
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config num-server-rounds=10  --stream
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
