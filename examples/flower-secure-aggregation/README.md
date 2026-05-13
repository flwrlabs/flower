---
tags: [advanced, secure_aggregation, privacy]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Secure aggregation with Flower (the SecAgg+ protocol)

The following steps describe how to use Flower's built-in Secure Aggregation components. This example demonstrates how to apply `SecAgg+` to the same federated learning workload as in the [quickstart-pytorch](https://flower.ai/apps/flwrlabs/quickstart-pytorch/) example. The `ServerApp` uses the [`SecAggPlusWorkflow`](https://flower.ai/docs/framework/ref-api/flwr.server.workflow.SecAggPlusWorkflow.html#secaggplusworkflow) while `ClientApp` uses the [`secaggplus_mod`](https://flower.ai/docs/framework/ref-api/flwr.client.mod.secaggplus_mod.html#flwr.client.mod.secaggplus_mod). To introduce the various steps involved in `SecAgg+`, this example introduces as a sub-class of `SecAggPlusWorkflow` the `SecAggPlusWorkflowWithLogs`. It is enabled by default, but you can disable (see later in this readme).

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/flower-secure-aggregation
```

This will create a new directory called `flower-secure-aggregation` containing the
following files:

```shell
flower-secure-aggregation
|
├── secaggexample
|   ├── __init__.py
|   ├── client_app.py    # Defines your ClientApp
|   ├── server_app.py    # Defines your ServerApp
|   ├── task.py          # Defines your model, training and data loading
|   └── workflow_with_log.py # Defines a workflow used when `is-demo=true`
├── pyproject.toml       # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `secaggexample` package.

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

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.25"  --stream
```

To adapt the example for a practical usage, set `is-demo=false` like shown below. You might want to adjust the `num-shares` and `reconstruction-threshold` settings to suit your requirements. You can override those via `--run-config` as well.

```bash
flwr run . --run-config is-demo=false  --stream
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
