---
tags: [DP, SecAgg, vision, fds]
dataset: [MNIST]
framework: [torch, torchvision]
---

# Flower Example on MNIST with Differential Privacy and Secure Aggregation

This example demonstrates a federated learning setup using the Flower, incorporating central differential privacy (DP) with client-side fixed clipping and secure aggregation (SA). It is intended for a small number of rounds for demonstration purposes.

This example is similar to the [quickstart-pytorch example](https://flower.ai/apps/flwrlabs/quickstart-pytorch/) and extends it by integrating central differential privacy and secure aggregation. For more details on differential privacy and secure aggregation in Flower, please refer to the documentation [here](https://flower.ai/docs/framework/how-to-use-differential-privacy.html) and [here](https://flower.ai/docs/framework/contributor-ref-secure-aggregation-protocols.html).

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/fl-dp-sa
```

This will create a new directory called `fl-dp-sa` containing the following files:

```shell
fl-dp-sa
├── fl_dp_sa
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training, and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `fl_dp_sa` package.

```shell
# From a new python environment, run:
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!NOTE]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

This example is designed to run with 100 virtual `SuperNodes`. First we need to change the configuration of the Simulation Runtime (which by default uses 10 nodes). This guide assumes your default `SuperLink` connection points to one ready for simulations. If you aren't sure, please refer to the [How-to run Flower locally](https://flower.ai/docs/framework/how-to-run-flower-locally.html) guide.

```bash
flwr federation simulation-config --num-supernodes=100
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "noise-multiplier=0.1 clipping-norm=5"  --stream
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
