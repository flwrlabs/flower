---
tags: [advanced, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# Flower Federations with Authentication 🧪

The following steps describe how to start a long-running Flower server (SuperLink) and a long-running Flower clients (SuperNode) with authentication enabled. The task is to train a simple CNN for image classification using PyTorch.

> [!TIP]
> Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to learn more about Flower's Deployment Engine, how setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) works. If you are already familiar with how the Deployment Engine works, you may want to learn how to run this same example using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## Project Setup

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/supernode-authentication
```

This will create a new directory called `supernode-authentication` with the following project structure:

```shell
supernode-authentication
├── authexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
├── generate_creds.py   # Generate certificates and keys
├── prepare_dataset.py  # Generate datasets for each SuperNode to use
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `authexample` package.

```bash
pip install -e .
```

## Generate TLS certificates and authentication keys

The `generate_creds.py` script generates:

1. TLS certificates for establishing a secure connection.
2. Private and public keys for SuperNode authentication.

> [!NOTE]
> This script is for development purposes only.

> [!TIP]
> You can configure the certificate details (e.g. Validity, Organization, or adding a public IP to `SERVER_SAN_IPS`) by editing the global variables at the top of `generate_creds.py`.

```bash
# Generate certificates and keys (creates 2 SuperNodes key pairs by default)
python generate_creds.py

# Or specify the number of SuperNodes key pairs
python generate_creds.py --supernodes {your_number_of_supernodes}
```

## Define a SuperLink connection in the Flower Configuration file

Let's first locate the Flower Configuration file and create a SuperLink connection with that will allow us to interface with the SuperLink using the TLS certificate we just created.

Locate the Flower Configuration file:

```shell
flwr config list
```

```console
# Example output:
Flower Config file: /path/to/your/.flwr/config.toml
SuperLink connections:
 supergrid
 local (default)
```

Create a new Superlink connection named `my-connection`:

```TOML
[superlink.my-connection]
address = "127.0.0.1:9093" # Control API of SuperLink
root-certificates = "/abs/path/to/certificates/ca.crt"
```

Make this new connection the default one by editing the top part of the `config.toml`. In this way, if you now execute `flwr config list` again you should see the following output:

```console
Flower Config file: /path/to/your/.flwr/config.toml
SuperLink connections:
 supergrid
 local
 my-connection (default)
```

## Start the long-running Flower server (SuperLink)

Starting long-running Flower server component (SuperLink) and enable authentication is very easy; all you need to do is to pass the `--enable-supernode-auth` flag. In this example we also enable secure TLS communications between `SuperLink`, the `SuperNodes` and the Flower CLI.

Let's first launch the `SuperLink`:

```bash
flower-superlink \
    --ssl-ca-certfile certificates/ca.crt \
    --ssl-certfile certificates/server.pem \
    --ssl-keyfile certificates/server.key \
    --enable-supernode-auth
```

At this point your server-side is idling. Next, let's connect two `SuperNode`s, and then we'll start a run.

## Start the long-running Flower client (SuperNode)

> [!NOTE]
> Typically each `SuperNode` runs in a different entity/organization which has access to a dataset. In this example we are going to artificially create N dataset splits and saved them into a new directory called `datasets/`. Then, each `SuperNode` will be pointed to the dataset it should load via the `--node-config` argument. We provide a script that does the download, partition and saving of CIFAR-10.

```bash
python prepare_dataset.py
```

### Pre-registering SuperNodes

Before connecting the `SuperNodes` we need to register them with the `SuperLink`. This means we'll tell the `SuperLink` about the identities of the `SuperNodes` that will be connected. We do this by sending to it the public keys of the `SuperNodes` that we want the `SuperLink` to authorize.

Let's register the first `SuperNode`. The command below will send the public key to the `SuperLink`.

```shell
flwr supernode register keys/supernode_credentials_1.pub
# It will print something like:
# ✅ SuperNode 16019329408659850374 registered successfully.
```

Then, we register the second `SuperNode` using the other public key:

```shell
flwr supernode register keys/supernode_credentials_2.pub
# It will print something like:
# ✅ SuperNode 8392976743692794070 registered successfully.
```

You could also use the Flower ClI to view the status of the `SuperNodes`.

```shell
flwr supernode list
📄 Listing all nodes...
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃       Node ID        ┃   Owner    ┃ Status  ┃ Elapsed  ┃   Status Changed @   ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ 16019329408659850374 │   none     │ created │          │ N/A                  │
├──────────────────────┼────────────┼─────────┼──────────┼──────────────────────┤
│ 8392976743692794070  │   none     │ created │          │ N/A                  │
└──────────────────────┴────────────┴─────────┴──────────┴──────────────────────┘
```

Once the `SuperNodes` are connected, you'll see the status changes. Let's connect them !

### Connecting SuperNodes

In a new terminal window, start the first long-running Flower client (SuperNode):

```bash
flower-supernode \
    --root-certificates certificates/ca.crt \
    --auth-supernode-private-key keys/supernode_credentials_1 \
    --superlink "127.0.0.1:9092" \
    --node-config 'dataset-path="datasets/cifar10_part_1"' \
    --clientappio-api-address="127.0.0.1:9094"
```

In yet another new terminal window, start the second long-running Flower client:

```bash
flower-supernode \
    --root-certificates certificates/ca.crt \
    --auth-supernode-private-key keys/supernode_credentials_2 \
    --superlink "127.0.0.1:9092" \
    --node-config 'dataset-path="datasets/cifar10_part_2"' \
    --clientappio-api-address="127.0.0.1:9095"
```

Now that you have connected the `SuperNodes`, you should see them with status `online`:

```shell
flwr supernode list
📄 Listing all nodes...
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃       Node ID        ┃   Owner    ┃ Status  ┃ Elapsed  ┃   Status Changed @   ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ 16019329408659850374 │   none     │ online  │ 00:00:30 │ 2025-10-13 13:40:47Z │
├──────────────────────┼────────────┼─────────┼──────────┼──────────────────────┤
│ 8392976743692794070  │   none     │ online  │ 00:00:22 │ 2025-10-13 13:52:21Z │
└──────────────────────┴────────────┴─────────┴──────────┴──────────────────────┘
```

If you generated more than 2 client credentials, you can add more clients by opening new terminal windows and running the command
above. Don't forget to specify the correct client private key for each client (SuperNode) you created.

> [!TIP]
> Note the `--node-config` passed when spawning the `SuperNode` is accessible to the `ClientApp` via the `context` argument, i.e., `context.node_config`. In this example, the `ClientApp` uses it to load the dataset and then proceed with the training of the model.
>
> ```python
> @app.train()
> def train(msg: Message, context: Context):
>     # Read the node_config to know where dataset is located
>     dataset_path = context.node_config["dataset-path"]
>     # then load the dataset
> ```

## Run the Flower App

With both the long-running server (SuperLink) and two SuperNodes up and running, we can now start the run. You can optionally use the `--stream` flag to stream logs from your `ServerApp` running on SuperLink.

```bash
flwr run . --stream
```
