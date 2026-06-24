##########################################
 Write your first Flower App with PyTorch
##########################################

.. |Grid_link| replace:: ``Grid``

.. _grid_link: ref-api/flwr.serverapp.Grid.html

.. |context_link| replace:: ``Context``

.. _context_link: ref-api/flwr.app.Context.html

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.app.Message.html

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.app.ArrayRecord.html

.. |metricrecord_link| replace:: ``MetricRecord``

.. _metricrecord_link: ref-api/flwr.app.MetricRecord.html

.. |configrecord_link| replace:: ``ConfigRecord``

.. _configrecord_link: ref-api/flwr.app.ConfigRecord.html

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: ref-api/flwr.clientapp.ClientApp.html

.. |fedavg_link| replace:: ``FedAvg``

.. _fedavg_link: ref-api/flwr.serverapp.strategy.FedAvg.html

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.serverapp.ServerApp.html

.. |strategy_start_link| replace:: ``start``

.. _strategy_start_link: ref-api/flwr.serverapp.strategy.Strategy.html#flwr.serverapp.strategy.Strategy.start

.. |result_link| replace:: ``Result``

.. _result_link: ref-api/flwr.serverapp.strategy.Result.html

Welcome to the next part of the Flower collaborative AI tutorial!

In the previous tutorials, you created a simulated federation on SuperGrid, ran a Flower
App, downloaded the ``@flwrlabs/demo`` app, and learned how ``ServerApp``,
``ClientApp``, strategies, and ``pyproject.toml`` fit together. In this tutorial, you
will use the same workflow with a more realistic Flower App: a PyTorch app that trains a
small image classifier on CIFAR-10.

.. tip::

    `Star Flower on GitHub <https://github.com/flwrlabs/flower>`__ ⭐️ and join the
    Flower community on `Flower Discuss <https://discuss.flower.ai/>`__ or `Flower Slack
    <https://flower.ai/join-slack>`__ to introduce yourself, ask questions, and get
    help.

Let's get started! 🌼

****************
 Create the App
****************

Use ``flwr new`` to fetch the PyTorch quickstart app from Flower Hub:

.. code-block:: shell

    $ flwr new @flwrlabs/quickstart-pytorch

After running the command, a new directory named ``quickstart-pytorch`` will be created:

.. code-block:: shell

    quickstart-pytorch
    ├── pytorchexample
    │   ├── __init__.py
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

This app has the same Flower structure as the NumPy demo from the previous tutorial, but
the workload is now a real PyTorch training task. The app trains a small convolutional
neural network on CIFAR-10, an image classification dataset with ten classes such as
airplane, automobile, bird, cat, dog, ship, and truck.

********************
 Quick App Overview
********************

.. note::

    A more detailed walkthrough of the app is available later in this tutorial.

Before running the app, it helps to know what each file is responsible for:

- ``pytorchexample/task.py`` contains the PyTorch-specific code: the neural network,
  CIFAR-10 data loading and partitioning, the local training loop, the evaluation loop,
  and server-side evaluation helpers.
- ``pytorchexample/client_app.py`` defines the ``ClientApp``. Its ``@app.train()``
  handler receives the current global model, loads one CIFAR-10 partition, trains the
  model locally, and replies with updated model parameters plus metrics. Its
  ``@app.evaluate()`` handler evaluates the received model on local validation data and
  replies with metrics.
- ``pytorchexample/server_app.py`` defines the ``ServerApp``. It creates the initial
  PyTorch model, wraps the model parameters in an ``ArrayRecord``, creates a ``FedAvg``
  strategy, and starts the federated learning run.
- ``pyproject.toml`` declares the app metadata and dependencies, points Flower to the
  ``ServerApp`` and ``ClientApp`` objects, and defines run configuration values such as
  the number of server rounds, batch size, local epochs, learning rate, and evaluation
  settings.

The important idea is the same as before: the ``ServerApp`` starts the run, ``FedAvg``
coordinates each federated learning round, and each ``ClientApp`` trains or evaluates
the model using the data available on its SuperNode.

This app uses `Flower Datasets <https://flower.ai/docs/datasets/>`__ to download
CIFAR-10 and split it into partitions, one for each simulated client. This is ideal for
simulations because it lets you experiment with federated learning even when you start
from a single centralized dataset. In a typical Flower App that runs outside of
simulation, you usually do not create artificial partitions. Instead, each ``ClientApp``
loads the data already available on the SuperNode where it runs.

**************************
 Run the App on SuperGrid
**************************

.. note::

    If you have not already done so, complete the :doc:`first tutorial
    <tutorial-series-get-started-with-flower>` to create a SuperGrid account and a
    simulated federation.

Open a terminal, activate your Python environment, and run the following command to
first login to SuperGrid:

.. code-block:: shell

    # This will open a browser window where you can enter your SuperGrid credentials.
    $ flwr login supergrid

Once you are logged in, run the following command to run the app on SuperGrid:

.. code-block:: shell

    # Navigate to the directory of the app you want to run
    $ cd /path/to/quickstart-pytorch
    # Run the app
    $ flwr run . supergrid

SuperGrid will start a new run for this app. Open the `SuperGrid dashboard
<https://flower.ai/federations/>`__, select your federation, and click the new run to
follow its progress and inspect the logs.

In the logs, you should see Flower start the ``FedAvg`` strategy and run several rounds
of federated learning. Each round includes local training on selected ``ClientApp``
instances, aggregation in the ``ServerApp``, and evaluation metrics such as
``eval_loss`` and ``eval_acc``.

You can override values from ``pyproject.toml`` at run time. For example:

.. code-block:: shell

    # Run the app for five rounds instead of the default three rounds
    $ flwr run . supergrid \
        --run-config "num-server-rounds=5"

    # Run the app for five rounds and a smaller batch size
    $ flwr run . supergrid \
        --run-config "num-server-rounds=5" \
        --run-config "batch-size=16"

.. tip::

    In SuperGrid, use the ``--federation`` flag to choose which federation runs your
    app. If you omit it, Flower uses ``@<your-account>/workspace``. Learn more in
    :doc:`Create and Manage Federations on SuperGrid
    <how-to-create-and-manage-federations>`.

*********************
 Run the App Locally
*********************

Running on SuperGrid is the recommended way to run collaborative AI workflows with
Flower. However, it is also useful to run the same app locally while you are developing
or debugging.

Navigate to the directory where the app was downloaded, then run the app locally with
the command below. Flower will start a managed local SuperLink -- a distilled version of
SuperGrid -- and execute the app with simulated SuperNodes on your machine. The first
run can take longer because the app needs to download CIFAR-10 and install the
dependencies of your App. With the flag ``--stream``, you can see the logs from the
local run in your terminal.

.. code-block:: shell

    $ cd /path/to/quickstart-pytorch
    $ flwr run . local --stream

The streamed output should include logs similar to this:

.. code-block:: shell

    INFO :      Starting FedAvg strategy:
    INFO :          ├── Number of rounds: 3
    INFO :      ...
    INFO :      [ROUND 1/3]
    INFO :      configure_train: Sampled 2 SuperNodes (out of 2)
    INFO :      aggregate_train: Received 2 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 2.149280}
    INFO :      configure_evaluate: Sampled 2 SuperNodes (out of 2)
    INFO :      aggregate_evaluate: Received 2 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 2.31319, 'eval_acc': 0.13004}
    INFO :      [ROUND 2/3]
    INFO :      ...
    INFO :      [ROUND 3/3]
    INFO :      ...
    INFO :      Strategy execution finished

.. note::

    In the above ``flwr run`` command you are not specifying a federation, this is
    because for local prototyping there is only one federation available. Because of
    this, the ``--federation`` flag is not required.

.. note::

    If you're on Windows and see unexpected terminal output, for example ``�
    □[32m□[1m``, check :ref:`this FAQ entry <faq-windows-unexpected-output>`.

For more details on using the Flower CLI against a locally running SuperLink, including
how to list your runs and view their logs, see :doc:`Run Flower Locally with a Managed
SuperLink <how-to-run-flower-locally>`.

****************************
 A Deeper Dive into the App
****************************

The ``@flwrlabs/quickstart-pytorch`` app demonstrates a simple federated learning
workflow. In federated learning, the server sends global model parameters to the client,
and the client updates the local model with parameters received from the server. It then
trains the model on the local data (which changes the model parameters locally) and
sends the updated/changed model parameters back to the server (or, alternatively, it
sends just the gradients back to the server, not the full model parameters).

Define the Flower ClientApp
===========================

Federated learning systems consist of a server and multiple clients (SuperNodes). In
Flower, we create a |serverapp_link|_ and a |clientapp_link|_ to run the server-side and
client-side code, respectively.

The core functionality of the ``ClientApp`` is to perform some action with the local
data that the SuperNode it runs on (e.g. an edge device, a server in a data center, or a
laptop) has access to. In this tutorial such action is to train and evaluate the small
CNN model defined earlier using the local training and validation data.

Loading the data
----------------

This app trains a small convolutional neural network on CIFAR-10. Since the tutorial
uses the Simulation Runtime, all data starts from one centralized dataset and is split
into partitions, one for each simulated SuperNode.

The ``load_data()`` function in ``task.py`` uses `Flower Datasets
<https://flower.ai/docs/datasets/>`_ to load one partition, split it into training and
validation data, apply the PyTorch transforms, and return two ``DataLoader`` objects:

.. code-block:: python

    def load_data(partition_id: int, num_partitions: int, batch_size: int):
        """Load partition CIFAR10 data."""
        # Only initialize `FederatedDataset` once
        global fds
        if fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
        partition = fds.load_partition(partition_id)
        # Divide data on each SuperNode: 80% train, 20% test
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        def apply_transforms(batch):
            """Apply transforms to the partition from FederatedDataset."""
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch

        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
        return trainloader, testloader

This partitioning is only needed for simulation. In deployment, each SuperNode would
usually load its own local data directly, for example from a path provided through
``--node-config``.

Training
--------

We can define how the ``ClientApp`` performs training by wrapping a function with the
``@app.train()`` decorator. In this case we name this function ``train`` because we'll
use it to train the model on the local data. The function always expects two arguments:

- A |message_link|_: The message received from the server. It contains the model
  parameters and any other configuration information sent by the server.
- A |context_link|_: The context object that contains information about the SuperNode
  executing the ``ClientApp`` and about the current run.

Through the context you can retrieve the config settings defined in the
``pyproject.toml`` of your app. The context can be used to persist the state of the
client across multiple calls to ``train`` or ``evaluate``. In Flower, ``ClientApps`` are
ephemeral objects that get instantiated for the execution of one ``Message`` and
destroyed when a reply is communicated back to the server.

Let's see an implementation of ``ClientApp`` that uses the previously defined PyTorch
CNN model, applies the parameters received from the ``ServerApp`` via the message, loads
its local data, trains the model with it (using the ``train_fn`` function), and
generates a reply ``Message`` containing the updated model parameters as well as some
metrics of interest.

.. code-block:: python

    from pytorchexample.task import train as train_fn

    # Flower ClientApp
    app = ClientApp()


    @app.train()
    def train(msg: Message, context: Context):
        """Train the model on local data."""

        # Load the model and initialize it with the received weights
        model = Net()
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load the data
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        batch_size = context.run_config["batch-size"]
        trainloader, _ = load_data(partition_id, num_partitions, batch_size)

        # Call the training function
        train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )

        # Construct and return reply Message
        model_record = ArrayRecord(model.state_dict())
        metrics = {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        return Message(content=content, reply_to=msg)

.. note::

    The ``partition-id`` and ``num-partitions`` values shown above are provided by the
    :doc:`Simulation Runtime <how-to-run-simulations>`. In a deployment setting, the
    ``ClientApp`` would usually load data that already exists on the SuperNode. For
    example, you could pass the path to that data when starting the SuperNode with
    ``--node-config "data-path=/path/to/data"`` and then call ``load_data`` with
    ``context.node_config["data-path"]``.

Note that the ``train_fn`` is simply an alias name pointing to the train function
defined earlier in this tutorial (where we defined the PyTorch training loop and
optimizer). To this function we pass the model we want to train locally and the data
loader, but also the number of local epochs and the learning rate (``lr``) to use. Note
how in this case the ``local-epochs`` setting is read from the run config via the
``Context`` while the ``lr`` is read from the ``ConfigRecord`` sent by the server via
the ``Message``. This can be used to adjust the learning rate on each round from the
server. When this dynamism isn't needed, reading the ``lr`` from the run config via the
``Context`` is also perfectly valid.

Once training is completed, the ``ClientApp`` constructs a reply ``Message``. This reply
typically includes a ``RecordDict`` with two records:

- An ``ArrayRecord`` containing the updated model parameters
- A ``MetricRecord`` with relevant metrics (in this case, the training loss and the
  number of examples used for training)

.. note::

    Returning the number of examples under the ``"num-examples"`` key is **required**,
    because strategies such as |fedavg_link|_ used by the ``ServerApp`` rely on this key
    to aggregate both models and metrics by default, unless you override the
    ``weighted_by_key`` argument (for example:
    ``FedAvg(weighted_by_key="my-different-key")``).

After constructing the reply ``Message``, the ``ClientApp`` returns it. Flower then
handles sending the reply back to the server automatically.

Evaluation
----------

In a typical federated learning setup, the ``ClientApp`` would also implement an
``@app.evaluate()`` function to evaluate the model received from the ``ServerApp`` on
local validation data. This is especially useful to monitor the performance of the
global model on each client during training. The implementation of the ``evaluate``
function is very similar to the ``train`` function, except that it calls the ``test_fn``
function defined earlier in this tutorial (which implements the PyTorch evaluation loop)
and it returns a ``Message`` containing only a ``MetricRecord`` with the evaluation
metrics (no ``ArrayRecord`` because the model parameters are not updated during
evaluation). Here's how the ``evaluate`` function looks like:

.. code-block:: python

    from pytorchexample.task import test as test_fn


    @app.evaluate()
    def evaluate(msg: Message, context: Context):
        """Evaluate the model on local data."""

        # Load the model and initialize it with the received weights
        model = Net()
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load the data
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        batch_size = context.run_config["batch-size"]
        _, valloader = load_data(partition_id, num_partitions, batch_size)

        # Call the evaluation function
        eval_loss, eval_acc = test_fn(
            model,
            valloader,
            device,
        )

        # Construct and return reply Message
        metrics = {
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "num-examples": len(valloader.dataset),
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"metrics": metric_record})
        return Message(content=content, reply_to=msg)

As you can see the ``evaluate`` implementation is near identical to the ``train``
implementation, except that it calls the ``test_fn`` function instead of the
``train_fn`` function and it returns a ``Message`` containing only a ``MetricRecord``
with metrics relevant to evaluation (``eval_loss``, ``eval_acc`` -- both scalars). We
also need to include the ``num-examples`` key in the metrics so the server can aggregate
the evaluation metrics correctly.

Define the Flower ServerApp
===========================

On the server side, we need to configure a strategy which encapsulates the federated
learning approach/algorithm, for example, *Federated Averaging* (FedAvg). Flower has a
number of built-in strategies, but we can also use our own strategy implementations to
customize nearly all aspects of the federated learning approach. For this tutorial, we
use the built-in ``FedAvg`` implementation and customize it slightly by specifying the
fraction of connected SuperNodes to involve in a round of training.

To construct a |serverapp_link|_, we define its ``@app.main()`` method. This method
receives as input arguments:

- a ``Grid`` object that will be used to interface with the SuperNodes running the
  ``ClientApp`` to involve them in a round of train/evaluate/query or other.
- a |context_link|_ object that provides access to the run configuration.

Before launching the strategy via the |strategy_start_link|_ method, we want to
initialize the global model. This will be the model that gets sent to the ``ClientApp``
running on the clients in the first round of federated learning. We can do this by
creating an instance of the model (``Net``), extracting the parameters in its
``state_dict``, and constructing an ``ArrayRecord`` with them. We can then make it
available to the strategy via the ``initial_arrays`` argument of the ``start()`` method.

We can also optionally pass to the ``start()`` method a ``ConfigRecord`` containing
settings that we would like to communicate to the clients. These will be sent as part of
the ``Message`` that also carries the model parameters.

.. code-block:: python

    app = ServerApp()


    @app.main()
    def main(grid: Grid, context: Context) -> None:
        """Main entry point for the ServerApp."""

        # Read run config
        fraction_evaluate: float = context.run_config["fraction-evaluate"]
        num_rounds: int = context.run_config["num-server-rounds"]
        lr: float = context.run_config["learning-rate"]

        # Load global model
        global_model = Net()
        arrays = ArrayRecord(global_model.state_dict())

        # Initialize FedAvg strategy
        strategy = FedAvg(fraction_evaluate=fraction_evaluate)

        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr}),
            num_rounds=num_rounds,
            evaluate_fn=global_evaluate,
        )

        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")

Most of the execution of the ``ServerApp`` happens inside the ``strategy.start()``
method. After the specified number of rounds (``num_rounds``), the ``start()`` method
returns a |result_link|_ object containing the final model parameters and metrics
received from the clients or generated by the strategy itself. We can then save the
final model to disk for later use.

Behind the scenes
=================

So how does this work? How does Flower execute this simulation?

When we execute ``flwr run`` against the default local connection configuration, Flower
submits the run to the managed local SuperLink. By default, the local SuperLink will
configure the simulation runtime to use two SuperNodes. Each will run an instance of the
``ClientApp`` we defined earlier.

The local SuperLink then starts the ``ServerApp`` and asks it to issue instructions to
those SuperNodes using the ``FedAvg`` strategy. In this example, ``FedAvg`` is
configured with two key parameters:

- ``fraction-train=1.0`` → select 100% of the available clients for training
- ``fraction-evaluate=1.0`` → select 100% of the available clients for evaluation

This means in our example, all clients (SuperNodes) will be sampled for both a round of
training and evaluation.

A typical round looks like this:

- **Training**

  1. ``FedAvg`` selects all clients (2 out of 2).
  2. Flower sends a ``TRAIN`` message to each selected ``ClientApp``.
  3. Each ``ClientApp`` calls the function decorated with ``@app.train()``, then returns
     a ``Message`` containing an ``ArrayRecord`` (the updated model parameters) and a
     ``MetricRecord`` (the training loss and number of examples).
  4. The ``ServerApp`` receives all replies.
  5. ``FedAvg`` aggregates all ``ArrayRecord`` into a new ``ArrayRecord`` representing
     the new global model and combines all ``MetricRecord``.

- **Evaluation**

  1. ``FedAvg`` selects all clients (2 out of 2).
  2. Flower sends an ``EVALUATE`` message to each ``ClientApp``.
  3. Each ``ClientApp`` calls the function decorated with ``@app.evaluate()`` and
     returns a ``Message`` containing a ``MetricRecord`` (the evaluation loss, accuracy,
     and number of examples).
  4. The ``ServerApp`` receives all replies.
  5. ``FedAvg`` aggregates all ``MetricRecord``.

Once both training and evaluation are done, the next round begins: another training
step, then another evaluation step, and so on, until the configured number of rounds is
reached.

***************
 Final remarks
***************

You have now run a PyTorch Flower App on SuperGrid and locally. Compared with the NumPy
demo, this app uses a real model, a real dataset, and real local training, but the
Flower structure is the same: ``ServerApp``, ``ClientApp``, strategy, and
``pyproject.toml``.

In the next tutorial, you will customize the federated learning strategy to change how
the server coordinates training and evaluation.

************
 Next steps
************

Before you continue, make sure to join the Flower community on Flower Discuss (`Join
Flower Discuss <https://discuss.flower.ai>`__) and on Slack (`Join Slack
<https://flower.ai/join-slack/>`__).

There's a dedicated ``#questions`` Slack channel if you need help, but we'd also love to
hear who you are in ``#introductions``!

The :doc:`Flower Collaborative AI Tutorial - Part 4: Use a federated learning strategy
<tutorial-series-use-a-federated-learning-strategy-pytorch>` goes into more depth about
strategies and the advanced behavior you can build with them.
