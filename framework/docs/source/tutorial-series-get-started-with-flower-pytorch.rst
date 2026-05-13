#########################
 Get started with Flower
#########################

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

Welcome to the Flower federated learning tutorial!

In this tutorial, we'll build a federated learning system using the Flower framework,
Flower Datasets and PyTorch. In part 1, we use PyTorch for model training and data
loading. In part 2, we federate this PyTorch project using Flower.

.. tip::

    `Star Flower on GitHub <https://github.com/flwrlabs/flower>`__ ⭐️ and join the
    Flower community on Flower Discuss and the Flower Slack to connect, ask questions,
    and get help:

    - `Join Flower Discuss <https://discuss.flower.ai/>`__ We'd love to hear from you in
      the ``Introduction`` topic! If anything is unclear, post in ``Flower Help -
      Beginners``.
    - `Join Flower Slack <https://flower.ai/join-slack>`__ We'd love to hear from you in
      the ``#introductions`` channel! If anything is unclear, head over to the
      ``#questions`` channel.

Let's get started! 🌼

*************
 Preparation
*************

Before we begin with any actual code, let's make sure that we have everything we need.

Install dependencies
====================

First, we install the Flower package ``flwr``:

.. code-block:: shell

    # In a new Python environment
    $ pip install -U "flwr[simulation]"

Then, run the command below:

.. code-block:: shell

    $ flwr new @flwrlabs/quickstart-pytorch

.. note::

    If you're on Windows and see unexpected terminal output (e.g.: ``� □[32m□[1m``),
    check :ref:`this FAQ entry <faq-windows-unexpected-output>`.

After running it you'll notice a new directory named ``quickstart-pytorch`` has been
created. It should have the following structure:

.. code-block:: shell

    quickstart-pytorch
    ├── pytorchexample
    │   ├── __init__.py
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

Next, we install the project and its dependencies, which are specified in the
``pyproject.toml`` file.

.. code-block:: shell

    $ cd quickstart-pytorch
    $ pip install -e .

Before we dive into federated learning, we'll take a look at the dataset that we'll be
using for this tutorial, which is the `CIFAR-10
<https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset, and run a simple centralized
training pipeline using PyTorch.

The ``CIFAR-10`` dataset
========================

Federated learning can be applied to many different types of tasks across different
domains. In this tutorial, we introduce federated learning by training a simple
convolutional neural network (CNN) on the popular CIFAR-10 dataset. CIFAR-10 can be used
to train image classifiers that distinguish between images from ten different classes:
'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', and
'truck'.

We simulate having multiple datasets from multiple organizations (also called the
“cross-silo” setting in federated learning) by splitting the original CIFAR-10 dataset
into multiple partitions. Each partition will represent the data from a single
organization. We're doing this purely for experimentation purposes. In the real world
there's no need for data splitting because each organization already has their own data
(the data is naturally partitioned).

Each organization will act as a client in the federated learning system. Having ten
organizations participate in a federation means having ten clients connected to the
federated learning server.

We use the `Flower Datasets <https://flower.ai/docs/datasets/>`_ library
(``flwr-datasets``) to partition CIFAR-10 into ten partitions using
``FederatedDataset``. Using the ``load_data()`` function defined in ``task.py``, we will
create a small training and test set for each of the ten organizations and wrap each of
these into a PyTorch ``DataLoader``:

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
        # Divide data on each node: 80% train, 20% test
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

We now have a function that can return a training set and validation set
(``trainloader`` and ``valloader``) representing one dataset from one of ten different
organizations. Each ``trainloader``/``valloader`` pair contains 4000 training examples
and 1000 validation examples. There's also a single ``testloader`` (we did not split the
test set). Again, this is only necessary for building research or educational systems,
actual federated learning systems have their data naturally distributed across multiple
partitions.

*****************************************
 The model, training, and test functions
*****************************************

Next, we're going to use PyTorch to define a simple convolutional neural network. This
introduction assumes basic familiarity with PyTorch, so it doesn't cover the
PyTorch-related aspects in full detail. If you want to dive deeper into PyTorch, we
recommend `this introductory tutorial
<https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html>`_.

Model
=====

We will use the simple CNN described in the aforementioned PyTorch tutorial (The
following code is already defined in ``task.py``):

.. code-block:: python

    class Net(nn.Module):
        """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

Training and test functions
===========================

The PyTorch quickstart also provides the usual training and test functions:

.. code-block:: python

    def train(net, trainloader, epochs, lr, device):
        """Train the model on the training set."""
        net.to(device)  # move model to GPU if available
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        net.train()
        running_loss = 0.0
        for _ in range(epochs):
            for batch in trainloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad()
                loss = criterion(net(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        avg_trainloss = running_loss / (epochs * len(trainloader))
        return avg_trainloss


    def test(net, testloader, device):
        """Validate the model on the test set."""
        net.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for batch in testloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader)
        return loss, accuracy

********************************
 Federated Learning with Flower
********************************

In federated learning, the server sends global model parameters to the client, and the
client updates the local model with parameters received from the server. It then trains
the model on the local data (which changes the model parameters locally) and sends the
updated/changed model parameters back to the server (or, alternatively, it sends just
the gradients back to the server, not the full model parameters).

Constructing Messages
=====================

In Flower, the server and clients communicate by sending and receiving |message_link|_
objects. A ``Message`` carries a ``RecordDict`` as its main payload. The ``RecordDict``
is like a Python dictionary that can contain multiple records of different types. There
are three main types of records:

- |arrayrecord_link|_: Contains model parameters as a dictionary of NumPy arrays
- |metricrecord_link|_: Contains training or evaluation metrics as a dictionary of
  integers, floats, lists of integers, or lists of floats.
- |configrecord_link|_: Contains configuration parameters as a dictionary of integers,
  floats, strings, booleans, or bytes. Lists of these types are also supported.

Let's see a few examples of how to work with these types of records and, ultimately,
construct a ``RecordDict`` that can be sent over a ``Message``.

.. code-block:: python

    from flwr.app import ArrayRecord, MetricRecord, ConfigRecord, RecordDict

    # ConfigRecord can be used to communicate configs between ServerApp and ClientApp
    # They can hold scalars, but also strings and booleans
    config = ConfigRecord(
        {"batch_size": 32, "use_augmentation": True, "data-path": "/my/dataset"}
    )

    # MetricRecords expect scalar-based metrics (i.e. int/float/list[int]/list[float])
    # By limiting the types Flower can aggregate MetricRecords automatically
    metrics = MetricRecord({"accuracy": 0.9, "losses": [0.1, 0.001], "perplexity": 2.31})

    # ArrayRecord objects are designed to communicate arrays/tensors/weights from ML models
    array_record = ArrayRecord(my_model.state_dict())  # for a PyTorch model
    array_record_other = ArrayRecord(my_model.to_numpy_ndarrays())  # for other ML models

    # A RecordDict is like a dictionary that holds named records.
    # This is the main payload of a Message
    rd = RecordDict({"my-config": config, "metrics": metrics, "my-model": array_record})

Define the Flower ClientApp
===========================

Federated learning systems consist of a server and multiple nodes or clients. In Flower,
we create a |serverapp_link|_ and a |clientapp_link|_ to run the server-side and
client-side code, respectively.

The core functionality of the ``ClientApp`` is to perform some action with the local
data that the node it runs from (e.g. an edge device, a server in a data center, or a
laptop) has access to. In this tutorial such action is to train and evaluate the small
CNN model defined earlier using the local training and validation data.

Training
--------

We can define how the ``ClientApp`` performs training by wrapping a function with the
``@app.train()`` decorator. In this case we name this function ``train`` because we'll
use it to train the model on the local data. The function always expects two arguments:

- A |message_link|_: The message received from the server. It contains the model
  parameters and any other configuration information sent by the server.
- A |context_link|_: The context object that contains information about the node
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

    from flower_tutorial.task import train as train_fn

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
    ``FedAvg(weighted_by="my-different-key")``).

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

    from flower_tutorial.task import test as test_fn


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
fraction of connected nodes to involve in a round of training.

To construct a |serverapp_link|_, we define its ``@app.main()`` method. This method
receives as input arguments:

- a ``Grid`` object that will be used to interface with the nodes running the
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

Run the training
================

With all of these components in place, we can now run the federated learning simulation
with Flower! The last step is to run our simulation in the command line, as follows:

.. code-block:: shell

    $ flwr run . --stream

This submits the run to the managed local SuperLink for the ``[superlink.local]``
profile (identified in the Flower configuration with ``address = ":local:"``), which
then executes the federated learning simulation with 10 clients, or SuperNodes, using
the Flower Simulation Runtime. Plain ``flwr run .`` would submit the run, print the run
ID, and return without streaming logs. For the full local workflow, see
:doc:`how-to-run-flower-locally`. If you run into SQL database errors during local
simulations, see :ref:`FAQ <faq-local-superlink-db-error>`.

You should expect streamed output similar to this:

.. code-block:: shell

    Starting local SuperLink on 127.0.0.1:39093...
    Successfully started run 1859953118041441032
    INFO :      Starting FedAvg strategy:
    INFO :          ├── Number of rounds: 3
    INFO :      [ROUND 1/3]
    INFO :      configure_train: Sampled 5 nodes (out of 10)
    INFO :      aggregate_train: Received 5 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 2.25811}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 2.304821, 'eval_acc': 0.0965}
    INFO :      [ROUND 2/3]
    INFO :      ...
    INFO :      [ROUND 3/3]
    INFO :      ...
    INFO :      Strategy execution finished in 17.18s
    INFO :      Final results:
    INFO :          ServerApp-side Evaluate Metrics:
    INFO :          {}
    Saving final model to disk...

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in ``pyproject.toml`` like this:

.. code-block:: shell

    # Run the simulation with 5 server rounds and 3 local epochs
    $ flwr run . --stream --run-config "num-server-rounds=5 local-epochs=3"

.. tip::

    Learn more about how to configure the execution of your Flower App by checking the
    `pyproject.toml <how-to-configure-pyproject-toml.html>`_ guide.

Behind the scenes
=================

So how does this work? How does Flower execute this simulation?

When we execute ``flwr run`` against the default local connection configuration (for
example, the one with ``address = ":local:"``), Flower submits the run to the managed
local SuperLink. By default, the local SuperLink will configure the simulation runtime
to use 10 clients. Each will run an instance of the ``ClientApp`` we defined earlier.

The local SuperLink then starts the ``ServerApp`` and asks it to issue instructions to
those nodes using the ``FedAvg`` strategy. In this example, ``FedAvg`` is configured
with two key parameters:

- ``fraction-train=0.5`` → select 50% of the available clients for training
- ``fraction-evaluate=1.0`` → select 100% of the available clients for evaluation

This means in our example, 5 out of 10 clients will be selected for training, and all 10
clients will later participate in evaluation.

A typical round looks like this:

- **Training**

  1. ``FedAvg`` randomly selects 5 clients (50% of 10).
  2. Flower sends a ``TRAIN`` message to each selected ``ClientApp``.
  3. Each ``ClientApp`` calls the function decorated with ``@app.train()``, then returns
     a ``Message`` containing an ``ArrayRecord`` (the updated model parameters) and a
     ``MetricRecord`` (the training loss and number of examples).
  4. The ``ServerApp`` receives all replies.
  5. ``FedAvg`` aggregates all ``ArrayRecord`` into a new ``ArrayRecord`` representing
     the new global model and combines all ``MetricRecord``.

- **Evaluation**

  1. ``FedAvg`` selects all 10 clients (100%).
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

Congratulations, you just trained a convolutional neural network, federated over 10
clients! With that, you understand the basics of federated learning with Flower. The
same approach you've seen can be used with other machine learning frameworks (not just
PyTorch) and tasks (not just CIFAR-10 image classification), for example NLP with
Hugging Face Transformers or speech with SpeechBrain.

In the next tutorial, we're going to cover some more advanced concepts. Want to
customize your strategy? Do learning rate decay at the strategy and communicate it to
the clients ? Or evaluate the aggregated model on the server side? We'll cover all this
and more in the next tutorial.

************
 Next steps
************

Before you continue, make sure to join the Flower community on Flower Discuss (`Join
Flower Discuss <https://discuss.flower.ai>`__) and on Slack (`Join Slack
<https://flower.ai/join-slack/>`__).

There's a dedicated ``#questions`` Slack channel if you need help, but we'd also love to
hear who you are in ``#introductions``!

The :doc:`Flower Federated Learning Tutorial - Part 2
<tutorial-series-use-a-federated-learning-strategy-pytorch>` goes into more depth about
strategies and all the advanced things you can build with them.
