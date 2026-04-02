:og:description: Learn how to train a Convolutional Neural Network on CIFAR-10 using federated learning with Flower and PyTorch in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a Convolutional Neural Network on CIFAR-10 using federated learning with Flower and PyTorch in this step-by-step tutorial.

.. _quickstart-pytorch:

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.app.Message.html

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.app.ArrayRecord.html

.. |context_link| replace:: ``Context``

.. _context_link: ref-api/flwr.app.Context.html

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: ref-api/flwr.clientapp.ClientApp.html

.. |fedavg_link| replace:: ``FedAvg``

.. _fedavg_link: ref-api/flwr.serverapp.strategy.FedAvg.html

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.serverapp.ServerApp.html

.. |strategy_start_link| replace:: ``start``

.. _strategy_start_link: ref-api/flwr.serverapp.strategy.Strategy.html#flwr.serverapp.strategy.Strategy.start

.. |strategy_link| replace:: ``Strategy``

.. _strategy_link: ref-api/flwr.serverapp.strategy.Strategy.html

.. |result_link| replace:: ``Result``

.. _result_link: ref-api/flwr.serverapp.strategy.Result.html

####################
 Quickstart PyTorch
####################

In this federated learning tutorial we will learn how to train a Convolutional Neural
Network on CIFAR-10 using Flower and PyTorch. It is recommended to create a virtual
environment and run everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`.

Let's use ``flwr new`` to create a complete Flower+PyTorch project. It will generate all
the files needed to run a federation of 10 nodes using |fedavg_link|_. By default, the
generated app uses a local simulation profile that ``flwr run`` submits to a managed
local SuperLink, which then executes the run with the Flower Simulation Runtime. The
dataset will be partitioned using Flower Dataset's `IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_.

Now that we have a rough idea of what this example is about, let's get started. First,
install Flower in your new environment:

.. code-block:: shell

    # In a new Python environment
    $ pip install flwr[simulation]

Then, run the command below:

.. code-block:: shell

    $ flwr new @flwrlabs/quickstart-pytorch

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

If you haven't yet installed the project and its dependencies, you can do so by:

.. code-block:: shell

    # From the directory where your pyproject.toml is
    $ pip install -e .

To run the project, do:

.. code-block:: shell

    # Run with default arguments and stream logs
    $ flwr run . --stream

Plain ``flwr run .`` submits the run, prints the run ID, and returns without streaming
logs. For the full local workflow, see :doc:`how-to-run-flower-locally`.

With default arguments you will see streamed output like this:

.. code-block:: shell

    Starting local SuperLink on 127.0.0.1:39093...
    Successfully started run 1859953118041441032
    INFO :      Starting FedAvg strategy:
    INFO :          ├── Number of rounds: 3
    INFO :      [ROUND 1/3]
    INFO :      configure_train: Sampled 5 nodes (out of 10)
    INFO :      aggregate_train: Received 5 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 2.149280}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 2.31319, 'eval_acc': 0.10004}
    INFO :      [ROUND 2/3]
    INFO :      ...
    INFO :      [ROUND 3/3]
    INFO :      ...
    INFO :      Strategy execution finished in 16.56s
    INFO :      Final results:
    INFO :          Server-side Evaluate Metrics:
    INFO :          {}
    Saving final model to disk...

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config "num-server-rounds=5 local-epochs=3"

What follows is an explanation of each component in the project you just created:
dataset partition, the model, defining the ``ClientApp`` and defining the ``ServerApp``.

**********
 The Data
**********

This tutorial uses `Flower Datasets <https://flower.ai/docs/datasets/>`_ to easily
download and partition the `CIFAR-10` dataset. In this example you'll make use of the
`IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_
to generate `num_partitions` partitions. You can choose `other partitioners
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html>`_ available in
Flower Datasets. Each ``ClientApp`` will call this function to create dataloaders with
the data that correspond to their data partition.

.. code-block:: python

    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch


    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)

***********
 The Model
***********

We defined a simple Convolutional Neural Network (CNN), but feel free to replace it with
a more sophisticated model if you'd like:

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

In addition to defining the model architecture, we also include two utility functions to
perform both training (i.e. ``train()``) and evaluation (i.e. ``test()``) using the
above model. These functions should look fairly familiar if you have some prior
experience with PyTorch. Note these functions do not have anything specific to Flower.
That being said, the training function will normally be called, as we'll see later, from
a Flower client passing its own data. In summary, your clients can use standard
training/testing functions to perform local training or evaluation:

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

***************
 The ClientApp
***************

The main changes we have to make to use `PyTorch` with `Flower` have to do with
converting the |arrayrecord_link|_ received in the |message_link|_ into a `PyTorch`
state_dict, and vice versa when generating the reply ``Message`` from the ClientApp. We
can make use of built-in methods in the ``ArrayRecord`` to make these conversions:

.. code-block:: python

    @app.train()
    def train(msg: Message, context: Context):

        # Instantiate a PyTorch model
        model = Net()
        # Extract ArrayRecord from Message and convert to PyTorch state_dict
        state_dict = msg.content["arrays"].to_torch_state_dict()
        # Load received state_dict into model
        model.load_state_dict(state_dict)

        # ...

        # Convert state_dict back into an ArrayRecord
        array_record = ArrayRecord(model.state_dict())

The rest of the functionality is directly inspired by the centralized case. The
|clientapp_link|_ comes with three core methods (``train``, ``evaluate``, and ``query``)
that we can implement for different purposes. For example: ``train`` to train the
received model using the local data; ``evaluate`` to assess its performance of the
received model on a validation set; and ``query`` to retrieve information about the node
executing the ``ClientApp``. In this tutorial we will only make use of ``train`` and
``evaluate``.

Let's see how the ``train`` method can be implemented. It receives as input arguments a
|message_link|_ from the ``ServerApp``. By default it carries:

- an ``ArrayRecord`` with the arrays of the model to federate. By default they can be
  retrieved with key ``"arrays"`` when accessing the message content.
- a ``ConfigRecord`` with the configuration sent from the ``ServerApp``. By default it
  can be retrieved with key ``"config"`` when accessing the message content.

The ``train`` method also receives the |context_link|_, giving access to configs for
your run and node. The run config hyperparameters are defined in the ``pyproject.toml``
of your Flower App. The node config can only be set when running Flower with the
Deployment Runtime and is not directly configurable during simulations.

.. code-block:: python

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
        # Include the locally-trained model
        model_record = ArrayRecord(model.state_dict())
        # Include some statistics such as the training loss
        # We also want to include the number of examples used for training
        # so the strategy in the ServerApp can do FedAvg
        metrics = {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
        }
        metric_record = MetricRecord(metrics)
        # RecordDict are the main payload type in Messages
        # We insert both the ArrayRecord and the MetricRecord into it
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        return Message(content=content, reply_to=msg)

The ``@app.evaluate()`` method would be near identical with two exceptions: (1) the
model is not locally trained, instead it is used to evaluate its performance on the
locally held-out validation set; (2) including the model in the reply Message is no
longer needed because it is not locally modified.

***************
 The ServerApp
***************

To construct a |serverapp_link|_ we define its ``@app.main()`` method. This method
receive as input arguments:

- a ``Grid`` object that will be used to interface with the nodes running the
  ``ClientApp`` to involve them in a round of train/evaluate/query or other.
- a ``Context`` object that provides access to the run configuration.

In this example we use the |fedavg_link|_ and configure it with a specific value of
``fraction_evaluate`` which is read from the run config. You can find the default value
defined in the ``pyproject.toml``. Then, the execution of the strategy is launched when
invoking its |strategy_start_link|_ method. To it we pass:

- the ``Grid`` object.
- an ``ArrayRecord`` carrying a randomly initialized model that will serve as the global
  model to federate.
- a ``ConfigRecord`` with the training hyperparameters to be sent to the clients. The
  strategy will also insert the current round number in this config before sending it to
  the participating nodes.
- the ``num_rounds`` parameter specifying how many rounds of ``FedAvg`` to perform.
- an ``evaluate_fn`` function that will be called to evaluate the global model on
  centralized test data after each round.

.. code-block:: python

    # Create ServerApp
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

Note the ``start`` method of the strategy returns a |result_link|_ object. This object
contains all the relevant information about the FL process, including the final model
weights as an ``ArrayRecord``, and federated training and evaluation metrics as
``MetricRecords``.

Congratulations! You've successfully built and run your first federated learning system.

.. tip::

    Check the :doc:`how-to-run-simulations` documentation to learn more about how to
    configure and run Flower simulations.

.. note::

    Check the `source code
    <https://github.com/flwrlabs/flower/blob/main/examples/quickstart-pytorch>`_ of the
    extended version of this tutorial in ``examples/quickstart-pytorch`` in the Flower
    GitHub repository.
