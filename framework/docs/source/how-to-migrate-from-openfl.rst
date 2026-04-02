:og:description: Migrate your OpenFL workloads to Flower in this step-by-step tutorial
.. meta::
    :description: Migrate your OpenFL workloads to Flower in this step-by-step tutorial

.. _how-to-migrate-from-openfl:

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

########################
 OpenFL Migration Guide
########################

It was `recently announced
<https://github.com/securefederatedai/openfederatedlearning>`_ that The Open Federated
Learning project (formerly known as OpenFL) is no longer being developed or maintained.
This guide, written in collaboration with the OpenFL developers, aims to create an easy
path for OpenFL users to bring their workloads into Flower.

***************************************
 Creating a Flower App for OpenFL code
***************************************

Let's start by creating a Flower app where the OpenFL code can be migrated to.

Install dependencies
====================

First, we install the Flower package ``flwr``:

.. code-block:: shell

    # In a new Python environment
    $ pip install -U "flwr[simulation]"

Then, run the command below:

.. code-block:: shell

    $ flwr new @flwrlabs/quickstart-pytorch

After running the command, a new directory called ``quickstart-pytorch`` will be
created. Here is a comparison between it and the relevant files in a typical
``openfl-example`` folder:

.. grid:: 2

    .. grid-item-card:: OpenFL Example

        .. code-block:: text
           :emphasize-lines: 15-17

           openfl-example
           ├── requirements.txt
           ├── .workspace
           ├── plan
           │   ├── plan.yaml
           │   ├── cols.yaml
           │   ├── defaults
           │   └── data.yaml
           ├── logs
           ├── cert
           ├── save
           ├── data
           └── src
               ├── __init__.py
               ├── taskrunner.py
               ├── utils.py
               └── dataloader.py

    .. grid-item-card:: Flower Quickstart (PyTorch)

        .. code-block:: text
           :emphasize-lines: 4-6

           quickstart-pytorch
           ├── pytorchexample
           │   ├── __init__.py
           │   ├── client_app.py
           │   ├── server_app.py
           │   └── task.py
           ├── pyproject.toml
           └── README.md

Let's start with an overview of which areas of OpenFL and Flower directory structures
you'll want to focus on. We will go through these in depth in later sections of the
guide:

- **Model**: In OpenFL, the model is usually defined in ``taskrunner.py``. In Flower,
  the model definition is usually located in ``task.py``.
- **Train and Evaluate Functions**: In OpenFL, these are part of the `TaskRunner`
  subclass in ``taskrunner.py``. For Flower, you'll find these in ``client_app.py`` and
  identified beneath the ``@app.train()`` and ``@app.evaluate`` decorators.
- **Aggregation Functions**: In OpenFL, most examples use the ``WeightedAverage()``
  aggregation algorithm by default. If you're using a different aggregation algorithm,
  you'll find it in ``plan.yaml`` by searching for `aggregation_type`. In Flower, the
  aggregation algorithm is defined as a |strategy_link|_.

Migrate your model
==================

The model is very straightforward to port from OpenFL to Flower. If you are working with
a PyTorch model, OpenFL has a ``PyTorchTaskRunner`` that inherits from ``nn.Module`` (in
``taskrunner.py``) - and includes other things like the ``train`` and ``validate``
functions. Flower assumes you bring a standard PyTorch model, so it's as easy as moving
the model definition to ``task.py`` in the ``quickstart-pytorch`` directory, and
changing the inheritance of the Net back to ``nn.Module``. For a concrete example, see
the following OpenFL TaskRunner code snippet:

.. code-block:: python
    :emphasize-lines: 2,50-60

    # OpenFL PyTorch TaskRunner
    class PyTorchCNN(PyTorchTaskRunner):
        """
        Simple CNN for classification.

        PyTorchTaskRunner inherits from nn.Module, so you can define your model
        in the same way that you would for PyTorch
        """

        def __init__(self, device="cpu", **kwargs):
            """Initialize.

            Args:
                device: The hardware device to use for training (Default = "cpu")
                **kwargs: Additional arguments to pass to the function

            """
            super().__init__(device=device, **kwargs)

            # Define the model
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            self.to(device)

            # `self.optimizer` must be set for optimizer weights to be federated
            self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

            # Set the loss function
            self.loss_fn = F.cross_entropy

        def forward(self, x):
            """
            Forward pass of the model.

            Args:
                x: Data input to the model for the forward pass
            """
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

        def train_(
            self, train_dataloader: Iterator[Tuple[np.ndarray, np.ndarray]]
        ) -> Metric:
            """TaskRunner train function"""
            ...

        def validate_(
            self, valid_dataloader: Iterator[Tuple[np.ndarray, np.ndarray]]
        ) -> Metric:
            """TaskRunner validation function"""
            ...

And the corresponding PyTorch model used by Flower:

.. code-block:: python

    # Standard PyTorch model definition in Flower (Found in task.py)
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

Migrate your training and test functions
========================================

Recent versions of OpenFL had a simple way of defining training and evaluation
functions. The setting and extraction of model weights was hidden from users, and a list
of ``Metric`` values resulting from training or validation could be explicitly returned
from the function. To make migration easy, see the highlighted blocks that can carry
over directly to the Flower ``client_app.py`` file:

.. code-block:: python
    :emphasize-lines: 33-42,60-74

    from openfl.federated import PyTorchTaskRunner
    from openfl.utilities import Metric


    class PyTorchCNN(PyTorchTaskRunner):
        """
        Simple CNN for classification.

        """

        def __init__(self, device="cpu", **kwargs):
            # Model definition
            ...

        def forward(self, x):
            # Forward function definition
            ...

        def train_(
            self, train_dataloader: Iterator[Tuple[np.ndarray, np.ndarray]]
        ) -> Metric:
            """
            Train single epoch.

            Override this function in order to use custom training.

            Args:
                train_dataloader: Train dataset batch generator. Yields (samples, targets) tuples of
                size = `self.data_loader.batch_size`.
            Returns:
                Metric: An object containing name and np.ndarray value.
            """
            losses = []
            for data, target in train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.detach().cpu().numpy())
            loss = np.mean(losses)
            return Metric(name=self.loss_fn.__name__, value=np.array(loss))

        def validate_(
            self, validation_dataloader: Iterator[Tuple[np.ndarray, np.ndarray]]
        ) -> Metric:
            """
            Perform validation on PyTorch Model

            Override this function for your own custom validation function

            Args:
                validation_dataloader: Validation dataset batch generator.
                                       Yields (samples, targets) tuples
            Returns:
                Metric: An object containing name and np.ndarray value
            """

            total_samples = 0
            val_score = 0
            with torch.no_grad():
                for data, target in validation_dataloader:
                    samples = target.shape[0]
                    total_samples += samples
                    data, target = data.to(self.device), target.to(
                        self.device, dtype=torch.int64
                    )
                    output = self(data)
                    # get the index of the max log-probability
                    pred = output.argmax(dim=1)
                    val_score += pred.eq(target).sum().cpu().numpy()

            accuracy = val_score / total_samples
            return Metric(name="accuracy", value=np.array(accuracy))

In Flower more control is given to users by default. With the introduction of the
Message API, the training and validation functions are assumed to be stateless, so there
is some initialization that must be handled by user code. The good news is that this
setup is standard and quite reusable across examples. Let's see how the relevant OpenFL
``train_`` function fits into Flower:

.. code-block:: python
    :emphasize-lines: 22-38

    # client_app.py

    ...


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

        # Adapt the OpenFL training function here
        ##############################################
        criterion = torch.nn.CrossEntropyLoss().to(device)
        lr = msg.content["config"]["lr"]
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        losses = []
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        train_loss = np.mean(losses)
        #############################################

        # Construct and return reply Message
        model_record = ArrayRecord(model.state_dict())
        metrics = {
            "train_loss": train_loss,
            "num-examples": len(trainloader.dataset),
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        return Message(content=content, reply_to=msg)

Notice the model is reininitialized, the dataloader is initialized and configured, and
hyperparameters are each set before the core training operation begins. At the
conclusion of the training, the model weights are extracted and packed into an
``ArrayRecord`` and the model metrics are captured in a ``MetricRecord``. It's necessary
to also send the `num-examples` as a metric, as this is needed for capturing the weight
to give to the model parameters for ``FedAvg``.

Here is the corresponding evaluation function, with the highlighted area representing
the migrated code from OpenFL:

.. code-block:: python
    :emphasize-lines: 17-32

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

        # Adapt the OpenFL evaluation function here
        ########################################################
        total_samples = 0
        val_score = 0
        with torch.no_grad():
            for data, target in valloader:
                samples = target.shape[0]
                total_samples += samples
                data, target = data.to(device), target.to(self.device, dtype=torch.int64)
                output = model(data)
                # get the index of the max log-probability
                pred = output.argmax(dim=1)
                val_score += pred.eq(target).sum().cpu().numpy()

        eval_acc = val_score / total_samples
        ########################################################

        # Construct and return reply Message
        metrics = {
            "eval_acc": eval_acc,
            "num-examples": len(valloader.dataset),
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"metrics": metric_record})
        return Message(content=content, reply_to=msg)

The code can be mostly pasted in unmodified! There are a few references to cleanup (i.e.
changing `self` to `model`) to fit with the Flower variables, but the logic remains the
same.

Migrating the Data Loaders
==========================

Unlike OpenFL, Flower does not require that you use their own Dataloaders when
developing your application. This means you can simply DataLoaders in the same way that
you would for PyTorch, Tensorflow, or any other framework. For research and
experimentation purposes, a single dataset can be sharded into multiple partitions. This
information is passed to each ``ClientApp`` through the ``Context``:

.. code-block:: python

    # In client_app.py
    @app.train()
    def train(msg: Message, context: Context):
        ...

        # Load the data
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        trainloader, _ = load_data(partition_id, num_partitions)

Flower also has its own library for partitioning single datasets in distributions
representative of what can be expected in real world settings. For more information, see
the `flwr-datasets <https://flower.ai/docs/datasets/>`_ documentation for details.

Client-side Code
================

In OpenFL, the client side code was known as a Collaborator. In Flower, the application
that data owners operate is referred to as a ``ClientApp``. Each of the files referred
to so far (``client_app.py``, ``task.py``) are launched by the clients using the `flwr
run` command. Beyond the code that is defined, Flower has the ability to insert dynamic
changes through a configuration file, called ``pyproject.toml``. This can include
application specific changes like hyperparameters, but also other information like
ServerApp address, etc. Importantly, this file is shared between parties operating the
``ClientApp`` and ``ServerApp``. This concept directly maps to the Federated Learning
Plan (FLPlan) concept in OpenFL captured in the ``plan.yaml`` file of every workspace.

.. code-block:: shell

    # Flower pyproject.toml

    ...

    [tool.flwr.app.config]
    num-server-rounds = 3
    fraction-evaluate = 0.5
    local-epochs = 1
    learning-rate = 0.1
    batch-size = 32

    ...

Server-side Code
================

In OpenFL, all of the aggregator-side code is configured via the `plan.yaml` file
through the specification of different arguments. In Flower, the exact tasks performed
by the server are more configurable through code. For example, aggregation algorithms
are added through a ``Strategy``, and the logic to save models is added explictly. Here
is a ``ServerApp`` (akin to an OpenFL Aggregator) compatible with the prior code
snippets:

.. code-block:: python

    import torch
    from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
    from flwr.serverapp import Grid, ServerApp
    from flwr.serverapp.strategy import FedAvg

    from pytorchexample.task import Net, load_centralized_dataset, test

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

You'll notice that most ``ServerApp`` examples have specific logic for working with a
given deep learning framework (in this case PyTorch) due to the saving of a final model.
This functionality is optional, but mirrors the automatic saving of a model at the end
of an OpenFL experiment. This ``ServerApp`` change requires only a few lines of
modifications, and Flower has support for an extensive set of deep learning frameworks
in it's `examples <https://github.com/flwrlabs/flower/tree/main/examples>`_ (Tensorflow,
FastAI, Huggingface, etc.) should you need reference code.

**************
 Further help
**************

For a complete PyTorch example that goes into depth on various Flower components, see
the `Get started with Flower
<https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html>`_
tutorial. While we expect this guide will help most users get migrated quickly to the
Flower ecosystem, certain complex OpenFL workloads may require more clarification or
help. If you have further questions, `join the Flower Slack
<https://flower.ai/join-slack/>`_ (and use the channel ``#questions``) or join our
`OpenFL Continuity Program
<https://docs.google.com/forms/d/e/1FAIpQLScprGGX_jFRoEUv4HbJkkhkg6O7e5eCiq7uP95_0xK5Qnt1gA/viewform>`_
to get in touch with our team!

.. admonition:: Important

    As we work with the OpenFL community, we'll be periodically updating this guide.
    Please feel free to share any feedback with us!

Happy migrating! 🚀
