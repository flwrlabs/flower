:og:description: Learn how to train an MLP on MNIST using federated learning with Flower and MLX in this step-by-step tutorial.
.. meta::
    :description: Learn how to train an MLP on MNIST using federated learning with Flower and MLX in this step-by-step tutorial.

.. _quickstart-mlx:

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

################
 Quickstart MLX
################

In this federated learning tutorial, we will learn how to train a simple MLP on MNIST
using Flower and MLX. It is recommended to create a virtual environment and run
everything within a :doc:`virtualenv <contributor-how-to-set-up-a-virtual-env>`.

Let's use ``flwr new`` to create a complete Flower+MLX project. It will generate all the
files needed to run a federation of 10 nodes using |fedavg_link|_. By default, the
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

    $ flwr new @flwrlabs/quickstart-mlx

After running it you'll notice a new directory named ``quickstart-mlx`` has been
created. It should have the following structure:

.. code-block:: shell

    quickstart-mlx
    ├── mlxexample
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

To run the project do:

.. code-block:: shell

    # Run with default arguments and stream logs
    $ flwr run . --stream

Plain ``flwr run .`` submits the run, prints the run ID, and returns without streaming
logs. For the full local workflow, see :doc:`how-to-run-flower-locally`.

With default arguments, you will see streamed output like this:

.. code-block:: shell

    Starting local SuperLink on 127.0.0.1:39093...
    Successfully started run 1859953118041441032
    INFO :      Starting FedAvg strategy:
    INFO :          ├── Number of rounds: 3
    INFO :      [ROUND 1/3]
    INFO :      configure_train: Sampled 10 nodes (out of 10)
    INFO :      aggregate_train: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'accuracy': 0.270375007390976, 'loss': 2.2390866}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'accuracy': 0.2720000118017197, 'loss': 2.24028}
    INFO :      [ROUND 2/3]
    INFO :      ...
    INFO :      [ROUND 3/3]
    INFO :      ...
    INFO :      Strategy execution finished in 9.96s
    INFO :      Final results:
    INFO :          ServerApp-side Evaluate Metrics:
    INFO :          {}
    Saving final model to disk...

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in the ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"

What follows is an explanation of each component in the project you just created:
dataset partitioning, the model, defining the ``ClientApp``, and defining the
``ServerApp``.

**********
 The Data
**********

We will use `Flower Datasets <https://flower.ai/docs/datasets/>`_ to easily download and
partition the `MNIST` dataset. In this example, you'll make use of the `IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_
to generate `num_partitions` partitions. You can choose from other partitioners
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html>`_ available in
Flower Datasets:

.. code-block:: python

    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )
    partition = fds.load_partition(partition_id)
    partition_splits = partition.train_test_split(test_size=0.2, seed=42)

    partition_splits["train"].set_format("numpy")
    partition_splits["test"].set_format("numpy")

    train_partition = partition_splits["train"].map(
        lambda img: {"img": img.reshape(-1, 28 * 28).squeeze().astype(np.float32) / 255.0},
        input_columns="image",
    )
    test_partition = partition_splits["test"].map(
        lambda img: {"img": img.reshape(-1, 28 * 28).squeeze().astype(np.float32) / 255.0},
        input_columns="image",
    )

    data = (
        train_partition["img"],
        train_partition["label"].astype(np.uint32),
        test_partition["img"],
        test_partition["label"].astype(np.uint32),
    )

    train_images, train_labels, test_images, test_labels = map(mx.array, data)

***********
 The Model
***********

We define the model as in the `centralized MLX example
<https://github.com/ml-explore/mlx-examples/tree/main/mnist>`_, it's a simple MLP:

.. code-block:: python

    class MLP(nn.Module):
        """A simple MLP."""

        def __init__(
            self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
        ):
            super().__init__()
            layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
            self.layers = [
                nn.Linear(idim, odim)
                for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
            ]

        def __call__(self, x):
            for l in self.layers[:-1]:
                x = mx.maximum(l(x), 0.0)
            return self.layers[-1](x)

We also define some utility functions to test our model and to iterate over batches.

.. code-block:: python

    def loss_fn(model, X, y):
        return mx.mean(nn.losses.cross_entropy(model(X), y))


    def eval_fn(model, X, y):
        return mx.mean(mx.argmax(model(X), axis=1) == y)


    def batch_iterate(batch_size, X, y):
        perm = mx.array(np.random.permutation(y.size))
        for s in range(0, y.size, batch_size):
            ids = perm[s : s + batch_size]
            yield X[ids], y[ids]

The ClientApp
=============

The main changes we have to make to use `MLX` with `Flower` will be found in the
``get_params()`` and ``set_params()`` functions. MLX doesn't provide an easy way to
convert the model parameters into a list of ``np.array`` objects (the format we need for
message serialization to work).

MLX stores its parameters as follows:

.. code-block:: shell

    {
    "layers": [
        {"weight": mlx.core.array, "bias": mlx.core.array},
        {"weight": mlx.core.array, "bias": mlx.core.array},
        ...,
        {"weight": mlx.core.array, "bias": mlx.core.array}
    ]
    }

Therefore, to get our list of ``np.array`` objects, we need to extract each array and
convert it into a NumPy array:

.. code-block:: python

    def get_params(model):
        layers = model.parameters()["layers"]
        return [np.array(val) for layer in layers for _, val in layer.items()]

For the ``set_params()`` function, we perform the reverse operation. We receive a list
of NumPy arrays and want to convert them into MLX parameters. Therefore, we iterate
through pairs of parameters and assign them to the `weight` and `bias` keys of each
layer dictionary:

.. code-block:: python

    def set_params(model, parameters):
        new_params = {}
        new_params["layers"] = [
            {"weight": mx.array(parameters[i]), "bias": mx.array(parameters[i + 1])}
            for i in range(0, len(parameters), 2)
        ]
        model.update(new_params)

The rest of the functionality is directly inspired by the centralized case. The
|clientapp_link|_ will train the model on local data using the standard MLX training
loop:

.. code-block:: python

    # Train the model on local data
    for _ in range(num_epochs):
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            _, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

Let's put everything together and see the complete implementation of the ``ClientApp``.
First, the behavior in a round of training is defined inside a function wrapped with the
``@app.train()`` decorator.

After reading configuration parameters from the |context_link|_, we instantiate the
model and apply the global parameters sent by the server using the ``set_params()``
function defined above. We then define the optimizer and loss function, load the local
data partition using the ``load_data()``, and train the model on the data. Finally, we
compute the accuracy and loss on the training data and construct a reply |message_link|_
containing an |arrayrecord_link|_ with the updated model parameters and a
``MetricRecord`` with the training accuracy and loss. Very importantly it also contains
the key `num-examples` which will be used by the server to perform weighted averaging of
the model parameters. The value of this key is the number of training examples in the
local data partition.

.. code-block:: python

    # Flower ClientApp
    app = ClientApp()


    @app.train()
    def train(msg: Message, context: Context):
        """Train the model on local data."""

        # Read config
        num_layers = context.run_config["num-layers"]
        input_dim = context.run_config["input-dim"]
        hidden_dim = context.run_config["hidden-dim"]
        batch_size = context.run_config["batch-size"]
        learning_rate = context.run_config["learning-rate"]
        num_epochs = context.run_config["local-epochs"]

        # Instantiate model and apply global parameters
        model = MLP(num_layers, input_dim, hidden_dim, output_dim=10)
        ndarrays = msg.content["arrays"].to_numpy_ndarrays()
        set_params(model, ndarrays)

        # Define optimizer and loss function
        optimizer = optim.SGD(learning_rate=learning_rate)
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

        # Load data
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        train_images, train_labels, _, _ = load_data(partition_id, num_partitions)

        # Train on local data
        for _ in range(num_epochs):
            for X, y in batch_iterate(batch_size, train_images, train_labels):
                _, grads = loss_and_grad_fn(model, X, y)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

        # Compute train accuracy and loss
        accuracy = eval_fn(model, train_images, train_labels)
        loss = loss_fn(model, train_images, train_labels)
        # Construct and return reply Message
        model_record = ArrayRecord(get_params(model))
        metrics = {
            "num-examples": len(train_images),
            "accuracy": float(accuracy.item()),
            "loss": float(loss.item()),
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        return Message(content=content, reply_to=msg)

The ``ClientApp`` also allows for evaluation of the model on local test data. This can
be done by defining a function wrapped with the ``@app.evaluate()`` decorator. The
signature of the function is identical to that of the ``train()`` function. As shown
below, the evaluation function is very similar to the training function, except that we
don't perform any training. We still need to update the model parameters with those sent
by the server, and then we compute the loss and accuracy using the functions defined
above. Finally, we construct a reply |message_link|_ containing a ``MetricRecord`` with
the evaluation accuracy and loss, as well as the key `num-examples`, which will be used
by the server to perform weighted averaging of the metrics.

.. code-block:: python

    @app.evaluate()
    def evaluate(msg: Message, context: Context):
        """Evaluate the model on local data."""

        # ... read config, instantiate model, load data

        # Evaluate the model on local data
        accuracy = eval_fn(model, test_images, test_labels)
        loss = loss_fn(model, test_images, test_labels)

        # Construct and return reply Message
        metrics = {
            "num-examples": len(test_images),
            "accuracy": float(accuracy.item()),
            "loss": float(loss.item()),
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"metrics": metric_record})
        return Message(content=content, reply_to=msg)

The ServerApp
-------------

***************
 The ServerApp
***************

To construct a |serverapp_link|_, we define its ``@app.main()`` method. This method
receives as input arguments:

- a ``Grid`` object that will be used to interface with the nodes running the
  ``ClientApp`` to involve them in a round of train/evaluate/query or other.
- a ``Context`` object that provides access to the run configuration.

In this example we use the |fedavg_link|_ and left it with its default parameters. Then,
after initializing the ``MLP`` that would serve as global model in the first round, the
execution of the strategy is launched when invoking its |strategy_start_link|_ method.
To it we pass:

- the ``Grid`` object.
- an ``ArrayRecord`` carrying a randomly initialized model that will serve as the global
  model to federate.
- the ``num_rounds`` parameter specifying how many rounds of ``FedAvg`` to perform.

.. code-block:: python

    # Create ServerApp
    app = ServerApp()


    @app.main()
    def main(grid: Grid, context: Context) -> None:
        """Main entry point for the ServerApp."""
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]
        num_layers = context.run_config["num-layers"]
        input_dim = context.run_config["input-dim"]
        hidden_dim = context.run_config["hidden-dim"]

        # Initialize global model
        model = MLP(num_layers, input_dim, hidden_dim, output_dim=10)
        params = get_params(model)
        arrays = ArrayRecord(params)

        # Initialize FedAvg strategy
        strategy = FedAvg()

        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=num_rounds,
        )

        # Save final model to disk
        print("\nSaving final model to disk...")
        ndarrays = result.arrays.to_numpy_ndarrays()
        set_params(model, ndarrays)
        model.save_weights("final_model.npz")

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
    <https://github.com/flwrlabs/flower/blob/main/examples/quickstart-mlx>`_ of the
    extended version of this tutorial in ``examples/quickstart-mlx`` in the Flower
    GitHub repository.
