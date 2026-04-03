:og:description: Learn how to train a Convolutional Neural Network on CIFAR-10 using federated learning with Flower and TensorFlow in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a Convolutional Neural Network on CIFAR-10 using federated learning with Flower and TensorFlow in this step-by-step tutorial.

.. _quickstart-tensorflow:

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.app.Message.html

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.app.ArrayRecord.html

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: ref-api/flwr.clientapp.ClientApp.html

.. |fedavg_link| replace:: ``FedAvg``

.. _fedavg_link: ref-api/flwr.serverapp.strategy.FedAvg.html

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.serverapp.ServerApp.html

.. |strategy_start_link| replace:: ``start``

.. _strategy_start_link: ref-api/flwr.serverapp.Strategy.html#flwr.serverapp.Strategy.start

.. |strategy_link| replace:: ``Strategy``

.. _strategy_link: ref-api/flwr.serverapp.Strategy.html

#######################
 Quickstart TensorFlow
#######################

In this tutorial we will learn how to train a Convolutional Neural Network on CIFAR-10
using the Flower framework and TensorFlow. First of all, it is recommended to create a
virtual environment and run everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`.

Let's use ``flwr new`` to create a complete Flower+TensorFlow project. It will generate
all the files needed to run a federation of 10 nodes using |fedavg_link|_. By default,
the generated app uses a local simulation profile that ``flwr run`` submits to a managed
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

    $ flwr new @flwrlabs/quickstart-tensorflow

After running it you'll notice a new directory named ``quickstart-tensorflow`` has been
created. It should have the following structure:

.. code-block:: shell

    quickstart-tensorflow
    ├── tfexample
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
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 2.0013, 'train_acc': 0.2624}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_acc': 0.1216, 'eval_loss': 2.2686}
    INFO :      [ROUND 2/3]
    INFO :      ...
    INFO :      [ROUND 3/3]
    INFO :      ...
    INFO :      Strategy execution finished in 16.60s
    INFO :      Final results:
    INFO :          ServerApp-side Evaluate Metrics:
    INFO :          {}
    Saving final model to disk as final_model.keras...

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config "num-server-rounds=5 batch-size=16"

**********
 The Data
**********

This tutorial uses `Flower Datasets <https://flower.ai/docs/datasets/>`_ to easily
download and partition the `CIFAR-10` dataset. In this example you'll make use of the
`IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_
to generate `num_partitions` partitions. You can choose `other partitioners
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html>`_ available in
Flower Datasets. Each ``ClientApp`` will call this function to create the ``NumPy``
arrays that correspond to their data partition.

.. code-block:: python

    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]

***********
 The Model
***********

Next, we need a model. We defined a simple Convolutional Neural Network (CNN), but feel
free to replace it with a more sophisticated model if you'd like:

.. code-block:: python

    def load_model(learning_rate: float = 0.001):
        # Define a simple CNN for CIFAR-10 and set Adam optimizer
        model = keras.Sequential(
            [
                keras.Input(shape=(32, 32, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(10, activation="softmax"),
            ]
        )
        optimizer = keras.optimizers.Adam(learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

***************
 The ClientApp
***************

The main changes we have to make to use `Tensorflow` with `Flower` have to do with
converting the |arrayrecord_link|_ received in the |message_link|_ into numpy ndarrays
for use with the built-in ``set_weights()`` function. After training, the
``get_weights()`` function can be used to extract then pack the updated numpy ndarrays
into a ``Message`` from the ClientApp. We can make use of built-in methods in the
``ArrayRecord`` to make these conversions:

.. code-block:: python

    @app.train()
    def train(msg: Message, context: Context):

        # Load the model
        model = load_model(context.run_config["learning-rate"])
        # Extract the ArrayRecord from Message and convert to numpy ndarrays
        model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

        # Train the model
        ...

        # Pack the model weights into an ArrayRecord
        model_record = ArrayRecord(model.get_weights())

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

The ``train`` method also receives the ``Context``, giving access to configs for your
run and node. The run config hyperparameters are defined in the ``pyproject.toml`` of
your Flower App. The node config can only be set when running Flower with the Deployment
Runtime and is not directly configurable during simulations.

.. code-block:: python

    # Flower ClientApp
    app = ClientApp()


    @app.train()
    def train(msg: Message, context: Context):
        """Train the model on local data."""

        # Reset local Tensorflow state
        keras.backend.clear_session()

        # Load the data
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        x_train, y_train, _, _ = load_data(partition_id, num_partitions)

        # Load the model
        model = load_model(context.run_config["learning-rate"])
        model.set_weights(msg.content["arrays"].to_numpy_ndarrays())
        epochs = context.run_config["local-epochs"]
        batch_size = context.run_config["batch-size"]
        verbose = context.run_config.get("verbose")

        # Train the model
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Get training metrics
        train_loss = history.history["loss"][-1] if "loss" in history.history else None
        train_acc = (
            history.history["accuracy"][-1] if "accuracy" in history.history else None
        )

        # Pack and send the model weights and metrics as a message
        model_record = ArrayRecord(model.get_weights())
        metrics = {"num-examples": len(x_train)}
        if train_loss is not None:
            metrics["train_loss"] = train_loss
        if train_acc is not None:
            metrics["train_acc"] = train_acc
        content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})
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
``fraction_train`` which is read from the run config. You can find the default value
defined in the ``pyproject.toml``. Then, the execution of the strategy is launched when
invoking its |strategy_start_link|_ method. To it we pass:

- the ``Grid`` object.
- an ``ArrayRecord`` carrying a randomly initialized model that will serve as the global
  model to federated.
- the ``num_rounds`` parameter specifying how many rounds of ``FedAvg`` to perform.

.. code-block:: python

    # Create the ServerApp
    app = ServerApp()


    @app.main()
    def main(grid: Grid, context: Context) -> None:
        """Main entry point for the ServerApp."""
        # Load config
        num_rounds = context.run_config["num-server-rounds"]
        fraction_train = context.run_config["fraction-train"]

        # Load initial model
        model = load_model()
        arrays = ArrayRecord(model.get_weights())

        # Define and start FedAvg strategy
        strategy = FedAvg(
            fraction_train=fraction_train,
        )

        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=num_rounds,
        )

        # Save the final model
        ndarrays = result.arrays.to_numpy_ndarrays()
        final_model_name = "final_model.keras"
        print(f"Saving final model to disk as {final_model_name}...")
        model.set_weights(ndarrays)
        model.save(final_model_name)

Note the ``start`` method of the strategy returns a result object. This object contains
all the relevant information about the FL process, including the final model weights as
an ``ArrayRecord``, and federated training and evaluation metrics as ``MetricRecords``.
You can easily log the metrics using Python's `pprint
<https://docs.python.org/3/library/pprint.html>`_ and save the final model weights using
Tensorflow's ``save()`` function.

Congratulations! You've successfully built and run your first federated learning system.

.. tip::

    Check the :doc:`how-to-run-simulations` documentation to learn more about how to
    configure and run Flower simulations.

.. note::

    Check the source code of the extended version of this tutorial in
    |quickstart_tf_link|_ in the Flower GitHub repository.

.. |quickstart_tf_link| replace:: ``examples/quickstart-tensorflow``

.. _quickstart_tf_link: https://github.com/flwrlabs/flower/blob/main/examples/quickstart-tensorflow
