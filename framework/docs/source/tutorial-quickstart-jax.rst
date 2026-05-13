:og:description: Learn how to train a linear regression using federated learning with Flower and JAX in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a linear regression using federated learning with Flower and JAX in this step-by-step tutorial.

.. _quickstart-jax:

################
 Quickstart JAX
################

In this federated learning tutorial we will learn how to train a CNN model on the MNIST
dataset using Flower and `JAX <https://jax.readthedocs.io/en/latest/>`_ with the `Flax
<https://flax.readthedocs.io/en/latest/index.html>`_ library. It is recommended to
create a virtual environment and run everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`.

Let's use ``flwr new`` to create a complete Flower+JAX project. It will generate all the
files needed to run a federation of 50 nodes using |fedavg|_. By default, the generated
app uses a local simulation profile that ``flwr run`` submits to a managed local
SuperLink, which then executes the run with the Flower Simulation Runtime. The MNIST
dataset will be partitioned using |flowerdatasets|_'s |iidpartitioner|_.

Now that we have a rough idea of what this example is about, let's get started. First,
install Flower in your new environment:

.. code-block:: shell

    # In a new Python environment
    $ pip install flwr[simulation]

Then, run the command below:

.. code-block:: shell

    $ flwr new @flwrlabs/quickstart-jax

After running it you'll notice a new directory named ``quickstart-jax`` has been
created. It should have the following structure:

.. code-block:: shell

    quickstart-jax
    ├── jaxexample
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
    INFO :          ├── Number of rounds: 5
    INFO :      [ROUND 1/5]
    INFO :      configure_train: Sampled 20 nodes (out of 50)
    INFO :      aggregate_train: Received 20 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 2.1116, 'train_acc': 0.2821}
    INFO :      configure_evaluate: Sampled 20 nodes (out of 50)
    INFO :      aggregate_evaluate: Received 20 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 1.3394, 'eval_acc': 0.4984}
    INFO :      [ROUND 2/5]
    INFO :      ...
    INFO :      [ROUND 5/5]
    INFO :      ...
    INFO :      Strategy execution finished in 60.58s
    INFO :      Final results:
    INFO :          ServerApp-side Evaluate Metrics:
    INFO :          {}
    Saving final model to disk...

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config "num-server-rounds=5 batch-size=64"

What follows is an explanation of each component in the project you just created:
dataset partition, the model, defining the ``ClientApp`` and defining the ``ServerApp``.

**********
 The Data
**********

This tutorial uses |flowerdatasets|_ to easily download and partition the MNIST dataset.
In this example you'll make use of the |iidpartitioner|_ to generate ``num_partitions``
partitions. You can choose |otherpartitioners|_ available in Flower Datasets.

.. code-block:: python

    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="mnist",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)

    partition["train"].set_format("jax")
    partition["test"].set_format("jax")


    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [
            jnp.expand_dims(jnp.float32(img), 3) / 255 for img in batch["image"]
        ]
        batch["label"] = [jnp.int16(label) for label in batch["label"]]
        return batch


    train_partition = (
        partition["train"]
        .batch(batch_size, num_proc=2, drop_last_batch=True)
        .with_transform(apply_transforms)
    )
    test_partition = (
        partition["test"]
        .batch(batch_size, num_proc=2, drop_last_batch=True)
        .with_transform(apply_transforms)
    )

***********
 The Model
***********

We use `Flax <https://flax.readthedocs.io/en/latest/index.html>`_ to define a simple CNN
model for image classification:

.. code-block:: python

    class CNN(nn.Module):
        """A simple CNN model."""

        @nn.compact
        def __call__(self, x):
            x = nn.Conv(features=6, kernel_size=(5, 5))(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = nn.Conv(features=16, kernel_size=(5, 5))(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))  # flatten
            x = nn.Dense(features=120)(x)
            x = nn.relu(x)
            x = nn.Dense(features=84)(x)
            x = nn.relu(x)
            x = nn.Dense(features=10)(x)
            return x


    def create_train_state(learning_rate: float) -> TrainState:
        """Creates initial `TrainState`."""

        tx = optax.sgd(learning_rate, momentum=0.9)
        model, model_params = create_model(rng)
        return TrainState.create(apply_fn=model.apply, params=model_params, tx=tx)

In addition to defining the model architecture, we also include utility functions to
perform both training (i.e. ``train()``) and evaluation using the above model.

.. code-block:: python

    @jax.jit
    def apply_model(
        state: TrainState, images: Array, labels: Array
    ) -> Tuple[Any, Array, Array]:
        """Computes gradients, loss and accuracy for a single batch."""

        def loss_fn(params):
            logits = state.apply_fn({"params": params}, images)
            one_hot = jax.nn.one_hot(labels, 10)
            loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        return grads, loss, accuracy


    @jax.jit
    def update_model(state: TrainState, grads: Any) -> TrainState:
        return state.apply_gradients(grads=grads)


    def train(state: TrainState, train_ds) -> Tuple[TrainState, float, float]:
        """Train for a single epoch."""

        epoch_loss = []
        epoch_accuracy = []

        for batch in train_ds:
            batch_images = batch["image"]
            batch_labels = batch["label"]
            grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
            state = update_model(state, grads)
            epoch_loss.append(loss)
            epoch_accuracy.append(accuracy)
        train_loss = np.mean(epoch_loss)
        train_accuracy = np.mean(epoch_accuracy)
        return state, float(train_loss), float(train_accuracy)

***************
 The ClientApp
***************

The main changes we have to make to use JAX with Flower have to do with converting the
|arrayrecord_link|_ received in the |message_link|_ into NumPy arrays and vice versa
when generating the reply ``Message`` from the ClientApp. We also have to introduce the
``get_params()`` and ``set_params()`` functions for setting parameter values for the JAX
model. In ``get_params()``, JAX model parameters are extracted and represented as a list
of NumPy arrays. The ``set_params()`` function is the opposite: given a list of NumPy
arrays it creates a new ``TrainState`` with those parameters. We will combine these
functions with the built-in methods in the ``ArrayRecord`` to make these conversions:

.. code-block:: python

    def get_params(params: Any) -> List[npt.NDArray[Any]]:
        """Get model parameters as list of numpy arrays."""
        return [np.array(param) for param in jax.tree_util.tree_leaves(params)]


    def set_params(
        train_state: TrainState, global_params: Sequence[npt.NDArray[Any]]
    ) -> TrainState:
        """Create a new trainstate with the global_params."""
        new_params_dict = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(train_state.params), list(global_params)
        )
        return train_state.replace(params=new_params_dict)

.. code-block:: python

    # Create train state object (model + optimizer)
    lr = float(context.run_config["learning-rate"])
    train_state = create_train_state(lr)

    # Extract ArrayRecord from Message and convert to NumPy arrays
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    # Set JAX model parameters using the converted NumPy arrays
    train_state = set_params(train_state, ndarrays)

    # ... do some training

    # Extract NumPy arrays from the JAX model and convert back into an ArrayRecord
    params = get_params(train_state.params)
    model_record = ArrayRecord(params)

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

        # Create train state object (model + optimizer)
        lr = float(context.run_config["learning-rate"])
        train_state = create_train_state(lr)
        # Extract numpy arrays from ArrayRecord before applying
        ndarrays = msg.content["arrays"].to_numpy_ndarrays()
        train_state = set_params(train_state, ndarrays)

        # Load the data
        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])
        batch_size = int(context.run_config["batch-size"])
        trainloader, _ = load_data(partition_id, num_partitions, batch_size)

        train_state, loss, acc = jax_train(train_state, trainloader)
        params = get_params(train_state.params)

        # Construct and return reply Message
        model_record = ArrayRecord(params)
        metrics = {
            "train_loss": float(loss),
            "train_acc": float(acc),
            "num-examples": int(len(trainloader) * batch_size),
        }
        metric_record = MetricRecord(metrics)
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

In this example we use the |fedavg|_ and configure it with specific values read from the
run config. You can find the default values defined in the ``pyproject.toml``. Then, the
execution of the strategy is launched when invoking its |strategy_start_link|_ method.
To it we pass:

- the ``Grid`` object.
- an ``ArrayRecord`` carrying a randomly initialized model that will serve as the global
  model to be federated.
- a ``ConfigRecord`` with the training hyperparameters (learning rate) to be sent to the
  clients. The strategy will also insert the current round number in this config before
  sending it to the participating nodes.
- the ``num_rounds`` parameter specifying how many rounds of ``FedAvg`` to perform.

.. code-block:: python

    # Create ServerApp
    app = ServerApp()


    @app.main()
    def main(grid: Grid, context: Context) -> None:
        """Main entry point for the ServerApp."""

        # Read run config
        fraction_evaluate: float = float(context.run_config["fraction-evaluate"])
        num_rounds: int = int(context.run_config["num-server-rounds"])
        lr: float = float(context.run_config["learning-rate"])

        rng = random.PRNGKey(0)
        rng, _ = random.split(rng)
        _, model_params = create_model(rng)
        params = get_params(model_params)

        # Initialize FedAvg strategy
        strategy = FedAvg(
            fraction_train=0.4,
            fraction_evaluate=fraction_evaluate,
        )

        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord(params),
            train_config=ConfigRecord({"lr": lr}),
            num_rounds=num_rounds,
        )

        # Save final model to disk
        print("\nSaving final model to disk...")
        ndarrays = result.arrays.to_numpy_ndarrays()
        np.savez("final_model.npz", *ndarrays)

Note the ``start`` method of the strategy returns a result object. This object contains
all the relevant information about the FL process, including the final model weights as
an ``ArrayRecord``, and federated training and evaluation metrics as ``MetricRecords``.
You can easily log the metrics using Python's `pprint
<https://docs.python.org/3/library/pprint.html>`_ and save the global model NumPy arrays
using ``np.savez()`` as shown above.

Congratulations! You've successfully built and run your first federated learning system
for JAX with Flower!

.. tip::

    Check the :doc:`how-to-run-simulations` documentation to learn more about how to
    configure and run Flower simulations.

.. note::

    Check the source code of the extended version of this tutorial in
    |quickstart_jax_link|_ in the Flower GitHub repository.

.. |fedavg| replace:: ``FedAvg``

.. _fedavg: ref-api/flwr.serverapp.strategy.FedAvg.html

.. |flowerdatasets| replace:: Flower Datasets

.. _flowerdatasets: https://flower.ai/docs/datasets/

.. |iidpartitioner| replace:: ``IidPartitioner``

.. _iidpartitioner: https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner

.. |otherpartitioners| replace:: other partitioners

.. _otherpartitioners: https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html

.. |quickstart_jax_link| replace:: ``examples/quickstart-jax``

.. _quickstart_jax_link: https://github.com/flwrlabs/flower/tree/main/examples/quickstart-jax

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.app.Message.html

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.app.ArrayRecord.html

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: ref-api/flwr.clientapp.ClientApp.html

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.serverapp.ServerApp.html

.. |strategy_start_link| replace:: ``start``

.. _strategy_start_link: ref-api/flwr.serverapp.strategy.Strategy.html#flwr.serverapp.strategy.Strategy.start

.. |strategy_link| replace:: ``Strategy``

.. _strategy_link: ref-api/flwr.serverapp.Strategy.html
