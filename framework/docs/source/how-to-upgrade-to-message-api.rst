:og:description: Upgrade seamlessly to Flower 1.21 with this guide for transitioning your setup to the latest features and enhancements powered by Flower's Message API.
.. meta::
    :description: Upgrade seamlessly to Flower 1.21 with this guide for transitioning your setup to the latest features and enhancements powered by Flower's Message API.

.. |numpyclient_link| replace:: ``NumPyClient``

.. _numpyclient_link: ref-api/flwr.client.NumPyClient.html

.. |client_link| replace:: ``Client``

.. _client_link: ref-api/flwr.client.Client.html

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: ref-api/flwr.client.ClientApp.html

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.server.ServerApp.html

.. |strategy_link| replace:: ``Strategy``

.. _strategy_link: ref-api/flwr.serverapp.strategy.Strategy.html

.. |fedavg_link| replace:: ``FedAvg``

.. _fedavg_link: ref-api/flwr.serverapp.strategy.FedAvg.html

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.common.Message.html

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.common.ArrayRecord.html

.. |metricrecord_link| replace:: ``MetricRecord``

.. _metricrecord_link: ref-api/flwr.common.MetricRecord.html

.. |configrecord_link| replace:: ``ConfigRecord``

.. _configrecord_link: ref-api/flwr.common.ConfigRecord.html

.. |recorddict_link| replace:: ``RecordDict``

.. _recorddict_link: ref-api/flwr.common.RecordDict.html

########################
 Upgrade to Message API
########################

Welcome to the migration guide for updating your Flower Apps to use Flower's Message
API! This guide will walk you through the necessary steps to transition from Flower Apps
based on ``Strategy`` and ``NumPyClient`` to their equivalent using the new Message API.
This guide is relevant when updating pre-``1.21`` Flower apps to the latest stable
version.

Let's dive in!

.. tip::

    If you would like to create a new Flower App using the `Message API`, run the ``flwr
    new`` command and choose the appropriate template. Alternatively, you may want to
    take a look at the `quickstart-pytorch
    <https://github.com/flwrlabs/flower/blob/main/hub/apps/quickstart-pytorch>`_
    example.

********************
 Summary of changes
********************

Thousands of Flower Apps have been created using the Strategies and |numpyclient_link|_
abstractions. With the introduction of the Message API, these apps can now take
advantage of a more powerful and flexible communication layer with the |message_link|_
abstraction being its cornerstone. Messages replace the previous ``FitIns`` and
``FitRes`` data structures (and their equivalent for the other operations) into a
single, unified and more versatile data structure.

To fully take advantage of the new Message API, you will need to update your app's code
to use the new message-based communication patterns. This guide will show you how to:

1. Update your |serverapp_link|_ to make use of the new ``Message``-based strategies.
   You won't need to use the ``server_fn`` anymore. The new strategies make it easier to
   customize how the different federated learning rounds are executed, to retrieve
   results from your run, and more.
2. Update your |clientapp_link|_ so it operates directly on ``Message`` objects received
   from the |serverapp_link|_. You will be able to keep most of the code from your
   |numpyclient_link|_ implementation but you won't need to create a new class anymore
   or use the helper ``client_fn`` function.

.. tip::

    The main payload ``Message`` objects carry are of type |recorddict_link|_. You can
    think of it as a dictionary that can hold other types of records, namely
    |arrayrecord_link|_, |metricrecord_link|_, and |configrecord_link|_. Let's see with
    a few short examples what's the intended usage behind each type of record.

    .. code-block:: python

        from flwr.app import ArrayRecord, MetricRecord, ConfigRecord, RecordDict

        # ConfigRecord can be used to communicate configs between ServerApp and ClientApp
        # They can hold scalars, but also strings and booleans
        config = ConfigRecord(
            {"batch_size": 32, "use_augmentation": True, "data-path": "/my/dataset"}
        )

        # MetricRecords are designed for scalar-based metrics only (i.e. int/float/list[int]/list[float])
        # By limiting the types Flower can aggregate MetricRecords automatically
        metrics = MetricRecord({"accuracy": 0.9, "losses": [0.1, 0.001], "perplexity": 2.31})

        # ArrayRecord objects are designed to communicate arrays/tensors/weights from ML models
        array_record = ArrayRecord(my_model.state_dict())  # for a PyTorch model
        array_record_other = ArrayRecord(my_model.to_numpy_ndarrays())  # for other ML models

        # A RecordDict is like a dictionary that holds named records.
        # This is the main payload of a Message
        rd = RecordDict({"my-config": config, "metrics": metrics, "my-model": array_record})

    Please refer to the documentation for each record for all the details on how they
    can be constructed and adapted to your usecase. In this guide we won't delve into
    the specifics of each record type, but rather focus on the overall migration
    process.

****************
 Install update
****************

The first step is to update the Flower version defined in the `pyproject.toml` in your
app:

.. code-block:: toml
    :caption: pyproject.toml
    :emphasize-lines: 2

    dependencies = [
        "flwr[simulation]>=1.21.0", # update Flower package
        # ...
    ]

Then, run the following command to install the updated dependencies:

.. code-block:: bash

    # Install the app with updated dependencies
    $ pip install -e .

***************************
 Update your ``ServerApp``
***************************

Starting with Flower 1.21, the ``ServerApp`` no longer requires a ``server_fn`` function
to make use of strategies. This is because a new collection of strategies (all sharing
the common |strategy_link|_ base class) has been created to operate directly on
``Message`` objects, allowing for a more streamlined and flexible approach to federated
learning rounds.

.. note::

    The new ``Message``-based strategies are located in the `flwr.serverapp.strategy
    <ref-api/flwr.serverapp.strategy.html>`_ module unlike the previous strategies which
    were located in the `flwr.server.strategy <ref-api/flwr.server.strategy.html>`_
    module. Over time more strategies will be added to the `flwr.serverapp.strategy`
    module. Users are encouraged to use these new strategies.

Since Flower 1.10, the recommended ``ServerApp`` implementation would look something
like the code snippet below. Naturally, more customization can be applied to the
Strategy by, for example, reading the config from the ``Context``. But to keep things
focused, we will use a simple example and assume we are federating a PyTorch model.

.. note::

    ``Context`` has moved to ``flwr.app`` and ``ServerApp`` to ``flwr.serverapp``.
    Importing them from ``flwr.common`` or ``flwr.server`` is deprecated.

.. code-block:: python

    from flwr.common import Context  # Deprecated, import from flwr.app instead
    from flwr.server import ServerApp  # Deprecated, import from flwr.serverapp instead
    from flwr.server import ServerAppComponents, ServerConfig, start_server
    from flwr.server.strategy import FedAvg


    def server_fn(context: Context):

        # Instantiate strategy with initial parameters
        model = MyModel()
        parameters = ndarrays_to_parameters(
            [v.cpu().numpy() for v in model.state_dict().values()]
        )
        strategy = FedAvg(fraction_fit=0.5, initial_parameters=parameters)
        # Set number of rounds and return
        config = ServerConfig(num_rounds=3)
        return ServerAppComponents(config=config, strategy=strategy)


    # Create ServerApp with helper function
    app = ServerApp(server_fn=server_fn)

With Flower 1.21 and later, the equivalent ``ServerApp`` using the new Message API would
look as shown below after following these steps:

1. Define the ``main`` method under the ``@app.main()`` decorator. If your ``server_fn``
   was reading config values from the ``Context`` you can still do so (consider copying
   those lines directly from your ``server_fn`` function)
2. Instantiate your model as usual and construct an ``ArrayRecord`` out of its
   parameters.
3. Replace your existing strategy with one from the ``flwr.serverapp.strategy`` module.
   For example with |fedavg_link|_. Pass the arguments related to node sampling to the
   constructor of your strategy.
4. Call the ``start`` method of the new strategy passing to it the ``ArrayRecord``
   representing the initial state of your global model, the number of FL rounds and, the
   ``Grid`` object (which is used internally to communicate with the nodes executing the
   ``ClientApp``).

Note how we no longer need the ``server_fn`` function. The ``Context`` is still
accessible, allowing you to customize how the ``ServerApp`` behaves at runtime. With the
new strategies, a new ``start`` method is available. It defines a for loop which sets
the steps involved in a round of FL. By default it behaves as the original strategies
do, i.e. a round of FL training followed by one of FL evaluation and a stage to evaluate
the global model. Note how the ``start`` method returns results. These are of type
``Result`` and by default contain the final global model (via ``result.arrays``) as well
as aggregated |metricrecord_link|_ from federated stages and, optionally, metrics from
evaluation stages done at the ``ServerApp``.

.. note::

    In addition to helper methods for working with PyTorch models, the
    |arrayrecord_link|_ class comes with a pair of methods to convert such record to and
    from a list of `NumPy` arrays (i.e. to ``to_numpy_ndarrays`` and
    ``from_numpy_ndarrays``). You may choose these methods if you aren't working with
    PyTorch models.

.. warning::

    Note that the new strategies have renamed several arguments related to node/client
    sampling, replacing the term `"fit"` with `"train"` and `"clients"` with `"nodes"`.
    The following arguments were renamed:

    - ``fraction_fit`` → ``fraction_train``
    - ``min_fit_clients`` → ``min_train_nodes``
    - ``min_evaluate_clients`` → ``min_evaluate_nodes``
    - ``min_available_clients`` → ``min_available_nodes``

.. code-block:: python
    :emphasize-lines: 3,9,10,14,17,20

    from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
    from flwr.serverapp import Grid, ServerApp
    from flwr.serverapp.strategy import FedAvg

    # Create ServerApp
    app = ServerApp()


    @app.main()
    def main(grid: Grid, context: Context) -> None:

        # Defined model to federate and extract parameters
        model = MyModel()
        arrays = ArrayRecord(global_model.state_dict())

        # Instantiate strategy
        strategy = FedAvg(fraction_train=0.5)

        # Start the strategy
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=3,
        )
        print(result)

***********************
 Update your ClientApp
***********************

Similar to the ``ServerApp``, the ``ClientApp`` no longer requires a helper function
(i.e. ``client_fn`` ) that instantiates a |numpyclient_link|_ or base |client_link|_
object. Instead, with the Message API, you get to define directly how the ClientApp
operates on ``Message`` objects received from the ``ServerApp``.

Remember ``NumPyClient`` came with two key built-in methods, ``fit`` and ``evaluate``,
that were respectively designed for doing federated training and evaluation using the
client's local data. With the new Message API, you can define similar methods directly
on the ``ClientApp`` via decorators to handle incoming ``Message`` objects.

Let's see a basic example showing first a minimal ``NumPyClient``-based ``ClientApp``
and then the upgraded design using the Message API.

.. note::

    ``Context`` has moved to ``flwr.app`` and ``ClientApp`` to ``flwr.clientapp``.
    Importing them from ``flwr.common`` or ``flwr.client`` is deprecated.

.. code-block:: python

    from flwr.client import ClientApp  # Deprecated, import from flwr.clientapp instead
    from flwr.client import NumPyClient
    from flwr.common import Context  # Deprecated, import from flwr.app instead
    from my_utils import train_fn, test_fn, get_weights, set_weights


    class MyFlowerClient(NumPyClient):

        def __init__(self):
            self.model = MyModel()
            self.train_loader = DataLoader(...)
            self.test_loader = DataLoader(...)

        def fit(self, parameters, config):
            """Fit the model to the local data using the parameters sent by ServerApp."""
            # Update model with the latest parameters
            set_weights(self.model, parameters)
            # Train the model locally
            train_fn(self.model, self.train_loader)
            # Return the updated parameters and number of training examples
            return get_weights(self.model), len(self.train_loader.dataset), {}

        def evaluate(self, parameters, config):
            """Evaluate the model on the local data using the parameters sent by ServerApp."""
            # Update model with the latest parameters
            set_weights(self.model, parameters)
            # Evaluate the model locally
            loss, accuracy = test_fn(self.model, self.test_loader)
            # Return the evaluation results
            return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}


    def client_fn(context: Context):
        # Return an instance of MyFlowerClient
        return MyFlowerClient().to_client()


    app = ClientApp(client_fn=client_fn)

Upgrading a ClientApp designed around the ``NumPyClient`` and ``client_fn`` abstractions
to the Message API would result in the following code. Note that the behavior of the
``ClientApp`` is defined directly in its methods (i.e. a secondary class based on
``NumPyClient`` is no longer needed).

The |clientapp_link|_ abstraction comes with built-in ``@app.train`` and
``@app.evaluate`` decorators. The arguments the associated methods receive have been
unified and they both operate on ``Message`` objects. Each method is responsible for
handling the incoming ``Message`` objects and returning the appropriate response (also
as a ``Message``). Note that you'll still be able to use the functions you might have
written to, for example, train your model using the ML framework of your choice. In this
example those are represented by ``train_fn`` and ``test_fn``. Follow these steps to
migrate your existing ``ClientApp``:

1. Introduce the ``@app.train`` and ``@app.evaluate`` decorators and respective methods.
2. Copy the lines of code you had in your ``client_fn`` reading config values from the
   ``Context`` into your ``train`` and ``evaluate`` methods implementations (created in
   step 1).
3. From the ``Message`` object, extract the relevant items (e.g. an ``ArrayRecord``
   defining the global model, a ``ConfigRecord`` containing configs for the current
   round) to use in your training and evaluation logic.
4. Copy the lines calling the functions that do the actual training/evaluation (in the
   code snippet below we named those ``train_fn`` and ``test_fn``).
5. Based on the method, construct a ``RecordDict`` and use it to construct the reply
   ``Message``.

.. note::

    The payload that ``Message`` objects carry is of type |recorddict_link|_ which can
    contain records of type ``ArrayRecord``, ``MetricRecord`` and ``ConfigRecord``.

.. code-block:: python
    :emphasize-lines: 9,10,18,23,33,34,37,38,46,56,57

    from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
    from flwr.clientapp import ClientApp
    from my_utils import train_fn, test_fn

    # Flower ClientApp
    app = ClientApp()


    @app.train()
    def train(msg: Message, context: Context) -> Message:
        """Train the model on local data."""

        # Init Model and data loader
        train_loader = DataLoader(...)
        model = MyModel()

        # Read ArrayRecord received from ServerApp
        arrays = msg.content["arrays"]
        # Load weights to model
        model.load_state_dict(arrays.to_torch_state_dict())

        # Do local training
        train_loss = train_fn(model, train_loader)

        # Construct reply Message: arrays and metrics
        model_record = ArrayRecord(model.state_dict())
        # You can include any metric (scalar or list of scalars)
        # relevant to your usecase.
        # A weighting metric (`num-examples` by default) is always
        # expected by FedAvg to do aggregation
        metrics = MetricRecord(
            {
                "train_loss": train_loss,
                "num-examples": len(train_loader.dataset),
            }
        )
        # Construct RecordDict and add ArrayRecord and MetricRecord
        content = RecordDict({"arrays": model_record, "metrics": metrics})
        return Message(content=content, reply_to=msg)


    @app.evaluate()
    def evaluate(msg: Message, context: Context) -> Message:
        """Evaluate the model on local data."""

        # Identical to @app.train but returning only metrics
        # after doing local evaluation
        # ...

        # Do local evaluation
        loss, accuracy = test_fn(model, test_loader)

        # Construct reply Message
        # Retrun metrics relevant to usecase
        # THe weighting metric is also sent and will be used
        # to do weighted aggregation of metrics
        metrics = MetricRecord(
            {
                "eval_loss": loss,
                "eval_accuracy": accuracy,
                "num-examples": len(test_loader.dataset),
            }
        )
        # Construct RecordDict and add MetricRecord
        content = RecordDict({"metrics": metrics})
        return Message(content=content, reply_to=msg)

This concludes the migration guide, we hope you found it useful! Happy federating!
