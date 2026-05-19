:og:description: Learn how to train a classification model on the Higgs dataset using federated learning with Flower and XGBoost in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a classification model on the Higgs dataset using federated learning with Flower and XGBoost in this step-by-step tutorial.

.. _quickstart-xgboost:

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.app.Message.html

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.app.ArrayRecord.html

.. |context_link| replace:: ``Context``

.. _context_link: ref-api/flwr.app.Context.html

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: ref-api/flwr.clientapp.ClientApp.html

.. |fedxgbbagging_link| replace:: ``FedXgbBagging``

.. _fedxgbbagging_link: ref-api/flwr.serverapp.strategy.FedXgbBagging.html

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.serverapp.ServerApp.html

.. |strategy_start_link| replace:: ``start``

.. _strategy_start_link: ref-api/flwr.serverapp.strategy.Strategy.html#flwr.serverapp.strategy.Strategy.start

.. |strategy_link| replace:: ``Strategy``

.. _strategy_link: ref-api/flwr.serverapp.strategy.Strategy.html

.. |result_link| replace:: ``Result``

.. _result_link: ref-api/flwr.serverapp.strategy.Result.html

####################
 Quickstart XGBoost
####################

In this federated learning tutorial, we will learn how to train a simple XGBoost
classifier on Higgs dataset using Flower and XGBoost. It is recommended to create a
virtual environment and run everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`.

Let's use ``flwr new`` to create a complete Flower+XGBoost project. It will generate all
the files needed to run a federation of 10 nodes using |fedxgbbagging_link|_ strategy.
By default, the generated app uses a local simulation profile that ``flwr run`` submits
to a managed local SuperLink, which then executes the run with the Flower Simulation
Runtime. The dataset will be partitioned using Flower Dataset's `IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_.

|fedxgbbagging_link|_ (bootstrap aggregation) is an ensemble method that improves
stability and accuracy in machine learning, here applied to XGBoost in FL. Each client
generates a bootstrap sample by subsampling its data and trains a tree per round, which
is then aggregated by the server and added to the global model.

Now that we have a rough idea of what this example is about, let's get started. First,
install Flower in your new environment:

.. code-block:: shell

    # In a new Python environment
    $ pip install flwr[simulation]

Then, run the command below:

.. code-block:: shell

    $ flwr new @flwrlabs/quickstart-xgboost

After running it you'll notice a new directory named ``quickstart-xgboost`` has been
created. It should have the following structure:

.. code-block:: shell

    quickstart-xgboost
    ├── quickstart_xgboost
    │   ├── __init__.py
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your data loading and utility functions
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
    INFO :      Starting FedXgbBagging strategy:
    INFO :              ├── Number of rounds: 3
    INFO :      [ROUND 1/3]
    INFO :      configure_train: Sampled 2 nodes (out of 10)
    INFO :      aggregate_train: Received 2 results and 0 failures
    INFO :              └──> Aggregated MetricRecord: {}
    INFO :      configure_evaluate: Sampled 2 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 2 results and 0 failures
    INFO :              └──> Aggregated MetricRecord: {'auc': 0.7677505289821278}
    INFO :      [ROUND 2/3]
    INFO :      ...
    INFO :      [ROUND 3/3]
    INFO :      ...
    INFO :      Strategy execution finished in 132.88s
    INFO :      Final results:
    INFO :              ServerApp-side Evaluate Metrics:
    INFO :              {}
    Saving final model to disk...

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in the ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config "num-server-rounds=5 params.eta=0.2"

What follows is an explanation of each component in the project you just created:
configurations, dataset partitioning, defining the ``ClientApp``, and defining the
``ServerApp``.

********************
 The Configurations
********************

We define all required configurations / hyper-parameters inside the ``pyproject.toml``
file:

.. code-block:: toml

    [tool.flwr.app.config]
    num-server-rounds = 3
    fraction-train = 0.1
    fraction-evaluate = 0.1
    local-epochs = 1

    # XGBoost parameters
    params.objective = "binary:logistic"
    params.eta = 0.1 # Learning rate
    params.max-depth = 8
    params.eval-metric = "auc"
    params.nthread = 16
    params.num-parallel-tree = 1
    params.subsample = 1
    params.tree-method = "hist"

The ``local-epochs`` represents the number of iterations for local tree boost. We use
CPU for the training in default. One can assign it to a GPU by setting ``tree-method``
to ``gpu_hist``. We use AUC as evaluation metric.

**********
 The Data
**********

We will use `Flower Datasets <https://flower.ai/docs/datasets/>`_ to easily download and
partition the `Higgs` dataset. In this example, you'll make use of the `IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_
to generate `num_partitions` partitions. You can choose from other `partitioners
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html>`_ available in
Flower Datasets:

.. code-block:: python

    partitioner = IidPartitioner(num_partitions=num_clients)
    fds = FederatedDataset(
        dataset="jxie/higgs",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id, split="train")
    partition.set_format("numpy")

    # Train/test splitting
    train_data, valid_data, num_train, num_val = train_test_split(
        partition, test_fraction=0.2, seed=42
    )

    # Reformat data to DMatrix for xgboost
    train_dmatrix = transform_dataset_to_dmatrix(train_data)
    valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

We train/test split using the given partition (client's local data), and reformat data
to DMatrix for the ``xgboost`` package. The functions of ``train_test_split`` and
``transform_dataset_to_dmatrix`` are defined as below:

.. code-block:: python

    def train_test_split(partition, test_fraction, seed):
        """Split the data into train and validation set given split rate."""
        train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
        partition_train = train_test["train"]
        partition_test = train_test["test"]

        num_train = len(partition_train)
        num_test = len(partition_test)

        return partition_train, partition_test, num_train, num_test


    def transform_dataset_to_dmatrix(data):
        """Transform dataset to DMatrix format for xgboost."""
        x = data["inputs"]
        y = data["label"]
        new_data = xgb.DMatrix(x, label=y)
        return new_data

***************
 The ClientApp
***************

The main changes we have to make to use `XGBoost` with `Flower` have to do with
converting the |arrayrecord_link|_ received in the |message_link|_ into a `XGBoost`
loadable binary object, and vice versa when generating the reply ``Message`` from the
ClientApp. We can make use of the following conversions:

.. code-block:: python

    @app.train()
    def train(msg: Message, context: Context):

        # Instantiate a XGBoost model
        bst = xgb.Booster(params=params)
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())

        # Load global model into booster
        bst.load_model(global_model)

        # ...

        # Convert XGB object back into an ArrayRecord
        # Note: we store the model as the first item in a list into ArrayRecord,
        # which can be accessed using index ["0"].
        local_model = bst.save_raw("json")
        model_np = np.frombuffer(local_model, dtype=np.uint8)
        model_record = ArrayRecord([model_np])

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
    def train(msg: Message, context: Context) -> Message:
        # Load model and data
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        train_dmatrix, _, num_train, _ = load_data(partition_id, num_partitions)

        # Read from run config
        num_local_round = context.run_config["local-epochs"]
        # Flatted config dict and replace "-" with "_"
        cfg = replace_keys(unflatten_dict(context.run_config))
        params = cfg["params"]

        global_round = msg.content["config"]["server-round"]
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                params,
                train_dmatrix,
                num_boost_round=num_local_round,
            )
        else:
            bst = xgb.Booster(params=params)
            global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = _local_boost(bst, num_local_round, train_dmatrix)

        # Save model
        local_model = bst.save_raw("json")
        model_np = np.frombuffer(local_model, dtype=np.uint8)

        # Construct reply message
        # Note: we store the model as the first item in a list into ArrayRecord,
        # which can be accessed using index ["0"].
        model_record = ArrayRecord([model_np])
        metrics = {
            "num-examples": num_train,
        }
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        return Message(content=content, reply_to=msg)

At the first round, we call ``xgb.train()`` to build up the first set of trees. From the
second round, we load the global model sent from server to new build Booster object, and
then update model weights on local training data with function ``_local_boost`` as
follows:

.. code-block:: python

    def _local_boost(bst_input, num_local_round, train_dmatrix):
        # Update trees based on local training data.
        for i in range(num_local_round):
            bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds()
            - num_local_round : bst_input.num_boosted_rounds()
        ]
        return bst

Given ``num_local_round``, we update trees by calling ``bst_input.update`` method. After
training, the last ``N=num_local_round`` trees will be extracted to send to the server.

The ``@app.evaluate()`` method would be near identical with two exceptions: (1) the
model is not locally trained, instead it is used to evaluate its performance on the
locally held-out validation set; (2) including the model in the reply Message is no
longer needed because it is not locally modified.

***************
 The ServerApp
***************

To construct a |serverapp_link|_, we define its ``@app.main()`` method. This method
receives as input arguments:

- a ``Grid`` object that will be used to interface with the nodes running the
  ``ClientApp`` to involve them in a round of train/evaluate/query or other.
- a ``Context`` object that provides access to the run configuration.

In this example we use the ``FedXgbBagging`` strategy. Then, we initialize an empty
global model as the XGBoost model will be initialized on client side in the first round.
After that, the execution of the strategy is launched when invoking its
|strategy_start_link|_ method. To it we pass:

- the ``Grid`` object.
- an ``ArrayRecord`` carrying a randomly initialized model that will serve as the global
      model to federate.
- the ``num_rounds`` parameter specifying how many rounds to perform.

.. code-block:: python

    # Create ServerApp
    app = ServerApp()


    @app.main()
    def main(grid: Grid, context: Context) -> None:
        # Read run config
        num_rounds = context.run_config["num-server-rounds"]
        fraction_train = context.run_config["fraction-train"]
        fraction_evaluate = context.run_config["fraction-evaluate"]
        # Flatted config dict and replace "-" with "_"
        cfg = replace_keys(unflatten_dict(context.run_config))
        params = cfg["params"]

        # Init global model
        # Init with an empty object; the XGBooster will be created
        # and trained on the client side.
        global_model = b""
        # Note: we store the model as the first item in a list into ArrayRecord,
        # which can be accessed using index ["0"].
        arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])

        # Initialize FedXgbBagging strategy
        strategy = FedXgbBagging(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
        )

        # Start strategy, run FedXgbBagging for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=num_rounds,
        )

        # Save final model to disk
        bst = xgb.Booster(params=params)
        global_model = bytearray(result.arrays["0"].numpy().tobytes())

        # Load global model into booster
        bst.load_model(global_model)

        # Save model
        print("\nSaving final model to disk...")
        bst.save_model("final_model.json")

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
    <https://github.com/flwrlabs/flower/blob/main/examples/xgboost-quickstart>`_ of the
    extended version of this tutorial in ``examples/xgboost-quickstart`` in the Flower
    GitHub repository.
