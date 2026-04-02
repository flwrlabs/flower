:og:description: Learn how to train a SqueezeNet model on MNIST using federated learning with Flower and fastai in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a SqueezeNet model on MNIST using federated learning with Flower and fastai in this step-by-step tutorial.

.. _quickstart-fastai:

###################
 Quickstart fastai
###################

In this federated learning tutorial we will learn how to train a SqueezeNet model on
MNIST using Flower and fastai. It is recommended to create a virtual environment and run
everything within a :doc:`virtualenv <contributor-how-to-set-up-a-virtual-env>`.

Then, clone the code example directly from GitHub:

.. code-block:: shell

    git clone --depth=1 https://github.com/flwrlabs/flower.git _tmp \
                 && mv _tmp/examples/quickstart-fastai . \
                 && rm -rf _tmp && cd quickstart-fastai

This will create a new directory called `quickstart-fastai` containing the following
files:

.. code-block:: shell

    quickstart-fastai
    ├── fastai_example
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

Next, activate your environment, then run:

.. code-block:: shell

    # Navigate to the example directory
    $ cd path/to/quickstart-fastai

    # Install project and dependencies
    $ pip install -e .

This example uses a local simulation profile that ``flwr run`` submits to a managed
local SuperLink, which then executes the run with the Flower Simulation Runtime,
creating a federation of 10 nodes using `FedAvg
<https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html#flwr.server.strategy.FedAvg>`_
as the aggregation strategy. The dataset will be partitioned using Flower Dataset's
`IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_.
Let's run the project:

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
    INFO :          └──> Aggregated MetricRecord: {}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 3.1197, 'eval_acc': 0.14874}
    INFO :      [ROUND 2/3]
    INFO :      ...
    INFO :      [ROUND 3/3]
    INFO :      ...
    INFO :      Strategy execution finished in 72.84s
    INFO :      Final results:
    INFO :          ServerApp-side Evaluate Metrics:
    INFO :          {}
    Saving final model to disk...

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config num-server-rounds=5

.. tip::

    Check the :doc:`how-to-run-simulations` documentation to learn more about how to
    configure and run Flower simulations.

.. note::

    Check the `source code
    <https://github.com/flwrlabs/flower/tree/main/examples/quickstart-fastai>`_ of this
    tutorial in ``examples/quickstart-fastai`` in the Flower GitHub repository.
