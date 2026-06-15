:og:description: Learn how to train an autoencoder on MNIST using federated learning with Flower and PyTorch Lightning in this step-by-step tutorial.
.. meta::
    :description: Learn how to train an autoencoder on MNIST using federated learning with Flower and PyTorch Lightning in this step-by-step tutorial.

.. _quickstart-pytorch-lightning:

##############################
 Quickstart PyTorch Lightning
##############################

In this federated learning tutorial we will learn how to train an AutoEncoder model on
MNIST using Flower and PyTorch Lightning. It is recommended to create a virtual
environment and run everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`.

Then, clone the code example directly from GitHub:

.. code-block:: shell

    git clone --depth=1 https://github.com/flwrlabs/flower.git _tmp \
                 && mv _tmp/examples/quickstart-pytorch-lightning . \
                 && rm -rf _tmp && cd quickstart-pytorch-lightning

This will create a new directory called `quickstart-pytorch-lightning` containing the
following files:

.. code-block:: shell

    quickstart-pytorch-lightning
    ├── pytorchlightning_example
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

Next, activate your environment, then run:

.. code-block:: shell

    # Navigate to the example directory
    $ cd path/to/quickstart-pytorch-lightning

    # Install project and dependencies
    $ pip install -e .

By default, this project uses a local simulation profile that ``flwr run`` submits to a
managed local SuperLink, which then executes the run with the Flower Simulation Runtime.
It creates a federation of 4 nodes using |fedavg|_ as the aggregation strategy. The
dataset will be partitioned using Flower Dataset's |iidpartitioner|_. To run the
project, do:

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
    INFO :      configure_train: Sampled 2 nodes (out of 4)
    INFO :      aggregate_train: Received 2 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 0.0487}
    INFO :      configure_evaluate: Sampled 2 nodes (out of 4)
    INFO :      aggregate_evaluate: Received 2 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 0.0495}
    INFO :      [ROUND 2/3]
    INFO :      ...
    INFO :      [ROUND 3/3]
    INFO :      ...
    INFO :      Strategy execution finished in 159.24s
    INFO :      Final results:
    INFO :          ServerApp-side Evaluate Metrics:
    INFO :          {}

Each simulated `ClientApp` (two per round) will also log a summary of their local
training process. Expect this output to be similar to:

.. code-block:: shell

    # The left part indicates the process ID running the `ClientApp`
    (ClientAppActor pid=38155) ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    (ClientAppActor pid=38155) ┃        Test metric        ┃       DataLoader 0        ┃
    (ClientAppActor pid=38155) ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    (ClientAppActor pid=38155) │         test_loss         │   0.045175597071647644    │
    (ClientAppActor pid=38155) └───────────────────────────┴───────────────────────────┘

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
    <https://github.com/flwrlabs/flower/tree/main/examples/quickstart-pytorch-lightning>`_
    of this tutorial in ``examples/quickstart-pytorch-lightning`` in the Flower GitHub
    repository.

.. |fedavg| replace:: ``FedAvg``

.. _fedavg: ref-api/flwr.serverapp.strategy.FedAvg.html

.. |iidpartitioner| replace:: ``IidPartitioner``

.. _iidpartitioner: https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner
