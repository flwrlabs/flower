##########################################
 Write your first Flower App with PyTorch
##########################################

Welcome to the next part of the Flower collaborative AI tutorial!

In the previous tutorials, you created a simulated federation on SuperGrid, ran a Flower
App, downloaded the ``@flwrlabs/demo`` app, and learned how ``ServerApp``,
``ClientApp``, strategies, and ``pyproject.toml`` fit together. In this tutorial, you
will use the same workflow with a more realistic Flower App: a PyTorch app that trains a
small image classifier on CIFAR-10.

Let's get started! 🌼

****************
 Create the App
****************

Use ``flwr new`` to fetch the PyTorch quickstart app from Flower Hub:

.. code-block:: shell

    $ flwr new @flwrlabs/quickstart-pytorch

After running the command, a new directory named ``quickstart-pytorch`` will be created:

.. code-block:: shell

    quickstart-pytorch
    ├── pytorchexample
    │   ├── __init__.py
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

This app has the same Flower structure as the NumPy demo from the previous tutorial, but
the workload is now a real PyTorch training task. The app trains a small convolutional
neural network on CIFAR-10, an image classification dataset with ten classes such as
airplane, automobile, bird, cat, dog, ship, and truck.

********************
 Quick App Overview
********************

Before running the app, it helps to know what each file is responsible for:

- ``pytorchexample/task.py`` contains the PyTorch-specific code: the neural network,
  CIFAR-10 data loading and partitioning, the local training loop, the evaluation loop,
  and server-side evaluation helpers.
- ``pytorchexample/client_app.py`` defines the ``ClientApp``. Its ``@app.train()``
  handler receives the current global model, loads one CIFAR-10 partition, trains the
  model locally, and replies with updated model parameters plus metrics. Its
  ``@app.evaluate()`` handler evaluates the received model on local validation data and
  replies with metrics.
- ``pytorchexample/server_app.py`` defines the ``ServerApp``. It creates the initial
  PyTorch model, wraps the model parameters in an ``ArrayRecord``, creates a ``FedAvg``
  strategy, and starts the federated learning run.
- ``pyproject.toml`` declares the app metadata and dependencies, points Flower to the
  ``ServerApp`` and ``ClientApp`` objects, and defines run configuration values such as
  the number of server rounds, batch size, local epochs, learning rate, and evaluation
  settings.

The important idea is the same as before: the ``ServerApp`` starts the run, ``FedAvg``
coordinates each federated learning round, and each ``ClientApp`` trains or evaluates
the model using the data available on its node.

This app uses `Flower Datasets <https://flower.ai/docs/datasets/>`__ to download
CIFAR-10 and split it into partitions, one for each simulated client. This is ideal for
simulations because it lets you experiment with federated learning even when you start
from a single centralized dataset. In a typical Flower App that runs outside of
simulation, you usually do not create artificial partitions. Instead, each ``ClientApp``
loads the data already available on the client node where it runs.

**************************
 Run the App on SuperGrid
**************************

.. note::

    If you have not already done so, complete the :doc:`first tutorial
    <tutorial-series-get-started-with-flower>` to create a SuperGrid account and a
    simulated federation.

Open a terminal, activate your Python environment, and run the following command to
first login to SuperGrid:

.. code-block:: shell

    # This will open a browser window where you can enter your SuperGrid credentials.
    $ flwr login

Once you are logged in, run the following command to run the app on SuperGrid and
accross the federation you created in the previous tutorial:

.. code-block:: shell

    # Navigate to the directory of the app you want to run
    $ cd /path/to/quickstart-pytorch
    # Run the app across the federation you created in the previous tutorial
    $ flwr run . --federation @<username>/<federation-name>
    # for example
    # flwr run . --federation @peter123/my-first-federation`

SuperGrid will start a new run for this app. Open the `SuperGrid dashboard
<https://flower.ai/federations/>`__, select your federation, and click the new run to
follow its progress and inspect the logs.

In the logs, you should see Flower start the ``FedAvg`` strategy and run several rounds
of federated learning. Each round includes local training on selected ``ClientApp``
instances, aggregation in the ``ServerApp``, and evaluation metrics such as
``eval_loss`` and ``eval_acc``.

You can override values from ``pyproject.toml`` at run time. For example:

.. code-block:: shell

    # Run the app for five rounds intead of the default three rounds
    $ flwr run . --federation @<username>/<federation-name> \
        --run-config "num-server-rounds=5"

    # Run the app for five rounds and a smaller batch size
    $ flwr run . --federation @<username>/<federation-name>
        --run-config "num-server-rounds=5" \
        --run-config "batch-size=16"

*********************
 Run the App Locally
*********************

Running on SuperGrid is the recommended way to run collaborative AI workflows with
Flower. However, it is also useful to run the same app locally while you are developing
or debugging.

From the ``quickstart-pytorch`` directory, install the app and its dependencies into
your Python environment:

.. code-block:: shell

    $ cd /path/to/quickstart-pytorch
    $ pip install -e .

Then run the app locally with the command below. Flower will start a managed local
SuperLink -- a distilled version of SuperGrid -- and execute the app with simulated
SuperNodes on your machine. The first run can take longer because the app needs to
download CIFAR-10. With the flag ``--stream``, you can see the logs from the local run
in your terminal.

.. code-block:: shell

    $ flwr run . local --stream

The streamed output should include logs similar to this:

.. code-block:: shell

    INFO :      Starting FedAvg strategy:
    INFO :          ├── Number of rounds: 3
    INFO :      [ROUND 1/3]
    INFO :      configure_train: Sampled 5 nodes (out of 10)
    INFO :      aggregate_train: Received 5 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'train_loss': 2.149280}
    INFO :      configure_evaluate: Sampled 10 nodes (out of 10)
    INFO :      aggregate_evaluate: Received 10 results and 0 failures
    INFO :          └──> Aggregated MetricRecord: {'eval_loss': 2.31319, 'eval_acc': 0.13004}
    INFO :      [ROUND 2/3]
    INFO :      ...
    INFO :      [ROUND 3/3]
    INFO :      ...
    INFO :      Strategy execution finished

.. note::

    In the above ``flwr run`` command you are not specifying a federation, this is
    becuase for local prototyping there is only one federation available. Because of
    this, the ``--federation`` flag is not required.

.. note::

    If you're on Windows and see unexpected terminal output, for example ``�
    □[32m□[1m``, check :ref:`this FAQ entry <faq-windows-unexpected-output>`.

For more details on using the Flower CLI against a locally running SuperLink, including
how to list your runs and view their logs, see :doc:`Run Flower Locally with a Managed
SuperLink <how-to-run-flower-locally>`.

***************
 Final remarks
***************

You have now run a PyTorch Flower App on SuperGrid and locally. Compared with the NumPy
demo, this app uses a real model, a real dataset, and real local training, but the
Flower structure is the same: ``ServerApp``, ``ClientApp``, strategy, and
``pyproject.toml``.

In the next tutorial, you will customize the federated learning strategy to change how
the server coordinates training and evaluation.

************
 Next steps
************

Before you continue, make sure to join the Flower community on Flower Discuss (`Join
Flower Discuss <https://discuss.flower.ai>`__) and on Slack (`Join Slack
<https://flower.ai/join-slack/>`__).

There's a dedicated ``#questions`` Slack channel if you need help, but we'd also love to
hear who you are in ``#introductions``!

The :doc:`Flower Collaborative AI Tutorial - Part 4: Use a federated learning strategy
<tutorial-series-use-a-federated-learning-strategy-pytorch>` goes into more depth about
strategies and the advanced behavior you can build with them.
