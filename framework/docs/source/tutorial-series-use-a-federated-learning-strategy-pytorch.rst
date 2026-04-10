###################################
 Use a federated learning strategy
###################################

.. |fedavg_link| replace:: ``FedAvg``

.. _fedavg_link: ref-api/flwr.serverapp.strategy.FedAvg.html

.. |fedadagrad_link| replace:: ``FedAdagrad``

.. _fedadagrad_link: ref-api/flwr.serverapp.strategy.FedAdagrad.html

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.serverapp.ServerApp.html

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.common.Message.html

.. |metricrecord_link| replace:: ``MetricRecord``

.. _metricrecord_link: ref-api/flwr.common.MetricRecord.html

.. |configrecord_link| replace:: ``ConfigRecord``

.. _configrecord_link: ref-api/flwr.common.ConfigRecord.html

.. |strategy_start_link| replace:: ``start``

.. _strategy_start_link: ref-api/flwr.serverapp.strategy.Strategy.html#flwr.serverapp.strategy.Strategy.start

Welcome to the next part of the federated learning tutorial. In previous parts of this
tutorial, we introduced federated learning with PyTorch and Flower (:doc:`part 1
<tutorial-series-get-started-with-flower-pytorch>`).

In part 2, we'll begin to customize the federated learning system we built in part 1
using the Flower framework, Flower Datasets, and PyTorch.

.. tip::

    `Star Flower on GitHub <https://github.com/flwrlabs/flower>`__ ⭐️ and join the
    Flower community on Flower Discuss and the Flower Slack to connect, ask questions,
    and get help:

    - `Join Flower Discuss <https://discuss.flower.ai/>`__ We'd love to hear from you in
      the ``Introduction`` topic! If anything is unclear, post in ``Flower Help -
      Beginners``.
    - `Join Flower Slack <https://flower.ai/join-slack>`__ We'd love to hear from you in
      the ``#introductions`` channel! If anything is unclear, head over to the
      ``#questions`` channel.

Let's move beyond FedAvg with Flower strategies! 🌼

*************
 Preparation
*************

Before we begin with the actual code, let's make sure that we have everything we need.

Installing dependencies
=======================

.. note::

    If you've completed part 1 of the tutorial, you can skip this step.

First, we install the Flower package ``flwr``:

.. code-block:: shell

    # In a new Python environment
    $ pip install -U "flwr[simulation]"

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

Next, we install the project and its dependencies, which are specified in the
``pyproject.toml`` file:

.. code-block:: shell

    $ cd quickstart-pytorch
    $ pip install -e .

So far, everything should look familiar if you've worked through the introductory
tutorial. With that, we're ready to introduce a number of new features.

*******************************
 Choosing a different strategy
*******************************

In part 1, we created a |serverapp_link|_ (in ``server_app.py``). In it, we defined the
strategy, the model to federatedly train, and then we launched the strategy by calling
its ``|strategy_start_link|`` method.

The strategy encapsulates the federated learning approach/algorithm, for example,
|fedavg_link|_. Let's try to use a different strategy this time. Modify the following
lines in your ``server_app.py`` to switch from ``FedAvg`` to |fedadagrad_link|_.

.. code-block:: python
    :emphasize-lines: 1,18

    from flwr.serverapp.strategy import FedAdagrad


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

        # Initialize FedAdagrad strategy
        strategy = FedAdagrad(fraction_evaluate=fraction_evaluate)

        # Start strategy, run FedAdagrad for `num_rounds`
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

Next, run the training with the following command:

.. code-block:: shell

    $ flwr run . --stream

**************************************
 Server-side parameter **evaluation**
**************************************

Flower can evaluate the aggregated model on the server side or on the client side.
Client-side and server-side evaluation are similar in some ways, but different in
others.

**Centralized Evaluation** (or *server-side evaluation*) is conceptually simple: it
works the same way that evaluation in centralized machine learning does. If there is a
server-side dataset that can be used for evaluation purposes, then that's great. We can
evaluate the newly aggregated model after each round of training without having to send
the model to clients. We're also fortunate in the sense that our entire evaluation
dataset is available at all times.

**Federated Evaluation** (or *client-side evaluation*) is more complex, but also more
powerful: it doesn't require a centralized dataset and allows us to evaluate models over
a larger set of data, which often yields more realistic evaluation results. In fact,
many scenarios require us to use **Federated Evaluation** if we want to get
representative evaluation results at all. But this power comes at a cost: once we start
to evaluate on the client side, we should be aware that our evaluation dataset can
change over consecutive rounds of learning if those clients are not always available.
Moreover, the dataset held by each client can also change over consecutive rounds. This
can lead to evaluation results that are not stable, so even if we would not change the
model, we'd see our evaluation results fluctuate over consecutive rounds.

We've seen how federated evaluation works on the client side (i.e., by implementing a
function wrapped with the ``@app.evaluate`` decorator in your ``ClientApp``). Now let's
see how we can evaluate the aggregated model parameters on the server side.

To do so, we need to create a new function in ``task.py`` that we can name
``global_evaluate``. This function is a callback that will be passed to the
|strategy_start_link|_ method of our strategy. This means that the strategy will call
this function after every round of federated learning passing two arguments: the current
round of federated learning and the aggregated model parameters.

Our ``global_evaluate`` function performs the following steps:

1. Load the aggregated model parameters into a PyTorch model
2. Load the entire CIFAR10 test dataset
3. Evaluate the model on the test dataset
4. Return the evaluation metrics as a |metricrecord_link|_

.. code-block:: python

    from flwr.app import ArrayRecord, MetricRecord


    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""

        # Load the model and initialize it with the received weights
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load entire test set
        test_dataloader = load_centralized_dataset()

        # Evaluate the global model on the test set
        test_loss, test_acc = test(model, test_dataloader, device)

        # Return the evaluation metrics
        return MetricRecord({"accuracy": test_acc, "loss": test_loss})

Remember we mentioned this ``global_evaluate`` will be called by the strategy. To do so
we need to pass it to the strategy's ``start`` method as shown below.

.. code-block:: python
    :emphasize-lines: 1,16

    from pytorchexample.task import global_evaluate


    @app.main()
    def main(grid: Grid, context: Context) -> None:
        """Main entry point for the ServerApp."""

        # ... unchanged

        # Start strategy, run FedAdagrad for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr}),
            num_rounds=num_rounds,
            evaluate_fn=global_evaluate,
        )

        # .. unchanged

Finally, we run the simulation.

.. code-block:: shell

    $ flwr run . --stream

You'll note that the server logs the metrics returned by the callback after each round.
Also, at the end of the run, note the ``ServerApp-side Evaluate Metrics`` shown:

.. code-block:: shell

    INFO :          ServerApp-side Evaluate Metrics:
    INFO :          { 0: {'accuracy': '1.0000e-01', 'loss': '2.3053e+00'},
    INFO :            1: {'accuracy': '1.0000e-01', 'loss': '2.3203e+00'},
    INFO :            2: {'accuracy': '2.3230e-01', 'loss': '2.0144e+00'},
    INFO :            3: {'accuracy': '2.5720e-01', 'loss': '1.9258e+00'}}

***************************************************
 Sending configurations to clients from strategies
***************************************************

In some situations, we want to configure client-side execution (training, evaluation)
from the server side. One example of this is the server asking the clients to train for
with a different learning rate based on the current round number. Flower provides a way
to send configuration values from the server to the clients as part of the
|message_link|_ that the ``ClientApp`` receives. Let's see how we can do this.

To the |strategy_start_link|_ method of our strategy we are already passing a
|configrecord_link|_ specifying the initial learning rate. This ``ConfigRecord`` will be
sent to the clients in all the ``Messages`` addressing the ``@app.train()`` function of
the ``ClientApp``. Let's say we want to decrease the learning rate by a factor of 0.5
every 5 rounds, then we need to override the ``configure_train`` method of our strategy
and embed such logic.

To do so, we create a new class inheriting from |fedadagrad_link|_ and override the
``configure_train`` method. We then use this new strategy in our ``ServerApp``. Let's
see how this looks like in code. Create a new file called ``custom_strategy.py`` in the
``pytorchexample`` directory and add the following code:

.. code-block:: python
    :emphasize-lines: 13,14

    from typing import Iterable
    from flwr.serverapp import Grid
    from flwr.serverapp.strategy import FedAdagrad
    from flwr.app import ArrayRecord, ConfigRecord, Message


    class CustomFedAdagrad(FedAdagrad):
        def configure_train(
            self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
        ) -> Iterable[Message]:
            """Configure the next round of federated training and maybe do LR decay."""
            # Decrease learning rate by a factor of 0.5 every 5 rounds
            if server_round % 5 == 0 and server_round > 0:
                config["lr"] *= 0.5
                print("LR decreased to:", config["lr"])
            # Pass the updated config and the rest of arguments to the parent class
            return super().configure_train(server_round, arrays, config, grid)

Next, we use this new strategy in our ``ServerApp`` by importing it in your
``server_app.py`` and use it instead of the standard ``FedAdagrad``.

Finally, run the training with the following command. Here we increase the number of
rounds to 15 to see the learning rate decay in action.

.. code-block:: shell

    $ flwr run . --stream --run-config="num-server-rounds=15"

You'll note that in the ``configure_train`` stage of rounds 5 and 10, the learning rate
is decreased by a factor of 0.5 and the new learning rate is printed to the terminal.

How do we know the ``ClientApp`` is using that new learning rate? Recall that in
``client_app.py``, we are reading the learning rate from the ``Message`` received by the
``@app.train()`` function:

.. code-block:: python
    :emphasize-lines: 11

    @app.train()
    def train(msg: Message, context: Context):

        # ... setup

        # Call the training function
        train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )

        # ... prepare reply Message
        return Message(content=content, reply_to=msg)

Congratulations! You have created your first custom strategy adding dynamism to the
``ConfigRecord`` that is sent to clients.

****************************
 Scaling federated learning
****************************

As a last step in this tutorial, let's see how we can use Flower to experiment with a
large number of clients. The most straightforward way to do this is by overriding the
default Simulation Runtime configuration via the ``--federation-config`` flag:

.. code-block:: shell

    # Run with 200 clients
    $ flwr run . --stream --federation-config="num-supernodes=200"

For more details on the Simulation Runtime and its configuration, check out the
:doc:`Simulation Runtime documentation <how-to-run-simulations>`.

Note that we can reuse the ``ClientApp`` for different ``num-supernodes`` since the
``Context`` carries the ``num-partitions`` key and for simulations with Flower, the
number of partitions is equal to the number of SuperNodes.

We now have 200 partitions, each holding 45 training and 5 validation examples. Given
that the number of training examples on each client is quite small, we should probably
train the model a bit longer, so we configure the clients to perform 3 local training
epochs. We should also adjust the fraction of clients selected for training during each
round (we don't want all 200 clients participating in every round), so we add
``fraction-train = 0.025`` and adjust ``fraction-evaluate`` to ``0.05``, which means
that only 2.5% of available clients will be selected for training each round (so 5
clients) and 5% of them for evaluation (so 10 clients). We can add and adjust values in
the ``pyproject.toml`` for ease of experimentation:

.. code-block:: toml

    [tool.flwr.app.config]
    num-server-rounds = 3
    fraction-train = 0.025  # <-- new
    fraction-evaluate = 0.05 # <-- updated
    local-epochs = 1
    learning-rate = 0.1
    batch-size = 32

Then, we update the initialization of our strategy in ``server_app.py`` to the
following:

.. code-block:: python

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        """Main entry point for the ServerApp."""

        # ... unchanged
        fraction_train: float = context.run_config["fraction-train"]
        # Initialize FedAdagrad strategy
        strategy = CustomFedAdagrad(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=20,  # Optional config
            min_evaluate_nodes=40,  # Optional config
            min_available_nodes=1000,  # Optional config
        )

        # ... rest unchanged

Finally, run the simulation with the following command:

.. code-block:: shell

    $ flwr run . --stream

*******
 Recap
*******

In this tutorial, we've seen how we can gradually enhance our system by customizing the
strategy, choosing a different strategy, applying learning rate decay at the strategy
level, and evaluating models on the server side. That's quite a bit of flexibility with
so little code, right?

In the later sections, we've seen how we can communicate arbitrary values between server
and clients to fully customize client-side execution. With that capability, we built a
large-scale Federated Learning simulation using the Flower Simulation Runtime and ran an
experiment involving 1000 clients in the same workload — all in the same Flower project!

************
 Next steps
************

Before you continue, make sure to join the Flower community on Flower Discuss (`Join
Flower Discuss <https://discuss.flower.ai>`__) and on Slack (`Join Slack
<https://flower.ai/join-slack/>`__).

There's a dedicated ``#questions`` Slack channel if you need help, but we'd also love to
hear who you are in ``#introductions``!

The :doc:`Flower Federated Learning Tutorial - Part 3
<tutorial-series-build-a-strategy-from-scratch-pytorch>` shows how to build a fully
custom ``Strategy`` from scratch.
