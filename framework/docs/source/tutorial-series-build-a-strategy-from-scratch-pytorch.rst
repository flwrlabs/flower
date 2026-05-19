#############################
 Customize a Flower Strategy
#############################

.. |configrecord_link| replace:: ``ConfigRecord``

.. _configrecord_link: ref-api/flwr.app.ConfigRecord.html

.. |strategy_start_link| replace:: ``start``

.. _strategy_start_link: ref-api/flwr.serverapp.strategy.Strategy.html#flwr.serverapp.strategy.Strategy.start

.. |fedadagrad_link| replace:: ``FedAdagrad``

.. _fedadagrad_link: ref-api/flwr.serverapp.strategy.FedAdagrad.html

Welcome to the third part of the Flower federated learning tutorial. In previous parts
of this tutorial, we introduced federated learning with PyTorch and the Flower framework
(:doc:`part 1 <tutorial-series-get-started-with-flower-pytorch>`) and we learned how
strategies can be used to customize the execution on both the server and the clients
(:doc:`part 2 <tutorial-series-use-a-federated-learning-strategy-pytorch>`).

In this tutorial, we'll continue to customize the federated learning system we built
previously by creating a much more customized version of ``FedAdagrad``.

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

Let's build a new ``Strategy`` with a customized |strategy_start_link|_ method that:

- saves a copy of the global model when a new best global accuracy is found;
- logs the metrics generated during the run to Weights & Biases!

*************
 Preparation
*************

Before we begin with the actual code, let's make sure that we have everything we need.

Installing dependencies
=======================

.. note::

    If you've completed part 1 and 2 of the tutorial, you can skip this step. But
    remember to include ``wandb`` as a dependency in your ``pyproject.toml`` file and
    install it in your environment.

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

Next, add the `wandb` dependency to the project by editing the ``pyproject.toml`` file
located in the root of the project. Add the following line to the list of dependencies:

.. code-block:: shell

    "wandb>=0.17.8"

Next, we install the project and its dependencies, which are specified in the
``pyproject.toml`` file:

.. code-block:: shell

    $ cd quickstart-pytorch
    $ pip install -e .

.. note::

    If this is your first time installing ``wandb``, you might be asked to create an
    account and then log in to your system. You can start this process by typing this in
    your terminal:

    .. code-block:: shell

        $ wandb login

**********************************************
 Customize the ``start`` method of a strategy
**********************************************

Flower strategies have a number of methods that can be overridden to customize their
behavior. In part 2, you learned how to customize the ``configure_train`` method to
perform learning rate decay and communicate the updated learning rate as part of the
|configrecord_link|_ sent to the clients in the ``Message``. In this tutorial you'll
learn how to customize the |strategy_start_link|_ method. If you inspect the `source
code
<https://github.com/flwrlabs/flower/blob/main/framework/py/flwr/serverapp/strategy/strategy.py#L135>`_
of this method you'll see that it contains a for loop where each iteration represents a
federated learning round. Each round consists of three distinct stages:

1. A training stage, where a subset of clients is selected to train the current global
   model on their local data.
2. An evaluation stage, where a subset of clients is selected to evaluate the updated
   global model on their local validation sets.
3. An optional stage to evaluate the global model on the server side. Note that this is
   what you enabled in part 2 of this tutorial by means of the ``central_evaluate``
   callback.

Let's extend the ``CustomFedAdagrad`` strategy we created earlier and introduce:

1. ``_update_best_acc``: An auxiliary method to save the global model whenever a new
   best accuracy is found.
2. ``set_save_path``: An auxiliary method to set the path where ``wandb`` logs and model
   checkpoints will be saved. This method will be called from the ``server_app.py``
   after instantiating the strategy.
3. A customized |strategy_start_link|_ method to log metrics to Weight & Biases (`W&B
   <https://wandb.ai/site>`__) and save the model checkpoints to disk.

.. code-block:: python
    :emphasize-lines: 31,35,65,68,126,155,168,170

    import io
    import time
    from logging import INFO
    from pathlib import Path
    from typing import Callable, Iterable, Optional

    import torch
    import wandb
    from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
    from flwr.common import log, logger
    from flwr.serverapp import Grid
    from flwr.serverapp.strategy import FedAdagrad, Result
    from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

    PROJECT_NAME = "FLOWER-advanced-pytorch"


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

        def set_save_path(self, path: Path):
            """Set the path where wandb logs and model checkpoints will be saved."""
            self.save_path = path

        def _update_best_acc(
            self, current_round: int, accuracy: float, arrays: ArrayRecord
        ) -> None:
            """Update best accuracy and save model checkpoint if current accuracy is
            higher."""
            if accuracy > self.best_acc_so_far:
                self.best_acc_so_far = accuracy
                logger.log(INFO, "💡 New best global model found: %f", accuracy)
                # Save the PyTorch model
                file_name = f"model_state_acc_{accuracy}_round_{current_round}.pth"
                torch.save(arrays.to_torch_state_dict(), self.save_path / file_name)
                logger.log(INFO, "💾 New best model saved to disk: %s", file_name)

        def start(
            self,
            grid: Grid,
            initial_arrays: ArrayRecord,
            num_rounds: int = 3,
            timeout: float = 3600,
            train_config: Optional[ConfigRecord] = None,
            evaluate_config: Optional[ConfigRecord] = None,
            evaluate_fn: Optional[
                Callable[[int, ArrayRecord], Optional[MetricRecord]]
            ] = None,
        ) -> Result:
            """Execute the federated learning strategy logging results to W&B and saving
            them to disk."""

            # Init W&B
            name = f"{str(self.save_path.parent.name)}/{str(self.save_path.name)}-ServerApp"
            wandb.init(project=PROJECT_NAME, name=name)

            # Keep track of best acc
            self.best_acc_so_far = 0.0

            log(INFO, "Starting %s strategy:", self.__class__.__name__)
            log_strategy_start_info(
                num_rounds, initial_arrays, train_config, evaluate_config
            )
            self.summary()
            log(INFO, "")

            # Initialize if None
            train_config = ConfigRecord() if train_config is None else train_config
            evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
            result = Result()

            t_start = time.time()
            # Evaluate starting global parameters
            if evaluate_fn:
                res = evaluate_fn(0, initial_arrays)
                log(INFO, "Initial global evaluation results: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[0] = res

            arrays = initial_arrays

            for current_round in range(1, num_rounds + 1):
                log(INFO, "")
                log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

                # -----------------------------------------------------------------
                # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
                # -----------------------------------------------------------------

                # Call strategy to configure training round
                # Send messages and wait for replies
                train_replies = grid.send_and_receive(
                    messages=self.configure_train(
                        current_round,
                        arrays,
                        train_config,
                        grid,
                    ),
                    timeout=timeout,
                )

                # Aggregate train
                agg_arrays, agg_train_metrics = self.aggregate_train(
                    current_round,
                    train_replies,
                )

                # Log training metrics and append to history
                if agg_arrays is not None:
                    result.arrays = agg_arrays
                    arrays = agg_arrays
                if agg_train_metrics is not None:
                    log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                    result.train_metrics_clientapp[current_round] = agg_train_metrics
                    # Log to W&B
                    wandb.log(dict(agg_train_metrics), step=current_round)

                # -----------------------------------------------------------------
                # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
                # -----------------------------------------------------------------

                # Call strategy to configure evaluation round
                # Send messages and wait for replies
                evaluate_replies = grid.send_and_receive(
                    messages=self.configure_evaluate(
                        current_round,
                        arrays,
                        evaluate_config,
                        grid,
                    ),
                    timeout=timeout,
                )

                # Aggregate evaluate
                agg_evaluate_metrics = self.aggregate_evaluate(
                    current_round,
                    evaluate_replies,
                )

                # Log training metrics and append to history
                if agg_evaluate_metrics is not None:
                    log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                    result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
                    # Log to W&B
                    wandb.log(dict(agg_evaluate_metrics), step=current_round)
                # -----------------------------------------------------------------
                # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
                # -----------------------------------------------------------------

                # Centralized evaluation
                if evaluate_fn:
                    log(INFO, "Global evaluation")
                    res = evaluate_fn(current_round, arrays)
                    log(INFO, "\t└──> MetricRecord: %s", res)
                    if res is not None:
                        result.evaluate_metrics_serverapp[current_round] = res
                        # Maybe save to disk if new best is found
                        self._update_best_acc(current_round, res["accuracy"], arrays)
                        # Log to W&B
                        wandb.log(dict(res), step=current_round)

            log(INFO, "")
            log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
            log(INFO, "")
            log(INFO, "Final results:")
            log(INFO, "")
            for line in io.StringIO(str(result)):
                log(INFO, "\t%s", line.strip("\n"))
            log(INFO, "")

            return result

With the extended ``CustomFedAdagrad`` strategy defined, we now need to set the path
where the model checkpoints will be saved as well as the name of the runs in ``W&B``. We
need to call the ``set_save_path`` method after instantiating the strategy and before
calling the ``start`` method. In ``server_app.py``, we can create a new directory called
``results`` and then a subdirectory with the current timestamp to store the results of
each run. We can then call the ``set_save_path``. In this tutorial we create the
directory based on the current date and time, this means that each time you do ``flwr
run`` a new directory will be used. Let's see how this looks in code:

.. code-block:: python
    :emphasize-lines: 22

    from datetime import datetime
    from pathlib import Path


    @app.main()
    def main(grid: Grid, context: Context) -> None:
        """Main entry point for the ServerApp."""

        # ... unchanged

        # Initialize FedAdagrad strategy
        # strategy = CustomFedAdagrad( ... )

        # Get the current date and time
        current_time = datetime.now()
        run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
        # Save path is based on the current directory
        save_path = Path.cwd() / f"outputs/{run_dir}"
        save_path.mkdir(parents=True, exist_ok=False)

        # Set the path where results and model checkpoints will be saved
        strategy.set_save_path(save_path)

        # ... rest unchanged

Finally, let's run the ``FlowerApp``:

.. code-block:: shell

    $ flwr run . --stream

Plain ``flwr run .`` submits the run, prints the run ID, and returns without streaming
logs. See :doc:`how-to-run-flower-locally` for the full local workflow.

After starting the run you will notice two things:

1. A new directory will be created in ``outputs/YYYY-MM-DD/HH-MM-SS`` where
   ``YYYY-MM-DD/HH-MM-SS`` is the current date and time. This directory will contain the
   model checkpoints saved during the run. Recall that a checkpoint is saved whenever a
   new best accuracy is found during the centralized evaluation stage.
2. A new run will be created in your `W&B project <https://wandb.ai/home>`_ where you
   can visualize the metrics logged during the run.

Congratulations! You've successfully created a custom Flower strategy by overriding the
|strategy_start_link|_ method. You've also learned how to log metrics to Weight & Biases
and how to save model checkpoints to disk.

*******
 Recap
*******

In this tutorial, we've seen how to customize the |strategy_start_link|_ method of a
Flower strategy. This method is the main entry point of any strategy and contains the
logic to execute the federated learning process. In this tutorial, you learned how to
log the metrics to Weight & Biases and how to save model checkpoints to disk.

In the next tutorial, we're going to cover how to communicate arbitrary Python objects
between the ``ClientApp`` and the ``ServerApp`` by serializing them and send them in a
``Message`` as a ``ConfigRecord``.

************
 Next steps
************

Before you continue, make sure to join the Flower community on Flower Discuss (`Join
Flower Discuss <https://discuss.flower.ai>`__) and on Slack (`Join Slack
<https://flower.ai/join-slack/>`__).

There's a dedicated ``#questions`` Slack channel if you need help, but we'd also love to
hear who you are in ``#introductions``!
