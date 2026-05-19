:og:description: Learn the two main approaches to evaluating models in Flower: centralized (server-side) evaluation and federated (client-side) evaluation, with built-in strategies and configuration options.
.. meta::
    :description: Learn the two main approaches to evaluating models in Flower: centralized (server-side) evaluation and federated (client-side) evaluation, with built-in strategies and configuration options.

######################
 Federated evaluation
######################

There are two main approaches to evaluating models in federated learning systems:
centralized (or server-side) evaluation and federated (or client-side) evaluation.

************************
 Centralized Evaluation
************************

Built-In Strategies
===================

All built-in strategies support centralized evaluation by providing an evaluation
function during initialization. An evaluation function is any function that can take the
current global model parameters as input and return evaluation results:

.. code-block:: python

    from flwr.app import Context
    from flwr.common import NDArrays, Scalar
    from flwr.server import ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.serverapp import ServerApp

    from typing import Dict, Optional, Tuple


    def get_evaluate_fn(model):
        """Return an evaluation function for server-side evaluation."""

        # Load data and model here to avoid the overhead of doing it in `evaluate` itself
        (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

        # Use the last 5k training examples as a validation set
        x_val, y_val = x_train[45000:50000], y_train[45000:50000]

        # The `evaluate` function will be called after every round
        def evaluate(
            server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            model.set_weights(parameters)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(x_val, y_val)
            return loss, {"accuracy": accuracy}

        return evaluate


    def server_fn(context: Context):
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]
        config = ServerConfig(num_rounds=num_rounds)

        # Load and compile model for server-side parameter evaluation
        model = tf.keras.applications.EfficientNetB0(
            input_shape=(32, 32, 3), weights=None, classes=10
        )
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

        # Create strategy
        strategy = FedAvg(
            # ... other FedAvg arguments
            evaluate_fn=get_evaluate_fn(model),
        )

        return ServerAppComponents(strategy=strategy, config=config)


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)

Custom Strategies
=================

The ``Strategy`` abstraction provides a method called ``evaluate`` that can directly be
used to evaluate the current global model parameters. The current server implementation
calls ``evaluate`` after parameter aggregation and before federated evaluation (see next
paragraph).

**********************
 Federated Evaluation
**********************

Implementing Federated Evaluation
=================================

Client-side evaluation happens in the ``Client.evaluate`` method and can be configured
from the server side.

.. code-block:: python

    from flwr.client import NumPyClient


    class FlowerClient(NumPyClient):
        def __init__(self, model, x_train, y_train, x_test, y_test):
            self.model = model
            self.x_train, self.y_train = x_train, y_train
            self.x_test, self.y_test = x_test, y_test

        def fit(self, parameters, config):
            # ...
            pass

        def evaluate(self, parameters, config):
            """Evaluate parameters on the locally held test set."""

            # Update local model with global parameters
            self.model.set_weights(parameters)

            # Get config values
            steps: int = config["val_steps"]

            # Evaluate global model parameters on the local test data and return results
            loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
            num_examples_test = len(self.x_test)
            return loss, num_examples_test, {"accuracy": accuracy}

Configuring Federated Evaluation
================================

Federated evaluation can be configured from the server side. Built-in strategies support
the following arguments:

- ``fraction_evaluate``: a ``float`` defining the fraction of clients that will be
  selected for evaluation. If ``fraction_evaluate`` is set to ``0.1`` and ``100``
  clients are connected to the server, then ``10`` will be randomly selected for
  evaluation. If ``fraction_evaluate`` is set to ``0.0``, federated evaluation will be
  disabled.
- ``min_evaluate_clients``: an ``int``: the minimum number of clients to be selected for
  evaluation. If ``fraction_evaluate`` is set to ``0.1``, ``min_evaluate_clients`` is
  set to 20, and ``100`` clients are connected to the server, then ``20`` clients will
  be selected for evaluation.
- ``min_available_clients``: an ``int`` that defines the minimum number of clients which
  need to be connected to the server before a round of federated evaluation can start.
  If fewer than ``min_available_clients`` are connected to the server, the server will
  wait until more clients are connected before it continues to sample clients for
  evaluation.
- ``on_evaluate_config_fn``: a function that returns a configuration dictionary which
  will be sent to the selected clients. The function will be called during each round
  and provides a convenient way to customize client-side evaluation from the server
  side, for example, to configure the number of validation steps performed.

.. code-block:: python

    from flwr.app import Context
    from flwr.server import ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.serverapp import ServerApp


    def evaluate_config(server_round: int):
        """Return evaluation configuration dict for each round.
        Perform five local evaluation steps on each client (i.e., use five
        batches) during rounds, one to three, then increase to ten local
        evaluation steps.
        """
        val_steps = 5 if server_round < 4 else 10
        return {"val_steps": val_steps}


    # Create strategy
    strategy = FedAvg(
        # ... other FedAvg arguments
        fraction_evaluate=0.2,
        min_evaluate_clients=2,
        min_available_clients=10,
        on_evaluate_config_fn=evaluate_config,
    )


    def server_fn(context: Context):
        num_rounds = context.run_config["num-server-rounds"]
        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)

Evaluating Local Model Updates During Training
==============================================

Model parameters can also be evaluated during training. ``Client.fit`` can return
arbitrary evaluation results as a dictionary:

.. code-block:: python

    from flwr.client import NumPyClient


    class FlowerClient(NumPyClient):
        def __init__(self, model, x_train, y_train, x_test, y_test):
            self.model = model
            self.x_train, self.y_train = x_train, y_train
            self.x_test, self.y_test = x_test, y_test

        def fit(self, parameters, config):
            """Train parameters on the locally held training set."""

            # Update local model parameters
            self.model.set_weights(parameters)

            # Train the model using hyperparameters from config
            history = self.model.fit(
                self.x_train, self.y_train, batch_size=32, epochs=2, validation_split=0.1
            )

            # Return updated model parameters and validation results
            parameters_prime = self.model.get_weights()
            num_examples_train = len(self.x_train)
            results = {
                "loss": history.history["loss"][0],
                "accuracy": history.history["accuracy"][0],
                "val_loss": history.history["val_loss"][0],
                "val_accuracy": history.history["val_accuracy"][0],
            }
            return parameters_prime, num_examples_train, results

        def evaluate(self, parameters, config):
            # ...
            pass

*******************
 Full Code Example
*******************

For a full code example that uses both centralized and federated evaluation, see the
`Advanced PyTorch Example
<https://github.com/flwrlabs/flower/tree/main/examples/advanced-pytorch>`_ (the same
approach can be applied to workloads implemented in any other framework).
