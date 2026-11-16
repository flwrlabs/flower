:og:description: Create custom federated learning strategies in Flower by modifying client sampling, training, aggregation, and evaluation for enhanced flexibility and control.
.. meta::
    :description: Create custom federated learning strategies in Flower by modifying client sampling, training, aggregation, and evaluation for enhanced flexibility and control.

.. |strategy_link| replace:: ``Strategy``

.. _strategy_link: ref-api/flwr.serverapp.strategy.Strategy.html

.. |fedavg_link| replace:: ``FedAvg``

.. _fedavg_link: ref-api/flwr.serverapp.strategy.FedAvg.html

#############################
 Flower Strategy Abstraction
#############################

The strategy abstraction enables the implementation of fully custom federated learning
strategies. In Flower, a *strategy* is essentially the federated learning algorithm that
runs inside the ``ServerApp``. Strategies define how to:

- Sample clients
- Configure instructions for training and evaluation
- Aggregate updates and metrics
- Evaluate models

Flower ships with a number of built-in strategies, all following the same API described
below. You can also implement your own strategies with full access to the same
capabilities.

******************************
 The ``Strategy`` abstraction
******************************

All strategy implementations must derive from the abstract base class |strategy_link|_.
This includes both built-in strategies and third-party/custom strategies. By extending
this base class, user-defined strategies gain the exact same power and flexibility as
the built-in ones.

The ``Strategy`` base class defines a ``start`` method and requires subclasses to
implement several abstract methods:

.. code-block:: python

    class Strategy(ABC):
        """Abstract base class for server strategy implementations."""

        @abstractmethod
        def configure_train(
            self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
        ) -> Iterable[Message]:
            """Configure the next round of training."""

        @abstractmethod
        def aggregate_train(
            self,
            server_round: int,
            replies: Iterable[Message],
        ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
            """Aggregate training results from client nodes."""

        @abstractmethod
        def configure_evaluate(
            self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
        ) -> Iterable[Message]:
            """Configure the next round of evaluation."""

        @abstractmethod
        def aggregate_evaluate(
            self,
            server_round: int,
            replies: Iterable[Message],
        ) -> Optional[MetricRecord]:
            """Aggregate evaluation metrics from client nodes."""

        @abstractmethod
        def summary(self) -> None:
            """Log a summary of the strategy configuration."""

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
            should_stop: Optional[Callable[[int, Result], bool]] = None,
        ) -> Result:
            """Execute the federated learning strategy."""
            # Implementation details
            pass

*************************
 Creating a new strategy
*************************

You can customize an existing strategy (e.g., |fedavg_link|_) by overriding one or
several of its methods. For full flexibility, you can also implement a strategy from
scratch. To implement a brand new strategy, simply define a class that derives from
``Strategy`` and implement the abstract methods:

.. code-block:: python

    class SotaStrategy(Strategy):

        def configure_train(self, server_round, arrays, config, grid):
            # Your implementation here
            pass

        def aggregate_train(self, server_round, replies):
            # Your implementation here
            pass

        def configure_evaluate(self, server_round, arrays, config, grid):
            # Your implementation here
            pass

        def aggregate_evaluate(self, server_round, replies):
            # Your implementation here
            pass

        def summary(self):
            print("SotaStrategy: This is the state-of-the-art strategy!")

The ``start`` method is already implemented in the base class and typically does not
need to be overridden. It orchestrates the federated learning process by invoking the
abstract methods in sequence.

*****************************
 Understand ``start`` method
*****************************

The ``start`` method of the ``Strategy`` base class follows this workflow:

1. Call ``evaluate_fn`` (if provided) to evaluate the initial model on the ServerApp
   side.
2. Call ``configure_train`` to generate training messages for ClientApps.
3. Send training messages to ClientApps.
4. ClientApps run their ``@app.train()`` function and return training replies.
5. Call ``aggregate_train`` to aggregate the training replies.
6. Call ``configure_evaluate`` to generate evaluation messages for ClientApps.
7. Send evaluation messages to ClientApps.
8. ClientApps run their ``@app.evaluate()`` function and return evaluation replies.
9. Call ``aggregate_evaluate`` to aggregate the evaluation replies.
10. Call ``evaluate_fn`` (if provided) to evaluate the aggregated model on the ServerApp
    side.
11. Call ``should_stop`` (if provided) with the completed round number and accumulated
    ``Result``. If it returns ``True``, stop executing rounds
12. Otherwise, repeat steps 2-11 up to ``num_rounds`` times
13. Return the final ``Result``, which contains the final model and metrics history

The following diagram illustrates the flow.

.. note::

    The sequence diagram below shows the interaction between ``ServerApp``, ``Strategy``
    (inside ``ServerApp``), and ``ClientApp``. In reality, they do **not** communicate
    directly over the network—Flower infrastructure (``SuperLink`` and ``SuperNode``)
    transparently manages all communication. You can read more about it in the
    :doc:`Flower Network Communication <ref-flower-network-communication>` guide.

.. mermaid::

    sequenceDiagram
        participant SA as ServerApp
        participant ST as Strategy
        participant CA as ClientApps

        SA->>ST: start(num_rounds, ...)

        opt
            ST->>ST: evaluate_fn()
        end

        loop rounds 1..N
            Note over ST: --- Training Phase ---

            ST->>ST: configure_train()
            ST->>CA: train_messages
            CA->>CA: @app.train() callback
            CA-->>ST: train_replies
            ST->>ST: aggregate_train(train_replies)

            Note over ST: --- Evaluation Phase ---

            ST->>ST: configure_evaluate()
            ST->>CA: evaluate_messages
            CA->>CA: @app.evaluate() callback
            CA-->>ST: evaluate_replies
            ST->>ST: aggregate_evaluate(evaluate_replies)

            opt
                ST->>ST: evaluate_fn()
            end
        end

        ST-->>SA: final Result

********************************
 The ``configure_train`` method
********************************

The ``configure_train`` method is responsible for preparing the next round of training.
But what does *configure* mean in this context? It means selecting which clients should
participate in the round and deciding what instructions they should receive.

Here is the method signature:

.. code-block:: python

    @abstractmethod
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training."""

This method takes four arguments:

- ``server_round``: The current round number
- ``arrays``: The current global model parameters
- ``config``: A configuration dictionary for the round
- ``grid``: The object responsible for managing communication with clients

The return value is an iterable of ``Message`` objects, where each message contains the
instructions to be sent to a specific client. A typical implementation of
``configure_train`` will:

- Use the ``grid`` to randomly sample a subset (or all) of the available clients
- Create one ``Message`` per selected client, containing the global model parameters and
  configuration values

More advanced strategies can implement custom client selection logic by using the
capabilities of ``grid``. A client only participates in a round if ``configure_train``
generates a message for its node ID.

.. note::

    Because the return value is defined per client, strategies can easily implement
    heterogeneous configurations. For example, different clients can receive different
    models or hyperparameters, enabling highly customized training behaviors.

********************************
 The ``aggregate_train`` method
********************************

The ``aggregate_train`` method is responsible for aggregating the training results
received from the clients selected in ``configure_train``.

Here is the method signature:

.. code-block:: python

    @abstractmethod
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate training results from client nodes."""

This method takes two arguments:

- ``server_round``: The current round number
- ``replies``: An iterable of ``Message`` objects from the participating clients

It returns a tuple consisting of:

1. ``ArrayRecord``: The updated global model parameters
2. ``MetricRecord``: Aggregated training metrics (such as loss or accuracy)

If aggregation cannot be performed (e.g., if too many clients failed during the round),
the method may decide to return ``(None, None)`` instead.

.. hint::

    You can use ``Message.has_error()`` to check if a reply contains an error and decide
    how to handle it during aggregation.

***********************************
 The ``configure_evaluate`` method
***********************************

The ``configure_evaluate`` method is responsible for preparing the next round of
evaluation. Similar to ``configure_train``, this involves selecting which clients should
participate and deciding what instructions they should receive for evaluation.

Here is the method signature:

.. code-block:: python

    @abstractmethod
    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of evaluation."""

This method takes four arguments:

- ``server_round``: The current round number
- ``arrays``: The current global model parameters to be evaluated
- ``config``: A configuration dictionary for evaluation
- ``grid``: The object that manages communication with clients

The return value is an iterable of ``Message`` objects, one for each selected client.
Each message typically contains the current global model parameters and any evaluation
configuration.

A typical implementation of ``configure_evaluate`` will:

- Use ``grid`` to select a subset (or all) of the available clients
- Create one ``Message`` per selected client containing the global model and evaluation
  configuration

As with training, more advanced strategies may apply custom client selection logic or
send different evaluation configurations to different clients.

.. note::

    Because each client receives its own message, strategies can implement heterogeneous
    evaluation setups. For example, some clients might evaluate on larger test sets,
    while others might use specialized metrics.

***********************************
 The ``aggregate_evaluate`` method
***********************************

The ``aggregate_evaluate`` method is responsible for aggregating the evaluation results
received from the clients selected in ``configure_evaluate``.

Here is the method signature:

.. code-block:: python

    @abstractmethod
    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Optional[MetricRecord]:
        """Aggregate evaluation metrics from client nodes."""

This method takes two arguments:

- ``server_round``: The current round number
- ``replies``: An iterable of ``Message`` objects returned by the clients after they
  executed evaluation

It returns a single ``MetricRecord`` that represents the aggregated evaluation metrics
across all participating clients. If aggregation cannot be performed (for example, due
to excessive client failures or missing metrics), the method may return ``None``.

.. hint::

    As with training, ``Message.has_error()`` can be used to detect and handle client
    errors during aggregation.
