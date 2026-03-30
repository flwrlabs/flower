:og:description: Customize federated learning in Flower with built-in strategies, callbacks, and custom server-side implementations for maximum flexibility and control.
.. meta::
    :description: Customize federated learning in Flower with built-in strategies, callbacks, and custom server-side implementations for maximum flexibility and control.

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.serverapp.ServerApp.html

.. |grid_link| replace:: ``Grid``

.. _grid_link: ref-api/flwr.serverapp.Grid.html

.. |strategy_link| replace:: ``Strategy``

.. _strategy_link: ref-api/flwr.serverapp.strategy.Strategy.html

.. |strategy_start_link| replace:: ``start``

.. _strategy_start_link: ref-api/flwr.serverapp.strategy.Strategy.html#flwr.serverapp.strategy.Strategy.start

.. |fedavg_link| replace:: ``FedAvg``

.. _fedavg_link: ref-api/flwr.serverapp.strategy.FedAvg.html

.. |fedadam_link| replace:: ``FedAdam``

.. _fedadam_link: ref-api/flwr.serverapp.strategy.FedAdam.html

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.app.Message.html

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.app.ArrayRecord.html

.. |metricrecord_link| replace:: ``MetricRecord``

.. _metricrecord_link: ref-api/flwr.app.MetricRecord.html

.. |strategy_explainer_link| replace:: Flower Strategy Abstraction

.. _strategy_explainer_link: explanation-flower-strategy-abstraction.html#understand-start-method

################
 Use strategies
################

Flower allows full customization of the learning process through the |strategy_link|_
abstraction. A number of built-in `strategies <ref-api/flwr.serverapp.strategy.html>`_
are provided in the core framework.

There are four ways to customize the way Flower orchestrates the learning process on the
server side:

- Use an existing strategy, for example, ``FedAvg``
- Customize an existing strategy with callback functions to its ``start`` method
- Customize an existing strategy by overriding one or more of its methods.
- Implement a novel strategy from scratch

.. note::

    Flower built-in strategies communicate one |arrayrecord_link|_ and one
    |metricrecord_link|_ in a ``Message`` to the ``ClientApps``. The strategies expect
    replies containing one ``MetricRecord`` and, if it's a round where ``ClientApps`` do
    local training, one ``ArrayRecord`` as well. The ``Message`` abstraction allows for
    unlimited records of any type. If you want to communicate multiple records you'd
    need to either expand an existing strategy or implement one from scratch.

**************************
 Use an existing strategy
**************************

Flower comes with a number of popular federated learning ``Strategies`` which can be
instantiated as follows as part of a simple |serverapp_link|_:

.. code-block:: python

    # Create ServerApp
    app = ServerApp()


    @app.main()
    def main(grid: Grid, context: Context) -> None:
        """Main entry point for the ServerApp."""

        # Load global model
        global_model = Net()
        arrays = ArrayRecord(global_model.state_dict())

        # Initialize FedAvg strategy with default settings
        strategy = FedAvg()

        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
        )

In the code above, instantiating ``FedAvg`` does not launch the logic built into the
strategy (i.e. sampling nodes, communicating |message_link|_, performing aggregation,
etc). In order to do so, we need to execute the |strategy_start_link|_ method.

The above ``ServerApp`` is very minimal, makes use of the default settings for
``FedAvg`` and only passes the required arguments to the ``start`` method. Let's see in
a bit more detail what options we have when instantiating strategies and when launching
it.

*************************************
 Parameterizing an existing strategy
*************************************

The constructor of strategies accepts different parameters based on, primarily, the
aggregation algorithm they implement. For example, |fedadam_link|_ accepts additional
arguments (i.e. to apply momentum during aggregation) compared to those that
|fedavg_link|_ requires. However, common to all strategies are settings to control how
nodes that run ``ClientApp`` instances get sampled. Let's take a look at this set of
arguments:

.. code-block:: python

    from flwr.serverapp.strategy import FedAvg

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_train=0.5,  # fraction of nodes to involve in a round of training
        fraction_evaluate=1.0,  # fraction of nodes to involve in a round of evaluation
        min_available_nodes=100,  # minimum connected nodes required before FL starts
    )

For most applications specifying one or all of the arguments shown above is sufficient.
A Flower strategy defined like the one above would wait for 100 nodes to be connected
before any federated stage begins. Then, 50% of the connected nodes will be involved in
a stage of federated training, followed by another stage of federated evaluation where
all connected nodes will participate. It is possible to set the ``min_train_nodes`` and
``min_evaluate_nodes`` arguments for finer control.

In addition to arguments to customize how the strategy performs sampling, we can define
at construction time which keys will be used to communicate different information
between the strategy in the ``ServerApp`` and the ``ClientApp``. Note that these keys
are used in both types of stages within the strategy ``start`` logic, i.e. federated
training and federated evaluation.

.. code-block:: python

    from flwr.serverapp.strategy import FedAvg

    # Initialize FedAvg strategy
    # Here we define our own keys instead of using the default
    strategy = FedAvg(
        arrayrecord_key="my-arrays",
        configrecord_key="super-config",
        weighted_by_key="num-batches",
    )

- ``arrayrecord_key``: the ``Message`` communicated to the ``ClientApp`` will contain an
  ``ArrayRecord`` containing the arrays of the global model under this key. By default
  the key is ``"arrays"``.
- ``configrecord_key``: the ``Message`` communicated to the ``ClientApp`` will contain a
  ``ConfigRecord`` containing config settings. By default the key is ``"config"``.
- ``weighted_by_key``: A key inside the |metricrecord_link|_ that the ``ClientApp``
  returns as part of its reply to the ``ServerApp``. The value under this key is used to
  perform weighted aggregation of ``MetricRecords`` and, after a round of federated
  training, ``ArrayRecords``. The default value is ``"num-examples"``.

With a strategy defined as in the code snippet above, the ``ClientApp`` should receive a
``Message`` with the following structure:

.. code-block:: python
    :emphasize-lines: 7,8,20

    # The content of a Message arriving to the ClientApp will have
    # the following structure and using the keys defined in the strategy
    msg = Message(
        # ....
        content=RecordDict(
            {
                "my-arrays": ArrayRecord(...),
                "super-config": ConfigRecord(...),
            }
        )
    )

    # The reply Message should contain a MetricRecord and inside it
    # an item associated with the key used to initialize the strategy
    reply_msg_content = RecordDict(
        {
            "locally-updated-params": ArrayRecord(...),
            "local-metrics": MetricRecord(
                {
                    "num-batches": N,
                    # ... Other metrics
                }
            ),
        }
    )

.. note::

    While the strategies fix the keys used to communicate the ``ArrayRecord`` and
    ``MetricRecord`` to the ``ClientApps``, the replies these send back to the
    ``ServerApp`` can use different keys. In the code snippet above we used
    ``"locally-updated-params"`` and ``"local-metrics"``. However, all ``ClientApps``
    need to use the same keys in their reply ``Messages`` otherwise the aggregation of
    replies (``ArrayRecord`` and ``MetricRecord``) cannot be performed.

Finally, the strategy constructor also allows passing two callbacks to control how the
``MetricRecords`` in the replies that ``ClientApps`` send are aggregated. Follow the
:doc:`how-to-aggregate-evaluation-results` guide for a walkthrough on how to define
these callbacks.

***************************************
 Using the strategy's ``start`` method
***************************************

As mentioned earlier, it is the ``start`` method of the strategy that launches the
federated learning process. Let's see what each argument passed to this method
represents.

.. tip::

    Check the |strategy_explainer_link|_ explainer for a deep dive into how the
    different stages implemented as part of the ``start`` method operate.

The only required arguments are the |grid_link|_ and an ``ArrayRecord``. The former is
an object that will be used to interface with the nodes running the ``ClientApp`` to
involve them in a round of train/evaluate/query or other. The latter contains the
parameters of the model we want to federate. Therefore, a minimal execution of the
``start`` method looks like this:

.. code-block:: python

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(...),
    )

In most settings, we want to customize how the ``start`` method is executed by passing
also the number of rounds to execute and, a pair of ``ConfigRecord`` objects to be sent
to the ``ClientApp`` during a step of training and evaluation respectively.

.. code-block:: python
    :emphasize-lines: 9,10,11

    # Define configs to send to ClientApp
    train_cfg = ConfigRecord({"lr": 0.1, "optim": "adam"})
    eval_cfg = ConfigRecord({"max-steps": 500, "local-checkpoint": True})

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(...),
        train_config=train_cfg,
        evaluate_config=eval_cfg,
        num_rounds=100,
    )

The ``start`` method also allows you to limit for how long the ``strategy`` will wait
for replies from the ``ClientApps`` until it proceeds with the rest of the stages. This
can be controlled with the argument ``timeout`` (which defaults to 3600s, i.e., 1h). For
example, if we want to increase the timeout to 2 hours, we would do:

.. code-block:: python
    :emphasize-lines: 12

    # Define configs to send to ClientApp
    train_cfg = ConfigRecord({"lr": 0.1, "optim": "adam"})
    eval_cfg = ConfigRecord({"max-steps": 500, "local-checkpoint": True})

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(...),
        train_config=train_cfg,
        evaluate_config=eval_cfg,
        num_rounds=100,
        timeout=7200,  # 2 hours
    )

Finally, the last argument in ``start`` is named ``evaluate_fn`` and it allows passing
to it a callback function to evaluate the aggregated model on some local data that the
``ServerApp`` might have access to. This callback is also useful if you want to save the
global model at the end of every round (or every N rounds). Let's see what the signature
of this callback is and how to use it:

.. code-block:: python

    # Callback definition. The function can have any name
    # but the arguments are fixed
    def my_callback(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""

        # Save checkpoint
        state_dict = arrays.to_torch_state_dict()
        torch.save(state_dict, f"model_at_round_{server_round}.pt")

        # eval model on local data
        model = MyModel()
        model.load_state_dict(state_dict)
        acc, loss = test(model, ...)

        # Return MetricRecord
        return MetricRecord({"acc": acc, "loss": loss})


    # Pass the callback to the start method
    strategy.start(..., evaluate_fn=my_callback)

.. tip::

    Take a look at the `quickstart-pytorch
    <https://github.com/flwrlabs/flower/blob/main/examples/quickstart-pytorch>`_ example
    on GitHub for a complete example using several of the concepts presented in this
    how-to guide.
