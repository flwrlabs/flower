:og:description: Upgrade seamlessly from Flower 0.x to Flower 1.0 with this guide.
.. meta::
    :description: Upgrade seamlessly from Flower 0.x to Flower 1.0 with this guide.

#######################
 Upgrade to Flower 1.0
#######################

.. note::

    This guide is for users who have already worked with Flower 0.x and want to upgrade
    to Flower 1.0. Newer versions of Flower (1.13 and later) are based on a new
    architecture and not covered in this guide. After upgrading Flower 0.x projects to
    Flower 1.0, please refer to :doc:`Upgrade to Flower 1.13
    <how-to-upgrade-to-flower-1.13>` to make your project compatible with the lastest
    version of Flower.

Flower 1.0 is here. Along with new features, Flower 1.0 provides a stable foundation for
future growth. Compared to Flower 0.19 (and other 0.x series releases), there are a few
breaking changes that make it necessary to change the code of existing 0.x-series
projects.

****************
 Install update
****************

Here's how to update an existing installation to Flower 1.0 using either pip or Poetry:

- pip: add ``-U`` when installing.

  - ``python -m pip install -U flwr`` (when using ``start_server`` and ``start_client``)
  - ``python -m pip install -U 'flwr[simulation]'`` (when using ``start_simulation``)

- Poetry: update the ``flwr`` dependency in ``pyproject.toml`` and then reinstall
  (ensure to delete ``poetry.lock`` via ``rm poetry.lock`` before running ``python -m
  poetry install``).

  - ``flwr = "^1.0.0"`` (when using ``start_server`` and ``start_client``)
  - ``flwr = { version = "^1.0.0", extras = ["simulation"] }`` (when using
    ``start_simulation``)

******************
 Required changes
******************

The following breaking changes require manual updates.

General
=======

Pass all arguments as keyword arguments (not as positional arguments). Here's an
example:

- Flower 0.19 (positional arguments): ``start_client("127.0.0.1:8080", FlowerClient())``
- Flower 1.0 (keyword arguments): ``start_client(server_address="127.0.0.1:8080",
  client=FlowerClient())``

Client
======

- Subclasses of ``NumPyClient``: change ``def get_parameters(self):``` to ``def
  get_parameters(self, config):``
- Subclasses of ``Client``: change ``def get_parameters(self):``` to ``def
  get_parameters(self, ins: GetParametersIns):``

Strategies / ``start_server`` / ``start_simulation``
====================================================

- Pass ``ServerConfig`` (instead of a dictionary) to ``start_server`` and
  ``start_simulation``. Here's an example:

  - Flower 0.19: ``start_server(..., config={"num_rounds": 3, "round_timeout": 600.0},
    ...)``
  - Flower 1.0: ``start_server(..., config=flwr.server.ServerConfig(num_rounds=3,
    round_timeout=600.0), ...)``

- Replace ``num_rounds=1`` in ``start_simulation`` with the new
  ``config=ServerConfig(...)`` (see previous item)
- Remove ``force_final_distributed_eval`` parameter from calls to ``start_server``.
  Distributed evaluation on all clients can be enabled by configuring the strategy to
  sample all clients for evaluation after the last round of training.
- Rename parameter/ndarray conversion functions:

  - ``parameters_to_weights`` --> ``parameters_to_ndarrays``
  - ``weights_to_parameters`` --> ``ndarrays_to_parameters``

- Strategy initialization: if the strategy relies on the default values for
  ``fraction_fit`` and ``fraction_evaluate``, set ``fraction_fit`` and
  ``fraction_evaluate`` manually to ``0.1``. Projects that do not manually create a
  strategy (by calling ``start_server`` or ``start_simulation`` without passing a
  strategy instance) should now manually initialize FedAvg with ``fraction_fit`` and
  ``fraction_evaluate`` set to ``0.1``.
- Rename built-in strategy parameters (e.g., ``FedAvg``):

  - ``fraction_eval`` --> ``fraction_evaluate``
  - ``min_eval_clients`` --> ``min_evaluate_clients``
  - ``eval_fn`` --> ``evaluate_fn``

- Rename ``rnd`` to ``server_round``. This impacts multiple methods and functions, for
  example, ``configure_fit``, ``aggregate_fit``, ``configure_evaluate``,
  ``aggregate_evaluate``, and ``evaluate_fn``.
- Add ``server_round`` and ``config`` to ``evaluate_fn``:

  - Flower 0.19: ``def evaluate(parameters: NDArrays) -> Optional[Tuple[float, Dict[str,
    Scalar]]]:``
  - Flower 1.0: ``def evaluate(server_round: int, parameters: NDArrays, config:
    Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:``

Custom strategies
=================

- The type of parameter ``failures`` has changed from ``List[BaseException]`` to
  ``List[Union[Tuple[ClientProxy, FitRes], BaseException]]`` (in ``aggregate_fit``) and
  ``List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]`` (in
  ``aggregate_evaluate``)
- The ``Strategy`` method ``evaluate`` now receives the current round of federated
  learning/evaluation as the first parameter:

  - Flower 0.19: ``def evaluate(self, parameters: Parameters) -> Optional[Tuple[float,
    Dict[str, Scalar]]]:``
  - Flower 1.0: ``def evaluate(self, server_round: int, parameters: Parameters) ->
    Optional[Tuple[float, Dict[str, Scalar]]]:``

***********************
 Optional improvements
***********************

Along with the necessary changes above, there are a number of potential improvements
that just became possible:

- Remove "placeholder" methods from subclasses of ``Client`` or ``NumPyClient``. If you,
  for example, use server-side evaluation, then empty placeholder implementations of
  ``evaluate`` are no longer necessary.
- Configure the round timeout via ``start_simulation``: ``start_simulation(...,
  config=flwr.server.ServerConfig(num_rounds=3, round_timeout=600.0), ...)``

**************
 Further help
**************

Most official `Flower code examples
<https://github.com/flwrlabs/flower/tree/main/hub/apps>`_ are already updated to Flower
1.0, they can serve as a reference for using the Flower 1.0 API. If there are further
questions, `join the Flower Slack <https://flower.ai/join-slack/>`_ and use the channel
``#questions``.
