:og:description: Upgrade seamlessly to Flower 1.13 with this guide for transitioning your setup to the latest features and enhancements.
.. meta::
    :description: Upgrade seamlessly to Flower 1.13 with this guide for transitioning your setup to the latest features and enhancements.

########################
 Upgrade to Flower 1.13
########################

Welcome to the migration guide for updating Flower to Flower 1.13! Whether you're a
seasoned user or just getting started, this guide will help you smoothly transition your
existing setup to take advantage of the latest features and improvements in Flower 1.13.

.. note::

    This guide shows how to make pre-``1.13`` Flower code compatible with Flower 1.13
    (and later) with only minimal code changes.

Let's dive in!

..
    Generate link text as literal. Refs:
    - https://stackoverflow.com/q/71651598
    - https://github.com/jgm/pandoc/issues/3973#issuecomment-337087394

.. |clientapp_link| replace:: ``ClientApp``

.. |serverapp_link| replace:: ``ServerApp``

.. |runsim_link| replace:: ``run_simulation()``

.. |flower_superlink_link| replace:: ``flower-superlink``

.. |flower_supernode_link| replace:: ``flower-supernode``

.. |flower_architecture_link| replace:: Flower Architecture

.. |flower_how_to_run_simulations_link| replace:: How-to Run Simulations

.. _clientapp_link: ref-api/flwr.client.ClientApp.html

.. _flower_architecture_link: explanation-flower-architecture.html

.. _flower_how_to_run_simulations_link: how-to-run-simulations.html

.. _flower_superlink_link: ref-api-cli.html#flower-superlink

.. _flower_supernode_link: ref-api-cli.html#flower-supernode

.. _runsim_link: ref-api/flwr.simulation.run_simulation.html

.. _serverapp_link: ref-api/flwr.server.ServerApp.html

****************
 Install update
****************

Here's how to update an existing installation of Flower to Flower 1.13 with ``pip``:

.. code-block:: bash

    $ python -m pip install -U flwr

or if you need Flower 1.13 with simulation:

.. code-block:: bash

    $ python -m pip install -U "flwr[simulation]"

Ensure you set the following version constraint in your ``requirements.txt``

.. code-block::

    # Without simulation support
    flwr>=1.13,<2.0

    # With simulation support
    flwr[simulation]>=1.13, <2.0

or ``pyproject.toml``:

.. code-block:: toml

    # Without simulation support
    dependencies = [
        "flwr>=1.13,2.0",
    ]

    # With simulation support
    dependencies = [
        "flwr[simulation]>=1.13,2.0",
    ]

******************
 Required changes
******************

Starting with Flower 1.8, the *infrastructure* and *application layers* have been
decoupled. Flower 1.13 enforces this separation further. Among other things, this allows
you to run the exact same code in a simulation as in a real deployment.

Instead of starting a client in code via ``start_client()``, you create a
|clientapp_link|_. Instead of starting a server in code via ``start_server()``, you
create a |serverapp_link|_. Both ``ClientApp`` and ``ServerApp`` are started by the
long-running components of the server and client: the `SuperLink` and `SuperNode`,
respectively.

.. tip::

    For more details on SuperLink and SuperNode, please see the
    |flower_architecture_link|_ .

The following non-breaking changes require manual updates and allow you to run your
project both in the traditional (now deprecated) way and in the new (recommended) Flower
1.13 way:

|clientapp_link|_
=================

- Wrap your existing client with |clientapp_link|_ instead of launching it via
  ``start_client()``. Here's an example:

.. code-block:: python
    :emphasize-lines: 7,11

    from flwr.app import Context
    from flwr.client import start_client
    from flwr.clientapp import ClientApp


    # Flower 1.10 and later (recommended)
    def client_fn(context: Context):
        return FlowerClient().to_client()


    app = ClientApp(client_fn=client_fn)


    # # Flower 1.8 - 1.9 (deprecated, no longer supported)
    # def client_fn(cid: str):
    #     return FlowerClient().to_client()
    #
    #
    # app = ClientApp(client_fn=client_fn)


    # Flower 1.7 (deprecated, only for backwards-compatibility)
    if __name__ == "__main__":
        start_client(
            server_address="127.0.0.1:8080",
            client=FlowerClient().to_client(),
        )

|serverapp_link|_
=================

- Wrap your existing strategy with |serverapp_link|_ instead of starting the server via
  ``start_server()``. Here's an example:

.. code-block:: python
    :emphasize-lines: 8,14

    from flwr.app import Context
    from flwr.compat.server import ServerAppComponents
    from flwr.server import ServerConfig, start_server
    from flwr.server.strategy import FedAvg
    from flwr.serverapp import ServerApp


    # Flower 1.10 and later (recommended)
    def server_fn(context: Context):
        strategy = FedAvg()
        config = ServerConfig()
        return ServerAppComponents(config=config, strategy=strategy)


    app = ServerApp(server_fn=server_fn)


    # # Flower 1.8 - 1.9 (deprecated, no longer supported)
    # app = flwr.server.ServerApp(
    #     config=config,
    #     strategy=strategy,
    # )


    # Flower 1.7 (deprecated, only for backwards-compatibility)
    if __name__ == "__main__":
        start_server(
            server_address="0.0.0.0:8080",
            config=config,
            strategy=strategy,
        )

Deployment
==========

- In a terminal window, start the SuperLink using |flower_superlink_link|_. Then, in two
  additional terminal windows, start two SuperNodes using |flower_supernode_link|_ (2x).
  There is no need to directly run ``client.py`` and ``server.py`` as Python scripts.
- Here's an example to start the server without HTTPS (insecure mode, only for
  prototyping):

.. tip::

    For a comprehensive walk-through on how to deploy Flower using Docker, please refer
    to the :doc:`docker/index` guide.

.. code-block:: bash
    :emphasize-lines: 2,5,12

    # Start a SuperLink
    $ flower-superlink --insecure

    # In a new terminal window, start a long-running SuperNode
    $ flower-supernode \
         --insecure \
         --superlink 127.0.0.1:9092 \
         --clientappio-api-address 127.0.0.1:9094 \
         <other-args>

    # In another terminal window, start another long-running SuperNode (at least 2 SuperNodes are required)
    $ flower-supernode \
         --insecure \
         --superlink 127.0.0.1:9092 \
         --clientappio-api-address 127.0.0.1:9095 \
         <other-args>

- Here's another example to start both SuperLink and SuperNodes with HTTPS. Use the
  ``--ssl-ca-certfile``, ``--ssl-certfile``, and ``--ssl-keyfile`` command line options
  to pass paths to (CA certificate, server certificate, and server private key).

.. code-block:: bash
    :emphasize-lines: 2,8,15

    # Start a secure SuperLink
    $ flower-superlink \
        --ssl-ca-certfile <your-ca-cert-filepath> \
        --ssl-certfile <your-server-cert-filepath> \
        --ssl-keyfile <your-privatekey-filepath>

    # In a new terminal window, start a long-running SuperNode
    $ flower-supernode \
         --superlink 127.0.0.1:9092 \
         --clientappio-api-address 127.0.0.1:9094 \
         --root-certificates <your-ca-cert-filepath> \
         <other-args>

    # In another terminal window, start another long-running SuperNode (at least 2 SuperNodes are required)
    $ flower-supernode \
         --superlink 127.0.0.1:9092 \
         --clientappio-api-address 127.0.0.1:9095 \
         --root-certificates <your-ca-cert-filepath> \
         <other-args>

Simulation (CLI)
================

Wrap your existing client and strategy with |clientapp_link|_ and |serverapp_link|_,
respectively. There is no need to use ``start_simulation()`` anymore. Here's an example:

.. tip::

    For a comprehensive guide on how to setup and run Flower simulations please read the
    |flower_how_to_run_simulations_link|_ guide.

.. code-block:: python
    :emphasize-lines: 10,16,20,23,29

    from flwr.app import Context
    from flwr.clientapp import ClientApp
    from flwr.compat.server import ServerAppComponents
    from flwr.server import ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.serverapp import ServerApp
    from flwr.simulation import start_simulation


    # Regular Flower client implementation
    class FlowerClient(NumPyClient):
        # ...
        pass


    # Flower 1.10 and later (recommended)
    def client_fn(context: Context):
        return FlowerClient().to_client()


    app = ClientApp(client_fn=client_fn)


    def server_fn(context: Context):
        strategy = FedAvg(...)
        config = ServerConfig(...)
        return ServerAppComponents(strategy=strategy, config=config)


    server_app = ServerApp(server_fn=server_fn)


    # # Flower 1.8 - 1.9 (deprecated, no longer supported)
    # def client_fn(cid: str):
    #     return FlowerClient().to_client()
    #
    #
    # client_app = ClientApp(client_fn=client_fn)
    #
    #
    # server_app = ServerApp(
    #     config=config,
    #     strategy=strategy,
    # )


    # Flower 1.7 (deprecated, only for backwards-compatibility)
    if __name__ == "__main__":
        hist = start_simulation(
            num_clients=10,
            # ...
        )

Depending on your Flower version, you can run your simulation as follows:

- For Flower 1.11 and later, run ``flwr run`` in the terminal. This is the recommended
  way to start simulations, other ways are deprecated and no longer recommended.
- DEPRECATED For Flower versions between 1.8 and 1.10, run ``flower-simulation`` in the
  terminal and point to the ``server_app`` / ``client_app`` object in the code instead
  of executing the Python script. In the code snippet below, there is an example
  (assuming the ``server_app`` and ``client_app`` objects are in a ``sim.py`` module).
- DEPRECATED For Flower versions before 1.8, run the Python script directly.

.. code-block:: bash
    :emphasize-lines: 2

    # Flower 1.11 and later (recommended)
    $ flwr run


    # # Flower 1.8 - 1.10 (deprecated, no longer supported)
    # $ flower-simulation \
    #     --server-app=sim:server_app \
    #     --client-app=sim:client_app \
    #     --num-supernodes=10


    # Flower 1.7 (deprecated)
    $ python sim.py

Depending on your Flower version, you can also define the default resources as follows:

- For Flower 1.11 and later, you can edit your ``pyproject.toml`` file and then run
  ``flwr run`` in the terminal as shown in the example below.
- DEPRECATED For Flower versions between 1.8 and 1.10, you can adjust the resources for
  each |clientapp_link|_ using the ``--backend-config`` command line argument instead of
  setting the ``client_resources`` argument in ``start_simulation()``.
- DEPRECATED For Flower versions before 1.8, you need to run ``start_simulation()`` and
  pass a dictionary of the required resources to the ``client_resources`` argument.

.. code-block:: bash
    :emphasize-lines: 2,8

    # Flower 1.11 and later (recommended)
    # [file: pyproject.toml]
    [tool.flwr.federations.local-sim-gpu]
    options.num-supernodes = 10
    options.backend.client-resources.num-cpus = 2
    options.backend.client-resources.num-gpus = 0.25

    $ flwr run

    # # Flower 1.8 - 1.10 (deprecated, no longer supported)
    # $ flower-simulation \
    #     --client-app=sim:client_app \
    #     --server-app=sim:server_app \
    #     --num-supernodes=10 \
    #     --backend-config='{"client_resources": {"num_cpus": 2, "num_gpus": 0.25}}'

.. code-block:: python

    # Flower 1.7 (in `sim.py`, deprecated)
    if __name__ == "__main__":
        hist = start_simulation(
            num_clients=10, client_resources={"num_cpus": 2, "num_gpus": 0.25}, ...
        )

Simulation (Notebook)
=====================

To run your simulation from within a notebook, please consider the following examples
depending on your Flower version:

- For Flower 1.11 and later, you need to run |runsim_link|_ in your notebook instead of
  ``start_simulation()``.
- DEPRECATED For Flower versions between 1.8 and 1.10, you need to run |runsim_link|_ in
  your notebook instead of ``start_simulation()`` and configure the resources.
- DEPRECATED For Flower versions before 1.8, you need to run ``start_simulation()`` and
  pass a dictionary of the required resources to the ``client_resources`` argument.

.. tip::

    For a comprehensive guide on how to setup and run Flower simulations please read the
    |flower_how_to_run_simulations_link|_ guide.

.. code-block:: python
    :emphasize-lines: 10,12,14-17

    from flwr.app import Context
    from flwr.clientapp import ClientApp
    from flwr.serverapp import ServerApp
    from flwr.simulation import run_simulation, start_simulation


    # Flower 1.10 and later (recommended)
    # Omitted: client_fn and server_fn

    client_app = ClientApp(client_fn=client_fn)

    server_app = ServerApp(server_fn=server_fn)

    run_simulation(
        server_app=server_app,
        client_app=client_app,
    )


    # # Flower v1.8 - v1.10 (deprecated, no longer supported)
    # NUM_CLIENTS = 10  # Replace by any integer greater than zero
    # backend_config = {"client_resources": {"num_cpus": 2, "num_gpus": 0.25}}
    #
    #
    # def client_fn(cid: str):
    #     # ...
    #     return FlowerClient().to_client()
    #
    #
    # client_app = ClientApp(client_fn=client_fn)
    #
    # server_app = ServerApp(
    #     config=config,
    #     strategy=strategy,
    # )
    #
    # run_simulation(
    #     server_app=server_app,
    #     client_app=client_app,
    #     num_supernodes=NUM_CLIENTS,
    #     backend_config=backend_config,
    # )


    # Flower v1.7 (deprecated)
    NUM_CLIENTS = 10  # Replace by any integer greater than zero
    backend_config = {"client_resources": {"num_cpus": 2, "num_gpus": 0.25}}
    start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=config,
        strategy=strategy,
        client_resources=backend_config["client_resources"],
    )

**************
 Further help
**************

Most official `Flower code examples <https://flower.ai/docs/examples/>`_ are already
updated to Flower 1.13 so they can serve as a reference for using the Flower 1.13 API.
If there are further questions, `join the Flower Slack <https://flower.ai/join-slack/>`_
(and use the channel ``#questions``) or post them on `Flower Discuss
<https://discuss.flower.ai/>`_ where you can find the community posting and answering
questions.

.. admonition:: Important

    As we continuously enhance Flower at a rapid pace, we'll be periodically updating
    this guide. Please feel free to share any feedback with us!

Happy migrating! 🚀
