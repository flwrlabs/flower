##################
 Configure Flower
##################

*************************
 flower-superlink config
*************************

See `flower-superlink CLI ref
<https://flower.ai/docs/framework/ref-api-cli.html#flower-superlink>`_.

This is used to pass config values to the long-running SuperLink. For
instance, it allows you to define the address the SuperLink will be
running on, or which database file to use.

*************************
 flower-supernode config
*************************

See ``flower-supernode`` CLI ref (not yet public on `flower.ai
<http://flower.ai>`_).

This is used to pass config values to the long-running SuperNode. For
instance, it allows you to define arbitrary key-value pairs using the
``--node-config`` argument.

**************************
 flower-simulation config
**************************

See `flower-simulation CLI ref
<https://flower.ai/docs/framework/ref-api-cli.html#flower-simulation>`_.

This is used to pass config values to the long-running Simulation
Engine.

**************************
 Run config (Flower 1.10)
**************************

See `how-to configure Flower Apps guide
<https://flower.ai/docs/framework/how-to-configure-apps>`_.

This is used to pass config values to the app for one particular run.
For instance, you can dynamically change the number of rounds or the
learning rate without hardcoding them in the code.

Description
===========

In ``pyproject.toml``, you can specify a set of configuration values
that an app can react to:

.. code:: toml

   # ... [other pyproject.toml configs]

   [tool.flwr.config]
   lr = 0.01
   num_rounds = 100

   [tool.flwr.config.clientapp]
   lr = 0.01  # final key: "clientapp.lr"
   num_rounds = 100

These values are accessible to both the ``ServerApp`` and the
``ClientApp`` via ``Context`` . ``ServerApp`` example:

.. code:: python

   app = ServerApp()

   @app.main()
   def main(driver, context):
       lr = context.run_config["lr"]
       print(lr)  # Output: 0.01

Apart from specifying default values in ``pyproject.toml``, you can
override these values in two ways:

#. Override via CLI overrides: ``flwr run --run-config "lr=0.02"`` (this
   overrides ``lr`` from ``0.01`` to ``0.02``, but keeps ``num_rounds``
   at the default value of ``100``

#. Override via CLI + additional override config file: ``flwr run
   --run-config "overrides.toml"``

*******************************
 Executor config (Flower 1.10)
*******************************

This can be used to pass config values to the SuperExec executor plugin
when starting the SuperExec. Executor config provides a way to set
static configuration values that remain constant for the entire
liefetime of a SuperExec (i.e., changing them would require shutting
down the SuperExec and restarting it with different values).

For instance, you can set the address of the SuperLink with it.

Description
===========

When starting the SuperExec, we need a way to pass config values to the
executor plugin:

.. code:: bash

   flower-superexec flwr.superexec.deployment:executor \
     --executor-config 'superlink="localhost:9093" verbose=true'

We can use the same approach to configure the simulation engine:

.. code:: bash

   flower-superexec flwr.superexec.simulation:executor \
     --executor-config 'backend-name="not-ray" verbose_logging=true'

When using 3rd-party executor plugins, the config values (key names and
value types) that need be be passed on to the executor plugin are not
known in advance:

.. code:: bash

   flower-superexec nvflare.flower:executor \
     --executor-config 'nvflare-workspace-path="~/flare" nvflare-simulation-mode=false'

**********************************
 Federations config (Flower 1.10)
**********************************

This enables you to specify different federations that an app can run
on. In this case, a federation refers to a SuperExec running on the same
machine or a different machine. Federation config provides a way to tell
``flwr run`` which SuperExec (address) to connect to, which options to
send along, and whether or not to lazily start a SuperExec on the same
machine (to support ``flwr run`` without the need to start a SuperExec
beforehand).

Description
===========

In ``pyproject.toml``, you can configure different SuperExecs
(federations) that ``flwr run`` is able to connect to:

.. code:: toml

   # ... [other pyproject.toml configs]

   # ... [also pyproject.toml run config defaults]
   # [flower.config]
   # lr = 0.01

   ######################################################################
   # Federations config below 👇
   ######################################################################

   [tool.flwr.federations]
   default = "local-simulation"  # Could also be "bloodcounts" / ...

   [tool.flwr.federations.local-simulation]
   options.num-supernodes = 2

   [tool.flwr.federations.bloodcounts]
   address = "1.2.3.4:5678"  # SuperExec address
   root-certificates = "path/to/certs"
   options.num-supernodes = 2

   [tool.flwr.federations.flwrtune-llm-leaderboard]
   address = "flowertune-llm-leaderboard.federations.flower.ai:9093"
   options = { num-gpus = 8 }

   [tool.flwr.federations.nvidia]
   address = "superexec.nvidia.com:9093"
   options = { email = "info@nvidia.com", password = "flower-rocks", force = true }

A minimal version would look like this:

.. code:: toml

   # ... [other pyproject.toml configs]

   # ... [also pyproject.toml run config defaults]
   # [flower.config]
   # lr = 0.01

   ######################################################################
   # Federations config below 👇
   ######################################################################

   [flower.federations]
   default = "local"

   [flower.federations.local]
   start_lazily = true
   address = "localhost:9093"

When using ``flwr run`` to start a run, you can easily switch between
different federations:

.. code:: bash

   # Connect to the SuperExec running on 1.2.3.4:5678
   flwr run . bloodcounts

   # Connect to the SuperExec (w/ simulation engine) hosted by Flower Labs
   flwr run . flwrtune-llm-leaderboard

   # Connect to the SuperExec hosted by Nvidia (running the NVFLARE executor)
   flwr run . nvidia

.. note::

   Note that the ``[flower.federations]`` config is independent of, for
   example, ``[flower.config]``. You can copy ``[flower.federations]``
   from one project to another to run the other project on the same
   federations using ``flwr run``.
