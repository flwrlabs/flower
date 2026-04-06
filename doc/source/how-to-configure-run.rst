############################
 Configure a Flower App run
############################

**********************
 Using the App config
**********************

A Flower App run can be configured using the ``tool.flwr.app.config``
field of the ``pyproject.toml`` file.

The syntax used is the standard TOML syntax:

.. code:: toml

   # Start of the `pyproject.toml` skipped

   [tool.flwr.app.config]
   num-server-rounds = 3
   local-epochs = 1
   lr = 0.1
   verbose = true
   run-name = "Small Run"

Those values will then be accessible through the ``run_config``
attribute of the ``Context`` object passed to the ``client_fn`` (for the
``client_app``) and the ``server_fn`` (for the ``server_app``):

.. code:: python

   def client_fn(context: Context):
       local_epochs = context.run_config["local-epochs"]
       lr = context.run_config["lr"]
       verbose = context.run_config["verbose"]

       # Construct a client object (e.g. of type flwr.client.NumPyClient)
       return CustomClient(local_epochs, lr, verbose)


   app = ClientApp(client_fn=client_fn)

Note that accessing the `run_config` to configure your `ServerApp` can
be done in the same way as above:

.. code:: python

   def server_fn(context: Context):
       num_rounds = context.run_config["num-server-rounds"]
       name = context.run_config["run-name"]

       strategy = FedAvg()
       config = ServerConfig(num_rounds=num_rounds)

       log(INFO, "Starting %s", name)

       return ServerAppComponents(strategy=strategy, config=config)


   app = ServerApp(server_fn=server_fn)

It is also possible to use dictionnaries inside the ``pyproject.toml``,
with the different supported TOML syntaxes:

.. code:: toml

   # Start of the `pyproject.toml` skipped

   [tool.flwr.app.config]
   first-top-level.first-key = 1
   first-top-level.second-key = "value"

   # Note that this syntax is discouraged by the TOML spec:
   second-top-level = { local-epochs = 1, verbose = true }

   [tool.flwr.app.config.third-top-level]
   run-name = "Small Run"
   lr = 0.01

All of those formats will be flattened and will be usable in such a way:

.. code:: python

   def client_fn(context: Context):
       first_key = context.run_config["first-top-level.first-key"]
       second_key = context.run_config["first-top-level.second-key"]

       local_epochs = context.run_config["second-top-level.local-epochs"]
       verbose = context.run_config["second-top-level.verbose"]

       run_name = context.run_config["third-top-level.run-name"]
       lr = context.run_config["third-top-level.lr"]

       return CustomClient(local_epochs, lr, verbose)


   app = ClientApp(client_fn=client_fn)

It is also possible to use the ``flwr.common.config.unflatten_dict``
function to convert those objects back to regular dictionnaries:

.. code:: python

   from flwr.common.config import unflatten_dict

   first_top_level = unflatten_dict(context.run_config["first-top-level"])

   # first_top_level = {"first-key": 1, "second_key": "value"}

.. note::

   While we support most TOML data types, we currently don't support
   lists.

********************************
 Using the run config overrides
********************************

It is possible to temporarly override the config values set inside the
``pyproject.toml`` using the ``--run-config`` argument of the ``flwr
run`` command:

.. code:: bash

   flwr run --run-config "local-epochs=5 verbose=false run-name='Bigger Run'"

Or, with single quotes on the outside:

.. code:: bash

   flwr run --run-config 'local-epochs=5 verbose=false run-name="Bigger Run"'

.. note::

   The types are interpreted exactly as before, using the TOML syntax.

Those values will then be usable in the ``run_config`` attribute of the
``Context`` objects as explained above.

It is also possible to use this alternative syntax to pass overrides to
``flwr run``:

.. code:: bash

   flwr run --run-config "local-epochs=5" --run-config "verbose=false run-name='Bigger Run'"

Lastly, a TOML file can also be provided to the ``--run-config``
argument:

.. code:: bash

   flwr run --run-config "big_run.toml"

In this example, the ``big_run.toml`` file would look like:

.. code:: toml

   local-epochs = 5
   verbose = false
   run-name = "Bigger Run"
